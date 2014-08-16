#ifndef XGBOOST_LEARNER_OBJECTIVE_INL_HPP_
#define XGBOOST_LEARNER_OBJECTIVE_INL_HPP_
/*!
 * \file objective-inl.hpp
 * \brief objective function implementations
 * \author Tianqi Chen, Kailong Chen
 */
#include <vector>
#include <cmath>
#include "./objective.h"

namespace xgboost {
namespace learner {
/*! \brief defines functions to calculate some commonly used functions */
struct LossType {
  /*! \brief indicate which type we are using */
  int loss_type;
  // list of constants
  static const int kLinearSquare = 0;
  static const int kLogisticNeglik = 1;
  static const int kLogisticClassify = 2;
  static const int kLogisticRaw = 3;
  /*!
   * \brief transform the linear sum to prediction
   * \param x linear sum of boosting ensemble
   * \return transformed prediction
   */
  inline float PredTransform(float x) const {
    switch (loss_type) {
      case kLogisticRaw:
      case kLinearSquare: return x;
      case kLogisticClassify:
      case kLogisticNeglik: return 1.0f / (1.0f + expf(-x));
      default: utils::Error("unknown loss_type"); return 0.0f;
    }
  }
  /*!
   * \brief calculate first order gradient of loss, given transformed prediction
   * \param predt transformed prediction
   * \param label true label
   * \return first order gradient
   */
  inline float FirstOrderGradient(float predt, float label) const {
    switch (loss_type) {
      case kLinearSquare: return predt - label;
      case kLogisticRaw: predt = 1.0f / (1.0f + expf(-predt));
      case kLogisticClassify:
      case kLogisticNeglik: return predt - label;
      default: utils::Error("unknown loss_type"); return 0.0f;
    }
  }
  /*!
   * \brief calculate second order gradient of loss, given transformed prediction
   * \param predt transformed prediction
   * \param label true label
   * \return second order gradient
   */
  inline float SecondOrderGradient(float predt, float label) const {
    switch (loss_type) {
      case kLinearSquare: return 1.0f;
      case kLogisticRaw: predt = 1.0f / (1.0f + expf(-predt));
      case kLogisticClassify:
      case kLogisticNeglik: return predt * (1 - predt);
      default: utils::Error("unknown loss_type"); return 0.0f;
    }
  }
  /*!
   * \brief transform probability value back to margin
   */
  inline float ProbToMargin(float base_score) const {
    if (loss_type == kLogisticRaw ||
        loss_type == kLogisticClassify ||
        loss_type == kLogisticNeglik ) {
      utils::Check(base_score > 0.0f && base_score < 1.0f,
                   "base_score must be in (0,1) for logistic loss");
      base_score = -logf(1.0f / base_score - 1.0f);
    }
    return base_score;
  }
  /*! \brief get default evaluation metric for the objective */
  inline const char *DefaultEvalMetric(void) const {
    if (loss_type == kLogisticClassify) return "error";
    if (loss_type == kLogisticRaw) return "auc";
    return "rmse";
  }
};

/*! \brief objective function that only need to */
class RegLossObj : public IObjFunction{
 public:
  explicit RegLossObj(int loss_type) {
    loss.loss_type = loss_type;
    scale_pos_weight = 1.0f;
  }
  virtual ~RegLossObj(void) {}
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp("scale_pos_weight", name)) {
      scale_pos_weight = static_cast<float>(atof(val));
    }
  }
  virtual void GetGradient(const std::vector<float>& preds,
                           const MetaInfo &info,
                           int iter,
                           std::vector<bst_gpair> *out_gpair) {
    utils::Check(preds.size() == info.labels.size(),
                 "labels are not correctly provided");
    std::vector<bst_gpair> &gpair = *out_gpair;
    gpair.resize(preds.size());
    // start calculating gradient
    const unsigned ndata = static_cast<unsigned>(preds.size());
    #pragma omp parallel for schedule(static)
    for (unsigned j = 0; j < ndata; ++j) {
      float p = loss.PredTransform(preds[j]);
      float w = info.GetWeight(j);
      if (info.labels[j] == 1.0f) w *= scale_pos_weight;
      gpair[j] = bst_gpair(loss.FirstOrderGradient(p, info.labels[j]) * w,
                           loss.SecondOrderGradient(p, info.labels[j]) * w);
    }
  }
  virtual const char* DefaultEvalMetric(void) {
    return loss.DefaultEvalMetric();
  }
  virtual void PredTransform(std::vector<float> *io_preds) {
    std::vector<float> &preds = *io_preds;
    const unsigned ndata = static_cast<unsigned>(preds.size());
    #pragma omp parallel for schedule(static)
    for (unsigned j = 0; j < ndata; ++j) {
      preds[j] = loss.PredTransform(preds[j]);
    }
  }

 protected:
  float scale_pos_weight;
  LossType loss;
};
}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_OBJECTIVE_INL_HPP_
