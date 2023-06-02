/**
 * Copyright 2014-2023 by XGBoost Contributors
 * \file objective.h
 * \brief interface of objective function used by xgboost.
 * \author Tianqi Chen, Kailong Chen
 */
#ifndef XGBOOST_OBJECTIVE_H_
#define XGBOOST_OBJECTIVE_H_

#include <dmlc/registry.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/model.h>
#include <xgboost/task.h>

#include <cstdint>  // std::int32_t
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace xgboost {

class RegTree;
struct Context;

/*! \brief interface of objective function */
class ObjFunction : public Configurable {
 protected:
  Context const* ctx_;

 public:
  static constexpr float DefaultBaseScore() { return 0.5f; }

 public:
  /*! \brief virtual destructor */
  ~ObjFunction() override = default;
  /*!
   * \brief Configure the objective with the specified parameters.
   * \param args arguments to the objective function.
   */
  virtual void Configure(const std::vector<std::pair<std::string, std::string> >& args) = 0;
  /*!
   * \brief Get gradient over each of predictions, given existing information.
   * \param preds prediction of current round
   * \param info information about labels, weights, groups in rank
   * \param iteration current iteration number.
   * \param out_gpair output of get gradient, saves gradient and second order gradient in
   */
  virtual void GetGradient(const HostDeviceVector<bst_float>& preds,
                           const MetaInfo& info,
                           int iteration,
                           HostDeviceVector<GradientPair>* out_gpair) = 0;

  /*! \return the default evaluation metric for the objective */
  virtual const char* DefaultEvalMetric() const = 0;
  /**
   * \brief Return the configuration for the default metric.
   */
  virtual Json DefaultMetricConfig() const { return Json{Null{}}; }

  // the following functions are optional, most of time default implementation is good enough
  /*!
   * \brief transform prediction values, this is only called when Prediction is called
   * \param io_preds prediction values, saves to this vector as well
   */
  virtual void PredTransform(HostDeviceVector<bst_float>*) const {}

  /*!
   * \brief transform prediction values, this is only called when Eval is called,
   *  usually it redirect to PredTransform
   * \param io_preds prediction values, saves to this vector as well
   */
  virtual void EvalTransform(HostDeviceVector<bst_float> *io_preds) {
    this->PredTransform(io_preds);
  }
  /*!
   * \brief transform probability value back to margin
   * this is used to transform user-set base_score back to margin
   * used by gradient boosting
   * \return transformed value
   */
  virtual bst_float ProbToMargin(bst_float base_score) const {
    return base_score;
  }
  /**
   * \brief Make initialize estimation of prediction.
   *
   * \param info MetaInfo that contains label.
   * \param base_score Output estimation.
   */
  virtual void InitEstimation(MetaInfo const& info, linalg::Tensor<float, 1>* base_score) const;
  /*!
   * \brief Return task of this objective.
   */
  virtual struct ObjInfo Task() const = 0;
  /**
   * \brief Return number of targets for input matrix.  Right now XGBoost supports only
   *        multi-target regression.
   */
  virtual bst_target_t Targets(MetaInfo const& info) const {
    if (info.labels.Shape(1) > 1) {
      LOG(FATAL) << "multioutput is not supported by current objective function";
    }
    return 1;
  }

  /**
   * \brief Update the leaf values after a tree is built. Needed for objectives with 0
   *        hessian.
   *
   *   Note that the leaf update is not well defined for distributed training as XGBoost
   *   computes only an average of quantile between workers. This breaks when some leaf
   *   have no sample assigned in a local worker.
   *
   * \param position The leaf index for each rows.
   * \param info MetaInfo providing labels and weights.
   * \param learning_rate The learning rate for current iteration.
   * \param prediction Model prediction after transformation.
   * \param group_idx The group index for this tree, 0 when it's not multi-target or multi-class.
   * \param p_tree Tree that needs to be updated.
   */
  virtual void UpdateTreeLeaf(HostDeviceVector<bst_node_t> const& /*position*/,
                              MetaInfo const& /*info*/, float /*learning_rate*/,
                              HostDeviceVector<float> const& /*prediction*/,
                              std::int32_t /*group_idx*/, RegTree* /*p_tree*/) const {}

  /*!
   * \brief Create an objective function according to name.
   * \param ctx  Pointer to runtime parameters.
   * \param name Name of the objective.
   */
  static ObjFunction* Create(const std::string& name, Context const* ctx);
};

/*!
 * \brief Registry entry for objective factory functions.
 */
struct ObjFunctionReg
    : public dmlc::FunctionRegEntryBase<ObjFunctionReg,
                                        std::function<ObjFunction* ()> > {
};

/*!
 * \brief Macro to register objective function.
 *
 * \code
 * // example of registering a objective
 * XGBOOST_REGISTER_OBJECTIVE(LinearRegression, "reg:squarederror")
 * .describe("Linear regression objective")
 * .set_body([]() {
 *     return new RegLossObj(LossType::kLinearSquare);
 *   });
 * \endcode
 */
#define XGBOOST_REGISTER_OBJECTIVE(UniqueId, Name)                      \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::ObjFunctionReg &              \
  __make_ ## ObjFunctionReg ## _ ## UniqueId ## __ =                    \
      ::dmlc::Registry< ::xgboost::ObjFunctionReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
#endif  // XGBOOST_OBJECTIVE_H_
