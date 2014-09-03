#ifndef XGBOOST_LEARNER_OBJECTIVE_H_
#define XGBOOST_LEARNER_OBJECTIVE_H_
/*!
 * \file objective.h
 * \brief interface of objective function used for gradient boosting
 * \author Tianqi Chen, Kailong Chen
 */
#include "dmatrix.h"

namespace xgboost {
namespace learner {
/*! \brief interface of objective function */
class IObjFunction{
 public:
  /*! \brief virtual destructor */
  virtual ~IObjFunction(void){}
  /*!
   * \brief set parameters from outside
   * \param name name of the parameter
   * \param val value of the parameter
   */
  virtual void SetParam(const char *name, const char *val) = 0;  
  /*!
   * \brief get gradient over each of predictions, given existing information
   * \param preds prediction of current round
   * \param info information about labels, weights, groups in rank
   * \param iter current iteration number
   * \param out_gpair output of get gradient, saves gradient and second order gradient in
   */
  virtual void GetGradient(const std::vector<float> &preds,
                           const MetaInfo &info,
                           int iter,
                           std::vector<bst_gpair> *out_gpair) = 0;
  /*! \return the default evaluation metric for the objective */
  virtual const char* DefaultEvalMetric(void) const = 0;
  // the following functions are optional, most of time default implementation is good enough
  /*!
   * \brief transform prediction values, this is only called when Prediction is called
   * \param io_preds prediction values, saves to this vector as well
   */
  virtual void PredTransform(std::vector<float> *io_preds){}
  /*!
   * \brief transform prediction values, this is only called when Eval is called, 
   *  usually it redirect to PredTransform
   * \param io_preds prediction values, saves to this vector as well
   */
  virtual void EvalTransform(std::vector<float> *io_preds) {
    this->PredTransform(io_preds);
  }
  /*!
   * \brief transform probability value back to margin
   * this is used to transform user-set base_score back to margin 
   * used by gradient boosting
   * \return transformed value
   */
  virtual float ProbToMargin(float base_score) const {
    return base_score;
  }
};
}  // namespace learner
}  // namespace xgboost

// this are implementations of objective functions
#include "objective-inl.hpp"
// factory function
namespace xgboost {
namespace learner {
/*! \brief factory funciton to create objective function by name */
inline IObjFunction* CreateObjFunction(const char *name) {
  using namespace std;
  if (!strcmp("reg:linear", name)) return new RegLossObj(LossType::kLinearSquare);
  if (!strcmp("reg:logistic", name)) return new RegLossObj(LossType::kLogisticNeglik);
  if (!strcmp("binary:logistic", name)) return new RegLossObj(LossType::kLogisticClassify);
  if (!strcmp("binary:logitraw", name)) return new RegLossObj(LossType::kLogisticRaw);
  if (!strcmp("multi:softmax", name)) return new SoftmaxMultiClassObj(0);
  if (!strcmp("multi:softprob", name)) return new SoftmaxMultiClassObj(1);
  if (!strcmp("rank:pairwise", name )) return new PairwiseRankObj();
  if (!strcmp("rank:ndcg", name)) return new LambdaRankObjNDCG();
  if (!strcmp("rank:map", name)) return new LambdaRankObjMAP();  
  utils::Error("unknown objective function type: %s", name);
  return NULL;
}
}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_OBJECTIVE_H_
