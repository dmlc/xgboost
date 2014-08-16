#ifndef XGBOOST_GBM_GBM_H_
#define XGBOOST_GBM_GBM_H_
/*!
 * \file gbm.h
 * \brief interface of gradient booster, that learns through gradient statistics
 * \author Tianqi Chen
 */
#include <vector>
#include "../data.h"

namespace xgboost {
/*! \brief namespace for gradient booster */
namespace gbm {
/*! 
 * \brief interface of gradient boosting model
 * \tparam FMatrix the data type updater taking
 */
template<typename FMatrix>
class IGradBooster {
 public:
  /*!
   * \brief set parameters from outside
   * \param name name of the parameter
   * \param val  value of the parameter
   */
  virtual void SetParam(const char *name, const char *val) = 0;
  /*!
   * \brief load model from stream
   * \param fi input stream
   */
  virtual void LoadModel(utils::IStream &fi) = 0;
  /*!
   * \brief save model to stream
   * \param fo output stream
   */
  virtual void SaveModel(utils::IStream &fo) const = 0;
  /*!
   * \brief initialize the model
   */
  virtual void InitModel(void) = 0;
  /*!
   * \brief peform update to the model(boosting)
   * \param gpair the gradient pair statistics of the data
   * \param fmat feature matrix that provide access to features
   * \param root_index pre-partitioned root_index of each instance,
   *   root_index.size() can be 0 which indicates that no pre-partition involved
   */
  virtual void DoBoost(const std::vector<bst_gpair> &gpair,
                       const FMatrix &fmat,
                       const std::vector<unsigned> &root_index) = 0;
  /*!
   * \brief generate predictions for given feature matrix
   * \param fmat feature matrix
   * \param buffer_offset buffer index offset of these instances, if equals -1
   *        this means we do not have buffer index allocated to the gbm
   *  a buffer index is assigned to each instance that requires repeative prediction
   *  the size of buffer is set by convention using IGradBooster.SetParam("num_pbuffer","size")
   * \param root_index pre-partitioned root_index of each instance,
   *   root_index.size() can be 0 which indicates that no pre-partition involved
   * \param out_preds output vector to hold the predictions
   */
  virtual void Predict(const FMatrix &fmat,
                       int64_t buffer_offset,
                       const std::vector<unsigned> &root_index,
                       std::vector<float> *out_preds) = 0;
  // destrcutor
  virtual ~IGradBooster(void){}
};
}  // namespace gbm
}  // namespace xgboost
#include "gbtree-inl.hpp"
namespace xgboost {
namespace gbm {
template<typename FMatrix>
inline IGradBooster<FMatrix>* CreateGradBooster(const char *name) {
  if (!strcmp("gbtree", name)) return new GBTree<FMatrix>();
  utils::Error("unknown booster type: %s", name);
  return NULL;
}
}  // namespace gbm
}  // namespace xgboost
#endif  // XGBOOST_GBM_GBM_H_
