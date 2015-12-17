/*!
 * Copyright by Contributors
 * \file gbm.h
 * \brief interface of gradient booster, that learns through gradient statistics
 * \author Tianqi Chen
 */
#ifndef XGBOOST_GBM_GBM_H_
#define XGBOOST_GBM_GBM_H_

#include <vector>
#include <string>
#include "../data.h"
#include "../utils/io.h"
#include "../utils/fmap.h"

namespace xgboost {
/*! \brief namespace for gradient booster */
namespace gbm {
/*!
 * \brief interface of gradient boosting model
 */
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
   * \param with_pbuffer whether the incoming data contains pbuffer
   */
  virtual void LoadModel(utils::IStream &fi, bool with_pbuffer) = 0; // NOLINT(*)
  /*!
   * \brief save model to stream
   * \param fo output stream
   * \param with_pbuffer whether save out pbuffer
   */
  virtual void SaveModel(utils::IStream &fo, bool with_pbuffer) const = 0; // NOLINT(*)
  /*!
   * \brief initialize the model
   */
  virtual void InitModel(void) = 0;
  /*!
   * \brief reset the predict buffer
   * this will invalidate all the previous cached results
   * and recalculate from scratch
   */
  virtual void ResetPredBuffer(size_t num_pbuffer) {}
  /*!
   * \brief whether the model allow lazy checkpoint
   * return true if model is only updated in DoBoost
   * after all Allreduce calls
   */
  virtual bool AllowLazyCheckPoint(void) const {
    return false;
  }
  /*!
   * \brief peform update to the model(boosting)
   * \param p_fmat feature matrix that provide access to features
   * \param buffer_offset buffer index offset of these instances, if equals -1
   *        this means we do not have buffer index allocated to the gbm
   * \param info meta information about training
   * \param in_gpair address of the gradient pair statistics of the data
   * the booster may change content of gpair
   */
  virtual void DoBoost(IFMatrix *p_fmat,
                       int64_t buffer_offset,
                       const BoosterInfo &info,
                       std::vector<bst_gpair> *in_gpair) = 0;
  /*!
   * \brief generate predictions for given feature matrix
   * \param p_fmat feature matrix
   * \param buffer_offset buffer index offset of these instances, if equals -1
   *        this means we do not have buffer index allocated to the gbm
   *  a buffer index is assigned to each instance that requires repeative prediction
   *  the size of buffer is set by convention using IGradBooster.SetParam("num_pbuffer","size")
   * \param info extra side information that may be needed for prediction
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction, when it equals 0, this means
   *    we do not limit number of trees, this parameter is only valid for gbtree, but not for gblinear
   */
  virtual void Predict(IFMatrix *p_fmat,
                       int64_t buffer_offset,
                       const BoosterInfo &info,
                       std::vector<float> *out_preds,
                       unsigned ntree_limit = 0) = 0;
  /*!
   * \brief online prediction funciton, predict score for one instance at a time
   *  NOTE: use the batch prediction interface if possible, batch prediction is usually
   *        more efficient than online prediction
   *        This function is NOT threadsafe, make sure you only call from one thread
   *
   * \param inst the instance you want to predict
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction
   * \param root_index the root index
   * \sa Predict
   */
  virtual void Predict(const SparseBatch::Inst &inst,
                       std::vector<float> *out_preds,
                       unsigned ntree_limit = 0,
                       unsigned root_index = 0)  = 0;
  /*!
   * \brief predict the leaf index of each tree, the output will be nsample * ntree vector
   *        this is only valid in gbtree predictor
   * \param p_fmat feature matrix
   * \param info extra side information that may be needed for prediction
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction, when it equals 0, this means
   *    we do not limit number of trees, this parameter is only valid for gbtree, but not for gblinear
   */
  virtual void PredictLeaf(IFMatrix *p_fmat,
                           const BoosterInfo &info,
                           std::vector<float> *out_preds,
                           unsigned ntree_limit = 0) = 0;
  /*!
   * \brief dump the model in text format
   * \param fmap feature map that may help give interpretations of feature
   * \param option extra option of the dumo model
   * \return a vector of dump for boosters
   */
  virtual std::vector<std::string> DumpModel(const utils::FeatMap& fmap, int option) = 0;
  // destrcutor
  virtual ~IGradBooster(void){}
};
/*!
 * \breif create a gradient booster from given name
 * \param name name of gradient booster
 */
IGradBooster* CreateGradBooster(const char *name);
}  // namespace gbm
}  // namespace xgboost
#endif  // XGBOOST_GBM_GBM_H_
