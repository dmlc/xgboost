#ifndef XGBOOST_GBM_GBM_H_
#define XGBOOST_GBM_GBM_H_
/*!
 * \file gbm.h
 * \brief interface of gradient booster, that learns through gradient statistics
 * \author Tianqi Chen
 */
#include <vector>
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
   * \param p_fmat feature matrix that provide access to features
   * \param info meta information about training
   * \param in_gpair address of the gradient pair statistics of the data
   * the booster may change content of gpair
   */
  virtual void DoBoost(IFMatrix *p_fmat,
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
