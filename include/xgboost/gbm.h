/*!
 * Copyright by Contributors
 * \file gbm.h
 * \brief Interface of gradient booster,
 *  that learns through gradient statistics.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_GBM_H_
#define XGBOOST_GBM_H_

#include <dmlc/registry.h>
#include <vector>
#include <utility>
#include <string>
#include <functional>
#include "./base.h"
#include "./data.h"
#include "./feature_map.h"

namespace xgboost {
/*!
 * \brief interface of gradient boosting model.
 */
class GradientBooster {
 public:
  /*! \brief virtual destructor */
  virtual ~GradientBooster() {}
  /*!
   * \brief set configuration from pair iterators.
   * \param begin The beginning iterator.
   * \param end The end iterator.
   * \tparam PairIter iterator<std::pair<std::string, std::string> >
   */
  template<typename PairIter>
  inline void Configure(PairIter begin, PairIter end);
  /*!
   * \brief Set the configuration of gradient boosting.
   *  User must call configure once before InitModel and Training.
   *
   * \param cfg configurations on both training and model parameters.
   */
  virtual void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) = 0;
  /*!
   * \brief load model from stream
   * \param fi input stream.
   */
  virtual void Load(dmlc::Stream* fi) = 0;
  /*!
   * \brief save model to stream.
   * \param fo output stream
   */
  virtual void Save(dmlc::Stream* fo) const = 0;
  /*!
   * \brief reset the predict buffer size.
   *  This will invalidate all the previous cached results
   *  and recalculate from scratch
   * \param num_pbuffer The size of predict buffer.
   */
  virtual void ResetPredBuffer(size_t num_pbuffer) {}
  /*!
   * \brief whether the model allow lazy checkpoint
   * return true if model is only updated in DoBoost
   * after all Allreduce calls
   */
  virtual bool AllowLazyCheckPoint() const {
    return false;
  }
  /*!
   * \brief perform update to the model(boosting)
   * \param p_fmat feature matrix that provide access to features
   * \param buffer_offset buffer index offset of these instances, if equals -1
   *        this means we do not have buffer index allocated to the gbm
   * \param in_gpair address of the gradient pair statistics of the data
   * the booster may change content of gpair
   */
  virtual void DoBoost(DMatrix* p_fmat,
                       int64_t buffer_offset,
                       std::vector<bst_gpair>* in_gpair) = 0;
  /*!
   * \brief generate predictions for given feature matrix
   * \param dmat feature matrix
   * \param buffer_offset buffer index offset of these instances, if equals -1
   *        this means we do not have buffer index allocated to the gbm
   *  a buffer index is assigned to each instance that requires repeative prediction
   *  the size of buffer is set by convention using GradientBooster.ResetPredBuffer(size);
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction, when it equals 0, this means
   *    we do not limit number of trees, this parameter is only valid for gbtree, but not for gblinear
   */
  virtual void Predict(DMatrix* dmat,
                       int64_t buffer_offset,
                       std::vector<float>* out_preds,
                       unsigned ntree_limit = 0) = 0;
  /*!
   * \brief online prediction function, predict score for one instance at a time
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
  virtual void Predict(const SparseBatch::Inst& inst,
                       std::vector<float>* out_preds,
                       unsigned ntree_limit = 0,
                       unsigned root_index = 0) = 0;
  /*!
   * \brief predict the leaf index of each tree, the output will be nsample * ntree vector
   *        this is only valid in gbtree predictor
   * \param dmat feature matrix
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction, when it equals 0, this means
   *    we do not limit number of trees, this parameter is only valid for gbtree, but not for gblinear
   */
  virtual void PredictLeaf(DMatrix* dmat,
                           std::vector<float>* out_preds,
                           unsigned ntree_limit = 0) = 0;
  /*!
   * \brief dump the model to text format
   * \param fmap feature map that may help give interpretations of feature
   * \param option extra option of the dump model
   * \return a vector of dump for boosters.
   */
  virtual std::vector<std::string> Dump2Text(const FeatureMap& fmap, int option) const = 0;
  /*!
   * \brief create a gradient booster from given name
   * \param name name of gradient booster
   * \return The created booster.
   */
  static GradientBooster* Create(const std::string& name);
};

// implementing configure.
template<typename PairIter>
inline void GradientBooster::Configure(PairIter begin, PairIter end) {
  std::vector<std::pair<std::string, std::string> > vec(begin, end);
  this->Configure(vec);
}

/*!
 * \brief Registry entry for tree updater.
 */
struct GradientBoosterReg
    : public dmlc::FunctionRegEntryBase<GradientBoosterReg,
                                        std::function<GradientBooster* ()> > {
};

/*!
 * \brief Macro to register gradient booster.
 *
 * \code
 * // example of registering a objective ndcg@k
 * XGBOOST_REGISTER_GBM(GBTree, "gbtree")
 * .describe("Boosting tree ensembles.")
 * .set_body([]() {
 *     return new GradientBooster<TStats>();
 *   });
 * \endcode
 */
#define XGBOOST_REGISTER_GBM(UniqueId, Name)                            \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::GradientBoosterReg &          \
  __make_ ## GradientBoosterReg ## _ ## UniqueId ## __ =                \
      ::dmlc::Registry< ::xgboost::GradientBoosterReg>::Get()->__REGISTER__(Name)

}  // namespace xgboost
#endif  // XGBOOST_GBM_H_
