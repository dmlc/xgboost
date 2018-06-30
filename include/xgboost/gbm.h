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
#include <memory>
#include "./base.h"
#include "./data.h"
#include "./objective.h"
#include "./feature_map.h"
#include "../../src/common/host_device_vector.h"

namespace xgboost {
/*!
 * \brief interface of gradient boosting model.
 */
class GradientBooster {
 public:
  /*! \brief virtual destructor */
  virtual ~GradientBooster() = default;
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
   * \param in_gpair address of the gradient pair statistics of the data
   * \param obj The objective function, optional, can be nullptr when use customized version
   * the booster may change content of gpair
   */
  virtual void DoBoost(DMatrix* p_fmat,
                       HostDeviceVector<GradientPair>* in_gpair,
                       ObjFunction* obj = nullptr) = 0;

  /*!
   * \brief generate predictions for given feature matrix
   * \param dmat feature matrix
   * \param out_preds output vector to hold the predictions
   * \param ntree_limit limit the number of trees used in prediction, when it equals 0, this means
   *    we do not limit number of trees, this parameter is only valid for gbtree, but not for gblinear
   */
  virtual void PredictBatch(DMatrix* dmat,
                            HostDeviceVector<bst_float>* out_preds,
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
  virtual void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
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
                           std::vector<bst_float>* out_preds,
                           unsigned ntree_limit = 0) = 0;

  /*!
   * \brief feature contributions to individual predictions; the output will be a vector
   *         of length (nfeats + 1) * num_output_group * nsample, arranged in that order
   * \param dmat feature matrix
   * \param out_contribs output vector to hold the contributions
   * \param ntree_limit limit the number of trees used in prediction, when it equals 0, this means
   *    we do not limit number of trees
   * \param approximate use a faster (inconsistent) approximation of SHAP values
   * \param condition condition on the condition_feature (0=no, -1=cond off, 1=cond on).
   * \param condition_feature feature to condition on (i.e. fix) during calculations
   */
  virtual void PredictContribution(DMatrix* dmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit = 0, bool approximate = false,
                           int condition = 0, unsigned condition_feature = 0) = 0;

  virtual void PredictInteractionContributions(DMatrix* dmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate) = 0;

  /*!
   * \brief dump the model in the requested format
   * \param fmap feature map that may help give interpretations of feature
   * \param with_stats extra statistics while dumping model
   * \param format the format to dump the model in
   * \return a vector of dump for boosters.
   */
  virtual std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                             bool with_stats,
                                             std::string format) const = 0;
  /*!
   * \brief create a gradient booster from given name
   * \param name name of gradient booster
   * \param cache_mats The cache data matrix of the Booster.
   * \param base_margin The base margin of prediction.
   * \return The created booster.
   */
  static GradientBooster* Create(
      const std::string& name,
      const std::vector<std::shared_ptr<DMatrix> >& cache_mats,
      bst_float base_margin);
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
    : public dmlc::FunctionRegEntryBase<
  GradientBoosterReg,
  std::function<GradientBooster* (const std::vector<std::shared_ptr<DMatrix> > &cached_mats,
                                  bst_float base_margin)> > {
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
