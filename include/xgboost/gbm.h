/*!
 * Copyright 2014-2021 by Contributors
 * \file gbm.h
 * \brief Interface of gradient booster,
 *  that learns through gradient statistics.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_GBM_H_
#define XGBOOST_GBM_H_

#include <dmlc/registry.h>
#include <dmlc/any.h>
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/model.h>

#include <vector>
#include <utility>
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>

namespace xgboost {

class Json;
class FeatureMap;
class ObjFunction;

struct GenericParameter;
struct LearnerModelParam;
struct PredictionCacheEntry;
class PredictionContainer;

/*!
 * \brief interface of gradient boosting model.
 */
class GradientBooster : public Model, public Configurable {
 protected:
  GenericParameter const* generic_param_;

 public:
  /*! \brief virtual destructor */
  ~GradientBooster() override = default;
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
   * \brief Slice a model using boosting index. The slice m:n indicates taking all trees
   *        that were fit during the boosting rounds m, (m+1), (m+2), ..., (n-1).
   * \param layer_begin Beginning of boosted tree layer used for prediction.
   * \param layer_end   End of booster layer. 0 means do not limit trees.
   * \param out         Output gradient booster
   */
  virtual void Slice(int32_t layer_begin, int32_t layer_end, int32_t step,
                     GradientBooster *out, bool* out_of_bound) const {
    LOG(FATAL) << "Slice is not supported by current booster.";
  }
  /*!
   * \brief whether the model allow lazy checkpoint
   * return true if model is only updated in DoBoost
   * after all Allreduce calls
   */
  virtual bool AllowLazyCheckPoint() const {
    return false;
  }
  /*! \brief Return number of boosted rounds.
   */
  virtual int32_t BoostedRounds() const = 0;
  /*!
   * \brief perform update to the model(boosting)
   * \param p_fmat feature matrix that provide access to features
   * \param in_gpair address of the gradient pair statistics of the data
   * \param prediction The output prediction cache entry that needs to be updated.
   * the booster may change content of gpair
   */
  virtual void DoBoost(DMatrix* p_fmat,
                       HostDeviceVector<GradientPair>* in_gpair,
                       PredictionCacheEntry*) = 0;

  /*!
   * \brief generate predictions for given feature matrix
   * \param dmat feature matrix
   * \param out_preds output vector to hold the predictions
   * \param training Whether the prediction value is used for training.  For dart booster
   *                 drop out is performed during training.
   * \param layer_begin Beginning of boosted tree layer used for prediction.
   * \param layer_end   End of booster layer. 0 means do not limit trees.
   */
  virtual void PredictBatch(DMatrix* dmat,
                            PredictionCacheEntry* out_preds,
                            bool training,
                            unsigned layer_begin,
                            unsigned layer_end) = 0;

  /*!
   * \brief Inplace prediction.
   *
   * \param           x                      A type erased data adapter.
   * \param           missing                Missing value in the data.
   * \param [in,out]  out_preds              The output preds.
   * \param           layer_begin (Optional) Beginning of boosted tree layer used for prediction.
   * \param           layer_end   (Optional) End of booster layer. 0 means do not limit trees.
   */
  virtual void InplacePredict(dmlc::any const &, std::shared_ptr<DMatrix>, float,
                              PredictionCacheEntry*,
                              uint32_t,
                              uint32_t) const {
    LOG(FATAL) << "Inplace predict is not supported by current booster.";
  }
  /*!
   * \brief online prediction function, predict score for one instance at a time
   *  NOTE: use the batch prediction interface if possible, batch prediction is usually
   *        more efficient than online prediction
   *        This function is NOT threadsafe, make sure you only call from one thread
   *
   * \param inst the instance you want to predict
   * \param out_preds output vector to hold the predictions
   * \param layer_begin Beginning of boosted tree layer used for prediction.
   * \param layer_end   End of booster layer. 0 means do not limit trees.
   * \sa Predict
   */
  virtual void PredictInstance(const SparsePage::Inst& inst,
                               std::vector<bst_float>* out_preds,
                               unsigned layer_begin, unsigned layer_end) = 0;
  /*!
   * \brief predict the leaf index of each tree, the output will be nsample * ntree vector
   *        this is only valid in gbtree predictor
   * \param dmat feature matrix
   * \param out_preds output vector to hold the predictions
   * \param layer_begin Beginning of boosted tree layer used for prediction.
   * \param layer_end   End of booster layer. 0 means do not limit trees.
   */
  virtual void PredictLeaf(DMatrix *dmat,
                           HostDeviceVector<bst_float> *out_preds,
                           unsigned layer_begin, unsigned layer_end) = 0;

  /*!
   * \brief feature contributions to individual predictions; the output will be a vector
   *         of length (nfeats + 1) * num_output_group * nsample, arranged in that order
   * \param dmat feature matrix
   * \param out_contribs output vector to hold the contributions
   * \param layer_begin Beginning of boosted tree layer used for prediction.
   * \param layer_end   End of booster layer. 0 means do not limit trees.
   * \param approximate use a faster (inconsistent) approximation of SHAP values
   * \param condition condition on the condition_feature (0=no, -1=cond off, 1=cond on).
   * \param condition_feature feature to condition on (i.e. fix) during calculations
   */
  virtual void PredictContribution(DMatrix* dmat,
                                   HostDeviceVector<bst_float>* out_contribs,
                                   unsigned layer_begin, unsigned layer_end,
                                   bool approximate = false, int condition = 0,
                                   unsigned condition_feature = 0) = 0;

  virtual void PredictInteractionContributions(
      DMatrix *dmat, HostDeviceVector<bst_float> *out_contribs,
      unsigned layer_begin, unsigned layer_end, bool approximate) = 0;

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
   * \brief Whether the current booster uses GPU.
   */
  virtual bool UseGPU() const = 0;
  /*!
   * \brief create a gradient booster from given name
   * \param name name of gradient booster
   * \param generic_param Pointer to runtime parameters
   * \param learner_model_param pointer to global model parameters
   * \return The created booster.
   */
  static GradientBooster* Create(
      const std::string& name,
      GenericParameter const* generic_param,
      LearnerModelParam const* learner_model_param);
};

/*!
 * \brief Registry entry for tree updater.
 */
struct GradientBoosterReg
    : public dmlc::FunctionRegEntryBase<
  GradientBoosterReg,
  std::function<GradientBooster* (LearnerModelParam const* learner_model_param)> > {
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
