/*!
 * Copyright 2017-2021 by Contributors
 * \file predictor.h
 * \brief Interface of predictor,
 *  performs predictions for a gradient booster.
 */
#pragma once
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <xgboost/generic_parameters.h>
#include <xgboost/host_device_vector.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>

// Forward declarations
namespace xgboost {
class TreeUpdater;
namespace gbm {
struct GBTreeModel;
}  // namespace gbm
}

namespace xgboost {
/**
 * \struct  PredictionCacheEntry
 *
 * \brief Contains pointer to input matrix and associated cached predictions.
 */
struct PredictionCacheEntry {
  // A storage for caching prediction values
  HostDeviceVector<bst_float> predictions;
  // The version of current cache, corresponding number of layers of trees
  uint32_t version { 0 };
  // A weak pointer for checking whether the DMatrix object has expired.
  std::weak_ptr< DMatrix > ref;

  PredictionCacheEntry() = default;
  /* \brief Update the cache entry by number of versions.
   *
   * \param v Added versions.
   */
  void Update(uint32_t v) {
    version += v;
  }
};

/* \brief A container for managed prediction caches.
 */
class PredictionContainer {
  std::unordered_map<DMatrix *, PredictionCacheEntry> container_;
  void ClearExpiredEntries();

 public:
  PredictionContainer() = default;
  /* \brief Add a new DMatrix to the cache, at the same time this function will clear out
   *        all expired caches by checking the `std::weak_ptr`.  Caching an existing
   *        DMatrix won't renew it.
   *
   *  Passing in a `shared_ptr` is critical here.  First to create a `weak_ptr` inside the
   *  entry this shared pointer is necessary.  More importantly, the life time of this
   *  cache is tied to the shared pointer.
   *
   *  Another way to make a safe cache is create a proxy to this entry, with anther shared
   *  pointer defined inside, and pass this proxy around instead of the real entry.  But
   *  seems to be too messy.  In XGBoost, functions like `UpdateOneIter` will have
   *  (memory) safe access to the DMatrix as long as it's passed in as a `shared_ptr`.
   *
   * \param m shared pointer to the DMatrix that needs to be cached.
   * \param device Which device should the cache be allocated on.  Pass
   *               GenericParameter::kCpuId for CPU or positive integer for GPU id.
   *
   * \return the cache entry for passed in DMatrix, either an existing cache or newly
   *         created.
   */
  PredictionCacheEntry& Cache(std::shared_ptr<DMatrix> m, int32_t device);
  /* \brief Get a prediction cache entry.  This entry must be already allocated by `Cache`
   *        method.  Otherwise a dmlc::Error is thrown.
   *
   * \param m pointer to the DMatrix.
   * \return The prediction cache for passed in DMatrix.
   */
  PredictionCacheEntry& Entry(DMatrix* m);
  /* \brief Get a const reference to the underlying hash map.  Clear expired caches before
   *        returning.
   */
  decltype(container_) const& Container();
};

/**
 * \class Predictor
 *
 * \brief Performs prediction on individual training instances or batches of instances for
 *        GBTree. Prediction functions all take a GBTreeModel and a DMatrix as input and
 *        output a vector of predictions. The predictor does not modify any state of the
 *        model itself.
 */
class Predictor {
 protected:
  /*
   * \brief Runtime parameters.
   */
  GenericParameter const* generic_param_;

 public:
  explicit Predictor(GenericParameter const* generic_param) :
      generic_param_{generic_param} {}
  virtual ~Predictor() = default;

  /**
   * \brief Configure and register input matrices in prediction cache.
   *
   * \param cfg   The configuration.
   */
  virtual void Configure(const std::vector<std::pair<std::string, std::string>>&);

  /**
   * \brief Initialize output prediction
   *
   * \param info Meta info for the DMatrix object used for prediction.
   * \param out_predt Prediction vector to be initialized.
   * \param model Tree model used for prediction.
   */
  virtual void InitOutPredictions(const MetaInfo &info,
                                  HostDeviceVector<bst_float> *out_predt,
                                  const gbm::GBTreeModel &model) const = 0;

  /**
   * \brief Generate batch predictions for a given feature matrix. May use
   * cached predictions if available instead of calculating from scratch.
   *
   * \param [in,out]  dmat        Feature matrix.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       The model to predict from.
   * \param           tree_begin  The tree begin index.
   * \param           tree_end    The tree end index.
   */
  virtual void PredictBatch(DMatrix* dmat, PredictionCacheEntry* out_preds,
                            const gbm::GBTreeModel& model, uint32_t tree_begin,
                            uint32_t tree_end = 0) const = 0;

  /**
   * \brief Inplace prediction.
   * \param           x                      Type erased data adapter.
   * \param           model                  The model to predict from.
   * \param           missing                Missing value in the data.
   * \param [in,out]  out_preds              The output preds.
   * \param           tree_begin (Optional) Beginning of boosted trees used for prediction.
   * \param           tree_end   (Optional) End of booster trees. 0 means do not limit trees.
   *
   * \return True if the data can be handled by current predictor, false otherwise.
   */
  virtual bool InplacePredict(dmlc::any const &x, std::shared_ptr<DMatrix> p_m,
                              const gbm::GBTreeModel &model, float missing,
                              PredictionCacheEntry *out_preds,
                              uint32_t tree_begin = 0,
                              uint32_t tree_end = 0) const = 0;
  /**
   * \brief online prediction function, predict score for one instance at a time
   * NOTE: use the batch prediction interface if possible, batch prediction is
   * usually more efficient than online prediction This function is NOT
   * threadsafe, make sure you only call from one thread.
   *
   * \param           inst        The instance to predict.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       The model to predict from
   * \param           tree_end    (Optional) The tree end index.
   */

  virtual void PredictInstance(const SparsePage::Inst& inst,
                               std::vector<bst_float>* out_preds,
                               const gbm::GBTreeModel& model,
                               unsigned tree_end = 0) const = 0;

  /**
   * \brief predict the leaf index of each tree, the output will be nsample *
   * ntree vector this is only valid in gbtree predictor.
   *
   * \param [in,out]  dmat        The input feature matrix.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       Model to make predictions from.
   * \param           tree_end    (Optional) The tree end index.
   */

  virtual void PredictLeaf(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                           const gbm::GBTreeModel& model,
                           unsigned tree_end = 0) const = 0;

  /**
   * \brief feature contributions to individual predictions; the output will be
   * a vector of length (nfeats + 1) * num_output_group * nsample, arranged in
   * that order.
   *
   * \param [in,out]  dmat               The input feature matrix.
   * \param [in,out]  out_contribs       The output feature contribs.
   * \param           model              Model to make predictions from.
   * \param           tree_end           The tree end index.
   * \param           tree_weights       (Optional) Weights to multiply each tree by.
   * \param           approximate        Use fast approximate algorithm.
   * \param           condition          Condition on the condition_feature (0=no, -1=cond off, 1=cond on).
   * \param           condition_feature  Feature to condition on (i.e. fix) during calculations.
   */

  virtual void PredictContribution(DMatrix* dmat,
                                   HostDeviceVector<bst_float>* out_contribs,
                                   const gbm::GBTreeModel& model,
                                   unsigned tree_end = 0,
                                   std::vector<bst_float>* tree_weights = nullptr,
                                   bool approximate = false,
                                   int condition = 0,
                                   unsigned condition_feature = 0) const = 0;

  virtual void PredictInteractionContributions(DMatrix* dmat,
                                               HostDeviceVector<bst_float>* out_contribs,
                                               const gbm::GBTreeModel& model,
                                               unsigned tree_end = 0,
                                               std::vector<bst_float>* tree_weights = nullptr,
                                               bool approximate = false) const = 0;


  /**
   * \brief Creates a new Predictor*.
   *
   * \param name           Name of the predictor.
   * \param generic_param  Pointer to runtime parameters.
   */
  static Predictor* Create(
      std::string const& name, GenericParameter const* generic_param);
};

/*!
 * \brief Registry entry for predictor.
 */
struct PredictorReg
    : public dmlc::FunctionRegEntryBase<
  PredictorReg, std::function<Predictor*(GenericParameter const*)>> {};

#define XGBOOST_REGISTER_PREDICTOR(UniqueId, Name)      \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::PredictorReg& \
      __make_##PredictorReg##_##UniqueId##__ =          \
          ::dmlc::Registry<::xgboost::PredictorReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
