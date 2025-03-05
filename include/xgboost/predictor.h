/**
 * Copyright 2017-2025, XGBoost Contributors
 * \file predictor.h
 * \brief Interface of predictor,
 *  performs predictions for a gradient booster.
 */
#pragma once
#include <dmlc/registry.h>    // for FunctionRegEntryBase
#include <xgboost/base.h>     // for bst_tree_t
#include <xgboost/cache.h>    // for DMatrixCache
#include <xgboost/context.h>  // for Context
#include <xgboost/context.h>
#include <xgboost/data.h>
#include <xgboost/host_device_vector.h>

#include <functional>  // for function
#include <memory>      // for shared_ptr
#include <string>
#include <vector>

// Forward declarations
namespace xgboost::gbm {
struct GBTreeModel;
}  // namespace xgboost::gbm

namespace xgboost {
/**
 * \brief Contains pointer to input matrix and associated cached predictions.
 */
struct PredictionCacheEntry {
  // A storage for caching prediction values
  HostDeviceVector<float> predictions;
  // The version of current cache, corresponding number of layers of trees
  std::uint32_t version{0};

  PredictionCacheEntry() = default;
  /**
   * \brief Update the cache entry by number of versions.
   *
   * \param v Added versions.
   */
  void Update(std::uint32_t v) { version += v; }
  void Reset() { version = 0; }
};

/**
 * \brief A container for managed prediction caches.
 */
class PredictionContainer : public DMatrixCache<PredictionCacheEntry> {
  // We cache up to 64 DMatrix for all threads
  std::size_t static constexpr DefaultSize() { return 64; }

 public:
  PredictionContainer() : DMatrixCache<PredictionCacheEntry>{DefaultSize()} {}
  std::shared_ptr<PredictionCacheEntry> Cache(std::shared_ptr<DMatrix> m, DeviceOrd device) {
    auto p_cache = this->CacheItem(m);
    if (!device.IsCPU()) {
      p_cache->predictions.SetDevice(device);
    }
    return p_cache;
  }
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
  Context const* ctx_;

 public:
  explicit Predictor(Context const* ctx) : ctx_{ctx} {}

  virtual ~Predictor() = default;

  /**
   * \brief Configure and register input matrices in prediction cache.
   *
   * \param cfg   The configuration.
   */
  virtual void Configure(Args const&);

  /**
   * \brief Initialize output prediction
   *
   * \param info Meta info for the DMatrix object used for prediction.
   * \param out_predt Prediction vector to be initialized.
   * \param model Tree model used for prediction.
   */
  virtual void InitOutPredictions(const MetaInfo& info, HostDeviceVector<float>* out_predt,
                                  const gbm::GBTreeModel& model) const;

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
                            gbm::GBTreeModel const& model, bst_tree_t tree_begin,
                            bst_tree_t tree_end = 0) const = 0;

  /**
   * \brief Inplace prediction.
   *
   * \param           p_fmat                 A proxy DMatrix that contains the data and related
   *                                         meta info.
   * \param           model                  The model to predict from.
   * \param           missing                Missing value in the data.
   * \param [in,out]  out_preds              The output preds.
   * \param           tree_begin (Optional) Beginning of boosted trees used for prediction.
   * \param           tree_end   (Optional) End of booster trees. 0 means do not limit trees.
   *
   * \return True if the data can be handled by current predictor, false otherwise.
   */
  virtual bool InplacePredict(std::shared_ptr<DMatrix> p_fmat, const gbm::GBTreeModel& model,
                              float missing, PredictionCacheEntry* out_preds,
                              bst_tree_t tree_begin = 0, bst_tree_t tree_end = 0) const = 0;

  /**
   * \brief predict the leaf index of each tree, the output will be nsample *
   * ntree vector this is only valid in gbtree predictor.
   *
   * \param [in,out]  dmat        The input feature matrix.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       Model to make predictions from.
   * \param           tree_end    (Optional) The tree end index.
   */

  virtual void PredictLeaf(DMatrix* dmat, HostDeviceVector<float>* out_preds,
                           gbm::GBTreeModel const& model, bst_tree_t tree_end = 0) const = 0;

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

  virtual void PredictContribution(DMatrix* dmat, HostDeviceVector<float>* out_contribs,
                                   gbm::GBTreeModel const& model, bst_tree_t tree_end = 0,
                                   std::vector<float> const* tree_weights = nullptr,
                                   bool approximate = false, int condition = 0,
                                   unsigned condition_feature = 0) const = 0;

  virtual void PredictInteractionContributions(DMatrix* dmat, HostDeviceVector<float>* out_contribs,
                                               gbm::GBTreeModel const& model,
                                               bst_tree_t tree_end = 0,
                                               std::vector<float> const* tree_weights = nullptr,
                                               bool approximate = false) const = 0;

  /**
   * \brief Creates a new Predictor*.
   *
   * \param name  Name of the predictor.
   * \param ctx   Pointer to runtime parameters.
   */
  static Predictor* Create(std::string const& name, Context const* ctx);
};

/*!
 * \brief Registry entry for predictor.
 */
struct PredictorReg
    : public dmlc::FunctionRegEntryBase<PredictorReg, std::function<Predictor*(Context const*)>> {};

#define XGBOOST_REGISTER_PREDICTOR(UniqueId, Name)      \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::PredictorReg& \
      __make_##PredictorReg##_##UniqueId##__ =          \
          ::dmlc::Registry<::xgboost::PredictorReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
