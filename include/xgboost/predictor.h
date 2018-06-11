/*!
 * Copyright by Contributors
 * \file predictor.h
 * \brief Interface of predictor,
 *  performs predictions for a gradient booster.
 */
#pragma once
#include <xgboost/base.h>
#include <xgboost/data.h>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../../src/gbm/gbtree_model.h"
#include "../../src/common/host_device_vector.h"

// Forward declarations
namespace xgboost {
class TreeUpdater;
}

namespace xgboost {

/**
 * \class Predictor
 *
 * \brief Performs prediction on individual training instances or batches of
 * instances for GBTree. The predictor also manages a prediction cache
 * associated with input matrices. If possible, it will use previously
 * calculated predictions instead of calculating new predictions.
 *        Prediction functions all take a GBTreeModel and a DMatrix as input and
 * output a vector of predictions. The predictor does not modify any state of
 * the model itself.
 */

class Predictor {
 public:
  virtual ~Predictor() = default;

  /**
   * \fn  virtual void Predictor::Init(const std::vector<std::pair<std::string,
   * std::string> >&cfg ,const std::vector<std::shared_ptr<DMatrix> > &cache);
   *
   * \brief Configure and register input matrices in prediction cache.
   *
   * \param cfg   The configuration.
   * \param cache Vector of DMatrix's to be used in prediction.
   */

  virtual void Init(const std::vector<std::pair<std::string, std::string>>& cfg,
                    const std::vector<std::shared_ptr<DMatrix>>& cache);

  /**
   * \brief Generate batch predictions for a given feature matrix. May use
   * cached predictions if available instead of calculating from scratch.
   *
   * \param [in,out]  dmat        Feature matrix.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       The model to predict from.
   * \param           tree_begin  The tree begin index.
   * \param           ntree_limit (Optional) The ntree limit. 0 means do not
   * limit trees.
   */

  virtual void PredictBatch(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                            const gbm::GBTreeModel& model, int tree_begin,
                            unsigned ntree_limit = 0) = 0;

  /**
   * \fn  virtual void Predictor::UpdatePredictionCache( const gbm::GBTreeModel
   * &model, std::vector<std::unique_ptr<TreeUpdater> >* updaters, int
   * num_new_trees) = 0;
   *
   * \brief Update the internal prediction cache using newly added trees. Will
   * use the tree updater to do this if possible. Should be called as a part of
   * the tree boosting process to facilitate the look up of predictions
   * at a later time.
   *
   * \param           model         The model.
   * \param [in,out]  updaters      The updater sequence for gradient boosting.
   * \param           num_new_trees Number of new trees.
   */

  virtual void UpdatePredictionCache(
      const gbm::GBTreeModel& model,
      std::vector<std::unique_ptr<TreeUpdater>>* updaters,
      int num_new_trees) = 0;

  /**
   * \fn  virtual void Predictor::PredictInstance( const SparsePage::Inst&
   * inst, std::vector<bst_float>* out_preds, const gbm::GBTreeModel& model,
   * unsigned ntree_limit = 0, unsigned root_index = 0) = 0;
   *
   * \brief online prediction function, predict score for one instance at a time
   * NOTE: use the batch prediction interface if possible, batch prediction is
   * usually more efficient than online prediction This function is NOT
   * threadsafe, make sure you only call from one thread.
   *
   * \param           inst        The instance to predict.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       The model to predict from
   * \param           ntree_limit (Optional) The ntree limit.
   * \param           root_index  (Optional) Zero-based index of the root.
   */

  virtual void PredictInstance(const SparsePage::Inst& inst,
                               std::vector<bst_float>* out_preds,
                               const gbm::GBTreeModel& model,
                               unsigned ntree_limit = 0,
                               unsigned root_index = 0) = 0;

  /**
   * \fn  virtual void Predictor::PredictLeaf(DMatrix* dmat,
   * std::vector<bst_float>* out_preds, const gbm::GBTreeModel& model, unsigned
   * ntree_limit = 0) = 0;
   *
   * \brief predict the leaf index of each tree, the output will be nsample *
   * ntree vector this is only valid in gbtree predictor.
   *
   * \param [in,out]  dmat        The input feature matrix.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       Model to make predictions from.
   * \param           ntree_limit (Optional) The ntree limit.
   */

  virtual void PredictLeaf(DMatrix* dmat, std::vector<bst_float>* out_preds,
                           const gbm::GBTreeModel& model,
                           unsigned ntree_limit = 0) = 0;

  /**
   * \fn  virtual void Predictor::PredictContribution( DMatrix* dmat,
   * std::vector<bst_float>* out_contribs, const gbm::GBTreeModel& model,
   * unsigned ntree_limit = 0) = 0;
   *
   * \brief feature contributions to individual predictions; the output will be
   * a vector of length (nfeats + 1) * num_output_group * nsample, arranged in
   * that order.
   *
   * \param [in,out]  dmat               The input feature matrix.
   * \param [in,out]  out_contribs       The output feature contribs.
   * \param           model              Model to make predictions from.
   * \param           ntree_limit        (Optional) The ntree limit.
   * \param           approximate        Use fast approximate algorithm.
   * \param           condition          Condition on the condition_feature (0=no, -1=cond off, 1=cond on).
   * \param           condition_feature  Feature to condition on (i.e. fix) during calculations.
   */

  virtual void PredictContribution(DMatrix* dmat,
                                   std::vector<bst_float>* out_contribs,
                                   const gbm::GBTreeModel& model,
                                   unsigned ntree_limit = 0,
                                   bool approximate = false,
                                   int condition = 0,
                                   unsigned condition_feature = 0) = 0;

  virtual void PredictInteractionContributions(DMatrix* dmat,
                                   std::vector<bst_float>* out_contribs,
                                   const gbm::GBTreeModel& model,
                                   unsigned ntree_limit = 0,
                                   bool approximate = false) = 0;

  /**
   * \fn  static Predictor* Predictor::Create(std::string name);
   *
   * \brief Creates a new Predictor*.
   *
   */

  static Predictor* Create(std::string name);

 protected:
  /**
   * \struct  PredictionCacheEntry
   *
   * \brief Contains pointer to input matrix and associated cached predictions.
   */
  struct PredictionCacheEntry {
    std::shared_ptr<DMatrix> data;
    HostDeviceVector<bst_float> predictions;
  };

  /**
   * \brief Map of matrices and associated cached predictions to facilitate
   * storing and looking up predictions.
   */

  std::unordered_map<DMatrix*, PredictionCacheEntry> cache_;
};

/*!
 * \brief Registry entry for predictor.
 */
struct PredictorReg
    : public dmlc::FunctionRegEntryBase<PredictorReg,
                                        std::function<Predictor*()>> {};

#define XGBOOST_REGISTER_PREDICTOR(UniqueId, Name)      \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::PredictorReg& \
      __make_##PredictorReg##_##UniqueId##__ =          \
          ::dmlc::Registry<::xgboost::PredictorReg>::Get()->__REGISTER__(Name)
}  // namespace xgboost
