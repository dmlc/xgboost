/*!
 * Copyright by Contributors
 * \file predictor.h
 * \brief Interface of predictor,
 *  performs predictions for a gradient booster.
 */
#pragma once
#include <xgboost/base.h>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include "../../src/gbm/gbtree_model.h"

// Forward declarations
namespace xgboost {
class DMatrix;
class TreeUpdater;
}
namespace xgboost {
namespace gbm {
struct GBTreeModel;
}
}  // namespace xgboost

namespace xgboost {

/**
 * \class Predictor
 *
 * \brief Performs prediction on individual training instances or batches of
 * instances for GBTree.
 *
 */

class Predictor {
 public:
  virtual ~Predictor() {}

  void InitCache(const std::vector<std::shared_ptr<DMatrix> > &cache);

  /**
   * \fn  virtual void Predictor::PredictBatch( DMatrix* dmat, std::vector<bst_float>* out_preds, const gbm::GBTreeModel &model, int tree_begin, unsigned ntree_limit = 0) = 0;
   *
   * \brief Generate batch predictions for a given feature matrix.
   *
   * \param [in,out]  dmat        Feature matrix.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       The model to predict from.
   * \param           tree_begin  The tree begin index.
   * \param           ntree_limit (Optional) The ntree limit. 0 means do not limit trees.
   */

  virtual void PredictBatch(
      DMatrix* dmat, std::vector<bst_float>* out_preds, const gbm::GBTreeModel &model,
      int tree_begin, unsigned ntree_limit = 0) = 0;

  /**
   * \fn  virtual void Predictor::UpdatePredictionCache( const gbm::GBTreeModel &model, std::vector<std::unique_ptr<TreeUpdater> >* updaters, int num_new_trees) = 0;
   *
   * \brief Update the internal prediction cache using newly added trees. Will use the tree updater
   *        to do this if possible.
   *
   * \param           model         The model.
   * \param [in,out]  updaters      The updater sequence for gradient boosting.
   * \param           num_new_trees Number of new trees.
   */

  virtual void UpdatePredictionCache(
      const gbm::GBTreeModel &model, std::vector<std::unique_ptr<TreeUpdater> >* updaters,
      int num_new_trees) = 0;

  /**
   * \fn  virtual void Predictor::PredictInstance( const SparseBatch::Inst& inst, std::vector<bst_float>* out_preds, const gbm::GBTreeModel& model, unsigned ntree_limit = 0, unsigned root_index = 0) = 0;
   *
   * \brief online prediction function, predict score for one instance at a time NOTE: use the batch
   *        prediction interface if possible, batch prediction is usually more efficient than online
   *        prediction This function is NOT threadsafe, make sure you only call from one thread.
   *
   * \param           inst        The instance to predict.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       The model to predict from
   * \param           ntree_limit (Optional) The ntree limit.
   * \param           root_index  (Optional) Zero-based index of the root.
   */

  virtual void PredictInstance(
    const SparseBatch::Inst& inst, std::vector<bst_float>* out_preds,
    const gbm::GBTreeModel& model, unsigned ntree_limit = 0, unsigned root_index = 0) = 0;

  /**
   * \fn  virtual void Predictor::PredictLeaf(DMatrix* dmat, std::vector<bst_float>* out_preds, const gbm::GBTreeModel& model, unsigned ntree_limit = 0) = 0;
   *
   * \brief predict the leaf index of each tree, the output will be nsample * ntree vector this is
   *        only valid in gbtree predictor.
   *
   * \param [in,out]  dmat        The input feature matrix.
   * \param [in,out]  out_preds   The output preds.
   * \param           model       Model to make predictions from.
   * \param           ntree_limit (Optional) The ntree limit.
   */

  virtual void PredictLeaf(DMatrix* dmat, std::vector<bst_float>* out_preds,
                           const gbm::GBTreeModel& model, unsigned ntree_limit = 0) = 0;

  /**
   * \fn  virtual void Predictor::PredictContribution( DMatrix* dmat, std::vector<bst_float>* out_contribs, const gbm::GBTreeModel& model, unsigned ntree_limit = 0) = 0;
   *
   * \brief feature contributions to individual predictions; the output will be a vector of length
   *        (nfeats + 1) * num_output_group * nsample, arranged in that order.
   *
   * \param [in,out]  dmat          The input feature matrix.
   * \param [in,out]  out_contribs  The output feature contribs.
   * \param           model         Model to make predictions from.
   * \param           ntree_limit   (Optional) The ntree limit.
   */

  virtual void PredictContribution(
    DMatrix* dmat, std::vector<bst_float>* out_contribs,
    const gbm::GBTreeModel& model, unsigned ntree_limit = 0) = 0;

  /**
   * \fn  static Predictor* Predictor::Create(std::string name);
   *
   * \brief Creates a new Predictor*.
   *
   */

  static Predictor* Create(std::string name);

 protected:
  struct PredictionCacheEntry {
    std::shared_ptr<DMatrix> data;
    std::vector<bst_float> predictions;
  };

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
