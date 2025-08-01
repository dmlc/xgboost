/**
 * Copyright 2017-2025, XGBoost Contributors
 *
 * @file gbtree_model.h
 */
#ifndef XGBOOST_GBM_GBTREE_MODEL_H_
#define XGBOOST_GBM_GBTREE_MODEL_H_

#include <dmlc/io.h>
#include <dmlc/parameter.h>
#include <xgboost/context.h>
#include <xgboost/learner.h>
#include <xgboost/model.h>
#include <xgboost/parameter.h>
#include <xgboost/tree_model.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../common/threading_utils.h"
#include "../data/cat_container.h"  // for CatContainer

namespace xgboost {

class Json;

namespace gbm {
/**
 * \brief Container for all trees built (not update) for one group.
 */
using TreesOneGroup = std::vector<std::unique_ptr<RegTree>>;
/**
 * \brief Container for all trees built (not update) for one iteration.
 */
using TreesOneIter = std::vector<TreesOneGroup>;

/*! \brief model parameters */
struct GBTreeModelParam : public dmlc::Parameter<GBTreeModelParam> {
 public:
  /**
   * \brief number of trees
   */
  std::int32_t num_trees{0};
  /**
   * \brief Number of trees for a forest.
   */
  std::int32_t num_parallel_tree{1};

  GBTreeModelParam() = default;

  // declare parameters, only declare those that need to be set.
  DMLC_DECLARE_PARAMETER(GBTreeModelParam) {
    DMLC_DECLARE_FIELD(num_trees).set_lower_bound(0).set_default(0).describe(
        "Number of trees for the entire booster model.");
    DMLC_DECLARE_FIELD(num_parallel_tree)
        .set_default(1)
        .set_lower_bound(1)
        .describe(
            "Number of parallel trees constructed during each iteration."
            " This option is used to support boosted random forest.");
  }
};

struct GBTreeModel : public Model {
 public:
  explicit GBTreeModel(LearnerModelParam const* learner_model, Context const* ctx)
      : learner_model_param{learner_model}, ctx_{ctx} {}
  void Configure(const Args& cfg) {
    // initialize model parameters if not yet been initialized.
    if (trees.size() == 0) {
      param.UpdateAllowUnknown(cfg);
    }
  }

  void InitTreesToUpdate() {
    if (trees_to_update.size() == 0u) {
      for (auto& tree : trees) {
        trees_to_update.push_back(std::move(tree));
      }
      trees.clear();
      param.num_trees = 0;
      tree_info.clear();

      iteration_indptr.clear();
      iteration_indptr.push_back(0);
    }
  }

  void SaveModel(Json* p_out) const override;
  void LoadModel(Json const& p_out) override;

  [[nodiscard]] std::vector<std::string> DumpModel(const FeatureMap& fmap, bool with_stats,
                                                   int32_t n_threads, std::string format) const {
    std::vector<std::string> dump(trees.size());
    common::ParallelFor(trees.size(), n_threads,
                        [&](size_t i) { dump[i] = trees[i]->DumpModel(fmap, with_stats, format); });
    return dump;
  }
  /**
   * \brief Add trees to the model.
   *
   * \return The number of new trees.
   */
  bst_tree_t CommitModel(TreesOneIter&& new_trees);

  void CommitModelGroup(std::vector<std::unique_ptr<RegTree>>&& new_trees, bst_target_t group_idx) {
    for (auto& new_tree : new_trees) {
      trees.push_back(std::move(new_tree));
      tree_info.push_back(group_idx);
    }
    param.num_trees += static_cast<int>(new_trees.size());
  }

  [[nodiscard]] std::int32_t BoostedRounds() const {
    if (trees.empty()) {
      CHECK_EQ(iteration_indptr.size(), 1);
    }
    return static_cast<std::int32_t>(iteration_indptr.size() - 1);
  }

  // base margin
  LearnerModelParam const* learner_model_param;
  // model parameter
  GBTreeModelParam param;
  /*! \brief vector of trees stored in the model */
  std::vector<std::unique_ptr<RegTree>> trees;
  /*! \brief for the update process, a place to keep the initial trees */
  std::vector<std::unique_ptr<RegTree>> trees_to_update;
  /**
   * @brief Group index for trees.
   */
  std::vector<int> tree_info;
  /**
   * @brief Number of trees accumulated for each iteration.
   */
  std::vector<bst_tree_t> iteration_indptr{0};

  [[nodiscard]] CatContainer const* Cats() const { return this->cats_.get(); }
  [[nodiscard]] CatContainer* Cats() { return this->cats_.get(); }
  [[nodiscard]] std::shared_ptr<CatContainer> CatsShared() const { return this->cats_; }
  void Cats(std::shared_ptr<CatContainer> cats) { this->cats_ = cats; }

 private:
  /**
   * @brief Categories in the training data.
   */
  std::shared_ptr<CatContainer> cats_{std::make_shared<CatContainer>()};
  Context const* ctx_;
};
}  // namespace gbm
}  // namespace xgboost

#endif  // XGBOOST_GBM_GBTREE_MODEL_H_
