/*!
 * Copyright 2018-2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/tree_updater.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../../../src/tree/param.h"
#include "../../../src/tree/split_evaluator.h"
#include "../../../src/tree/updater_quantile_hist.h"
#include "../helpers.h"
#include "test_partitioner.h"
#include "xgboost/data.h"

namespace xgboost {
namespace tree {
TEST(QuantileHist, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};

  auto cuts = common::SketchOnDMatrix(Xy.get(), 64, ctx.Threads());

  for (auto const& page : Xy->GetBatches<SparsePage>()) {
    GHistIndexMatrix gmat;
    gmat.Init(page, {}, cuts, 64, false, 0.5, ctx.Threads());
    bst_feature_t const split_ind = 0;
    {
      auto min_value = gmat.cut.MinValues()[split_ind];
      RegTree tree;
      CommonRowPartitioner partitioner{&ctx, gmat, &tree, 8, false};
      GetSplit(&tree, min_value, &candidates);

      std::unordered_map<uint32_t, bool> smalest_nodes_mask;
      smalest_nodes_mask[2] = true;
      const bool loss_guide = false;
      std::unordered_map<uint32_t, int32_t> split_conditions_;
      std::unordered_map<uint32_t, uint64_t> split_ind_;
      split_ind_[0] = split_ind;
      const size_t max_depth = 8;
      std::vector<uint16_t> complete_trees_depth_wise_(3, 0);
      complete_trees_depth_wise_[0] = 1;
      complete_trees_depth_wise_[1] = 2;
      std::unordered_map<uint32_t, uint16_t> curr_level_nodes;
      curr_level_nodes[0] = 1;
      curr_level_nodes[1] = 2;
      partitioner.UpdatePosition<false, uint8_t, false, true>(&ctx, gmat, candidates, &tree,
                                                              0, &smalest_nodes_mask, false,
                                                              &split_conditions_, &split_ind_,
                                                              8, &complete_trees_depth_wise_);

      auto const & assignments = partitioner.GetNodeAssignments();
      std::vector<size_t> result(3, 0);
      size_t count = 0;
      for (auto node_id : assignments) {
        CHECK_NE(node_id, 0);
        CHECK_LT(node_id, 3);
        ++result[node_id];
        ++count;
      }
      ASSERT_EQ(count, assignments.size());
      ASSERT_EQ(result[0], 0);
      ASSERT_EQ(result[2], assignments.size());
    }
    {
      auto ptr = gmat.cut.Ptrs()[split_ind + 1];
      float split_value = gmat.cut.Values().at(ptr / 2);
      RegTree tree;
      CommonRowPartitioner partitioner{&ctx, gmat, &tree, 8, false};
      GetSplit(&tree, split_value, &candidates);

      std::unordered_map<uint32_t, bool> smalest_nodes_mask;
      smalest_nodes_mask[2] = true;
      const bool loss_guide = false;
      std::unordered_map<uint32_t, int32_t> split_conditions_;
      std::unordered_map<uint32_t, uint64_t> split_ind_;
      split_ind_[0] = split_ind;
      const size_t max_depth = 8;
      std::vector<uint16_t> complete_trees_depth_wise_(3, 0);
      complete_trees_depth_wise_[0] = 1;
      complete_trees_depth_wise_[1] = 2;
      std::unordered_map<uint32_t, uint16_t> curr_level_nodes;
      curr_level_nodes[0] = 1;
      curr_level_nodes[1] = 2;
      partitioner.UpdatePosition<false, uint8_t, false, true>(&ctx, gmat, candidates, &tree,
                                                              0, &smalest_nodes_mask, false,
                                                              &split_conditions_, &split_ind_,
                                                              8, &complete_trees_depth_wise_);
      auto const & assignments = partitioner.GetNodeAssignments();
      size_t it = 0;
      for (auto node_id : assignments) {
        CHECK_NE(node_id, 0);
        CHECK_LT(node_id, 3);
        if (node_id == 1) {
          auto value = gmat.cut.Values().at(gmat.index[it]);
          ASSERT_LE(value, split_value);
        } else if (node_id == 2) {
          auto value = gmat.cut.Values().at(gmat.index[it]);
          ASSERT_GT(value, split_value);
        } else {
          ASSERT_EQ(1,0);
        }
        ++it;
      }
    }
  }
}
}  // namespace tree
}  // namespace xgboost
