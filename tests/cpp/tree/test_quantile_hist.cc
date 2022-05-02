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
// <<<<<<< HEAD
// =======
// >>>>>>> fb16e1ca... partition optimizations
TEST(QuantileHist, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};

  auto cuts = common::SketchOnDMatrix(Xy.get(), 64, ctx.Threads());

  for (auto const& page : Xy->GetBatches<SparsePage>()) {
    GHistIndexMatrix gmat;
// std::cout << "gmat.Init" << std::endl;
    gmat.Init(page, {}, cuts, 64, false, 0.5, ctx.Threads());
// std::cout << "gmat.Init finished!" << std::endl;
    bst_feature_t const split_ind = 0;
    // common::ColumnMatrix column_indices;
// <<<<<<< HEAD
    // column_indices.Init(page, gmat, 0.5, ctx.Threads());
// =======
//     column_indices.Init(page, 0.5, 1);
// >>>>>>> fb16e1ca... partition optimizations
    {
// std::cout << "min_value computing..." << std::endl;
      auto min_value = gmat.cut.MinValues()[split_ind];
      RegTree tree;
      RowPartitioner partitioner{&ctx, gmat, &tree, 8, false};
      GetSplit(&tree, min_value, &candidates);
// std::cout << "min_value:" << min_value << std::endl;
// <<<<<<< HEAD
//       partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);
//       ASSERT_EQ(partitioner.Size(), 3);
//       ASSERT_EQ(partitioner[1].Size(), 0);
//       ASSERT_EQ(partitioner[2].Size(), n_samples);
//     }
//     {
//       HistRowPartitioner partitioner{n_samples, base_rowid, ctx.Threads()};
//       auto ptr = gmat.cut.Ptrs()[split_ind + 1];
//       float split_value = gmat.cut.Values().at(ptr / 2);
// =======

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

  // void UpdatePosition(GenericParameter const* ctx, GHistIndexMatrix const& gmat,
  //   std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree,
  //   int depth,
  //   NodeMaskListT* smalest_nodes_mask_ptr,
  //   const bool loss_guide,
  //   SplitFtrListT* split_conditions_,
  //   SplitIndListT* split_ind_, const size_t max_depth,
  //   NodeIdListT* child_node_ids_,
  //   bool is_left_small = true,
  //   bool check_is_left_small = false) {

// std::cout << "test::partitioner.UpdatePosition" << std::endl;
      partitioner.UpdatePosition<false, uint8_t, false, true>(&ctx, gmat, candidates, &tree,
                                                              0, &smalest_nodes_mask, false,
                                                              &split_conditions_, &split_ind_,
                                                              8, &complete_trees_depth_wise_);
// std::cout << "test::partitioner.UpdatePosition finished!" << std::endl;

      auto const & assignments = partitioner.GetNodeAssignments();
      std::vector<size_t> result(3, 0);
      size_t count = 0;
      for (auto node_id : assignments) {
        CHECK_NE(node_id, 0);
        CHECK_LT(node_id, 3);
        ++result[node_id];
        ++count;
      }
      bool is_cat = tree.GetSplitTypes()[0] == FeatureType::kCategorical;
      ASSERT_EQ(count, assignments.size());
      ASSERT_EQ(result[0], 0);
      ASSERT_EQ(result[2], assignments.size());
    }
    {
      // RowPartitioner partitioner{n_samples, base_rowid, 1};
      auto ptr = gmat.cut.Ptrs()[split_ind + 1];
      float split_value = gmat.cut.Values().at(ptr / 2);
// >>>>>>> fb16e1ca... partition optimizations
      RegTree tree;
      RowPartitioner partitioner{&ctx, gmat, &tree, 8, false};
      GetSplit(&tree, split_value, &candidates);
      auto left_nidx = tree[RegTree::kRoot].LeftChild();
// <<<<<<< HEAD
//       partitioner.UpdatePosition<false, true>(&ctx, gmat, column_indices, candidates, &tree);

//       auto elem = partitioner[left_nidx];
//       ASSERT_LT(elem.Size(), n_samples);
//       ASSERT_GT(elem.Size(), 1);
//       for (auto it = elem.begin; it != elem.end; ++it) {
//         auto value = gmat.cut.Values().at(gmat.index[*it]);
//         ASSERT_LE(value, split_value);
//       }
//       auto right_nidx = tree[RegTree::kRoot].RightChild();
//       elem = partitioner[right_nidx];
//       for (auto it = elem.begin; it != elem.end; ++it) {
//         auto value = gmat.cut.Values().at(gmat.index[*it]);
//         ASSERT_GT(value, split_value) << *it;
// =======

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

      // partitioner.UpdatePosition<false, uint8_t, false, true>(&ctx, page, candidates, &tree,
      //                                                         0, &smalest_nodes_mask, false,
      //                                                         &split_conditions_, &split_ind_,
      //                                                         8, &complete_trees_depth_wise_, &curr_level_nodes);
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
// >>>>>>> fb16e1ca... partition optimizations
      }
    }
  }
}
}  // namespace tree
}  // namespace xgboost
