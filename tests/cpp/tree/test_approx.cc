/*!
 * Copyright 2021-2022, XGBoost contributors.
 */
#include <gtest/gtest.h>

#include "../../../src/tree/updater_approx.h"
#include "../helpers.h"
#include "test_partitioner.h"

namespace xgboost {
namespace tree {
TEST(Approx, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});
  std::unordered_map<uint32_t, CPUExpandEntry> candidates;
  std::vector<CPUExpandEntry> candidates_vec(1, {0, 0, 0.4});
  candidates[0] = {0, 0, 0.4};

  auto grad = GenerateRandomGradients(n_samples);
  std::vector<float> hess(grad.Size());
  std::transform(grad.HostVector().cbegin(), grad.HostVector().cend(), hess.begin(),
                 [](auto gpair) { return gpair.GetHess(); });
  auto const& spage = *(Xy->GetBatches<SparsePage>().begin());

  for (auto const& page : Xy->GetBatches<GHistIndexMatrix>({64, hess, true})) {
    bst_feature_t const split_ind = 0;
    {
      auto min_value = page.cut.MinValues()[split_ind];
      RegTree tree;
      tree.ExpandNode(
          /*nid=*/0, /*split_index=*/0, /*split_value=*/min_value,
          /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          /*left_sum=*/0.0f,
          /*right_sum=*/0.0f);
      // common::ColumnMatrix column_matrix;
      // column_matrix.Init(spage, page, 1, 1);
      RowPartitioner partitioner(&ctx, page, &tree, 2, false);
      candidates[0].split.split_value = min_value;
      candidates[0].split.sindex = 0;
      candidates[0].split.sindex |= (1U << 31);
      std::unordered_map<uint32_t, bool> mask;//(1 << 4, false);
      std::vector<uint16_t> cnodes;
      cnodes.resize(1 << (4), 0);
      cnodes[1] = 1;
      candidates_vec[0].split.split_value = min_value;
      candidates_vec[0].split.sindex = 0;
      candidates_vec[0].split.sindex |= (1U << 31);
      std::unordered_map<uint32_t, int32_t> split_conditions_;
      std::unordered_map<uint32_t, uint64_t> split_ind_;
      partitioner.UpdatePosition<false, uint8_t, false, false>(&ctx, page,
        candidates_vec,
        &tree, 0, &mask, false, &split_conditions_, &split_ind_, 2, &cnodes,
        true, false);

      auto const & assignments = partitioner.GetNodeAssignments();
      std::vector<size_t> result(3, 0);
      for (auto node_id : assignments) {
        CHECK_LE(node_id, 2);
        ++result[node_id];
      }
      ASSERT_EQ(result[1], 0);
      ASSERT_EQ(result[2], n_samples);
    }
    {
      auto ptr = page.cut.Ptrs()[split_ind + 1];
      float split_value = page.cut.Values().at(ptr / 2);
      RegTree tree;
// <<<<<<< HEAD
//       GetSplit(&tree, split_value, &candidates);
//       partitioner.UpdatePosition(&ctx, page, candidates, &tree);

//       auto left_nidx = tree[RegTree::kRoot].LeftChild();
//       auto elem = partitioner[left_nidx];
//       ASSERT_LT(elem.Size(), n_samples);
//       ASSERT_GT(elem.Size(), 1);
//       for (auto it = elem.begin; it != elem.end; ++it) {
//         auto value = page.cut.Values().at(page.index[*it]);
//         ASSERT_LE(value, split_value);
//       }

//       auto right_nidx = tree[RegTree::kRoot].RightChild();
//       elem = partitioner[right_nidx];
//       for (auto it = elem.begin; it != elem.end; ++it) {
//         auto value = page.cut.Values().at(page.index[*it]);
//         ASSERT_GT(value, split_value) << *it;
// =======

      tree.ExpandNode(
          /*nid=*/RegTree::kRoot, /*split_index=*/split_ind,
          /*split_value=*/split_value,
          /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          /*left_sum=*/0.0f,
          /*right_sum=*/0.0f);
      // common::ColumnMatrix column_matrix;
      // column_matrix.Init(page, 1, 1);
      RowPartitioner partitioner(&ctx, page, &tree, 2, false);
      candidates[0].split.split_value = split_value;
      candidates[0].split.sindex = 0;
      candidates[0].split.sindex |= (1U << 31);
      std::unordered_map<uint32_t, bool> mask;//(1 << 4, false);
      std::vector<uint16_t> cnodes;
      cnodes.resize(1 << (4), 0);
      cnodes[1] = 1;
      candidates_vec[0].split.split_value = split_value;
      candidates_vec[0].split.sindex = 0;
      candidates_vec[0].split.sindex |= (1U << 31);
      std::unordered_map<uint32_t, int32_t> split_conditions_;
      std::unordered_map<uint32_t, uint64_t> split_ind_;
      partitioner.UpdatePosition<false, uint8_t, false, false>(&ctx, page,
        candidates_vec,
        &tree, 0, &mask, false, &split_conditions_, &split_ind_, 2, &cnodes,
        true, false);

      auto const & assignments = partitioner.GetNodeAssignments();
      size_t row_id = 0;
      for (auto node_id : assignments) {
        if (node_id == 1) { /* left child */
          auto value = page.cut.Values().at(page.index[row_id++]);
          ASSERT_LE(value, split_value);
        } else {            /* right child */
          auto value = page.cut.Values().at(page.index[row_id++]);
          ASSERT_GT(value, split_value);          
        }
// >>>>>>> 0755d8b2... partition optimizations
      }
    }
  }
}
// namespace {
// void TestLeafPartition(size_t n_samples) {
//   size_t const n_features = 2, base_rowid = 0;
//   common::RowSetCollection row_set;
//   ApproxRowPartitioner partitioner{n_samples, base_rowid};

//   auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
//   GenericParameter ctx;
//   std::vector<CPUExpandEntry> candidates{{0, 0, 0.4}};
//   RegTree tree;
//   std::vector<float> hess(n_samples, 0);
//   // emulate sampling
//   auto not_sampled = [](size_t i) {
//     size_t const kSampleFactor{3};
//     return i % kSampleFactor != 0;
//   };
//   size_t n{0};
//   for (size_t i = 0; i < hess.size(); ++i) {
//     if (not_sampled(i)) {
//       hess[i] = 1.0f;
//       ++n;
//     }
//   }

//   std::vector<size_t> h_nptr;
//   float split_value{0};
//   for (auto const& page : Xy->GetBatches<GHistIndexMatrix>({Context::kCpuId, 64})) {
//     bst_feature_t const split_ind = 0;
//     auto ptr = page.cut.Ptrs()[split_ind + 1];
//     split_value = page.cut.Values().at(ptr / 2);
//     GetSplit(&tree, split_value, &candidates);
//     partitioner.UpdatePosition(&ctx, page, candidates, &tree);
//     std::vector<bst_node_t> position;
//     partitioner.LeafPartition(&ctx, tree, hess, &position);
//     std::sort(position.begin(), position.end());
//     size_t beg = std::distance(
//         position.begin(),
//         std::find_if(position.begin(), position.end(), [&](bst_node_t nidx) { return nidx >= 0; }));
//     std::vector<size_t> nptr;
//     common::RunLengthEncode(position.cbegin() + beg, position.cend(), &nptr);
//     std::transform(nptr.begin(), nptr.end(), nptr.begin(), [&](size_t x) { return x + beg; });
//     auto n_uniques = std::unique(position.begin() + beg, position.end()) - (position.begin() + beg);
//     ASSERT_EQ(nptr.size(), n_uniques + 1);
//     ASSERT_EQ(nptr[0], beg);
//     ASSERT_EQ(nptr.back(), n_samples);

//     h_nptr = nptr;
//   }

//   if (h_nptr.front() == n_samples) {
//     return;
//   }

//   ASSERT_GE(h_nptr.size(), 2);

//   for (auto const& page : Xy->GetBatches<SparsePage>()) {
//     auto batch = page.GetView();
//     size_t left{0};
//     for (size_t i = 0; i < batch.Size(); ++i) {
//       if (not_sampled(i) && batch[i].front().fvalue < split_value) {
//         left++;
//       }
//     }
//     ASSERT_EQ(left, h_nptr[1] - h_nptr[0]);  // equal to number of sampled assigned to left
//   }
// }
// }  // anonymous namespace

// TEST(Approx, LeafPartition) {
//   for (auto n_samples : {0ul, 1ul, 128ul, 256ul}) {
//     TestLeafPartition(n_samples);
//   }
// }
}  // namespace tree
}  // namespace xgboost
