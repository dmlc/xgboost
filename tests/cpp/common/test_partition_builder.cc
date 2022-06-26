#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <utility>

#include "../../../src/common/opt_partition_builder.h"
#include "../helpers.h"

namespace xgboost {
namespace common {

// <<<<<<< HEAD
TEST(OptPartitionBuilder, BasicTest) {
  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat =
      RandomDataGenerator(kNRows, kNCols, 0).Seed(3).GenerateDMatrix();
  auto const &gmat = *(p_fmat->GetBatches<GHistIndexMatrix>(
                        BatchParam{GenericParameter::kCpuId, kMaxBins}).begin());
  // auto const& page = *(p_fmat->GetBatches<SparsePage>().begin());
  std::vector<GradientPair> row_gpairs =
    { {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {2.27f, 0.28f},
    {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f} };
  RegTree tree;
  tree.ExpandNode(0, 0, 0, true, 0, 0, 0, 0, 0, 0, 0);

  common::OptPartitionBuilder opt_partition_builder;
  opt_partition_builder.template Init<uint8_t>(gmat, gmat.Transpose(), &tree,
    1, 3, false);
  const uint8_t* data = reinterpret_cast<const uint8_t*>(gmat.Transpose().GetIndexData());

  const size_t fid = 0;
  const size_t split = 0;
  std::unordered_map<uint32_t, int32_t> split_conditions;
  std::unordered_map<uint32_t, uint64_t> split_ind;
  std::vector<uint16_t> node_ids(kNRows, 0);
  std::unordered_map<uint32_t, bool> smalest_nodes_mask;
  smalest_nodes_mask[1] = true;
  std::unordered_map<uint32_t, uint16_t> nodes;//(1, 0);
  std::vector<uint32_t> split_nodes(1, 0);
  auto pred = [&](auto ridx, auto bin_id, auto nid, auto split_cond) {
    return false;
  };
  opt_partition_builder.template CommonPartition<
    uint8_t, false, true, false>(0, 0, kNRows, data,
                          node_ids.data(),
                          &split_conditions,
                          &split_ind,
                          &smalest_nodes_mask,// row_gpairs,
                          gmat.Transpose(), split_nodes, pred, 1);
  opt_partition_builder.UpdateRowBuffer(node_ids,
                                        gmat, gmat.cut.Ptrs().size() - 1,
                                        0, node_ids, false);
  size_t left_cnt = 0, right_cnt = 0;
  const size_t bin_id_min = gmat.cut.Ptrs()[0];
  const size_t bin_id_max = gmat.cut.Ptrs()[1];

  // manually compute how many samples go left or right
  for (size_t rid = 0; rid < kNRows; ++rid) {
    for (size_t offset = gmat.row_ptr[rid]; offset < gmat.row_ptr[rid + 1]; ++offset) {
      const size_t bin_id = gmat.index[offset];
        if (bin_id >= bin_id_min && bin_id < bin_id_max) {
          if (bin_id <= split) {
            left_cnt++;
          } else {
            right_cnt++;
          }
        }
// =======
// TEST(PartitionBuilder, BasicTest) {
//   constexpr size_t kBlockSize = 16;
//   constexpr size_t kNodes = 5;
//   constexpr size_t kTasks = 3 + 5 + 10 + 1 + 2;

//   std::vector<size_t> tasks = { 3, 5, 10, 1, 2 };

//   PartitionBuilder<kBlockSize> builder;
//   builder.Init(kTasks, kNodes, [&](size_t i) {
//     return tasks[i];
//   });

//   std::vector<size_t> rows_for_left_node = { 2, 12, 0, 16, 8 };

//   for(size_t nid = 0; nid < kNodes; ++nid) {
//     size_t value_left = 0;
//     size_t value_right = 0;

//     size_t left_total = tasks[nid] * rows_for_left_node[nid];

//     for(size_t j = 0; j < tasks[nid]; ++j) {
//       size_t begin = kBlockSize*j;
//       size_t end = kBlockSize*(j+1);
//       const size_t id = builder.GetTaskIdx(nid, begin);
//       builder.AllocateForTask(id);

//       auto left  = builder.GetLeftBuffer(nid, begin, end);
//       auto right = builder.GetRightBuffer(nid, begin, end);

//       size_t n_left   = rows_for_left_node[nid];
//       size_t n_right = kBlockSize - rows_for_left_node[nid];

//       for(size_t i = 0; i < n_left; i++) {
//         left[i] = value_left++;
//       }

//       for(size_t i = 0; i < n_right; i++) {
//         right[i] = left_total + value_right++;
//       }

//       builder.SetNLeftElems(nid, begin, n_left);
//       builder.SetNRightElems(nid, begin, n_right);
// >>>>>>> 0725fd60819f9758fbed6ee54f34f3696a2fb2f8
    }
  }
  ASSERT_EQ(opt_partition_builder.summ_size, left_cnt);
  ASSERT_EQ(kNRows - opt_partition_builder.summ_size, right_cnt);
}

}  // namespace common
}  // namespace xgboost
