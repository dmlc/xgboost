#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <utility>

#include "../../../src/common/opt_partition_builder.h"
#include "../helpers.h"

namespace xgboost {
namespace common {

TEST(OptPartitionBuilder, BasicTest) {
  size_t constexpr kNRows = 8, kNCols = 16;
  int32_t constexpr kMaxBins = 4;
  auto p_fmat =
      RandomDataGenerator(kNRows, kNCols, 0).Seed(3).GenerateDMatrix();
  auto const &gmat = *(p_fmat->GetBatches<GHistIndexMatrix>(
                        BatchParam{GenericParameter::kCpuId, kMaxBins}).begin());
  auto const& page = *(p_fmat->GetBatches<SparsePage>().begin());
  std::vector<GradientPair> row_gpairs =
    { {1.23f, 0.24f}, {0.24f, 0.25f}, {0.26f, 0.27f}, {2.27f, 0.28f},
    {0.27f, 0.29f}, {0.37f, 0.39f}, {-0.47f, 0.49f}, {0.57f, 0.59f} };
  // ColumnMatrix column_matrix;
  // column_matrix.Init(page, gmat, 0.5, 1);

  // column_matrix.Init(gmat, 0, 1);
  RegTree tree;
  tree.ExpandNode(0, 0, 0, true, 0, 0, 0, 0, 0, 0, 0);

  common::OptPartitionBuilder opt_partition_builder;

  opt_partition_builder.template Init<uint8_t>(gmat, gmat.Transpose(), &tree,
    1, 3, false);
  const uint8_t* data = reinterpret_cast<const uint8_t*>(gmat.Transpose().GetIndexData());
  const size_t fid = 0;
  const size_t split = 0;
  std::unordered_map<uint32_t, int32_t> split_conditions;//(1, split - gmat.cut.Ptrs()[fid]);
  std::unordered_map<uint32_t, uint64_t> split_ind;//(1, fid*((gmat.IsDense() ? kNRows : 1)));
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
    }
  }
  std::cout << "opt_partition_builder.summ_size: " << opt_partition_builder.summ_size << std::endl;
  std::cout << "left_cnt: " << left_cnt << std::endl;
  std::cout << "right_cnt: " << right_cnt << std::endl;
  ASSERT_EQ(opt_partition_builder.summ_size, left_cnt);
  ASSERT_EQ(kNRows - opt_partition_builder.summ_size, right_cnt);
}

}  // namespace common
}  // namespace xgboost
