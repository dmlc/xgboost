#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <utility>

#include "../../../src/common/hist_util.h"
#include "../helpers.h"

namespace xgboost {
namespace common {

class HistCutMatrixMock : public HistCutMatrix {
 public:
  size_t SearchGroupIndFromBaseRow(
      std::vector<bst_uint> const& group_ptr, size_t const base_rowid) {
    return HistCutMatrix::SearchGroupIndFromBaseRow(group_ptr, base_rowid);
  }
};

TEST(HistCutMatrix, SearchGroupInd) {
  size_t constexpr kNumGroups = 4;
  size_t constexpr kNumRows = 17;
  size_t constexpr kNumCols = 15;

  auto pp_mat = CreateDMatrix(kNumRows, kNumCols, 0);

  auto& p_mat = *pp_mat;
  std::vector<bst_int> group(kNumGroups);
  group[0] = 2;
  group[1] = 3;
  group[2] = 7;
  group[3] = 5;

  p_mat->Info().SetInfo(
      "group", group.data(), DataType::kUInt32, kNumGroups);

  HistCutMatrixMock hmat;

  size_t group_ind = hmat.SearchGroupIndFromBaseRow(p_mat->Info().group_ptr_, 0);
  ASSERT_EQ(group_ind, 0);

  group_ind = hmat.SearchGroupIndFromBaseRow(p_mat->Info().group_ptr_, 5);
  ASSERT_EQ(group_ind, 2);

  EXPECT_ANY_THROW(hmat.SearchGroupIndFromBaseRow(p_mat->Info().group_ptr_, 17));

  delete pp_mat;
}

}  // namespace common
}  // namespace xgboost
