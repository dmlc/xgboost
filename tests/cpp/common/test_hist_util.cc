#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <utility>

#include "../../../src/common/hist_util.h"
#include "../helpers.h"

namespace xgboost {
namespace common {

TEST(HistCutMatrix, Init) {
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

  HistCutMatrix hmat;
  // Don't throw when finding group
  EXPECT_NO_THROW(hmat.Init(p_mat.get(), 4));

  delete pp_mat;
}

}  // namespace common
}  // namespace xgboost
