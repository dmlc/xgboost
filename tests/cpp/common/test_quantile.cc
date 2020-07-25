#include <gtest/gtest.h>
#include "test_hist_util.h"
#include "test_quantile.h"
#include "../../../src/common/hist_util.h"

namespace xgboost {
namespace common {
TEST(CPUQuantile, FromOneHot) {
  HistogramCuts cuts;
  DenseCuts dense(&cuts);
  std::vector<float> x = BasicOneHotEncodedData();
  auto m = GetDMatrixFromData(x, 5, 3);
  int32_t max_bins = 16;
  dense.Build(m.get(), max_bins);

  std::vector<uint32_t> const& h_cuts_ptr = cuts.Ptrs();
  std::vector<float> h_cuts_values = cuts.Values();
  ValidateBasicOneHot(h_cuts_ptr, h_cuts_values);
}
}  // namespace common
}  // namespace xgboost
