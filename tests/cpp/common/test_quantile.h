#include <gtest/gtest.h>
#include <vector>
#include "../helpers.h"

namespace xgboost {
namespace common {
inline auto BasicOneHotEncodedData() {
  std::vector<float> x {
    0, 1, 0,
    0, 1, 0,
    0, 1, 0,
    0, 0, 1,
    1, 0, 0
  };
  return x;
}

inline void ValidateBasicOneHot(std::vector<uint32_t> const& h_cuts_ptr, std::vector<float> const& h_cuts_values) {
  size_t const cols = 3;
  ASSERT_EQ(h_cuts_ptr.size(),  cols + 1);
  ASSERT_EQ(h_cuts_values.size(), cols * 2);

  for (size_t i = 1; i < h_cuts_ptr.size(); ++i) {
    auto feature =
        common::Span<float const>(h_cuts_values)
            .subspan(h_cuts_ptr[i - 1], h_cuts_ptr[i] - h_cuts_ptr[i - 1]);
    EXPECT_EQ(feature.size(), 2);
    EXPECT_EQ(feature[0], 1.0f);
    EXPECT_GT(feature[1], 1.0f);  // greater than any value.
  }
}
}  // namespace common
}  // namespace xgboost
