
#include <gtest/gtest.h>
#include "../../../src/data/simple_dmatrix.h"

namespace xgboost {
namespace common {

std::vector<float> GenerateRandomSingleColumn(int n, float low = -100,
                                              float high = 100) {
  std::vector<float> x(n);
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(low, high);
  std::generate(x.begin(), x.end(), [&]() { return dist(rng); });
  return x;
}

std::vector<float> GenerateRandomCategoricalSingleColumn(int n,
                                                         int num_categories) {
  std::vector<float> x(n);
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> dist(0, num_categories - 1);
  std::generate(x.begin(), x.end(), [&]() { return dist(rng); });
  // Make sure each category is present
  for(auto i = 0ull; i < num_categories; i++)
  {
    x[i] = i;
  }
  return x;
}

data::SimpleDMatrix GetDMatrixFromData(const std::vector<float>& x) {
  data::DenseAdapter adapter(x.data(), x.size(), 1);
  return data::SimpleDMatrix(&adapter, std::numeric_limits<float>::quiet_NaN(),
                             1);
}

void TestRank(const std::vector<float>& cuts,
              const std::vector<float>& sorted_x, float eps) {
  // Ignore the first and last cut, they are special
  size_t j = 0;
  for (auto i = 1ull; i < cuts.size() - 1; i++) {
    int expected_rank = (i * sorted_x.size()) / cuts.size() + 1;
    while (cuts[i] > sorted_x[j]) {
      j++;
    }
    int actual_rank = j;
    EXPECT_LT(std::abs(expected_rank - actual_rank), sorted_x.size() * eps);
  }
}
}  // namespace common
}  // namespace xgboost
