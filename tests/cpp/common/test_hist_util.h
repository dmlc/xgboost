
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

data::SimpleDMatrix GetDMatrixFromData(const std::vector<float>& x) {
  data::DenseAdapter adapter(x.data(), x.size(), x.size(), 1);
  return data::SimpleDMatrix(&adapter, std::numeric_limits<float>::quiet_NaN(),
                             1);
}

std::vector<float> CutsFromSort(const std::vector<float>& x_sorted,
                                int num_bins) {
  if (x_sorted.size() <= num_bins) return x_sorted;
  std::vector<float> cuts(num_bins);
  for(auto i = 0ull; i < num_bins; i++)
  {
    double rank = double(i)/num_bins;
    cuts[i] = x_sorted[size_t(rank*x_sorted.size())];
  }
  return cuts;
}
}  // namespace common
}  // namespace xgboost
