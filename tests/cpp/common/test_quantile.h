#ifndef XGBOOST_TESTS_CPP_COMMON_TEST_QUANTILE_H_
#define XGBOOST_TESTS_CPP_COMMON_TEST_QUANTILE_H_

#include <algorithm>
#include <string>
#include <vector>

#include "../helpers.h"

namespace xgboost {
namespace common {
template <typename Fn> void RunWithSeedsAndBins(size_t rows, Fn fn) {
  std::vector<int32_t> seeds(2);
  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist(3, 1000);
  std::generate(seeds.begin(), seeds.end(), [&](){ return dist(&lcg); });

  std::vector<size_t> bins(2);
  for (size_t i = 0; i < bins.size() - 1; ++i) {
    bins[i] = i * 35 + 2;
  }
  bins.back() = rows + 160;  // provide a bin number greater than rows.

  std::vector<MetaInfo> infos(2);
  auto& h_weights = infos.front().weights_.HostVector();
  h_weights.resize(rows);

  SimpleRealUniformDistribution<float> weight_dist(0, 10);
  std::generate(h_weights.begin(), h_weights.end(), [&]() { return weight_dist(&lcg); });

  for (auto seed : seeds) {
    for (auto n_bin : bins) {
      for (auto const& info : infos) {
        fn(seed, n_bin, info);
      }
    }
  }
}
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_TESTS_CPP_COMMON_TEST_QUANTILE_H_
