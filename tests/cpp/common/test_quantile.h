#include <rabit/rabit.h>
#include <algorithm>
#include <string>
#include <vector>

#include "../helpers.h"

namespace xgboost {
namespace common {
inline void InitRabitContext(std::string msg, size_t n_workers) {
  auto port = std::getenv("DMLC_TRACKER_PORT");
  std::string port_str;
  if (port) {
    port_str = port;
  } else {
    LOG(WARNING) << msg << " as `DMLC_TRACKER_PORT` is not set up.";
    return;
  }

  std::vector<std::string> envs{
      "DMLC_TRACKER_PORT=" + port_str,
      "DMLC_TRACKER_URI=127.0.0.1",
      "DMLC_NUM_WORKER=" + std::to_string(n_workers)};
  char* c_envs[] {&(envs[0][0]), &(envs[1][0]), &(envs[2][0])};
  rabit::Init(3, c_envs);
}

template <typename Fn> void RunWithSeedsAndBins(size_t rows, Fn fn) {
  std::vector<int32_t> seeds(4);
  SimpleLCG lcg;
  SimpleRealUniformDistribution<float> dist(3, 1000);
  std::generate(seeds.begin(), seeds.end(), [&](){ return dist(&lcg); });

  std::vector<size_t> bins(8);
  for (size_t i = 0; i < bins.size() - 1; ++i) {
    bins[i] = i * 35 + 2;
  }
  bins.back() = rows + 160;  // provide a bin number greater than rows.

  std::vector<MetaInfo> infos(2);
  auto& h_weights = infos.front().weights_.HostVector();
  h_weights.resize(rows);
  std::generate(h_weights.begin(), h_weights.end(), [&]() { return dist(&lcg); });

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
