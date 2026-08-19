/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include "random.h"

#include <algorithm>  // for sort, max, copy
#include <cstddef>    // for size_t
#include <cstdint>    // for uint64_t
#include <limits>     // for numeric_limits
#include <memory>     // for shared_ptr
#include <string>     // for string, to_string

#include "error_msg.h"                   // for WarnOldRngState
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json, Object, String, get, IsA

namespace xgboost::common {
namespace {
/**
 * @brief Read the two space-separated decimal integers written by @ref SaveRng.
 *
 * Parsed by hand rather than through `std::istream`, whose extraction of numbers consults
 * the global locale. Anything else is rejected, including the several hundred state words
 * that older versions wrote here.
 */
[[nodiscard]] bool ParseRngState(std::string const &str, std::uint64_t *p_seed,
                                 std::uint64_t *p_n_advanced) {
  std::uint64_t values[2]{};
  std::size_t pos = 0;

  for (auto &value : values) {
    while (pos < str.size() && str[pos] == ' ') {
      pos++;
    }
    auto const begin = pos;
    while (pos < str.size() && str[pos] >= '0' && str[pos] <= '9') {
      auto digit = static_cast<std::uint64_t>(str[pos] - '0');
      if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
        return false;  // Overflow.
      }
      value = value * 10 + digit;
      pos++;
    }
    if (pos == begin) {
      return false;  // Not a number.
    }
  }

  while (pos < str.size() && str[pos] == ' ') {
    pos++;
  }
  if (pos != str.size()) {
    return false;  // Trailing content.
  }

  *p_seed = values[0];
  *p_n_advanced = values[1];
  return true;
}
}  // anonymous namespace

void SaveRng(Json *p_out, SerializableRandomEngine const &rng) {
  auto &out = *p_out;
  // The surrounding object holds XGBoost parameters, whose values are all strings.
  out["rng_state"] = String{std::to_string(rng.Seed()) + " " + std::to_string(rng.NumAdvanced())};
}

bool LoadRng(Json const &in, SerializableRandomEngine *rng) {
  auto const &obj = get<Object const>(in);
  auto it = obj.find("rng_state");
  if (it == obj.cend() || !IsA<String>(it->second)) {
    return false;
  }

  std::uint64_t seed = 0;
  std::uint64_t n_advanced = 0;
  if (!ParseRngState(get<String const>(it->second), &seed, &n_advanced)) {
    // Older versions wrote the text produced by `operator<<` for `std::mt19937`, which
    // cannot be read back on a different standard library implementation. See
    // `SerializableRandomEngine`.
    error::WarnOldRngState();
    return false;
  }

  rng->Restore(static_cast<SerializableRandomEngine::result_type>(seed), n_advanced);
  return true;
}

std::shared_ptr<HostDeviceVector<bst_feature_t>> ColumnSampler::ColSample(
    Context const *ctx, std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features,
    float colsample) {
  if (colsample == 1.0f) {
    return p_features;
  }

  int n = std::max(1, static_cast<int>(colsample * p_features->Size()));
  auto p_new_features = std::make_shared<HostDeviceVector<bst_feature_t>>();

  if (ctx->IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
    cuda_impl::SampleFeature(ctx, n, p_features, p_new_features, this->feature_weights_,
                             &this->weight_buffer_, &this->idx_buffer_);
    return p_new_features;
#else
    AssertGPUSupport();
    return nullptr;
#endif  // defined(XGBOOST_USE_CUDA)
  }

  auto seed = ctx->Rng()();
  RandomEngine rng(seed);
  const auto &features = p_features->HostVector();
  CHECK_GT(features.size(), 0);

  auto &new_features = *p_new_features;

  if (!feature_weights_.Empty()) {
    auto const &h_features = p_features->HostVector();
    auto const &h_feature_weight = feature_weights_.ConstHostVector();
    auto &weight = this->weight_buffer_.HostVector();
    weight.resize(h_features.size());
    for (size_t i = 0; i < h_features.size(); ++i) {
      weight[i] = h_feature_weight[h_features[i]];
    }
    new_features.HostVector() =
        WeightedSamplingWithoutReplacement(ctx, &rng, p_features->HostVector(), weight, n);
  } else {
    new_features.Resize(features.size());
    std::copy(features.begin(), features.end(), new_features.HostVector().begin());
    std::shuffle(new_features.HostVector().begin(), new_features.HostVector().end(), rng);
    new_features.Resize(n);
  }
  std::sort(new_features.HostVector().begin(), new_features.HostVector().end());
  return p_new_features;
}
}  // namespace xgboost::common
