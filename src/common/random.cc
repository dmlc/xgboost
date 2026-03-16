/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include "random.h"

#include <algorithm>  // for sort, max, copy
#include <cstdint>    // for int64_t, uint32_t
#include <memory>     // for shared_ptr
#include <sstream>    // for stringstream
#include <string>     // for string

#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json, Object, Integer, String, get

namespace xgboost::common {
void SaveRng(Json *p_out, RandomEngine const &rng) {
  std::stringstream ss;
  ss << std::hex << rng;
  auto &out = *p_out;
  out["rng_state"] = String{ss.str()};
}

void LoadRng(Json const &in, RandomEngine *rng) {
  std::stringstream ss{get<String const>(in["rng_state"])};
  ss >> std::hex >> *rng;
}

[[nodiscard]] std::shared_ptr<ColumnSampler> LoadColumnSamplerOptional(
    Json const &jconfig, std::shared_ptr<ColumnSampler> column_sampler) {
  auto const &config = get<Object const>(jconfig);
  auto it = config.find("column_sampler");
  if (it != config.cend()) {
    column_sampler = std::make_shared<common::ColumnSampler>();
    column_sampler->LoadConfig(it->second);
  }
  return column_sampler;
}

void ColumnSampler::SaveConfig(Json *p_out) const {
  auto &out = *p_out;
  out["seed"] = Integer{static_cast<std::int64_t>(seed_)};
  SaveRng(&out, this->rng_);
}

void ColumnSampler::LoadConfig(Json const &in) {
  auto const &obj = get<Object const>(in);
  seed_ = static_cast<std::uint32_t>(get<Integer const>(obj.at("seed")));
  rng_.seed(seed_);
  std::stringstream ss{get<String const>(obj.at("rng_state"))};
  ss >> std::hex >> rng_;
}

std::shared_ptr<HostDeviceVector<bst_feature_t>> ColumnSampler::ColSample(
    std::shared_ptr<HostDeviceVector<bst_feature_t>> p_features, float colsample) {
  if (colsample == 1.0f) {
    return p_features;
  }

  int n = std::max(1, static_cast<int>(colsample * p_features->Size()));
  auto p_new_features = std::make_shared<HostDeviceVector<bst_feature_t>>();

  if (ctx_->IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
    cuda_impl::SampleFeature(ctx_, n, p_features, p_new_features, this->feature_weights_,
                             &this->weight_buffer_, &this->idx_buffer_, &rng_);
    return p_new_features;
#else
    AssertGPUSupport();
    return nullptr;
#endif  // defined(XGBOOST_USE_CUDA)
  }

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
    CHECK(ctx_);
    new_features.HostVector() =
        WeightedSamplingWithoutReplacement(ctx_, p_features->HostVector(), weight, n);
  } else {
    new_features.Resize(features.size());
    std::copy(features.begin(), features.end(), new_features.HostVector().begin());
    std::shuffle(new_features.HostVector().begin(), new_features.HostVector().end(), rng_);
    new_features.Resize(n);
  }
  std::sort(new_features.HostVector().begin(), new_features.HostVector().end());
  return p_new_features;
}
}  // namespace xgboost::common
