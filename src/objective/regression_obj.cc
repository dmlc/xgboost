/*!
 * Copyright 2018 XGBoost contributors
 */

// Dummy file to keep the CUDA conditional compile trick.

#include <dmlc/registry.h>

#include "../common/linalg_op.h"
#include "rabit/rabit.h"
#include "xgboost/data.h"
#include "xgboost/objective.h"
#include "xgboost/tree_model.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(regression_obj);

float WeightedQuantile(float quantile, common::Span<size_t const> row_set,
                       linalg::VectorView<float const> labels,
                       linalg::VectorView<float const> weights) {
  std::vector<size_t> sorted_idx(row_set.size());
  std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
  std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                   [&](size_t i, size_t j) { return labels(row_set[i]) < labels(row_set[j]); });
  std::vector<float> weighted_cdf(row_set.size());
  weighted_cdf[0] = weights(row_set[sorted_idx[0]]);
  for (size_t i = 1; i < row_set.size(); ++i) {
    weighted_cdf[i] = weighted_cdf[i - 1] + weights(row_set[sorted_idx[i]]);
  }
  float thresh = weighted_cdf.back() * quantile;
  size_t pos =
      std::upper_bound(weighted_cdf.cbegin(), weighted_cdf.cend(), thresh) - weighted_cdf.cbegin();
  pos = std::min(pos, static_cast<size_t>(row_set.size() - 1));
  if (pos == 0 || pos == static_cast<size_t>(row_set.size() - 1)) {
    return labels(row_set[sorted_idx[pos]]);
  }
  CHECK_GE(thresh, weighted_cdf[pos - 1]);
  CHECK_LT(thresh, weighted_cdf[pos]);
  float v1 = labels(row_set[sorted_idx[pos - 1]]);
  float v2 = labels(row_set[sorted_idx[pos]]);
  if (weighted_cdf[pos + 1] - weighted_cdf[pos] >= 1.0f) {
    return (thresh - weighted_cdf[pos]) / (weighted_cdf[pos + 1] - weighted_cdf[pos]) * (v2 - v2) +
           v1;
  } else {
    return v2;
  }
};

float Quantile(float quantile, common::Span<size_t const> row_set, linalg::VectorView<float const> labels) {
  float result;
  LOG(FATAL) << "Not implemented";
  // fixme: pick an algorithm from R quantile.
  return result;
}

void UpdateTreeLeafHost(common::Span<RowIndexCache const> row_index, MetaInfo const& info,
                        uint32_t target, float alpha, RegTree* p_tree) {
  auto& tree = *p_tree;
  std::vector<float> quantiles;
  for (auto const& part : row_index) {
    std::vector<float> results;
    for (auto const& seg : part.indptr) {
      auto h_row_set = part.row_index.HostSpan().subspan(seg.begin, seg.n);
      float q{0};
      if (info.weights_.Empty()) {
        q = Quantile(alpha, h_row_set, info.labels.HostView().Slice(linalg::All(), target));
      } else {
        q = WeightedQuantile(alpha, h_row_set, info.labels.HostView().Slice(linalg::All(), target),
                             linalg::MakeVec(&info.weights_));
      }
      results.push_back(q);
    }
    // fixme: verify this is correct for external memory
    if (quantiles.empty()) {
      quantiles.resize(results.size(), 0);
    }
    for (size_t i = 0; i < results.size(); ++i) {
      quantiles[i] += results[i];
    }
    // use the mean value
    rabit::Allreduce<rabit::op::Sum>(results.data(), results.size());
    auto world = rabit::GetWorldSize();
    std::transform(results.begin(), results.end(), results.begin(),
                   [&](float q) { return q / world; });
  }

  // fixme: verify this is correct for external memory
  for (size_t i = 0; i < row_index.front().indptr.size(); ++i) {
    auto seg = row_index.front().indptr[i];
    auto q = quantiles[i];
    tree[seg.nidx].SetLeaf(q);  // fixme: exact tree method
  }
}

class MeanAbsoluteError : public ObjFunction {
 public:
  void Configure(Args const&) override {}

  uint32_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
  }

  struct ObjInfo Task() const override {
    return {ObjInfo::kRegression, true};
  }

  void GetGradient(HostDeviceVector<bst_float> const& preds, const MetaInfo& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    out_gpair->SetDevice(ctx_->gpu_id);
    out_gpair->Resize(info.labels.Size());
    auto gpair = linalg::MakeVec(out_gpair);

    preds.SetDevice(ctx_->gpu_id);
    auto predt = linalg::MakeVec(&preds);
    auto sign = [](auto x) {
      return (x > static_cast<decltype(x)>(0)) - (x < static_cast<decltype(x)>(0));
    };

    info.weights_.SetDevice(ctx_->gpu_id);
    common::OptionalWeights weight{ctx_->IsCPU() ? info.weights_.ConstHostSpan()
                                                 : info.weights_.ConstDeviceSpan()};

    linalg::ElementWiseKernel(ctx_, info.labels.View(ctx_->gpu_id),
                              [=] XGBOOST_DEVICE(size_t i, float const y) mutable {
                                auto grad = sign(predt(i) - y) * weight[i];
                                auto hess = weight[i];
                                gpair(i) = GradientPair{grad, hess};
                              });
  }

  void UpdateTreeLeaf(common::Span<RowIndexCache const> row_index, MetaInfo const& info,
                      uint32_t target, RegTree* p_tree) const override {
    UpdateTreeLeafHost(row_index, info, target, 0.5, p_tree);
  }

  const char* DefaultEvalMetric() const override { return "mae"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:absoluteerror");
  }

  void LoadConfig(Json const& in) override {}
};

XGBOOST_REGISTER_OBJECTIVE(MeanAbsoluteError, "reg:absoluteerror")
    .describe("Mean absoluate error.")
    .set_body([]() { return new MeanAbsoluteError(); });

struct QuantileRegressionParameter : public XGBoostParameter<QuantileRegressionParameter> {
  float quantile;
};

class QuantileRegression : public ObjFunction {
  QuantileRegressionParameter param_;

 public:
  void Configure(Args const&) override {}

  uint32_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
  }

  struct ObjInfo Task() const override {
    return {ObjInfo::kRegression, true};
  }

  void GetGradient(HostDeviceVector<bst_float> const& preds, const MetaInfo& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {}

  void UpdateTreeLeaf(common::Span<RowIndexCache const> row_index, MetaInfo const& info,
                      uint32_t target, RegTree* p_tree) const override {
    UpdateTreeLeafHost(row_index, info, target, param_.quantile, p_tree);
  }

  const char* DefaultEvalMetric() const override { return "undefined"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:quantile");
    out["quantile_regression_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override { FromJson(in["quantile_regression_param"], &param_); }
};

XGBOOST_REGISTER_OBJECTIVE(QuantileRegression, "reg:quantile")
    .describe("Quantile regression.")
    .set_body([]() { return new QuantileRegression(); });
}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "regression_obj.cu"
#endif  // XGBOOST_USE_CUDA
