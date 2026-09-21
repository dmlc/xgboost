/**
 * Copyright 2015-2025, XGBoost Contributors
 * \file elementwise_metric.cu
 * \brief evaluation metrics for elementwise binary or regression.
 * \author Kailong Chen, Tianqi Chen
 *
 *  The expressions like wsum == 0 ? esum : esum / wsum is used to handle empty dataset.
 */
#include <dmlc/registry.h>

#include <array>
#include <cmath>
#include <numeric>  // for accumulate

#include "../collective/aggregator.h"
#include "../common/expectile_loss_utils.h"  // ExpectileLossParam
#include "../common/math.h"
#include "../common/nvtx_utils.h"           // for xgboost_NVTX_FN_RANGE
#include "../common/optional_weight.h"      // OptionalWeights
#include "../common/quantile_loss_utils.h"  // QuantileLossParam
#include "../common/threading_utils.h"
#include "metric_common.h"              // MetricNoCache
#include "xgboost/collective/result.h"  // for SafeColl
#include "xgboost/metric.h"

#if defined(XGBOOST_USE_CUDA)
#include <thrust/transform_reduce.h>

#include <cuda/std/functional>  // for plus

#include "../common/cuda_compat.cuh"   // for CUDA compatibility
#include "../common/cuda_context.cuh"  // for CUDAContext
#else
#include "../common/common.h"  // for AssertGPUSupport
#endif                         // XGBOOST_USE_CUDA

namespace xgboost::metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(elementwise_metric);

namespace {
/**
 * \brief Reduce function for element wise metrics.
 *
 *   The loss function should handle all the computation for each sample, including
 *   applying the weights.  A tuple of {error_i, weight_i} is expected as return.
 */
template <typename Fn>
PackedReduceResult Reduce(Context const* ctx, MetaInfo const& info, Fn&& loss,
                          size_t num_preds = 1) {
  CheckRowWeights(info);
  PackedReduceResult result;
  // This function doesn't have sycl-specific implementation yet.
  // For that reason we transfer data to host in case of sycl is used for propper execution.
  auto labels = info.labels.View(ctx->Device().IsSycl() ? DeviceOrd::CPU() : ctx->Device());
  if (ctx->IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
    dh::counting_iterator<size_t> begin(0);
    dh::counting_iterator<size_t> end = begin + labels.Size() * num_preds;
    result = thrust::transform_reduce(
        ctx->CUDACtx()->CTP(), begin, end,
        [=] XGBOOST_DEVICE(size_t i) {
          auto idx = linalg::UnravelIndex(i, labels.Shape());
          auto sample_id = std::get<0>(idx);
          auto target_id = std::get<1>(idx);
          auto res = loss(i, sample_id, target_id);
          float v{std::get<0>(res)}, wt{std::get<1>(res)};
          return PackedReduceResult{v, wt};
        },
        PackedReduceResult{}, cuda::std::plus<PackedReduceResult>());
#else
    common::AssertGPUSupport();
#endif  //  defined(XGBOOST_USE_CUDA)
  } else {
    auto n_threads = ctx->Threads();
    std::vector<double> score_tloc(n_threads, 0.0);
    std::vector<double> weight_tloc(n_threads, 0.0);
    // We sum over losses over all samples and targets instead of performing this for each
    // target since the first one approach more accurate while the second approach is used
    // for approximation in distributed setting.  For rmse:
    // - sqrt(1/w(sum_t0 + sum_t1 + ... + sum_tm))       // multi-target
    // - sqrt(avg_t0) + sqrt(avg_t1) + ... sqrt(avg_tm)  // distributed

    auto size = info.labels.Size() * num_preds;
    std::size_t constexpr kBlockSize = 2048;
    common::ParallelFor1d<kBlockSize>(size, n_threads, [&](auto&& block) {
      double sum_score = 0, sum_weight = 0;
      for (std::size_t i = block.begin(), n = block.end(); i < n; ++i) {
        auto [sample_id, target_id] = linalg::UnravelIndex(i, labels.Shape());

        auto [v, wt] = loss(i, sample_id, target_id);
        sum_score += v;
        sum_weight += wt;
      }

      auto t_idx = omp_get_thread_num();
      score_tloc[t_idx] += sum_score;
      weight_tloc[t_idx] += sum_weight;
    });

    double residue_sum = std::accumulate(score_tloc.cbegin(), score_tloc.cend(), 0.0);
    double weights_sum = std::accumulate(weight_tloc.cbegin(), weight_tloc.cend(), 0.0);
    result = PackedReduceResult{residue_sum, weights_sum};
  }
  return result;
}
}  // anonymous namespace

class QuantileError : public MetricNoCache {
  HostDeviceVector<float> alpha_;
  common::QuantileLossParam param_;

 public:
  std::set<std::string> Configure(Args const& args) override {
    auto used = UpdateAndGetUsedParameters(&param_, args);
    param_.Validate();
    alpha_.HostVector() = param_.quantile_alpha.Get();
    return used;
  }

  double Eval(HostDeviceVector<bst_float> const& preds, const MetaInfo& info) override {
    CHECK(!alpha_.Empty());
    CHECK_EQ(info.labels.Shape(0), info.num_row_) << "Invalid shape of labels.";
    CHECK_EQ(preds.Size(), info.labels.Size() * alpha_.Size())
        << "Prediction size must equal label size times the number of alpha values.";
    if (info.num_row_ == 0) {
      // empty DMatrix on distributed env
      std::array<double, 2> dat{0.0, 0.0};
      auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(dat.data(), dat.size()));
      collective::SafeColl(rc);
      CHECK_GT(dat[1], 0);
      return dat[0] / dat[1];
    }

    auto const* ctx = ctx_;
    auto y_true = info.labels.View(ctx->Device());
    preds.SetDevice(ctx->Device());
    alpha_.SetDevice(ctx->Device());
    auto alpha = ctx->IsCPU() ? alpha_.ConstHostSpan() : alpha_.ConstDeviceSpan();
    std::size_t n_targets = info.labels.Shape(1);
    CHECK_NE(n_targets, 0);
    auto y_predt = linalg::MakeTensorView(ctx, &preds, static_cast<std::size_t>(info.num_row_),
                                          alpha_.Size(), n_targets);

    info.weights_.SetDevice(ctx->Device());
    common::OptionalWeights weight{ctx->IsCPU() ? info.weights_.ConstHostSpan()
                                                : info.weights_.ConstDeviceSpan()};

    auto result = Reduce(
        ctx, info,
        [=] XGBOOST_DEVICE(std::size_t i, std::size_t sample_id, std::size_t target_id) {
          auto idx = linalg::UnravelIndex(i, y_predt.Shape());
          sample_id = std::get<0>(idx);
          std::size_t quantile_id = std::get<1>(idx);
          target_id = std::get<2>(idx);

          auto loss = [a = alpha[quantile_id]](float p, float y) {
            auto d = y - p;
            float sign = d >= 0.0f;
            auto res = (a * sign * d) - (1.0f - a) * (1.0f - sign) * d;
            return res;
          };
          auto w = weight[sample_id];
          auto l =
              loss(y_predt(sample_id, quantile_id, target_id), y_true(sample_id, target_id)) * w;
          return std::make_tuple(l, w);
        },
        alpha_.Size());
    std::array<double, 2> dat{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx, linalg::MakeVec(dat.data(), dat.size()));
    collective::SafeColl(rc);
    CHECK_GT(dat[1], 0);
    return dat[0] / dat[1];
  }

  const char* Name() const override { return "quantile"; }
  void LoadConfig(Json const& in) override {
    auto const& obj = get<Object const>(in);
    auto it = obj.find("quantile_loss_param");
    if (it != obj.cend()) {
      FromJson(it->second, &param_);
      auto const& name = get<String const>(in["name"]);
      CHECK_EQ(name, "quantile");
    }
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["quantile_loss_param"] = ToJson(param_);
  }
};

XGBOOST_REGISTER_METRIC(QuantileError, "quantile")
    .describe("Quantile regression error.")
    .set_body([](const char*) { return new QuantileError{}; });

class ExpectileError : public MetricNoCache {
  HostDeviceVector<float> alpha_;
  common::ExpectileLossParam param_;

 public:
  std::set<std::string> Configure(Args const& args) override {
    auto used = UpdateAndGetUsedParameters(&param_, args);
    param_.Validate();
    alpha_.HostVector() = param_.expectile_alpha.Get();
    return used;
  }

  double Eval(HostDeviceVector<bst_float> const& preds, const MetaInfo& info) override {
    CHECK(!alpha_.Empty());
    CHECK_EQ(info.labels.Shape(0), info.num_row_) << "Invalid shape of labels.";
    CHECK_EQ(preds.Size(), info.labels.Size() * alpha_.Size())
        << "Prediction size must equal label size times the number of alpha values.";
    if (info.num_row_ == 0) {
      // empty DMatrix on distributed env
      std::array<double, 2> dat{0.0, 0.0};
      auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(dat.data(), dat.size()));
      collective::SafeColl(rc);
      CHECK_GT(dat[1], 0);
      return dat[0] / dat[1];
    }

    auto const* ctx = ctx_;
    auto y_true = info.labels.View(ctx->Device());
    preds.SetDevice(ctx->Device());
    alpha_.SetDevice(ctx->Device());
    auto alpha = ctx->IsCPU() ? alpha_.ConstHostSpan() : alpha_.ConstDeviceSpan();
    std::size_t n_targets = info.labels.Shape(1);
    CHECK_NE(n_targets, 0);
    auto y_predt = linalg::MakeTensorView(ctx, &preds, static_cast<std::size_t>(info.num_row_),
                                          alpha_.Size(), n_targets);

    info.weights_.SetDevice(ctx->Device());
    common::OptionalWeights weight{ctx->IsCPU() ? info.weights_.ConstHostSpan()
                                                : info.weights_.ConstDeviceSpan()};

    auto result = Reduce(
        ctx, info,
        [=] XGBOOST_DEVICE(std::size_t i, std::size_t sample_id, std::size_t target_id) {
          auto idx = linalg::UnravelIndex(i, y_predt.Shape());
          sample_id = std::get<0>(idx);
          std::size_t expectile_id = std::get<1>(idx);
          target_id = std::get<2>(idx);

          auto pred = y_predt(sample_id, expectile_id, target_id);
          auto label = y_true(sample_id, target_id);
          auto diff = pred - label;
          auto expectile = alpha[expectile_id];
          auto weight_scale = diff >= 0.0f ? (1.0f - expectile) : expectile;
          auto sample_weight = weight[sample_id];
          auto loss = weight_scale * diff * diff * sample_weight;
          return std::make_tuple(loss, sample_weight);
        },
        alpha_.Size());
    std::array<double, 2> dat{result.Residue(), result.Weights()};
    auto rc = collective::GlobalSum(ctx, linalg::MakeVec(dat.data(), dat.size()));
    collective::SafeColl(rc);
    CHECK_GT(dat[1], 0);
    return dat[0] / dat[1];
  }

  const char* Name() const override { return "expectile"; }
  void LoadConfig(Json const& in) override {
    auto const& obj = get<Object const>(in);
    auto it = obj.find("expectile_loss_param");
    if (it != obj.cend()) {
      FromJson(it->second, &param_);
      auto const& name = get<String const>(in["name"]);
      CHECK_EQ(name, "expectile");
      param_.Validate();
      alpha_.HostVector() = param_.expectile_alpha.Get();
    }
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(this->Name());
    out["expectile_loss_param"] = ToJson(param_);
  }
};

XGBOOST_REGISTER_METRIC(ExpectileError, "expectile")
    .describe("Expectile regression error.")
    .set_body([](const char*) { return new ExpectileError{}; });
}  // namespace xgboost::metric
