/**
 * Copyright 2026, XGBoost Contributors
 * \file multiclass_metric.cc
 * \brief CPU implementations and registrations for multiclass metrics.
 */
#include "multiclass_metric.h"

#include <dmlc/registry.h>

#include <array>
#include <atomic>
#include <numeric>
#include <vector>

#include "../collective/aggregator.h"
#include "../common/kernel.h"
#include "../common/threading_utils.h"
#include "xgboost/collective/result.h"
#include "xgboost/metric.h"

namespace xgboost::metric {
DMLC_REGISTRY_FILE_TAG(multiclass_metric);
namespace {
template <typename EvalRowPolicy>
PackedReduceResult EvalCpu(Context const* ctx, HostDeviceVector<float> const& preds,
                           MetaInfo const& info, std::size_t n_class,
                           HostDeviceVector<std::int32_t>* /*label_error_buffer*/) {
  auto const& weights = info.weights_;
  auto const& labels = *info.labels.Data();
  auto n_threads = ctx->Threads();
  size_t ndata = labels.Size();

  const auto& h_labels = labels.HostVector();
  const auto& h_weights = weights.HostVector();
  const auto& h_preds = preds.HostVector();

  std::atomic<int> label_error{0};
  bool const is_null_weight = weights.Size() == 0;

  std::vector<double> scores_tloc(n_threads, 0);
  std::vector<double> weights_tloc(n_threads, 0);
  common::ParallelFor(ndata, n_threads, [&](size_t idx) {
    bst_float weight = is_null_weight ? 1.0f : h_weights[idx];
    auto label = static_cast<int>(h_labels[idx]);
    if (label >= 0 && label < static_cast<int>(n_class)) {
      auto t_idx = omp_get_thread_num();
      scores_tloc[t_idx] +=
          EvalRowPolicy::EvalRow(label, h_preds.data() + idx * n_class, n_class) * weight;
      weights_tloc[t_idx] += weight;
    } else {
      label_error = label;
    }
  });

  double residue_sum = std::accumulate(scores_tloc.cbegin(), scores_tloc.cend(), 0.0);
  double weights_sum = std::accumulate(weights_tloc.cbegin(), weights_tloc.cend(), 0.0);

  CheckMultiClassLabel(label_error, n_class);
  PackedReduceResult res{residue_sum, weights_sum};

  return res;
}

auto const kRegisterErrorCpu = common::KernelRegistration<MultiClassErrorEvalKernel>{
    DeviceOrd::kCPU, &EvalCpu<EvalMatchError>};
auto const kRegisterLogLossCpu = common::KernelRegistration<MultiClassLogLossEvalKernel>{
    DeviceOrd::kCPU, &EvalCpu<EvalMultiLogLoss>};
}  // namespace

template <typename Derived>
struct EvalMClassBase : public MetricNoCache {
  ~EvalMClassBase() noexcept override = default;

  double Eval(const HostDeviceVector<float>& preds, const MetaInfo& info) override {
    CheckRowWeights(info);
    if (info.labels.Size() == 0) {
      CHECK_EQ(preds.Size(), 0);
    } else {
      CHECK_EQ(info.labels.Shape(1), 1)
          << "`merror` and `mlogloss` do not support multi-target labels.";
      CHECK(preds.Size() % info.labels.Size() == 0) << "label and prediction size not match";
    }
    std::array<double, 2> dat{0.0, 0.0};
    if (info.labels.Size() != 0) {
      const size_t nclass = preds.Size() / info.labels.Size();
      CHECK_GE(nclass, 1U) << "mlogloss and merror are only used for multi-class classification,"
                           << " use logloss for binary classification";
      auto result = common::DispatchKernel<MultiClassEvalKernel<Derived>>(ctx_, preds, info, nclass,
                                                                          &label_error_);
      dat[0] = result.Residue();
      dat[1] = result.Weights();
    }
    auto rc = collective::GlobalSum(ctx_, linalg::MakeVec(dat.data(), dat.size()));
    collective::SafeColl(rc);
    return dat[0] / dat[1];
  }
  [[nodiscard]] char const* Name() const override { return Derived::Name(); }

 private:
  HostDeviceVector<std::int32_t> label_error_;
};

XGBOOST_REGISTER_METRIC(MatchError, "merror")
    .describe("Multiclass classification error.")
    .set_body([](const char*) { return new EvalMClassBase<EvalMatchError>(); });

XGBOOST_REGISTER_METRIC(MultiLogLoss, "mlogloss")
    .describe("Multiclass negative loglikelihood.")
    .set_body([](const char*) { return new EvalMClassBase<EvalMultiLogLoss>(); });
}  // namespace xgboost::metric
