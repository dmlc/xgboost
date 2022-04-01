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
  float result;
  // fixme: pick an algorithm from R quantile.
  return result;
};

float Quantile(float quantile, common::Span<size_t const> row_set, linalg::VectorView<float const> labels) {
  float result;
  // fixme: pick an algorithm from R quantile.
  return result;
}

class MeanAbsoluteError : public ObjFunction {
 public:
  void Configure(Args const&) override {}

  uint32_t Targets(MetaInfo const& info) const override {
    return std::max(static_cast<size_t>(1), info.labels.Shape(1));
  }

  void GetGradient(HostDeviceVector<bst_float> const& preds, const MetaInfo& info, int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    out_gpair->SetDevice(ctx_->gpu_id);
    out_gpair->Resize(info.labels.Size());
    auto gpair = linalg::MakeVec(out_gpair);

    preds.SetDevice(ctx_->gpu_id);
    auto predt = linalg::MakeVec(&preds);

    linalg::ElementWiseKernel(ctx_, info.labels.View(ctx_->gpu_id),
                              [=] XGBOOST_DEVICE(size_t i, float const y) {});
  }

  void UpdateTreeLeaf(RowIndexCache const& row_index, MetaInfo const& info, uint32_t target,
                      RegTree* p_tree) override {
    auto& tree = *p_tree;
    std::vector<float> results;
    for (auto const& seg : row_index.indptr) {
      auto h_row_set = row_index.row_index.HostSpan().subspan(seg.begin, seg.n);
      float q{0};
      if (info.weights_.Empty()) {
        q = Quantile(0.5f, h_row_set, info.labels.HostView().Slice(linalg::All(), target));
      } else {
        q = WeightedQuantile(0.5f, h_row_set, info.labels.HostView().Slice(linalg::All(), target),
                             linalg::MakeVec(&info.weights_));
      }
      results.push_back(q);
    }
    // use the mean value
    rabit::Allreduce<rabit::op::Sum>(results.data(), results.size());
    auto world = rabit::GetWorldSize();
    std::transform(results.begin(), results.end(), results.begin(),
                   [&](float q) { return q / world; });
    for (size_t i = 0; i < row_index.indptr.size(); ++i) {
      auto seg = row_index.indptr[i];
      auto q = results[i];
      tree[seg.nidx].SetLeaf(q);  // fixme: exact tree method
    }
  }

  const char* DefaultEvalMetric() const override { return "mae"; }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String("reg:mae");
  }

  void LoadConfig(Json const& in) override {}
};
}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "regression_obj.cu"
#endif  // XGBOOST_USE_CUDA
