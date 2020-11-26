/*!
 * Copyright 2019-2021 XGBoost contributors
 */
#include "rank_obj.h"

#include <dmlc/registry.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../common/charconv.h"
#include "../common/common.h"
#include "../common/linalg_op.h"
#include "../common/math.h"
#include "../common/ranking_utils.h"
#include "xgboost/json.h"
#include "xgboost/objective.h"
#include "xgboost/parameter.h"

namespace xgboost {
namespace obj {
DMLC_REGISTER_PARAMETER(LambdaMARTParam);

class LambdaMARTNDCG : public ObjFunction {
 private:
  LambdaMARTParam ndcg_param_;
  std::string metric_;
  struct NDCGCache {
    size_t truncation{0};
    MetaInfo const* p_info;
    std::vector<float> inv_idcg;
  } h_cache_;
  std::shared_ptr<DeviceNDCGCache> d_cache_{nullptr};

 public:
  static char const* Name() { return "lambdamart:ndcg"; }

  void Configure(Args const& args) override {
    ndcg_param_.UpdateAllowUnknown(args);
    metric_ = "ndcg@" + std::to_string(ndcg_param_.lambdamart_truncation);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Name());
    out["lambdamart_param"] = ToJson(ndcg_param_);
  }

  void LoadConfig(Json const& in) override {
    auto const& obj = get<Object const>(in);
    if (obj.find("lambdamart_param") != obj.cend()) {
      FromJson(in["lambdamart_param"], &ndcg_param_);
    } else {
      // Being compatible with XGBoost version < 1.6.
      auto const& j_parameter = get<Object const>(obj.at("lambda_rank_param"));
      ndcg_param_.lambdamart_truncation =
          std::stol(get<String const>(j_parameter.at("num_pairsample")));
    }
  }

  void CalcLambdaForGroup(common::Span<float const> predt, common::Span<float const> label,
                          common::Span<GradientPair> gpair, MetaInfo const& info,
                          bst_group_t query_id) {
    auto cnt = info.group_ptr_.at(query_id + 1) - info.group_ptr_.at(query_id);
    std::fill(gpair.begin(), gpair.end(), GradientPair{});
    const double inv_IDCG = h_cache_.inv_idcg[query_id];
    auto sorted_idx = common::ArgSort<size_t>(predt, std::greater<>{});
    for (size_t i = 0; i < cnt - 1 && i < ndcg_param_.lambdamart_truncation; ++i) {
      for (size_t j = i + 1; j < cnt; ++j) {
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) {
          continue;
        }
        LambdaNDCG(label, predt, sorted_idx, i, j, inv_IDCG, gpair);
      }
    }
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds, const MetaInfo& info, int,
                   HostDeviceVector<GradientPair>* out_gpair) override {
#if defined(XGBOOST_USE_CUDA)
    auto device = tparam_->gpu_id;
    if (device != GenericParameter::kCpuId) {
      LambdaMARTGetGradientNDCGGPUKernel(preds, info, ndcg_param_.lambdamart_truncation, &d_cache_,
                                         device, out_gpair);
      return;
    }
#endif  // defined(XGBOOST_USE_CUDA)

    CHECK_NE(info.group_ptr_.size(), 0);
    size_t n_groups = info.group_ptr_.size() - 1;
    out_gpair->Resize(info.num_row_);
    auto h_gpair = out_gpair->HostSpan();
    auto h_predt = preds.ConstHostSpan();
    auto h_label = info.labels.HostView();
    auto h_weight = info.weights_.ConstHostSpan();

    if (h_cache_.p_info != &info || h_cache_.truncation != ndcg_param_.lambdamart_truncation) {
      h_cache_.inv_idcg.clear();
      h_cache_.inv_idcg.resize(n_groups, std::numeric_limits<float>::quiet_NaN());
      CheckNDCGLabelsCPUKernel(ndcg_param_, h_label.Values());
      common::ParallelFor(n_groups, tparam_->Threads(), common::Sched::Guided(), [&](auto g) {
        size_t cnt = info.group_ptr_.at(g + 1) - info.group_ptr_[g];
        auto label = h_label.Slice(
            linalg::Range(static_cast<size_t>(info.group_ptr_[g]), info.group_ptr_[g] + cnt));
        std::vector<float> sorted_labels(label.Size());
        auto span = label.Values();
        std::copy(span.cbegin(), span.cend(), sorted_labels.begin());
        std::stable_sort(sorted_labels.begin(), sorted_labels.end(), std::greater<>{});
        float inv_IDCG = CalcInvIDCG(sorted_labels, ndcg_param_.lambdamart_truncation);
        h_cache_.inv_idcg[g] = inv_IDCG;
      });
      h_cache_.p_info = &info;
      h_cache_.truncation = ndcg_param_.lambdamart_truncation;
    }

    common::ParallelFor(n_groups, tparam_->Threads(), [&](auto g) {
      size_t cnt = info.group_ptr_.at(g + 1) - info.group_ptr_[g];
      auto predts = h_predt.subspan(info.group_ptr_[g], cnt);
      auto gpairs = h_gpair.subspan(info.group_ptr_[g], cnt);
      auto labels = h_label.Values().subspan(info.group_ptr_[g], cnt);
      this->CalcLambdaForGroup(predts, labels, gpairs, info, g);

      if (!h_weight.empty()) {
        CHECK_EQ(h_weight.size(), info.group_ptr_.size() - 1);
        std::transform(gpairs.begin(), gpairs.end(), gpairs.begin(),
                       [&](GradientPair const& gpair) { return gpair * h_weight[g]; });
      }
    });
  }

  const char* DefaultEvalMetric() const override {
    CHECK(ndcg_param_.GetInitialised());
    return metric_.c_str();
  }

  ObjInfo Task() const override { return ObjInfo{ObjInfo::kRanking}; }
};

class LambdaMARTPairwise : public ObjFunction {

};

class LambdaMARTMaps : public ObjFunction {

};

void CheckNDCGLabelsCPUKernel(LambdaMARTParam const& p, common::Span<float const> labels) {
  auto label_is_integer =
      std::none_of(labels.data(), labels.data() + labels.size(), [](auto const& v) {
        auto l = std::floor(v);
        return std::fabs(l - v) > kRtEps || v < 0.0f;
      });
  CHECK(label_is_integer) << "When using relevance degree as target, labels "
                             "must be either 0 or positive integer.";
}

XGBOOST_REGISTER_OBJECTIVE(LambdaMARTNDCG, LambdaMARTNDCG::Name())
    .describe("LambdaMART with NDCG as objective")
    .set_body([]() { return new LambdaMARTNDCG(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaMARTNDCG_obsolated, "rank:ndcg")
    .describe("LambdaMART with NDCG as objective")
    .set_body([]() { return new LambdaMARTNDCG(); });
DMLC_REGISTRY_FILE_TAG(rank_obj);
}  // namespace obj
}  // namespace xgboost
