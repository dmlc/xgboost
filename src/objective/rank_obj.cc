/*!
 * Copyright 2019-2020 XGBoost contributors
 */

#include <dmlc/registry.h>

#include <functional>
#include <algorithm>
#include <vector>
#include <limits>
#include <memory>
#include <string>

#include "xgboost/objective.h"
#include "xgboost/parameter.h"
#include "xgboost/json.h"

#include "../common/common.h"
#include "../common/math.h"
#include "../common/ranking_utils.h"
#include "rank_obj.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(rank_obj);
enum class NDCGLabelType { kRelevance = 0, kGain = 1 };
}  // namespace obj
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::obj::NDCGLabelType);

#ifndef XGBOOST_USE_CUDA
#include "rank_obj.cu"
#endif  // XGBOOST_USE_CUDA

namespace xgboost {
namespace obj {
struct NDCGParam : public XGBoostParameter<NDCGParam> {
  size_t ndcg_truncation;
  NDCGLabelType ndcg_label_type;

  DMLC_DECLARE_PARAMETER(NDCGParam) {
    DMLC_DECLARE_FIELD(ndcg_truncation).set_lower_bound(1).set_default(1)
        .describe("The truncation level for NDCG.");
    DMLC_DECLARE_FIELD(ndcg_label_type).set_default(NDCGLabelType::kRelevance)
        .add_enum("relevance", NDCGLabelType::kRelevance)
        .add_enum("gain", NDCGLabelType::kGain);
  }
};

DMLC_REGISTER_PARAMETER(NDCGParam);

class LambdaMARTNDCG : public ObjFunction {
 private:
  NDCGParam ndcg_param_;
  std::string metric_;
  struct IDCGCache {
    size_t truncation{0};
    MetaInfo const* p_info;
    std::vector<float> inv_idcg;
  } idcg_cache_;

  std::shared_ptr<DeviceNDCGCache> d_cache_{nullptr};

 public:
  static char const* Name() {
    return "lambdamart:ndcg";
  }

  void Configure(Args const& args) override {
    ndcg_param_.UpdateAllowUnknown(args);
    metric_ = "ndcg@" + std::to_string(ndcg_param_.ndcg_truncation);
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Name());
    out["ndcg_param"] = ToJson(ndcg_param_);
  }

  void LoadConfig(Json const& in) override {
    auto const& obj = get<Object const>(in);
    if (obj.find("ndcg_param") != obj.cend()) {
      FromJson(in["ndcg_param"], &ndcg_param_);
    } else {
      // Being compatible with XGBoost version < 1.4.
      auto const& j_parameter = get<Object const>(obj.at("lambda_rank_param"));
      ndcg_param_.ndcg_truncation = get<Number const>(j_parameter.at("num_pairsample"));
    }
  }

  void CalcLambdaForGroup(common::Span<float const> predt, common::Span<float const> label,
                          common::Span<GradientPair> gpair,
                          MetaInfo const &info, bst_group_t query_id) {
    auto cnt = info.group_ptr_.at(query_id+1) - info.group_ptr_.at(query_id);
    std::fill(gpair.begin(), gpair.end(), GradientPair{});
    const double inv_IDCG = idcg_cache_.inv_idcg[query_id];
    auto sorted_idx = common::ArgSort<size_t>(predt, std::greater<>{});
    for (size_t i = 0; i < cnt - 1 && i < ndcg_param_.ndcg_truncation; ++i) {
      for (size_t j = i + 1; j < cnt; ++j) {
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        LambdaNDCG(label, predt, sorted_idx, i, j, inv_IDCG, gpair);
      }
    }
  }

  void GetGradient(const HostDeviceVector<bst_float> &preds,
                   const MetaInfo &info, int,
                   HostDeviceVector<GradientPair> *out_gpair) override {
#if defined(XGBOOST_USE_CUDA)
    auto device = tparam_->gpu_id;
    if (device != GenericParameter::kCpuId) {
      LambdaMARTGetGradientNDCGGPUKernel(
          preds, info, ndcg_param_.ndcg_truncation, &d_cache_, device, out_gpair);
      return;
    }
#endif  // defined(XGBOOST_USE_CUDA)

    CHECK_NE(info.group_ptr_.size(), 0);
    size_t n_groups = info.group_ptr_.size() - 1;
    out_gpair->Resize(info.num_row_);
    auto h_gpair = out_gpair->HostSpan();
    auto h_predt = preds.ConstHostSpan();
    auto h_label = info.labels_.ConstHostSpan();

    if (idcg_cache_.p_info != &info ||
        idcg_cache_.truncation != ndcg_param_.ndcg_truncation) {
      idcg_cache_.inv_idcg.clear();
      idcg_cache_.inv_idcg.resize(n_groups, std::numeric_limits<float>::quiet_NaN());
#pragma omp parallel for schedule(guided)
      for (size_t g = 0; g < n_groups; ++g) {
        size_t cnt = info.group_ptr_.at(g + 1) - info.group_ptr_[g];
        auto label = h_label.subspan(info.group_ptr_[g], cnt);
        std::vector<float> sorted_labels(label.size());
        std::copy(label.cbegin(), label.cend(), sorted_labels.begin());
        std::stable_sort(sorted_labels.begin(), sorted_labels.end(),
                         std::greater<>{});
        float inv_IDCG =
            CalcInvIDCG(sorted_labels, ndcg_param_.ndcg_truncation);
        idcg_cache_.inv_idcg[g] = inv_IDCG;
      }
      idcg_cache_.p_info = &info;
      idcg_cache_.truncation = ndcg_param_.ndcg_truncation;
    }

#pragma omp parallel for schedule(guided)
    for (size_t g = 0; g < n_groups; ++g) {
      size_t cnt = info.group_ptr_.at(g + 1) - info.group_ptr_[g];
      auto predt = h_predt.subspan(info.group_ptr_[g], cnt);
      auto gpair = h_gpair.subspan(info.group_ptr_[g], cnt);
      auto label = h_label.subspan(info.group_ptr_[g], cnt);
      this->CalcLambdaForGroup(predt, label, gpair, info, g);
    }
  }

  const char* DefaultEvalMetric() const override {
    CHECK(ndcg_param_.GetInitialised());
    return metric_.c_str();
  }
};

XGBOOST_REGISTER_OBJECTIVE(LambdaMARTNDCG, LambdaMARTNDCG::Name())
    .describe("LambdaMART with NDCG as objective")
    .set_body([]() { return new LambdaMARTNDCG(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaMARTNDCG_obsolated, "rank:ndcg")
    .describe("LambdaMART with NDCG as objective")
    .set_body([]() {
      LOG(WARNING) << "Objective `rank:ndcg` is deprecated in 1.4.  Use "
                      "`lambdamart:ndcg` instead.";
      return new LambdaMARTNDCG();
    });
}  // namespace obj
}  // namespace xgboost
