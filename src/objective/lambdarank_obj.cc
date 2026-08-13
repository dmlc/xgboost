/**
 * Copyright 2023-2026, XGBoost contributors
 */
#include "lambdarank_obj.h"

#include <dmlc/registry.h>  // for DMLC_REGISTRY_FILE_TAG

#include <algorithm>    // for transform, copy, fill_n, min, max
#include <cmath>        // for pow, log2
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <map>          // for operator!=
#include <memory>       // for shared_ptr, __shared_ptr_access, allocator
#include <ostream>      // for operator<<, basic_ostream
#include <string>       // for char_traits, operator<, basic_string, string
#include <tuple>        // for apply, make_tuple
#include <type_traits>  // for is_floating_point
#include <utility>      // for pair, swap

#include "../common/error_msg.h"         // for GroupWeight, LabelScoreSize
#include "../common/linalg_op.h"         // for begin, cbegin, cend, SaveVector
#include "../common/optional_weight.h"   // for MakeOptionalWeights, OptionalWeights
#include "../common/ranking_utils.h"     // for RankingCache, LambdaRankParam, MAPCache, NDCGC...
#include "../common/threading_utils.h"   // for ParallelFor, Sched
#include "init_estimation.h"             // for FitIntercept
#include "xgboost/base.h"                // for bst_group_t, GradientPair, kRtEps, GradientPai...
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json, get, Value, ToJson, F32Array, FromJson, IsA
#include "xgboost/linalg.h"              // for Vector, Range, TensorView, VectorView, All
#include "xgboost/logging.h"             // for LogCheck_EQ, CHECK_EQ, CHECK, LogCheck_LE, CHE...
#include "xgboost/objective.h"           // for ObjFunctionReg, XGBOOST_REGISTER_OBJECTIVE
#include "xgboost/span.h"                // for Span, operator!=
#include "xgboost/string_view.h"         // for operator<<, StringView
#include "xgboost/task.h"                // for ObjInfo

namespace xgboost::obj {
/**
 * \brief Base class for pair-wise learning to rank.
 *
 *   See `From RankNet to LambdaRank to LambdaMART: An Overview` for a description of the
 *   algorithm.
 *
 */
template <typename Loss, typename Cache>
class LambdaRankObj : public FitIntercept {
  MetaInfo const* p_info_{nullptr};
  bool warned_unbiased_{false};

  void DisableUnbiased() {
    if (param_.lambdarank_unbiased) {
      if (!warned_unbiased_) {
        LOG(WARNING) << "`lambdarank_unbiased` was removed in 3.5.0. Falling back to standard "
                        "LambdaRank.";
        warned_unbiased_ = true;
      }
      param_.lambdarank_unbiased = false;
    }
  }

 protected:
  ltr::LambdaRankParam param_;
  // cache
  std::shared_ptr<ltr::RankingCache> p_cache_;

  [[nodiscard]] std::shared_ptr<Cache> GetCache() const {
    auto ptr = std::static_pointer_cast<Cache>(p_cache_);
    CHECK(ptr);
    return ptr;
  }

  // Calculate lambda gradient for each group on CPU.
  template <bool norm_by_diff, typename Delta>
  void CalcLambdaForGroup(std::uint32_t seed, common::Span<float const> g_predt,
                          linalg::VectorView<float const> g_label, float w,
                          common::Span<std::size_t const> g_rank, bst_group_t g, Delta delta,
                          linalg::VectorView<GradientPair> g_gpair) {
    std::fill_n(g_gpair.Values().data(), g_gpair.Size(), GradientPair{});

    // Normalization, first used by LightGBM.
    // https://github.com/lightgbm-org/LightGBM/pull/2331#issuecomment-523259298
    double sum_lambda{0.0};

    auto delta_op = [&](auto const&... args) {
      return delta(args..., g);
    };

    auto loop = [&](std::size_t i, std::size_t j) {
      // higher/lower on the target ranked list
      std::size_t rank_high = i, rank_low = j;
      if (g_label(g_rank[rank_high]) == g_label(g_rank[rank_low])) {
        return;
      }
      if (g_label(g_rank[rank_high]) < g_label(g_rank[rank_low])) {
        std::swap(rank_high, rank_low);
      }

      auto pg = LambdaGrad<norm_by_diff>(g_label, g_predt, g_rank, rank_high, rank_low, delta_op);
      auto ng = Repulse(pg);

      std::size_t idx_high = g_rank[rank_high];
      std::size_t idx_low = g_rank[rank_low];
      g_gpair(idx_high) += pg;
      g_gpair(idx_low) += ng;

      sum_lambda += -2.0 * static_cast<double>(pg.GetGrad());
    };

    MakePairs(ctx_, seed, p_cache_, g, g_label, g_rank, loop);
    if (param_.lambdarank_normalization) {
      double norm = 1.0;
      if (param_.IsMean()) {
        // Normalize using the number of pairs for mean.
        auto n_pairs = this->p_cache_->Param().NumPair();
        auto scale = 1.0 / static_cast<double>(n_pairs);
        norm = scale;
      } else {
        // Normalize using gradient for top-k.
        if (sum_lambda > 0.0) {
          norm = std::log2(1.0 + sum_lambda) / sum_lambda;
        }
      }
      if (norm != 1.0) {
        std::transform(linalg::begin(g_gpair), linalg::end(g_gpair), linalg::begin(g_gpair),
                       [norm](GradientPair const& g) { return g * norm; });
      }
    }

    auto w_norm = p_cache_->WeightNorm();
    std::transform(g_gpair.Values().data(), g_gpair.Values().data() + g_gpair.Size(),
                   g_gpair.Values().data(),
                   [&](GradientPair const& gpair) { return gpair * w * w_norm; });
  }

 public:
  std::set<std::string> Configure(Args const& args) override {
    auto used = UpdateAndGetUsedParameters(&param_, args);
    this->DisableUnbiased();
    return used;
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Loss::Name());
    out["lambdarank_param"] = ToJson(param_);
  }
  void LoadConfig(Json const& in) override {
    auto const& obj = get<Object const>(in);
    if (obj.find("lambdarank_param") != obj.cend()) {
      FromJson(in["lambdarank_param"], &param_);
    }
    this->DisableUnbiased();
  }

  [[nodiscard]] ObjInfo Task() const override { return ObjInfo{ObjInfo::kRanking}; }

  [[nodiscard]] bst_target_t Targets(MetaInfo const& info) const override {
    CHECK_LE(info.labels.Shape(1), 1) << "multi-output for LTR is not yet supported.";
    return 1;
  }

  [[nodiscard]] const char* RankEvalMetric(StringView metric) const {
    static thread_local std::string name;
    if (param_.HasTruncation()) {
      name = ltr::MakeMetricName(metric, param_.NumPair(), false);
    } else {
      name = ltr::MakeMetricName(metric, param_.NotSet(), false);
    }
    return name.c_str();
  }

  void GetGradient(HostDeviceVector<float> const& predt, MetaInfo const& info, std::int32_t,
                   linalg::Matrix<GradientPair>* out_gpair) override {
    CHECK_EQ(info.labels.Size(), predt.Size()) << error::LabelScoreSize();

    // init/renew cache
    if (!p_cache_ || p_info_ != &info || p_cache_->Param() != param_) {
      p_cache_ = std::make_shared<Cache>(ctx_, info, param_);
      p_info_ = &info;
    }
    auto n_groups = p_cache_->Groups();
    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), n_groups) << error::GroupWeight();
    }

    std::uint32_t seed{0};
    if (param_.IsMean()) {
      seed = static_cast<std::uint32_t>(ctx_->Rng()());
    }
    static_cast<Loss*>(this)->GetGradientImpl(seed, predt, info, out_gpair);
  }
};

class LambdaRankNDCG : public LambdaRankObj<LambdaRankNDCG, ltr::NDCGCache> {
 public:
  template <bool exp_gain>
  void CalcLambdaForGroupNDCG(std::uint32_t seed, common::Span<float const> g_predt,
                              linalg::VectorView<float const> g_label, float w,
                              common::Span<std::size_t const> g_rank,
                              linalg::VectorView<GradientPair> g_gpair,
                              linalg::VectorView<double const> inv_IDCG,
                              common::Span<double const> discount, bst_group_t g) {
    auto delta = [&](auto y_high, auto y_low, std::size_t rank_high, std::size_t rank_low,
                     bst_group_t g) {
      static_assert(std::is_floating_point_v<decltype(y_high)>);
      return DeltaNDCG<exp_gain>(y_high, y_low, rank_high, rank_low, inv_IDCG(g), discount);
    };

    if (this->param_.lambdarank_score_normalization) {
      this->CalcLambdaForGroup<true>(seed, g_predt, g_label, w, g_rank, g, delta, g_gpair);
    } else {
      this->CalcLambdaForGroup<false>(seed, g_predt, g_label, w, g_rank, g, delta, g_gpair);
    }
  }

  void GetGradientImpl(std::uint32_t seed, const HostDeviceVector<float>& predt,
                       const MetaInfo& info, linalg::Matrix<GradientPair>* out_gpair) {
    if (ctx_->IsCUDA()) {
      cuda_impl::LambdaRankGetGradientNDCG(ctx_, seed, predt, info, GetCache(), out_gpair);
      return;
    }

    auto device = ctx_->Device().IsSycl() ? DeviceOrd::CPU() : ctx_->Device();
    bst_group_t n_groups = p_cache_->Groups();
    auto gptr = p_cache_->DataGroupPtr(ctx_);

    out_gpair->SetDevice(device);
    out_gpair->Reshape(info.num_row_, 1);

    auto h_gpair = out_gpair->HostView();
    auto h_predt = predt.ConstHostSpan();
    auto h_label = info.labels.HostView();
    auto h_weight = common::MakeOptionalWeights(device, info.weights_);
    auto make_range = [&](bst_group_t g) {
      return linalg::Range(gptr[g], gptr[g + 1]);
    };

    auto dct = GetCache()->Discount(ctx_);
    auto rank_idx = p_cache_->SortedIdx(ctx_, h_predt);
    auto inv_IDCG = GetCache()->InvIDCG(ctx_);

    common::ParallelFor(n_groups, ctx_->Threads(), common::Sched::Guided(), [&](auto g) {
      std::size_t cnt = gptr[g + 1] - gptr[g];
      auto w = h_weight[g];
      auto g_predt = h_predt.subspan(gptr[g], cnt);
      auto g_gpair =
          h_gpair.Slice(linalg::Range(static_cast<std::size_t>(gptr[g]), gptr[g] + cnt), 0);
      auto g_label = h_label.Slice(make_range(g), 0);
      auto g_rank = rank_idx.subspan(gptr[g], cnt);

      auto args =
          std::make_tuple(this, seed, g_predt, g_label, w, g_rank, g_gpair, inv_IDCG, dct, g);

      if (param_.ndcg_exp_gain) {
        std::apply(&LambdaRankNDCG::CalcLambdaForGroupNDCG<true>, args);
      } else {
        std::apply(&LambdaRankNDCG::CalcLambdaForGroupNDCG<false>, args);
      }
    });
  }

  static char const* Name() { return "rank:ndcg"; }
  [[nodiscard]] const char* DefaultEvalMetric() const override {
    return this->RankEvalMetric("ndcg");
  }
  [[nodiscard]] Json DefaultMetricConfig() const override {
    Json config{Object{}};
    config["name"] = String{DefaultEvalMetric()};
    config["lambdarank_param"] = ToJson(param_);
    return config;
  }
};

namespace cuda_impl {
#if !defined(XGBOOST_USE_CUDA)
void LambdaRankGetGradientNDCG(Context const*, std::uint32_t, HostDeviceVector<float> const&,
                               const MetaInfo&, std::shared_ptr<ltr::NDCGCache>,
                               linalg::Matrix<GradientPair>*) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace cuda_impl

namespace cpu_impl {
void MAPStat(Context const* ctx, linalg::VectorView<float const> label,
             common::Span<std::size_t const> rank_idx, std::shared_ptr<ltr::MAPCache> p_cache) {
  auto h_n_rel = p_cache->NumRelevant(ctx);
  auto gptr = p_cache->DataGroupPtr(ctx);

  CHECK_EQ(h_n_rel.size(), gptr.back());
  CHECK_EQ(h_n_rel.size(), label.Size());

  auto h_acc = p_cache->Acc(ctx);

  common::ParallelFor(p_cache->Groups(), ctx->Threads(), [&](auto g) {
    auto cnt = gptr[g + 1] - gptr[g];
    auto g_n_rel = h_n_rel.subspan(gptr[g], cnt);
    auto g_rank = rank_idx.subspan(gptr[g], cnt);
    auto g_label = label.Slice(linalg::Range(gptr[g], gptr[g + 1]));

    // The number of relevant documents at each position
    g_n_rel[0] = g_label(g_rank[0]);
    for (std::size_t k = 1; k < g_rank.size(); ++k) {
      g_n_rel[k] = g_n_rel[k - 1] + g_label(g_rank[k]);
    }

    // \sum l_k/k
    auto g_acc = h_acc.subspan(gptr[g], cnt);
    g_acc[0] = g_label(g_rank[0]) / 1.0;

    for (std::size_t k = 1; k < g_rank.size(); ++k) {
      g_acc[k] = g_acc[k - 1] + (g_label(g_rank[k]) / static_cast<double>(k + 1));
    }
  });
}
}  // namespace cpu_impl

class LambdaRankMAP : public LambdaRankObj<LambdaRankMAP, ltr::MAPCache> {
 public:
  void GetGradientImpl(std::uint32_t seed, const HostDeviceVector<float>& predt,
                       const MetaInfo& info, linalg::Matrix<GradientPair>* out_gpair) {
    if (ctx_->IsCUDA()) {
      return cuda_impl::LambdaRankGetGradientMAP(ctx_, seed, predt, info, GetCache(), out_gpair);
    }

    auto gptr = p_cache_->DataGroupPtr(ctx_).data();
    bst_group_t n_groups = p_cache_->Groups();

    CHECK_EQ(info.labels.Shape(1), 1) << "multi-target for learning to rank is not yet supported.";
    auto device = ctx_->Device().IsSycl() ? DeviceOrd::CPU() : ctx_->Device();
    out_gpair->SetDevice(device);
    out_gpair->Reshape(info.num_row_, this->Targets(info));

    auto h_gpair = out_gpair->HostView();
    auto h_label = info.labels.HostView().Slice(linalg::All(), 0);
    auto h_predt = predt.ConstHostSpan();
    auto rank_idx = p_cache_->SortedIdx(ctx_, h_predt);
    auto h_weight = common::MakeOptionalWeights(device, info.weights_);

    auto make_range = [&](bst_group_t g) {
      return linalg::Range(gptr[g], gptr[g + 1]);
    };

    cpu_impl::MAPStat(ctx_, h_label, rank_idx, GetCache());
    auto n_rel = GetCache()->NumRelevant(ctx_);
    auto acc = GetCache()->Acc(ctx_);

    auto delta_map = [&](auto y_high, auto y_low, std::size_t rank_high, std::size_t rank_low,
                         bst_group_t g) {
      if (rank_high > rank_low) {
        std::swap(rank_high, rank_low);
        std::swap(y_high, y_low);
      }
      auto cnt = gptr[g + 1] - gptr[g];
      // In a hot loop
      auto g_n_rel = common::Span<double const>{n_rel.data() + gptr[g], cnt};
      auto g_acc = common::Span<double const>{acc.data() + gptr[g], cnt};
      auto d = DeltaMAP(y_high, y_low, rank_high, rank_low, g_n_rel, g_acc);
      return d;
    };
    using D = decltype(delta_map);

    common::ParallelFor(n_groups, ctx_->Threads(), [&](auto g) {
      auto cnt = gptr[g + 1] - gptr[g];
      auto w = h_weight[g];
      auto g_predt = h_predt.subspan(gptr[g], cnt);
      auto g_gpair = h_gpair.Slice(linalg::Range(gptr[g], gptr[g] + cnt), 0);
      auto g_label = h_label.Slice(make_range(g));
      auto g_rank = rank_idx.subspan(gptr[g], cnt);

      auto args = std::make_tuple(this, seed, g_predt, g_label, w, g_rank, g, delta_map, g_gpair);

      if (this->param_.lambdarank_score_normalization) {
        std::apply(&LambdaRankMAP::CalcLambdaForGroup<true, D>, args);
      } else {
        std::apply(&LambdaRankMAP::CalcLambdaForGroup<false, D>, args);
      }
    });
  }
  static char const* Name() { return "rank:map"; }
  [[nodiscard]] const char* DefaultEvalMetric() const override {
    return this->RankEvalMetric("map");
  }
};

#if !defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
void MAPStat(Context const*, MetaInfo const&, common::Span<std::size_t const>,
             std::shared_ptr<ltr::MAPCache>) {
  common::AssertGPUSupport();
}

void LambdaRankGetGradientMAP(Context const*, std::uint32_t, HostDeviceVector<float> const&,
                              const MetaInfo&, std::shared_ptr<ltr::MAPCache>,
                              linalg::Matrix<GradientPair>*) {
  common::AssertGPUSupport();
}
}  // namespace cuda_impl
#endif  // !defined(XGBOOST_USE_CUDA)

/**
 * \brief The RankNet loss.
 */
class LambdaRankPairwise : public LambdaRankObj<LambdaRankPairwise, ltr::RankingCache> {
 public:
  void GetGradientImpl(std::uint32_t seed, const HostDeviceVector<float>& predt,
                       const MetaInfo& info, linalg::Matrix<GradientPair>* out_gpair) {
    if (ctx_->IsCUDA()) {
      return cuda_impl::LambdaRankGetGradientPairwise(ctx_, seed, predt, info, GetCache(),
                                                      out_gpair);
    }

    auto gptr = p_cache_->DataGroupPtr(ctx_);
    bst_group_t n_groups = p_cache_->Groups();

    out_gpair->SetDevice(ctx_->Device());
    out_gpair->Reshape(info.num_row_, this->Targets(info));

    auto h_gpair = out_gpair->HostView();
    auto h_label = info.labels.HostView().Slice(linalg::All(), 0);
    auto h_predt = predt.ConstHostSpan();
    auto h_weight = common::MakeOptionalWeights(ctx_->Device(), info.weights_);

    auto make_range = [&](bst_group_t g) {
      return linalg::Range(gptr[g], gptr[g + 1]);
    };
    auto rank_idx = p_cache_->SortedIdx(ctx_, h_predt);

    auto delta = [](auto...) {
      return 1.0;
    };
    using D = decltype(delta);

    common::ParallelFor(n_groups, ctx_->Threads(), [&](auto g) {
      auto cnt = gptr[g + 1] - gptr[g];
      auto w = h_weight[g];
      auto g_predt = h_predt.subspan(gptr[g], cnt);
      auto g_gpair = h_gpair.Slice(linalg::Range(gptr[g], gptr[g] + cnt), 0);
      auto g_label = h_label.Slice(make_range(g));
      auto g_rank = rank_idx.subspan(gptr[g], cnt);

      auto args = std::make_tuple(this, seed, g_predt, g_label, w, g_rank, g, delta, g_gpair);
      if (this->param_.lambdarank_score_normalization) {
        std::apply(&LambdaRankPairwise::CalcLambdaForGroup<true, D>, args);
      } else {
        std::apply(&LambdaRankPairwise::CalcLambdaForGroup<false, D>, args);
      }
    });
  }

  static char const* Name() { return "rank:pairwise"; }
  [[nodiscard]] const char* DefaultEvalMetric() const override {
    return this->RankEvalMetric("ndcg");
  }

  [[nodiscard]] Json DefaultMetricConfig() const override {
    Json config{Object{}};
    config["name"] = String{DefaultEvalMetric()};
    config["lambdarank_param"] = ToJson(param_);
    return config;
  }
};

#if !defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
void LambdaRankGetGradientPairwise(Context const*, std::uint32_t, HostDeviceVector<float> const&,
                                   const MetaInfo&, std::shared_ptr<ltr::RankingCache>,
                                   linalg::Matrix<GradientPair>*) {
  common::AssertGPUSupport();
}
}  // namespace cuda_impl
#endif  // !defined(XGBOOST_USE_CUDA)

XGBOOST_REGISTER_OBJECTIVE(LambdaRankNDCG, LambdaRankNDCG::Name())
    .describe("LambdaRank with NDCG loss as objective")
    .set_body([]() { return new LambdaRankNDCG{}; });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankPairwise, LambdaRankPairwise::Name())
    .describe("LambdaRank with RankNet loss as objective")
    .set_body([]() { return new LambdaRankPairwise{}; });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankMAP, LambdaRankMAP::Name())
    .describe("LambdaRank with MAP loss as objective.")
    .set_body([]() { return new LambdaRankMAP{}; });

DMLC_REGISTRY_FILE_TAG(lambdarank_obj);
}  // namespace xgboost::obj
