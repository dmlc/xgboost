/**
 * Copyright (c) 2023, XGBoost contributors
 */
#include "lambdarank_obj.h"

#include <dmlc/registry.h>                 // for DMLC_REGISTRY_FILE_TAG

#include <algorithm>                       // for transform, copy, fill_n, min, max
#include <cmath>                           // for pow, log2
#include <cstddef>                         // for size_t
#include <cstdint>                         // for int32_t
#include <map>                             // for operator!=
#include <memory>                          // for shared_ptr, __shared_ptr_access, allocator
#include <ostream>                         // for operator<<, basic_ostream
#include <string>                          // for char_traits, operator<, basic_string, string
#include <tuple>                           // for apply, make_tuple
#include <type_traits>                     // for is_floating_point
#include <utility>                         // for pair, swap
#include <vector>                          // for vector

#include "../common/error_msg.h"           // for GroupWeight, LabelScoreSize
#include "../common/linalg_op.h"           // for begin, cbegin, cend
#include "../common/optional_weight.h"     // for MakeOptionalWeights, OptionalWeights
#include "../common/ranking_utils.h"       // for RankingCache, LambdaRankParam, MAPCache, NDCGC...
#include "../common/threading_utils.h"     // for ParallelFor, Sched
#include "../common/transform_iterator.h"  // for IndexTransformIter
#include "init_estimation.h"               // for FitIntercept
#include "xgboost/base.h"                  // for bst_group_t, GradientPair, kRtEps, GradientPai...
#include "xgboost/context.h"               // for Context
#include "xgboost/data.h"                  // for MetaInfo
#include "xgboost/host_device_vector.h"    // for HostDeviceVector
#include "xgboost/json.h"                  // for Json, get, Value, ToJson, F32Array, FromJson, IsA
#include "xgboost/linalg.h"                // for Vector, Range, TensorView, VectorView, All
#include "xgboost/logging.h"               // for LogCheck_EQ, CHECK_EQ, CHECK, LogCheck_LE, CHE...
#include "xgboost/objective.h"             // for ObjFunctionReg, XGBOOST_REGISTER_OBJECTIVE
#include "xgboost/span.h"                  // for Span, operator!=
#include "xgboost/string_view.h"           // for operator<<, StringView
#include "xgboost/task.h"                  // for ObjInfo

namespace xgboost::obj {
namespace cpu_impl {
void LambdaRankUpdatePositionBias(Context const* ctx, linalg::VectorView<double const> li_full,
                                  linalg::VectorView<double const> lj_full,
                                  linalg::Vector<double>* p_ti_plus,
                                  linalg::Vector<double>* p_tj_minus, linalg::Vector<double>* p_li,
                                  linalg::Vector<double>* p_lj,
                                  std::shared_ptr<ltr::RankingCache> p_cache) {
  auto ti_plus = p_ti_plus->HostView();
  auto tj_minus = p_tj_minus->HostView();
  auto li = p_li->HostView();
  auto lj = p_lj->HostView();

  auto gptr = p_cache->DataGroupPtr(ctx);
  auto n_groups = p_cache->Groups();
  auto regularizer = p_cache->Param().Regularizer();

  // Aggregate over query groups
  for (bst_group_t g{0}; g < n_groups; ++g) {
    auto begin = gptr[g];
    auto end = gptr[g + 1];
    std::size_t group_size = end - begin;
    auto n = std::min(group_size, p_cache->MaxPositionSize());

    auto g_li = li_full.Slice(linalg::Range(begin, end));
    auto g_lj = lj_full.Slice(linalg::Range(begin, end));

    for (std::size_t i{0}; i < n; ++i) {
      li(i) += g_li(i);
      lj(i) += g_lj(i);
    }
  }

  // The ti+ is not guaranteed to decrease since it depends on the |\delta Z|
  //
  // The update normalizes the ti+ to make ti+(0) equal to 1, which breaks the probability
  // meaning. The reasoning behind the normalization is not clear, here we are just
  // following the authors.
  for (std::size_t i = 0; i < ti_plus.Size(); ++i) {
    if (li(0) >= Eps64()) {
      ti_plus(i) = std::pow(li(i) / li(0), regularizer);  // eq.30
    }
    if (lj(0) >= Eps64()) {
      tj_minus(i) = std::pow(lj(i) / lj(0), regularizer);  // eq.31
    }
    assert(!std::isinf(ti_plus(i)));
    assert(!std::isinf(tj_minus(i)));
  }
}
}  // namespace cpu_impl

/**
 * \brief Base class for pair-wise learning to rank.
 *
 *   See `From RankNet to LambdaRank to LambdaMART: An Overview` for a description of the
 *   algorithm.
 *
 *   In addition to ranking, this also implements `Unbiased LambdaMART: An Unbiased
 *   Pairwise Learning-to-Rank Algorithm`.
 */
template <typename Loss, typename Cache>
class LambdaRankObj : public FitIntercept {
  MetaInfo const* p_info_{nullptr};

  // Update position biased for unbiased click data
  void UpdatePositionBias() {
    li_full_.SetDevice(ctx_->gpu_id);
    lj_full_.SetDevice(ctx_->gpu_id);
    li_.SetDevice(ctx_->gpu_id);
    lj_.SetDevice(ctx_->gpu_id);

    if (ctx_->IsCPU()) {
      cpu_impl::LambdaRankUpdatePositionBias(ctx_, li_full_.View(ctx_->gpu_id),
                                             lj_full_.View(ctx_->gpu_id), &ti_plus_, &tj_minus_,
                                             &li_, &lj_, p_cache_);
    } else {
      cuda_impl::LambdaRankUpdatePositionBias(ctx_, li_full_.View(ctx_->gpu_id),
                                              lj_full_.View(ctx_->gpu_id), &ti_plus_, &tj_minus_,
                                              &li_, &lj_, p_cache_);
    }

    li_full_.Data()->Fill(0.0);
    lj_full_.Data()->Fill(0.0);

    li_.Data()->Fill(0.0);
    lj_.Data()->Fill(0.0);
  }

 protected:
  // L / tj-* (eq. 30)
  linalg::Vector<double> li_;
  // L / ti+* (eq. 31)
  linalg::Vector<double> lj_;
  // position bias ratio for relevant doc, ti+ (eq. 30)
  linalg::Vector<double> ti_plus_;
  // position bias ratio for irrelevant doc, tj- (eq. 31)
  linalg::Vector<double> tj_minus_;
  // li buffer for all samples
  linalg::Vector<double> li_full_;
  // lj buffer for all samples
  linalg::Vector<double> lj_full_;

  ltr::LambdaRankParam param_;
  // cache
  std::shared_ptr<ltr::RankingCache> p_cache_;

  [[nodiscard]] std::shared_ptr<Cache> GetCache() const {
    auto ptr = std::static_pointer_cast<Cache>(p_cache_);
    CHECK(ptr);
    return ptr;
  }

  // get group view for li/lj
  linalg::VectorView<double> GroupLoss(bst_group_t g, linalg::Vector<double>* v) const {
    auto gptr = p_cache_->DataGroupPtr(ctx_);
    auto begin = gptr[g];
    auto end = gptr[g + 1];
    if (param_.lambdarank_unbiased) {
      return v->HostView().Slice(linalg::Range(begin, end));
    }
    return v->HostView();
  }

  // Calculate lambda gradient for each group on CPU.
  template <bool unbiased, typename Delta>
  void CalcLambdaForGroup(std::int32_t iter, common::Span<float const> g_predt,
                          linalg::VectorView<float const> g_label, float w,
                          common::Span<std::size_t const> g_rank, bst_group_t g, Delta delta,
                          common::Span<GradientPair> g_gpair) {
    std::fill_n(g_gpair.data(), g_gpair.size(), GradientPair{});
    auto p_gpair = g_gpair.data();

    auto ti_plus = ti_plus_.HostView();
    auto tj_minus = tj_minus_.HostView();

    auto li = GroupLoss(g, &li_full_);
    auto lj = GroupLoss(g, &lj_full_);

    // Normalization, first used by LightGBM.
    // https://github.com/microsoft/LightGBM/pull/2331#issuecomment-523259298
    double sum_lambda{0.0};

    auto delta_op = [&](auto const&... args) { return delta(args..., g); };

    auto loop = [&](std::size_t i, std::size_t j) {
      // higher/lower on the target ranked list
      std::size_t rank_high = i, rank_low = j;
      if (g_label(g_rank[rank_high]) == g_label(g_rank[rank_low])) {
        return;
      }
      if (g_label(g_rank[rank_high]) < g_label(g_rank[rank_low])) {
        std::swap(rank_high, rank_low);
      }

      double cost;
      auto pg = LambdaGrad<unbiased>(g_label, g_predt, g_rank, rank_high, rank_low, delta_op,
                                     ti_plus, tj_minus, &cost);
      auto ng = Repulse(pg);

      std::size_t idx_high = g_rank[rank_high];
      std::size_t idx_low = g_rank[rank_low];
      p_gpair[idx_high] += pg;
      p_gpair[idx_low] += ng;

      if (unbiased) {
        auto k = ti_plus.Size();
        // We can probably use all the positions. If we skip the update due to having
        // high/low > k, we might be losing out too many pairs. On the other hand, if we
        // cap the position, then we might be accumulating too many tail bias into the
        // last tracked position.
        // We use `idx_high` since it represents the original position from the label
        // list, and label list is assumed to be sorted.
        if (idx_high < k && idx_low < k) {
          if (tj_minus(idx_low) >= Eps64()) {
            li(idx_high) += cost / tj_minus(idx_low);  // eq.30
          }
          if (ti_plus(idx_high) >= Eps64()) {
            lj(idx_low) += cost / ti_plus(idx_high);  // eq.31
          }
        }
      }

      sum_lambda += -2.0 * static_cast<double>(pg.GetGrad());
    };

    MakePairs(ctx_, iter, p_cache_, g, g_label, g_rank, loop);
    if (sum_lambda > 0.0) {
      double norm = std::log2(1.0 + sum_lambda) / sum_lambda;
      std::transform(g_gpair.data(), g_gpair.data() + g_gpair.size(), g_gpair.data(),
                     [norm](GradientPair const& g) { return g * norm; });
    }

    auto w_norm = p_cache_->WeightNorm();
    std::transform(g_gpair.begin(), g_gpair.end(), g_gpair.begin(),
                   [&](GradientPair const& gpair) { return gpair * w * w_norm; });
  }

 public:
  void Configure(Args const& args) override { param_.UpdateAllowUnknown(args); }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(Loss::Name());
    out["lambdarank_param"] = ToJson(param_);

    auto save_bias = [](linalg::Vector<double> const& in, Json out) {
      auto& out_array = get<F32Array>(out);
      out_array.resize(in.Size());
      auto h_in = in.HostView();
      std::copy(linalg::cbegin(h_in), linalg::cend(h_in), out_array.begin());
    };

    if (param_.lambdarank_unbiased) {
      out["ti+"] = F32Array();
      save_bias(ti_plus_, out["ti+"]);
      out["tj-"] = F32Array();
      save_bias(tj_minus_, out["tj-"]);
    }
  }
  void LoadConfig(Json const& in) override {
    auto const& obj = get<Object const>(in);
    if (obj.find("lambdarank_param") != obj.cend()) {
      FromJson(in["lambdarank_param"], &param_);
    }

    if (param_.lambdarank_unbiased) {
      auto load_bias = [](Json in, linalg::Vector<double>* out) {
        if (IsA<F32Array>(in)) {
          // JSON
          auto const& array = get<F32Array>(in);
          out->Reshape(array.size());
          auto h_out = out->HostView();
          std::copy(array.cbegin(), array.cend(), linalg::begin(h_out));
        } else {
          // UBJSON
          auto const& array = get<Array>(in);
          out->Reshape(array.size());
          auto h_out = out->HostView();
          std::transform(array.cbegin(), array.cend(), linalg::begin(h_out),
                         [](Json const& v) { return get<Number const>(v); });
        }
      };
      load_bias(in["ti+"], &ti_plus_);
      load_bias(in["tj-"], &tj_minus_);
    }
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

  void GetGradient(HostDeviceVector<float> const& predt, MetaInfo const& info, std::int32_t iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
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

    if (ti_plus_.Size() == 0 && param_.lambdarank_unbiased) {
      CHECK_EQ(iter, 0);
      ti_plus_ = linalg::Constant<double>(ctx_, 1.0, p_cache_->MaxPositionSize());
      tj_minus_ = linalg::Constant<double>(ctx_, 1.0, p_cache_->MaxPositionSize());

      li_ = linalg::Zeros<double>(ctx_, p_cache_->MaxPositionSize());
      lj_ = linalg::Zeros<double>(ctx_, p_cache_->MaxPositionSize());

      li_full_ = linalg::Zeros<double>(ctx_, info.num_row_);
      lj_full_ = linalg::Zeros<double>(ctx_, info.num_row_);
    }
    static_cast<Loss*>(this)->GetGradientImpl(iter, predt, info, out_gpair);

    if (param_.lambdarank_unbiased) {
      this->UpdatePositionBias();
    }
  }
};

class LambdaRankNDCG : public LambdaRankObj<LambdaRankNDCG, ltr::NDCGCache> {
 public:
  template <bool unbiased, bool exp_gain>
  void CalcLambdaForGroupNDCG(std::int32_t iter, common::Span<float const> g_predt,
                              linalg::VectorView<float const> g_label, float w,
                              common::Span<std::size_t const> g_rank,
                              common::Span<GradientPair> g_gpair,
                              linalg::VectorView<double const> inv_IDCG,
                              common::Span<double const> discount, bst_group_t g) {
    auto delta = [&](auto y_high, auto y_low, std::size_t rank_high, std::size_t rank_low,
                     bst_group_t g) {
      static_assert(std::is_floating_point<decltype(y_high)>::value);
      return DeltaNDCG<exp_gain>(y_high, y_low, rank_high, rank_low, inv_IDCG(g), discount);
    };
    this->CalcLambdaForGroup<unbiased>(iter, g_predt, g_label, w, g_rank, g, delta, g_gpair);
  }

  void GetGradientImpl(std::int32_t iter, const HostDeviceVector<float>& predt,
                       const MetaInfo& info, HostDeviceVector<GradientPair>* out_gpair) {
    if (ctx_->IsCUDA()) {
      cuda_impl::LambdaRankGetGradientNDCG(
          ctx_, iter, predt, info, GetCache(), ti_plus_.View(ctx_->gpu_id),
          tj_minus_.View(ctx_->gpu_id), li_full_.View(ctx_->gpu_id), lj_full_.View(ctx_->gpu_id),
          out_gpair);
      return;
    }

    bst_group_t n_groups = p_cache_->Groups();
    auto gptr = p_cache_->DataGroupPtr(ctx_);

    out_gpair->Resize(info.num_row_);
    auto h_gpair = out_gpair->HostSpan();
    auto h_predt = predt.ConstHostSpan();
    auto h_label = info.labels.HostView();
    auto h_weight = common::MakeOptionalWeights(ctx_, info.weights_);
    auto make_range = [&](bst_group_t g) { return linalg::Range(gptr[g], gptr[g + 1]); };

    auto dct = GetCache()->Discount(ctx_);
    auto rank_idx = p_cache_->SortedIdx(ctx_, h_predt);
    auto inv_IDCG = GetCache()->InvIDCG(ctx_);

    common::ParallelFor(n_groups, ctx_->Threads(), common::Sched::Guided(), [&](auto g) {
      std::size_t cnt = gptr[g + 1] - gptr[g];
      auto w = h_weight[g];
      auto g_predt = h_predt.subspan(gptr[g], cnt);
      auto g_gpair = h_gpair.subspan(gptr[g], cnt);
      auto g_label = h_label.Slice(make_range(g), 0);
      auto g_rank = rank_idx.subspan(gptr[g], cnt);

      auto args =
          std::make_tuple(this, iter, g_predt, g_label, w, g_rank, g_gpair, inv_IDCG, dct, g);

      if (param_.lambdarank_unbiased) {
        if (param_.ndcg_exp_gain) {
          std::apply(&LambdaRankNDCG::CalcLambdaForGroupNDCG<true, true>, args);
        } else {
          std::apply(&LambdaRankNDCG::CalcLambdaForGroupNDCG<true, false>, args);
        }
      } else {
        if (param_.ndcg_exp_gain) {
          std::apply(&LambdaRankNDCG::CalcLambdaForGroupNDCG<false, true>, args);
        } else {
          std::apply(&LambdaRankNDCG::CalcLambdaForGroupNDCG<false, false>, args);
        }
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
void LambdaRankGetGradientNDCG(Context const*, std::int32_t, HostDeviceVector<float> const&,
                               const MetaInfo&, std::shared_ptr<ltr::NDCGCache>,
                               linalg::VectorView<double const>,  // input bias ratio
                               linalg::VectorView<double const>,  // input bias ratio
                               linalg::VectorView<double>, linalg::VectorView<double>,
                               HostDeviceVector<GradientPair>*) {
  common::AssertGPUSupport();
}

void LambdaRankUpdatePositionBias(Context const*, linalg::VectorView<double const>,
                                  linalg::VectorView<double const>, linalg::Vector<double>*,
                                  linalg::Vector<double>*, linalg::Vector<double>*,
                                  linalg::Vector<double>*, std::shared_ptr<ltr::RankingCache>) {
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
  void GetGradientImpl(std::int32_t iter, const HostDeviceVector<float>& predt,
                       const MetaInfo& info, HostDeviceVector<GradientPair>* out_gpair) {
    CHECK(param_.ndcg_exp_gain) << "NDCG gain can not be set for the MAP objective.";
    if (ctx_->IsCUDA()) {
      return cuda_impl::LambdaRankGetGradientMAP(
          ctx_, iter, predt, info, GetCache(), ti_plus_.View(ctx_->gpu_id),
          tj_minus_.View(ctx_->gpu_id), li_full_.View(ctx_->gpu_id), lj_full_.View(ctx_->gpu_id),
          out_gpair);
    }

    auto gptr = p_cache_->DataGroupPtr(ctx_).data();
    bst_group_t n_groups = p_cache_->Groups();

    out_gpair->Resize(info.num_row_);
    auto h_gpair = out_gpair->HostSpan();
    auto h_label = info.labels.HostView().Slice(linalg::All(), 0);
    auto h_predt = predt.ConstHostSpan();
    auto rank_idx = p_cache_->SortedIdx(ctx_, h_predt);
    auto h_weight = common::MakeOptionalWeights(ctx_, info.weights_);

    auto make_range = [&](bst_group_t g) { return linalg::Range(gptr[g], gptr[g + 1]); };

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
      auto g_gpair = h_gpair.subspan(gptr[g], cnt);
      auto g_label = h_label.Slice(make_range(g));
      auto g_rank = rank_idx.subspan(gptr[g], cnt);

      auto args = std::make_tuple(this, iter, g_predt, g_label, w, g_rank, g, delta_map, g_gpair);

      if (param_.lambdarank_unbiased) {
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

void LambdaRankGetGradientMAP(Context const*, std::int32_t, HostDeviceVector<float> const&,
                              const MetaInfo&, std::shared_ptr<ltr::MAPCache>,
                              linalg::VectorView<double const>,  // input bias ratio
                              linalg::VectorView<double const>,  // input bias ratio
                              linalg::VectorView<double>, linalg::VectorView<double>,
                              HostDeviceVector<GradientPair>*) {
  common::AssertGPUSupport();
}
}  // namespace cuda_impl
#endif  // !defined(XGBOOST_USE_CUDA)

/**
 * \brief The RankNet loss.
 */
class LambdaRankPairwise : public LambdaRankObj<LambdaRankPairwise, ltr::RankingCache> {
 public:
  void GetGradientImpl(std::int32_t iter, const HostDeviceVector<float>& predt,
                       const MetaInfo& info, HostDeviceVector<GradientPair>* out_gpair) {
    CHECK(param_.ndcg_exp_gain) << "NDCG gain can not be set for the pairwise objective.";
    if (ctx_->IsCUDA()) {
      return cuda_impl::LambdaRankGetGradientPairwise(
          ctx_, iter, predt, info, GetCache(), ti_plus_.View(ctx_->gpu_id),
          tj_minus_.View(ctx_->gpu_id), li_full_.View(ctx_->gpu_id), lj_full_.View(ctx_->gpu_id),
          out_gpair);
    }

    auto gptr = p_cache_->DataGroupPtr(ctx_);
    bst_group_t n_groups = p_cache_->Groups();

    out_gpair->Resize(info.num_row_);
    auto h_gpair = out_gpair->HostSpan();
    auto h_label = info.labels.HostView().Slice(linalg::All(), 0);
    auto h_predt = predt.ConstHostSpan();
    auto h_weight = common::MakeOptionalWeights(ctx_, info.weights_);

    auto make_range = [&](bst_group_t g) { return linalg::Range(gptr[g], gptr[g + 1]); };
    auto rank_idx = p_cache_->SortedIdx(ctx_, h_predt);

    auto delta = [](auto...) { return 1.0; };
    using D = decltype(delta);

    common::ParallelFor(n_groups, ctx_->Threads(), [&](auto g) {
      auto cnt = gptr[g + 1] - gptr[g];
      auto w = h_weight[g];
      auto g_predt = h_predt.subspan(gptr[g], cnt);
      auto g_gpair = h_gpair.subspan(gptr[g], cnt);
      auto g_label = h_label.Slice(make_range(g));
      auto g_rank = rank_idx.subspan(gptr[g], cnt);

      auto args = std::make_tuple(this, iter, g_predt, g_label, w, g_rank, g, delta, g_gpair);
      if (param_.lambdarank_unbiased) {
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
};

#if !defined(XGBOOST_USE_CUDA)
namespace cuda_impl {
void LambdaRankGetGradientPairwise(Context const*, std::int32_t, HostDeviceVector<float> const&,
                                   const MetaInfo&, std::shared_ptr<ltr::RankingCache>,
                                   linalg::VectorView<double const>,  // input bias ratio
                                   linalg::VectorView<double const>,  // input bias ratio
                                   linalg::VectorView<double>, linalg::VectorView<double>,
                                   HostDeviceVector<GradientPair>*) {
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
