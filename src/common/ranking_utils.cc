/**
 * Copyright 2023 by XGBoost contributors
 */
#include "ranking_utils.h"

#include <algorithm>          // for copy_n, max, min, none_of, all_of
#include <cstddef>            // for size_t
#include <cstdio>             // for sscanf
#include <functional>         // for greater
#include <string>             // for char_traits, string

#include "algorithm.h"        // for ArgSort
#include "linalg_op.h"        // for cbegin, cend
#include "optional_weight.h"  // for MakeOptionalWeights
#include "threading_utils.h"  // for ParallelFor
#include "xgboost/base.h"     // for bst_group_t
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for MetaInfo
#include "xgboost/linalg.h"   // for All, TensorView, Range
#include "xgboost/logging.h"  // for CHECK_EQ

namespace xgboost::ltr {
void RankingCache::InitOnCPU(Context const* ctx, MetaInfo const& info) {
  if (info.group_ptr_.empty()) {
    group_ptr_.Resize(2, 0);
    group_ptr_.HostVector()[1] = info.num_row_;
  } else {
    group_ptr_.HostVector() = info.group_ptr_;
  }

  auto const& gptr = group_ptr_.ConstHostVector();
  for (std::size_t i = 1; i < gptr.size(); ++i) {
    std::size_t n = gptr[i] - gptr[i - 1];
    max_group_size_ = std::max(max_group_size_, n);
  }

  double sum_weights = 0;
  auto n_groups = Groups();
  auto weight = common::MakeOptionalWeights(ctx, info.weights_);
  for (bst_omp_uint k = 0; k < n_groups; ++k) {
    sum_weights += weight[k];
  }
  weight_norm_ = static_cast<double>(n_groups) / sum_weights;
}

common::Span<std::size_t const> RankingCache::MakeRankOnCPU(Context const* ctx,
                                                            common::Span<float const> predt) {
  auto gptr = this->DataGroupPtr(ctx);
  auto rank = this->sorted_idx_cache_.HostSpan();
  CHECK_EQ(rank.size(), predt.size());

  common::ParallelFor(this->Groups(), ctx->Threads(), [&](auto g) {
    auto cnt = gptr[g + 1] - gptr[g];
    auto g_predt = predt.subspan(gptr[g], cnt);
    auto g_rank = rank.subspan(gptr[g], cnt);
    auto sorted_idx = common::ArgSort<std::size_t>(
        ctx, g_predt.data(), g_predt.data() + g_predt.size(), std::greater<>{});
    CHECK_EQ(g_rank.size(), sorted_idx.size());
    std::copy_n(sorted_idx.data(), sorted_idx.size(), g_rank.data());
  });

  return rank;
}

#if !defined(XGBOOST_USE_CUDA)
void RankingCache::InitOnCUDA(Context const*, MetaInfo const&) { common::AssertGPUSupport(); }
common::Span<std::size_t const> RankingCache::MakeRankOnCUDA(Context const*,
                                                             common::Span<float const>) {
  common::AssertGPUSupport();
  return {};
}
#endif  // !defined()

void NDCGCache::InitOnCPU(Context const* ctx, MetaInfo const& info) {
  auto const h_group_ptr = this->DataGroupPtr(ctx);

  discounts_.Resize(MaxGroupSize(), 0);
  auto& h_discounts = discounts_.HostVector();
  for (std::size_t i = 0; i < MaxGroupSize(); ++i) {
    h_discounts[i] = CalcDCGDiscount(i);
  }

  auto n_groups = h_group_ptr.size() - 1;
  auto h_labels = info.labels.HostView().Slice(linalg::All(), 0);

  CheckNDCGLabels(this->Param(), h_labels,
                  [](auto beg, auto end, auto op) { return std::none_of(beg, end, op); });

  inv_idcg_.Reshape(n_groups);
  auto h_inv_idcg = inv_idcg_.HostView();
  std::size_t topk = this->Param().TopK();
  auto const exp_gain = this->Param().ndcg_exp_gain;

  common::ParallelFor(n_groups, ctx->Threads(), [&](auto g) {
    auto g_labels = h_labels.Slice(linalg::Range(h_group_ptr[g], h_group_ptr[g + 1]));
    auto sorted_idx = common::ArgSort<std::size_t>(ctx, linalg::cbegin(g_labels),
                                                   linalg::cend(g_labels), std::greater<>{});

    double idcg{0.0};
    for (std::size_t i = 0; i < std::min(g_labels.Size(), topk); ++i) {
      if (exp_gain) {
        idcg += h_discounts[i] * CalcDCGGain(g_labels(sorted_idx[i]));
      } else {
        idcg += h_discounts[i] * g_labels(sorted_idx[i]);
      }
    }
    h_inv_idcg(g) = CalcInvIDCG(idcg);
  });
}

#if !defined(XGBOOST_USE_CUDA)
void NDCGCache::InitOnCUDA(Context const*, MetaInfo const&) { common::AssertGPUSupport(); }
#endif  // !defined(XGBOOST_USE_CUDA)

DMLC_REGISTER_PARAMETER(LambdaRankParam);

void MAPCache::InitOnCPU(Context const*, MetaInfo const& info) {
  auto const& h_label = info.labels.HostView().Slice(linalg::All(), 0);
  CheckMapLabels(h_label, [](auto beg, auto end, auto op) { return std::all_of(beg, end, op); });
}

#if !defined(XGBOOST_USE_CUDA)
void MAPCache::InitOnCUDA(Context const*, MetaInfo const&) { common::AssertGPUSupport(); }
#endif  // !defined(XGBOOST_USE_CUDA)

std::string ParseMetricName(StringView name, StringView param, position_t* topn, bool* minus) {
  std::string out_name;
  if (!param.empty()) {
    std::ostringstream os;
    if (std::sscanf(param.c_str(), "%u[-]?", topn) == 1) {
      os << name << '@' << param;
      out_name = os.str();
    } else {
      os << name << param;
      out_name = os.str();
    }
    if (*param.crbegin() == '-') {
      *minus = true;
    }
  } else {
    out_name = name.c_str();
  }
  return out_name;
}

std::string MakeMetricName(StringView name, position_t topn, bool minus) {
  std::ostringstream ss;
  if (topn == LambdaRankParam::NotSet()) {
    ss << name;
  } else {
    ss << name << "@" << topn;
  }
  if (minus) {
    ss << "-";
  }
  std::string out_name = ss.str();
  return out_name;
}
}  // namespace xgboost::ltr
