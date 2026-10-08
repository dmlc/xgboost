/**
 * Copyright 2020-2026, XGBoost contributors
 */
#include "rank_metric.h"

#include <dmlc/omp.h>
#include <dmlc/registry.h>

#include <algorithm>   // for min, max
#include <array>       // for array
#include <cmath>       // for log, sqrt
#include <functional>  // for less, greater
#include <memory>      // for allocator, unique_ptr, shared_ptr, __shared_...
#include <set>         // for set
#include <sstream>     // for ostringstream
#include <string>      // for char_traits, operator<, basic_string, to_string
#include <utility>     // for pair, make_pair
#include <vector>      // for vector

#include "../collective/aggregator.h"
#include "../collective/communicator-inl.h"
#include "../common/algorithm.h"  // for ArgSort, Sort
#include "../common/kernel.h"
#include "../common/linalg_op.h"         // for cbegin, cend
#include "../common/numeric.h"           // for TransformReduce
#include "../common/optional_weight.h"   // for OptionalWeights, MakeOptionalWeights
#include "metric_common.h"               // for MetricNoCache, PackedReduceResult
#include "xgboost/base.h"                // for bst_float, bst_omp_uint, bst_group_t, Args
#include "xgboost/cache.h"               // for DMatrixCache
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo, DMatrix
#include "xgboost/host_device_vector.h"  // for HostDeviceVector
#include "xgboost/json.h"                // for Json, FromJson, IsA, ToJson, get, Null, Object
#include "xgboost/linalg.h"              // for Tensor, TensorView, Range, VectorView, MakeT...
#include "xgboost/logging.h"             // for CHECK, ConsoleLogger, LOG_INFO, CHECK_EQ
#include "xgboost/metric.h"              // for MetricReg, XGBOOST_REGISTER_METRIC, Metric
#include "xgboost/string_view.h"         // for StringView

namespace {
using PredIndPair = std::pair<xgboost::bst_float, xgboost::ltr::rel_degree_t>;
using PredIndPairContainer = std::vector<PredIndPair>;
}  // anonymous namespace

namespace xgboost::metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(rank_metric);

namespace {
double EvalAMSCpu(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info,
                  float ratio) {
  using namespace std;  // NOLINT(*)

  const auto ndata = static_cast<bst_omp_uint>(info.labels.Size());
  PredIndPairContainer rec(ndata);

  const auto& h_preds = preds.ConstHostVector();
  common::ParallelFor(ndata, ctx->Threads(),
                      [&](bst_omp_uint i) { rec[i] = std::make_pair(h_preds[i], i); });
  common::Sort(ctx, rec.begin(), rec.end(),
               [](auto const& l, auto const& r) { return l.first > r.first; });
  auto ntop = static_cast<unsigned>(ratio * ndata);
  if (ntop == 0) ntop = ndata;
  const double br = 10.0;
  unsigned thresindex = 0;
  double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
  const auto& labels = info.labels.View(DeviceOrd::CPU());
  for (unsigned i = 0; i < static_cast<unsigned>(ndata - 1) && i < ntop; ++i) {
    const unsigned ridx = rec[i].second;
    const bst_float wt = info.GetWeight(ridx);
    if (labels(ridx) > 0.5f) {
      s_tp += wt;
    } else {
      b_fp += wt;
    }
    if (rec[i].first != rec[i + 1].first) {
      double ams = sqrt(2 * ((s_tp + b_fp + br) * log(1.0 + s_tp / (b_fp + br)) - s_tp));
      if (tams < ams) {
        thresindex = i;
        tams = ams;
      }
    }
  }
  if (ntop == ndata) {
    LOG(INFO) << "best-ams-ratio=" << static_cast<bst_float>(thresindex) / ndata;
    return static_cast<bst_float>(tams);
  } else {
    return static_cast<bst_float>(
        sqrt(2 * ((s_tp + b_fp + br) * log(1.0 + s_tp / (b_fp + br)) - s_tp)));
  }
}

double EvalCoxCpu(Context const* ctx, HostDeviceVector<float> const& preds, MetaInfo const& info) {
  using namespace std;  // NOLINT(*)

  const auto ndata = static_cast<bst_omp_uint>(info.labels.Size());
  const auto& label_order = info.LabelAbsSort(ctx);

  // pre-compute a sum for the denominator
  double exp_p_sum = 0;  // we use double because we might need the precision with large datasets

  const auto& h_preds = preds.ConstHostVector();
  for (omp_ulong i = 0; i < ndata; ++i) {
    exp_p_sum += h_preds[i];
  }

  double out = 0;
  double accumulated_sum = 0;
  bst_omp_uint num_events = 0;
  const auto& labels = info.labels.HostView();
  for (bst_omp_uint i = 0; i < ndata; ++i) {
    const size_t ind = label_order[i];
    const auto label = labels(ind);
    if (label > 0) {
      out -= log(h_preds[ind]) - log(exp_p_sum);
      ++num_events;
    }

    // only update the denominator after we move forward in time (labels are sorted)
    accumulated_sum += h_preds[ind];
    if (i == ndata - 1 || std::abs(label) < std::abs(labels(label_order[i + 1]))) {
      exp_p_sum -= accumulated_sum;
      accumulated_sum = 0;
    }
  }

  return out / num_events;  // normalize by the number of events
}

auto const kRegisterAMSCpu =
    common::KernelRegistration<AMSEvalKernel>{DeviceOrd::kCPU, &EvalAMSCpu};
auto const kRegisterCoxCpu =
    common::KernelRegistration<CoxEvalKernel>{DeviceOrd::kCPU, &EvalCoxCpu};
}  // namespace

/*! \brief AMS: also records best threshold */
struct EvalAMS : public MetricNoCache {
 public:
  explicit EvalAMS(const char* param) {
    CHECK(param != nullptr)  // NOLINT
        << "AMS must be in format ams@k";
    ratio_ = atof(param);
    std::ostringstream os;
    os << "ams@" << ratio_;
    name_ = os.str();
  }

  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    CheckRowWeights(info);
    CHECK(!collective::IsDistributed()) << "metric AMS do not support distributed evaluation";
    return common::DispatchKernel<AMSEvalKernel>(ctx_, preds, info, ratio_);
  }

  [[nodiscard]] const char* Name() const override { return name_.c_str(); }

 private:
  std::string name_;
  float ratio_;
};

/*! \brief Cox: Partial likelihood of the Cox proportional hazards model */
struct EvalCox : public MetricNoCache {
 public:
  EvalCox() = default;
  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    CHECK(!collective::IsDistributed()) << "Cox metric does not support distributed evaluation";
    return common::DispatchKernel<CoxEvalKernel>(ctx_, preds, info);
  }

  [[nodiscard]] const char* Name() const override { return "cox-nloglik"; }
};

XGBOOST_REGISTER_METRIC(AMS, "ams")
    .describe("AMS metric for higgs.")
    .set_body([](const char* param) { return new EvalAMS(param); });

XGBOOST_REGISTER_METRIC(Cox, "cox-nloglik")
    .describe("Negative log partial likelihood of Cox proportional hazards model.")
    .set_body([](const char*) { return new EvalCox(); });

// ranking metrics that requires cache
template <typename Cache>
class EvalRankWithCache : public Metric {
 protected:
  ltr::LambdaRankParam param_;
  bool minus_{false};
  std::string name_;

  DMatrixCache<Cache> cache_{DMatrixCache<Cache>::DefaultSize()};

 public:
  EvalRankWithCache(StringView name, const char* param) {
    auto constexpr kMax = ltr::LambdaRankParam::NotSet();
    std::uint32_t topn{kMax};
    this->name_ = ltr::ParseMetricName(name, param, &topn, &minus_);
    if (topn != kMax) {
      param_.UpdateAllowUnknown(Args{{"lambdarank_num_pair_per_sample", std::to_string(topn)},
                                     {"lambdarank_pair_method", "topk"}});
    }
    param_.UpdateAllowUnknown(Args{});
  }
  void LoadConfig(Json const& in) override {
    if (IsA<Null>(in)) {
      return;
    }
    auto const& obj = get<Object const>(in);
    auto it = obj.find("lambdarank_param");
    if (it != obj.cend()) {
      FromJson(it->second, &param_);
    }
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String{this->Name()};
    out["lambdarank_param"] = ToJson(param_);
  }

  double Evaluate(HostDeviceVector<float> const& preds, std::shared_ptr<DMatrix> p_fmat) override {
    double result{0.0};
    auto const& info = p_fmat->Info();
    if (!info.labels.Empty()) {
      CHECK_EQ(info.labels.Shape(1), 1) << "Ranking metrics do not support multi-target labels.";
    }
    CHECK_EQ(preds.Size(), info.labels.Size());

    auto p_cache = cache_.CacheItem(p_fmat, ctx_, info, param_);
    if (p_cache->Param() != param_) {
      p_cache = cache_.ResetItem(p_fmat, ctx_, info, param_);
    }
    CHECK(p_cache->Param() == param_);

    result = this->Eval(preds, info, p_cache);
    return result;
  }

  [[nodiscard]] const char* Name() const override { return name_.c_str(); }

  virtual double Eval(HostDeviceVector<float> const& preds, MetaInfo const& info,
                      std::shared_ptr<Cache> p_cache) = 0;
};

namespace {
double Finalize(Context const* ctx, MetaInfo const&, double score, double sw) {
  std::array<double, 2> dat{score, sw};
  auto rc = collective::GlobalSum(ctx, linalg::MakeVec(dat.data(), 2));
  collective::SafeColl(rc);
  std::tie(score, sw) = std::tuple_cat(dat);
  if (sw > 0.0) {
    score = score / sw;
  }

  CHECK_LE(score, 1.0 + kRtEps)
      << "Invalid output score, might be caused by invalid query group weight.";
  score = std::min(1.0, score);

  return score;
}
}  // namespace

namespace {
PackedReduceResult PreScoreCpu(Context const* ctx, MetaInfo const& info,
                               HostDeviceVector<float> const& predt,
                               std::shared_ptr<ltr::PreCache> p_cache) {
  auto gptr = p_cache->DataGroupPtr(ctx);
  auto h_label = info.labels.HostView().Slice(linalg::All(), 0);
  auto rank_idx = p_cache->SortedIdx(ctx, predt.ConstHostSpan());

  auto weight = common::MakeOptionalWeights(ctx->Device(), info.weights_);
  return common::TransformReduce<1>(
      p_cache->Groups(), ctx->Threads(), PackedReduceResult{}, [&](auto g) -> PackedReduceResult {
        auto g_label = h_label.Slice(linalg::Range(gptr[g], gptr[g + 1]));
        auto g_rank = rank_idx.subspan(gptr[g], gptr[g + 1] - gptr[g]);

        auto n = std::min(static_cast<std::size_t>(p_cache->Param().TopK()), g_label.Size());
        double n_hits{0.0};
        for (std::size_t i = 0; i < n; ++i) {
          n_hits += g_label(g_rank[i]) * weight[g];
        }
        return {n_hits / static_cast<double>(n), weight[g]};
      });
}

auto const kRegisterPrecisionCpu =
    common::KernelRegistration<PrecisionEvalKernel>{DeviceOrd::kCPU, &PreScoreCpu};

PackedReduceResult NDCGScoreCpu(Context const* ctx, MetaInfo const& info,
                                HostDeviceVector<float> const& preds, bool minus,
                                std::shared_ptr<ltr::NDCGCache> p_cache) {
  // group local ndcg
  auto group_ptr = p_cache->DataGroupPtr(ctx);
  bst_group_t n_groups = group_ptr.size() - 1;

  auto h_inv_idcg = p_cache->InvIDCG(ctx);
  auto p_discount = p_cache->Discount(ctx).data();

  auto h_label = info.labels.HostView();
  auto h_predt = linalg::MakeTensorView(ctx, &preds, preds.Size());
  auto weights = common::MakeOptionalWeights(ctx->Device(), info.weights_);

  return common::TransformReduce<1>(
      n_groups, ctx->Threads(), PackedReduceResult{}, [&](auto g) -> PackedReduceResult {
        auto g_predt = h_predt.Slice(linalg::Range(group_ptr[g], group_ptr[g + 1]));
        auto g_labels = h_label.Slice(linalg::Range(group_ptr[g], group_ptr[g + 1]), 0);
        auto sorted_idx = common::ArgSort<std::size_t>(ctx, linalg::cbegin(g_predt),
                                                       linalg::cend(g_predt), std::greater<>{});
        double ndcg{.0};
        double inv_idcg = h_inv_idcg(g);
        if (inv_idcg <= 0.0) {
          return {minus ? 0.0 : 1.0, weights[g]};
        }
        std::size_t n{
            std::min(sorted_idx.size(), static_cast<std::size_t>(p_cache->Param().TopK()))};
        if (p_cache->Param().ndcg_exp_gain) {
          for (std::size_t i = 0; i < n; ++i) {
            ndcg += p_discount[i] * ltr::CalcDCGGain(g_labels(sorted_idx[i])) * inv_idcg;
          }
        } else {
          for (std::size_t i = 0; i < n; ++i) {
            ndcg += p_discount[i] * g_labels(sorted_idx[i]) * inv_idcg;
          }
        }
        return {ndcg * weights[g], weights[g]};
      });
}

auto const kRegisterNDCGCpu =
    common::KernelRegistration<NDCGEvalKernel>{DeviceOrd::kCPU, &NDCGScoreCpu};

PackedReduceResult MAPScoreCpu(Context const* ctx, MetaInfo const& info,
                               HostDeviceVector<float> const& predt, bool minus,
                               std::shared_ptr<ltr::MAPCache> p_cache) {
  auto gptr = p_cache->DataGroupPtr(ctx);
  auto h_label = info.labels.HostView().Slice(linalg::All(), 0);

  auto rank_idx = p_cache->SortedIdx(ctx, predt.ConstHostSpan());
  auto weight = common::MakeOptionalWeights(ctx->Device(), info.weights_);
  if (!weight.Empty()) {
    CHECK_EQ(weight.weights.size(), p_cache->Groups());
  }
  return common::TransformReduce<1>(
      p_cache->Groups(), ctx->Threads(), PackedReduceResult{}, [&](auto g) -> PackedReduceResult {
        auto g_label = h_label.Slice(linalg::Range(gptr[g], gptr[g + 1]));
        auto g_rank = rank_idx.subspan(gptr[g], gptr[g + 1] - gptr[g]);

        auto n = std::min(static_cast<std::size_t>(p_cache->Param().TopK()), g_label.Size());
        double n_hits{0.0}, map{0.0};
        for (std::size_t i = 0; i < n; ++i) {
          auto p = g_label(g_rank[i]);
          n_hits += p;
          map += n_hits / static_cast<double>((i + 1)) * p;
        }
        for (std::size_t i = n; i < g_label.Size(); ++i) {
          n_hits += g_label(g_rank[i]);
        }
        if (n_hits > 0.0) {
          map /= std::min(n_hits, static_cast<double>(p_cache->Param().TopK()));
        } else {
          map = minus ? 0.0 : 1.0;
        }
        return {map * weight[g], weight[g]};
      });
}

auto const kRegisterMAPCpu =
    common::KernelRegistration<MAPEvalKernel>{DeviceOrd::kCPU, &MAPScoreCpu};
}  // namespace

class EvalPrecision : public EvalRankWithCache<ltr::PreCache> {
 public:
  using EvalRankWithCache::EvalRankWithCache;

  double Eval(HostDeviceVector<float> const& predt, MetaInfo const& info,
              std::shared_ptr<ltr::PreCache> p_cache) final {
    auto n_groups = p_cache->Groups();
    if (!info.weights_.Empty()) {
      CHECK_EQ(info.weights_.Size(), n_groups) << error::GroupWeight();
    }

    auto result = common::DispatchKernel<PrecisionEvalKernel>(ctx_, info, predt, p_cache);
    return Finalize(ctx_, info, result.Residue(), result.Weights());
  }
};

/**
 * \brief Implement the NDCG score function for learning to rank.
 *
 *     Ties are ignored, which can lead to different result with other implementations.
 */
class EvalNDCG : public EvalRankWithCache<ltr::NDCGCache> {
 public:
  using EvalRankWithCache::EvalRankWithCache;

  std::set<std::string> Configure(Args const& args) override {
    // do not configure, otherwise the ndcg param like top-k will be forced into the same
    // as the one in objective. The metric has its own syntax for parameter.
    std::set<std::string> used;
    for (auto const& [key, value] : args) {
      // Make a special case for the exp gain parameter, which is not exposed in the
      // metric configuration syntax.
      if (key == "ndcg_exp_gain") {
        this->param_.UpdateAllowUnknown(Args{{key, value}});
        used.insert(key);
      }
    }
    return used;
  }

  double Eval(HostDeviceVector<float> const& preds, MetaInfo const& info,
              std::shared_ptr<ltr::NDCGCache> p_cache) override {
    auto result = common::DispatchKernel<NDCGEvalKernel>(ctx_, info, preds, minus_, p_cache);
    return Finalize(ctx_, info, result.Residue(), result.Weights());
  }
};

class EvalMAPScore : public EvalRankWithCache<ltr::MAPCache> {
 public:
  using EvalRankWithCache::EvalRankWithCache;

  double Eval(HostDeviceVector<float> const& predt, MetaInfo const& info,
              std::shared_ptr<ltr::MAPCache> p_cache) override {
    auto result = common::DispatchKernel<MAPEvalKernel>(ctx_, info, predt, minus_, p_cache);
    return Finalize(ctx_, info, result.Residue(), result.Weights());
  }
};

XGBOOST_REGISTER_METRIC(Precision, "pre")
    .describe("precision@k for rank.")
    .set_body([](const char* param) { return new EvalPrecision("pre", param); });

XGBOOST_REGISTER_METRIC(EvalMAP, "map")
    .describe("map@k for ranking.")
    .set_body([](char const* param) { return new EvalMAPScore{"map", param}; });

XGBOOST_REGISTER_METRIC(EvalNDCG, "ndcg")
    .describe("ndcg@k for ranking.")
    .set_body([](char const* param) { return new EvalNDCG{"ndcg", param}; });
}  // namespace xgboost::metric
