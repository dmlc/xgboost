/**
 * Copyright 2020-2023 by XGBoost contributors
 */
// When device ordinal is present, we would want to build the metrics on the GPU. It is *not*
// possible for a valid device ordinal to be present for non GPU builds. However, it is possible
// for an invalid device ordinal to be specified in GPU builds - to train/predict and/or compute
// the metrics on CPU. To accommodate these scenarios, the following is done for the metrics
// accelerated on the GPU.
// - An internal GPU registry holds all the GPU metric types (defined in the .cu file)
// - An instance of the appropriate GPU metric type is created when a device ordinal is present
// - If the creation is successful, the metric computation is done on the device
// - else, it falls back on the CPU
// - The GPU metric types are *only* registered when xgboost is built for GPUs
//
// This is done for 2 reasons:
// - Clear separation of CPU and GPU logic
// - Sorting datasets containing large number of rows is (much) faster when parallel sort
//   semantics is used on the CPU. The __gnu_parallel/concurrency primitives needed to perform
//   this cannot be used when the translation unit is compiled using the 'nvcc' compiler (as the
//   corresponding headers that brings in those function declaration can't be included with CUDA).
//   This precludes the CPU and GPU logic to coexist inside a .cu file

#include "rank_metric.h"

#include <dmlc/omp.h>
#include <dmlc/registry.h>

#include <algorithm>                         // for stable_sort, copy, fill_n, min, max
#include <array>                             // for array
#include <cmath>                             // for log, sqrt
#include <functional>                        // for less, greater
#include <limits>                            // for numeric_limits
#include <map>                               // for operator!=, _Rb_tree_const_iterator
#include <memory>                            // for allocator, unique_ptr, shared_ptr, __shared_...
#include <numeric>                           // for accumulate
#include <ostream>                           // for operator<<, basic_ostream, ostringstream
#include <string>                            // for char_traits, operator<, basic_string, to_string
#include <utility>                           // for pair, make_pair
#include <vector>                            // for vector

#include "../collective/aggregator.h"        // for ApplyWithLabels
#include "../common/algorithm.h"             // for ArgSort, Sort
#include "../common/linalg_op.h"             // for cbegin, cend
#include "../common/math.h"                  // for CmpFirst
#include "../common/optional_weight.h"       // for OptionalWeights, MakeOptionalWeights
#include "dmlc/common.h"                     // for OMPException
#include "metric_common.h"                   // for MetricNoCache, GPUMetric, PackedReduceResult
#include "xgboost/base.h"                    // for bst_float, bst_omp_uint, bst_group_t, Args
#include "xgboost/cache.h"                   // for DMatrixCache
#include "xgboost/context.h"                 // for Context
#include "xgboost/data.h"                    // for MetaInfo, DMatrix
#include "xgboost/host_device_vector.h"      // for HostDeviceVector
#include "xgboost/json.h"                    // for Json, FromJson, IsA, ToJson, get, Null, Object
#include "xgboost/linalg.h"                  // for Tensor, TensorView, Range, VectorView, MakeT...
#include "xgboost/logging.h"                 // for CHECK, ConsoleLogger, LOG_INFO, CHECK_EQ
#include "xgboost/metric.h"                  // for MetricReg, XGBOOST_REGISTER_METRIC, Metric
#include "xgboost/string_view.h"             // for StringView

namespace {

using PredIndPair = std::pair<xgboost::bst_float, xgboost::ltr::rel_degree_t>;
using PredIndPairContainer = std::vector<PredIndPair>;

/*
 * Adapter to access instance weights.
 *
 *  - For ranking task, weights are per-group
 *  - For binary classification task, weights are per-instance
 *
 * WeightPolicy::GetWeightOfInstance() :
 *   get weight associated with an individual instance, using index into
 *   `info.weights`
 * WeightPolicy::GetWeightOfSortedRecord() :
 *   get weight associated with an individual instance, using index into
 *   sorted records `rec` (in ascending order of predicted labels). `rec` is
 *   of type PredIndPairContainer
 */

class PerInstanceWeightPolicy {
 public:
  inline static xgboost::bst_float
  GetWeightOfInstance(const xgboost::MetaInfo& info,
                      unsigned instance_id, unsigned) {
    return info.GetWeight(instance_id);
  }
  inline static xgboost::bst_float
  GetWeightOfSortedRecord(const xgboost::MetaInfo& info,
                          const PredIndPairContainer& rec,
                          unsigned record_id, unsigned) {
    return info.GetWeight(rec[record_id].second);
  }
};

class PerGroupWeightPolicy {
 public:
  inline static xgboost::bst_float
  GetWeightOfInstance(const xgboost::MetaInfo& info,
                      unsigned, unsigned group_id) {
    return info.GetWeight(group_id);
  }

  inline static xgboost::bst_float
  GetWeightOfSortedRecord(const xgboost::MetaInfo& info,
                          const PredIndPairContainer&,
                          unsigned, unsigned group_id) {
    return info.GetWeight(group_id);
  }
};
}  // anonymous namespace

namespace xgboost::metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(rank_metric);

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
    CHECK(!collective::IsDistributed()) << "metric AMS do not support distributed evaluation";
    using namespace std;  // NOLINT(*)

    const auto ndata = static_cast<bst_omp_uint>(info.labels.Size());
    PredIndPairContainer rec(ndata);

    const auto &h_preds = preds.ConstHostVector();
    common::ParallelFor(ndata, ctx_->Threads(),
                        [&](bst_omp_uint i) { rec[i] = std::make_pair(h_preds[i], i); });
    common::Sort(ctx_, rec.begin(), rec.end(), common::CmpFirst);
    auto ntop = static_cast<unsigned>(ratio_ * ndata);
    if (ntop == 0) ntop = ndata;
    const double br = 10.0;
    unsigned thresindex = 0;
    double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
    const auto& labels = info.labels.View(Context::kCpuId);
    for (unsigned i = 0; i < static_cast<unsigned>(ndata-1) && i < ntop; ++i) {
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
          sqrt(2 * ((s_tp + b_fp + br) * log(1.0 + s_tp/(b_fp + br)) - s_tp)));
    }
  }

  const char* Name() const override {
    return name_.c_str();
  }

 private:
  std::string name_;
  float ratio_;
};

/*! \brief Evaluate rank list */
struct EvalRank : public MetricNoCache, public EvalRankConfig {
 private:
  // This is used to compute the ranking metrics on the GPU - for training jobs that run on the GPU.
  std::unique_ptr<MetricNoCache> rank_gpu_;

 public:
  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    CHECK_EQ(preds.Size(), info.labels.Size())
        << "label size predict size not match";

    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(preds.Size());
    const auto &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;

    CHECK_NE(gptr.size(), 0U) << "must specify group when constructing rank file";
    CHECK_EQ(gptr.back(), preds.Size())
        << "EvalRank: group structure must match number of prediction";

    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    double sum_metric = 0.0f;

    // Check and see if we have the GPU metric registered in the internal registry
    if (ctx_->gpu_id >= 0) {
      if (!rank_gpu_) {
        rank_gpu_.reset(GPUMetric::CreateGPUMetric(this->Name(), ctx_));
      }
      if (rank_gpu_) {
        sum_metric = rank_gpu_->Eval(preds, info);
      }
    }

    CHECK(ctx_);
    std::vector<double> sum_tloc(ctx_->Threads(), 0.0);

    if (!rank_gpu_ || ctx_->gpu_id < 0) {
      const auto& labels = info.labels.View(Context::kCpuId);
      const auto &h_preds = preds.ConstHostVector();

      dmlc::OMPException exc;
#pragma omp parallel num_threads(ctx_->Threads())
      {
        exc.Run([&]() {
          // each thread takes a local rec
          PredIndPairContainer rec;
#pragma omp for schedule(static)
          for (bst_omp_uint k = 0; k < ngroups; ++k) {
            exc.Run([&]() {
              rec.clear();
              for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
                rec.emplace_back(h_preds[j], static_cast<int>(labels(j)));
              }
              sum_tloc[omp_get_thread_num()] += this->EvalGroup(&rec);
            });
          }
        });
      }
      sum_metric = std::accumulate(sum_tloc.cbegin(), sum_tloc.cend(), 0.0);
      exc.Rethrow();
    }

    return collective::GlobalRatio(info, sum_metric, static_cast<double>(ngroups));
  }

  const char* Name() const override {
    return name.c_str();
  }

 protected:
  explicit EvalRank(const char* name, const char* param) {
    this->name = ltr::ParseMetricName(name, param, &topn, &minus);
  }

  virtual double EvalGroup(PredIndPairContainer *recptr) const = 0;
};

/*! \brief Precision at N, for both classification and rank */
struct EvalPrecision : public EvalRank {
 public:
  explicit EvalPrecision(const char* name, const char* param) : EvalRank(name, param) {}

  double EvalGroup(PredIndPairContainer *recptr) const override {
    PredIndPairContainer &rec(*recptr);
    // calculate Precision
    std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
    unsigned nhit = 0;
    for (size_t j = 0; j < rec.size() && j < this->topn; ++j) {
      nhit += (rec[j].second != 0);
    }
    return static_cast<double>(nhit) / this->topn;
  }
};

/*! \brief Cox: Partial likelihood of the Cox proportional hazards model */
struct EvalCox : public MetricNoCache {
 public:
  EvalCox() = default;
  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    CHECK(!collective::IsDistributed()) << "Cox metric does not support distributed evaluation";
    using namespace std;  // NOLINT(*)

    const auto ndata = static_cast<bst_omp_uint>(info.labels.Size());
    const auto &label_order = info.LabelAbsSort(ctx_);

    // pre-compute a sum for the denominator
    double exp_p_sum = 0;  // we use double because we might need the precision with large datasets

    const auto &h_preds = preds.ConstHostVector();
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

    return out/num_events;  // normalize by the number of events
  }

  const char* Name() const override {
    return "cox-nloglik";
  }
};

XGBOOST_REGISTER_METRIC(AMS, "ams")
.describe("AMS metric for higgs.")
.set_body([](const char* param) { return new EvalAMS(param); });

XGBOOST_REGISTER_METRIC(Precision, "pre")
.describe("precision@k for rank.")
.set_body([](const char* param) { return new EvalPrecision("pre", param); });

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
  void Configure(Args const&) override {
    // do not configure, otherwise the ndcg param will be forced into the same as the one in
    // objective.
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
    collective::ApplyWithLabels(info, &result, sizeof(double), [&] {
      auto p_cache = cache_.CacheItem(p_fmat, ctx_, info, param_);
      if (p_cache->Param() != param_) {
        p_cache = cache_.ResetItem(p_fmat, ctx_, info, param_);
      }
      CHECK(p_cache->Param() == param_);
      CHECK_EQ(preds.Size(), info.labels.Size());

      result = this->Eval(preds, info, p_cache);
    });
    return result;
  }

  virtual double Eval(HostDeviceVector<float> const& preds, MetaInfo const& info,
                      std::shared_ptr<Cache> p_cache) = 0;
};

namespace {
double Finalize(MetaInfo const& info, double score, double sw) {
  std::array<double, 2> dat{score, sw};
  collective::GlobalSum(info, &dat);
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

/**
 * \brief Implement the NDCG score function for learning to rank.
 *
 *     Ties are ignored, which can lead to different result with other implementations.
 */
class EvalNDCG : public EvalRankWithCache<ltr::NDCGCache> {
 public:
  using EvalRankWithCache::EvalRankWithCache;
  const char* Name() const override { return name_.c_str(); }

  double Eval(HostDeviceVector<float> const& preds, MetaInfo const& info,
              std::shared_ptr<ltr::NDCGCache> p_cache) override {
    if (ctx_->IsCUDA()) {
      auto ndcg = cuda_impl::NDCGScore(ctx_, info, preds, minus_, p_cache);
      return Finalize(info, ndcg.Residue(), ndcg.Weights());
    }

    // group local ndcg
    auto group_ptr = p_cache->DataGroupPtr(ctx_);
    bst_group_t n_groups = group_ptr.size() - 1;
    auto ndcg_gloc = p_cache->Dcg(ctx_);
    std::fill_n(ndcg_gloc.Values().data(), ndcg_gloc.Size(), 0.0);

    auto h_inv_idcg = p_cache->InvIDCG(ctx_);
    auto p_discount = p_cache->Discount(ctx_).data();

    auto h_label = info.labels.HostView();
    auto h_predt = linalg::MakeTensorView(ctx_, &preds, preds.Size());
    auto weights = common::MakeOptionalWeights(ctx_, info.weights_);

    common::ParallelFor(n_groups, ctx_->Threads(), [&](auto g) {
      auto g_predt = h_predt.Slice(linalg::Range(group_ptr[g], group_ptr[g + 1]));
      auto g_labels = h_label.Slice(linalg::Range(group_ptr[g], group_ptr[g + 1]), 0);
      auto sorted_idx = common::ArgSort<std::size_t>(ctx_, linalg::cbegin(g_predt),
                                                     linalg::cend(g_predt), std::greater<>{});
      double ndcg{.0};
      double inv_idcg = h_inv_idcg(g);
      if (inv_idcg <= 0.0) {
        ndcg_gloc(g) = minus_ ? 0.0 : 1.0;
        return;
      }
      std::size_t n{std::min(sorted_idx.size(), static_cast<std::size_t>(param_.TopK()))};
      if (param_.ndcg_exp_gain) {
        for (std::size_t i = 0; i < n; ++i) {
          ndcg += p_discount[i] * ltr::CalcDCGGain(g_labels(sorted_idx[i])) * inv_idcg;
        }
      } else {
        for (std::size_t i = 0; i < n; ++i) {
          ndcg += p_discount[i] * g_labels(sorted_idx[i]) * inv_idcg;
        }
      }
      ndcg_gloc(g) += ndcg * weights[g];
    });
    double sum_w{0};
    if (weights.Empty()) {
      sum_w = n_groups;
    } else {
      sum_w = std::accumulate(weights.weights.cbegin(), weights.weights.cend(), 0.0);
    }
    auto ndcg = std::accumulate(linalg::cbegin(ndcg_gloc), linalg::cend(ndcg_gloc), 0.0);
    return Finalize(info, ndcg, sum_w);
  }
};

class EvalMAPScore : public EvalRankWithCache<ltr::MAPCache> {
 public:
  using EvalRankWithCache::EvalRankWithCache;
  const char* Name() const override { return name_.c_str(); }

  double Eval(HostDeviceVector<float> const& predt, MetaInfo const& info,
              std::shared_ptr<ltr::MAPCache> p_cache) override {
    if (ctx_->IsCUDA()) {
      auto map = cuda_impl::MAPScore(ctx_, info, predt, minus_, p_cache);
      return Finalize(info, map.Residue(), map.Weights());
    }

    auto gptr = p_cache->DataGroupPtr(ctx_);
    auto h_label = info.labels.HostView().Slice(linalg::All(), 0);
    auto h_predt = linalg::MakeTensorView(ctx_, &predt, predt.Size());

    auto map_gloc = p_cache->Map(ctx_);
    std::fill_n(map_gloc.data(), map_gloc.size(), 0.0);
    auto rank_idx = p_cache->SortedIdx(ctx_, predt.ConstHostSpan());

    common::ParallelFor(p_cache->Groups(), ctx_->Threads(), [&](auto g) {
      auto g_label = h_label.Slice(linalg::Range(gptr[g], gptr[g + 1]));
      auto g_rank = rank_idx.subspan(gptr[g]);

      auto n = std::min(static_cast<std::size_t>(param_.TopK()), g_label.Size());
      double n_hits{0.0};
      for (std::size_t i = 0; i < n; ++i) {
        auto p = g_label(g_rank[i]);
        n_hits += p;
        map_gloc[g] += n_hits / static_cast<double>((i + 1)) * p;
      }
      for (std::size_t i = n; i < g_label.Size(); ++i) {
        n_hits += g_label(g_rank[i]);
      }
      if (n_hits > 0.0) {
        map_gloc[g] /= std::min(n_hits, static_cast<double>(param_.TopK()));
      } else {
        map_gloc[g] = minus_ ? 0.0 : 1.0;
      }
    });

    auto sw = 0.0;
    auto weight = common::MakeOptionalWeights(ctx_, info.weights_);
    if (!weight.Empty()) {
      CHECK_EQ(weight.weights.size(), p_cache->Groups());
    }
    for (std::size_t i = 0; i < map_gloc.size(); ++i) {
      map_gloc[i] = map_gloc[i] * weight[i];
      sw += weight[i];
    }
    auto sum = std::accumulate(map_gloc.cbegin(), map_gloc.cend(), 0.0);
    return Finalize(info, sum, sw);
  }
};

XGBOOST_REGISTER_METRIC(EvalMAP, "map")
    .describe("map@k for ranking.")
    .set_body([](char const* param) {
      return new EvalMAPScore{"map", param};
    });

XGBOOST_REGISTER_METRIC(EvalNDCG, "ndcg")
    .describe("ndcg@k for ranking.")
    .set_body([](char const* param) {
      return new EvalNDCG{"ndcg", param};
    });
}  // namespace xgboost::metric
