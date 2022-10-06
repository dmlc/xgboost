/*!
 * Copyright 2020 XGBoost contributors
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

#include <dmlc/registry.h>
#include <xgboost/metric.h>

#include <cmath>
#include <vector>

#include "../collective/communicator-inl.h"
#include "../common/math.h"
#include "../common/threading_utils.h"
#include "metric_common.h"
#include "xgboost/host_device_vector.h"

namespace {

using PredIndPair = std::pair<xgboost::bst_float, uint32_t>;
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

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(rank_metric);

/*! \brief AMS: also records best threshold */
struct EvalAMS : public Metric {
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
    common::ParallelFor(ndata, tparam_->Threads(),
                        [&](bst_omp_uint i) { rec[i] = std::make_pair(h_preds[i], i); });
    XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);
    auto ntop = static_cast<unsigned>(ratio_ * ndata);
    if (ntop == 0) ntop = ndata;
    const double br = 10.0;
    unsigned thresindex = 0;
    double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
    const auto& labels = info.labels.View(GenericParameter::kCpuId);
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
struct EvalRank : public Metric, public EvalRankConfig {
 private:
  // This is used to compute the ranking metrics on the GPU - for training jobs that run on the GPU.
  std::unique_ptr<xgboost::Metric> rank_gpu_;

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
    if (tparam_->gpu_id >= 0) {
      if (!rank_gpu_) {
        rank_gpu_.reset(GPUMetric::CreateGPUMetric(this->Name(), tparam_));
      }
      if (rank_gpu_) {
        sum_metric = rank_gpu_->Eval(preds, info);
      }
    }

    CHECK(tparam_);
    std::vector<double> sum_tloc(tparam_->Threads(), 0.0);

    if (!rank_gpu_ || tparam_->gpu_id < 0) {
      const auto& labels = info.labels.View(GenericParameter::kCpuId);
      const auto &h_preds = preds.ConstHostVector();

      dmlc::OMPException exc;
#pragma omp parallel num_threads(tparam_->Threads())
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

    if (collective::IsDistributed()) {
      double dat[2]{sum_metric, static_cast<double>(ngroups)};
      // approximately estimate the metric using mean
      collective::Allreduce<collective::Operation::kSum>(dat, 2);
      return dat[0] / dat[1];
    } else {
      return sum_metric / ngroups;
    }
  }

  const char* Name() const override {
    return name.c_str();
  }

 protected:
  explicit EvalRank(const char* name, const char* param) {
    using namespace std;  // NOLINT(*)

    if (param != nullptr) {
      std::ostringstream os;
      if (sscanf(param, "%u[-]?", &topn) == 1) {
        os << name << '@' << param;
        this->name = os.str();
      } else {
        os << name << param;
        this->name = os.str();
      }
      if (param[strlen(param) - 1] == '-') {
        minus = true;
      }
    } else {
      this->name = name;
    }
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

/*! \brief NDCG: Normalized Discounted Cumulative Gain at N */
struct EvalNDCG : public EvalRank {
 private:
  double CalcDCG(const PredIndPairContainer &rec) const {
    double sumdcg = 0.0;
    for (size_t i = 0; i < rec.size() && i < this->topn; ++i) {
      const unsigned rel = rec[i].second;
      if (rel != 0) {
        sumdcg += ((1 << rel) - 1) / std::log2(i + 2.0);
      }
    }
    return sumdcg;
  }

 public:
  explicit EvalNDCG(const char* name, const char* param) : EvalRank(name, param) {}

  double EvalGroup(PredIndPairContainer *recptr) const override {
    PredIndPairContainer &rec(*recptr);
    std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
    double dcg = CalcDCG(rec);
    std::stable_sort(rec.begin(), rec.end(), common::CmpSecond);
    double idcg = CalcDCG(rec);
    if (idcg == 0.0f) {
      if (this->minus) {
        return 0.0f;
      } else {
        return 1.0f;
      }
    }
    return dcg/idcg;
  }
};

/*! \brief Mean Average Precision at N, for both classification and rank */
struct EvalMAP : public EvalRank {
 public:
  explicit EvalMAP(const char* name, const char* param) : EvalRank(name, param) {}

  double EvalGroup(PredIndPairContainer *recptr) const override {
    PredIndPairContainer &rec(*recptr);
    std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
    unsigned nhits = 0;
    double sumap = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
      if (rec[i].second != 0) {
        nhits += 1;
        if (i < this->topn) {
          sumap += static_cast<double>(nhits) / (i + 1);
        }
      }
    }
    if (nhits != 0) {
      sumap /= nhits;
      return sumap;
    } else {
      if (this->minus) {
        return 0.0;
      } else {
        return 1.0;
      }
    }
  }
};

/*! \brief Cox: Partial likelihood of the Cox proportional hazards model */
struct EvalCox : public Metric {
 public:
  EvalCox() = default;
  double Eval(const HostDeviceVector<bst_float>& preds, const MetaInfo& info) override {
    CHECK(!collective::IsDistributed()) << "Cox metric does not support distributed evaluation";
    using namespace std;  // NOLINT(*)

    const auto ndata = static_cast<bst_omp_uint>(info.labels.Size());
    const auto &label_order = info.LabelAbsSort();

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

XGBOOST_REGISTER_METRIC(NDCG, "ndcg")
.describe("ndcg@k for rank.")
.set_body([](const char* param) { return new EvalNDCG("ndcg", param); });

XGBOOST_REGISTER_METRIC(MAP, "map")
.describe("map@k for rank.")
.set_body([](const char* param) { return new EvalMAP("map", param); });

XGBOOST_REGISTER_METRIC(Cox, "cox-nloglik")
.describe("Negative log partial likelihood of Cox proportional hazards model.")
.set_body([](const char*) { return new EvalCox(); });
}  // namespace metric
}  // namespace xgboost
