/*!
 * Copyright 2019 XGBoost contributors
 */
// Include the metrics that aren't accelerated on the GPU here, with the rest in rank_metric.cu

#include <rabit/rabit.h>
#include <dmlc/registry.h>

#include <vector>

#include <xgboost/metric.h>
#include "../common/math.h"
#include "../common/timer.h"
#include "metric_common.h"

namespace {
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
  GetWeightOfInstance(const xgboost::MetaInfo &info,
                      unsigned instance_id, unsigned group_id) {
    return info.GetWeight(instance_id);
  }

  inline static xgboost::bst_float
  GetWeightOfSortedRecord(const xgboost::MetaInfo& info,
                          const PredIndPairContainer &rec,
                          unsigned record_id, unsigned group_id) {
    return info.GetWeight(rec[record_id].second);
  }
};

class PerGroupWeightPolicy {
 public:
  inline static xgboost::bst_float
  GetWeightOfInstance(const xgboost::MetaInfo& info,
                      unsigned instance_id, unsigned group_id) {
    return info.GetWeight(group_id);
  }

  inline static xgboost::bst_float
  GetWeightOfSortedRecord(const xgboost::MetaInfo& info,
                          const PredIndPairContainer &rec,
                          unsigned record_id, unsigned group_id) {
    return info.GetWeight(group_id);
  }
};
}  // end of anonymous namespace

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

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK(!distributed) << "metric AMS do not support distributed evaluation";
    using namespace std;  // NOLINT(*)

    const auto ndata = static_cast<bst_omp_uint>(info.labels_.Size());
    PredIndPairContainer rec(ndata);

    const std::vector<bst_float>& h_preds = preds.ConstHostVector();
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      rec[i] = std::make_pair(h_preds[i], i);
    }
    XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);
    auto ntop = static_cast<unsigned>(ratio_ * ndata);
    if (ntop == 0) ntop = ndata;
    const double br = 10.0;
    unsigned thresindex = 0;
    double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
    const auto& labels = info.labels_.ConstHostVector();
    for (unsigned i = 0; i < static_cast<unsigned>(ndata-1) && i < ntop; ++i) {
      const unsigned ridx = rec[i].second;
      const bst_float wt = info.GetWeight(ridx);
      if (labels[ridx] > 0.5f) {
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

/*! \brief Cox: Partial likelihood of the Cox proportional hazards model */
struct EvalCox : public Metric {
 public:
  EvalCox() = default;
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK(!distributed) << "Cox metric does not support distributed evaluation";
    using namespace std;  // NOLINT(*)

    const auto ndata = static_cast<bst_omp_uint>(info.labels_.Size());
    const std::vector<size_t> &label_order = info.LabelAbsSort();

    // pre-compute a sum for the denominator
    double exp_p_sum = 0;  // we use double because we might need the precision with large datasets

    const std::vector<bst_float>& h_preds = preds.ConstHostVector();
    for (omp_ulong i = 0; i < ndata; ++i) {
      exp_p_sum += h_preds[i];
    }

    double out = 0;
    double accumulated_sum = 0;
    bst_omp_uint num_events = 0;
    const auto& labels = info.labels_.ConstHostVector();
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      const size_t ind = label_order[i];
      const auto label = labels[ind];
      if (label > 0) {
        out -= log(h_preds[ind]) - log(exp_p_sum);
        ++num_events;
      }

      // only update the denominator after we move forward in time (labels are sorted)
      accumulated_sum += h_preds[ind];
      if (i == ndata - 1 || std::abs(label) < std::abs(labels[label_order[i + 1]])) {
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

// This internal metric computation type is used to compute the AUC(PR) metrics on CPU. This
// isn't meant to be used externally and thus enclosed within the internal namespace.
//
// The AUC(PR) metric type registered in rank_metric.cu decides and delegates which type is to
// be used for AUC(PR) metric computation on the GPU/CPU. The reason for splitting the
// functionality across the two files is the following:
// Sorting datasets containing large number of rows is (much) faster when parallel sort semantics
// is used on the CPU. The __gnu_parallel/concurrency primitives needed to perform this cannot be
// used when the translation unit is compiled using the 'nvcc' compiler (as the corresponding
// headers that brings in those function declaration can't be included with CUDA).
// Thus, this is moved to a separate (non CUDA) file which can then use those primitives when
// built using the standard C++ compiler. Hence, non-GPU builds and GPU builds that trains on
// CPU will use this facility.
// The CUDA file has the logic to compute this metric on the GPU will use it when a valid device
// ordinal is specified. When one isn't provided, it will delegate the responsibility to this type.
// The way it does that is by looking up the metrics registry at *runtime* for this type with a
// specific key - auc-cpu or aucpr-cpu.
// Note: It doesn't resolve this type during the construction of the EvalAuc or EvalAucPR that
// is used to drive the AUC(PR) metrics computation on CPU/GPU. This is because, the order of
// registration of the metric types into the metrics registry isn't deterministic; this internal
// type may or may not be present when EvalAuc or EvalAucPR is getting constructed. Hence, the
// resolution of the AUC metric computation type on the CPU is deferred until runtime.
namespace internal {
/*! \brief Area Under Curve, for both classification and rank computed on CPU */
struct EvalAucCpu : public Metric {
 private:
  template <typename WeightPolicy>
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) {
    // All sanity is done by the caller
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.empty() ? tgptr : info.group_ptr_;

    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum of all AUC's across all query groups
    double sum_auc = 0.0;
    int auc_error = 0;
    const auto& labels = info.labels_.ConstHostVector();
    const std::vector<bst_float>& h_preds = preds.ConstHostVector();

    #pragma omp parallel reduction(+:sum_auc, auc_error) if (ngroups > 1)
    {
      // Each thread works on a distinct group and sorts the predictions in that group
      PredIndPairContainer rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint group_id = 0; group_id < ngroups; ++group_id) {
        // Same thread can work on multiple groups one after another; hence, resize
        // the predictions array based on the current group
        rec.resize(gptr[group_id + 1] - gptr[group_id]);
        #pragma omp parallel for schedule(static) if (!omp_in_parallel())
        for (bst_omp_uint j = gptr[group_id]; j < gptr[group_id + 1]; ++j) {
          rec[j - gptr[group_id]] = {h_preds[j], j};
        }

        if (omp_in_parallel()) {
          std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
        } else {
          XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);
        }

        // calculate AUC
        double sum_pospair = 0.0;
        double sum_npos = 0.0, sum_nneg = 0.0, buf_pos = 0.0, buf_neg = 0.0;
        for (size_t j = 0; j < rec.size(); ++j) {
          const bst_float wt = WeightPolicy::GetWeightOfSortedRecord(info, rec, j, group_id);
          const bst_float ctr = labels[rec[j].second];
          // keep bucketing predictions in same bucket
          if (j != 0 && rec[j].first != rec[j - 1].first) {
            sum_pospair += buf_neg * (sum_npos + buf_pos * 0.5);
            sum_npos += buf_pos;
            sum_nneg += buf_neg;
            buf_neg = buf_pos = 0.0f;
          }
          buf_pos += ctr * wt;
          buf_neg += (1.0f - ctr) * wt;
        }
        sum_pospair += buf_neg * (sum_npos + buf_pos * 0.5);
        sum_npos += buf_pos;
        sum_nneg += buf_neg;
        // check weird conditions
        if (sum_npos <= 0.0 || sum_nneg <= 0.0) {
          auc_error += 1;
        } else {
          // this is the AUC
          sum_auc += sum_pospair / (sum_npos * sum_nneg);
        }
      }
    }

    // Report average AUC across all groups
    // In distributed mode, workers which only contains pos or neg samples
    // will be ignored when aggregate AUC.
    bst_float dat[2] = {0.0f, 0.0f};
    if (auc_error < static_cast<int>(ngroups)) {
      dat[0] = static_cast<bst_float>(sum_auc);
      dat[1] = static_cast<bst_float>(static_cast<int>(ngroups) - auc_error);
    }
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    CHECK_GT(dat[1], 0.0f)
      << "AUC: the dataset only contains pos or neg samples";
    return dat[0] / dat[1];
  }

 public:
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    // For ranking task, weights are per-group
    // For binary classification task, weights are per-instance
    const bool is_ranking_task =
      !info.group_ptr_.empty() && info.weights_.Size() != info.num_row_;
    if (is_ranking_task) {
      return Eval<PerGroupWeightPolicy>(preds, info, distributed);
    } else {
      return Eval<PerInstanceWeightPolicy>(preds, info, distributed);
    }
  }

  const char *Name() const override { return "auc-cpu"; }
};

/*! \brief Area Under PR Curve, for both classification and rank computed on CPU */
struct EvalAucPRCpu : public Metric {
  // implementation of AUC-PR for weighted data
  // translated from PRROC R Package
  // see https://doi.org/10.1371/journal.pone.0092209
 private:
  template <typename WeightPolicy>
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) {
    // All sanity is done by the caller
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());
    const std::vector<unsigned> &gptr =
        info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;

    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);

    // sum of all AUC's across all query groups
    double sum_auc = 0.0;
    int auc_error = 0;

    const auto& h_labels = info.labels_.ConstHostVector();
    const std::vector<bst_float>& h_preds = preds.ConstHostVector();

    #pragma omp parallel reduction(+:sum_auc, auc_error) if (ngroups > 1)
    {
      // Each thread works on a distinct group and sorts the predictions in that group
      PredIndPairContainer rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint group_id = 0; group_id < ngroups; ++group_id) {
        double total_pos = 0.0;
        double total_neg = 0.0;
        // Same thread can work on multiple groups one after another; hence, resize
        // the predictions array based on the current group
        rec.resize(gptr[group_id + 1] - gptr[group_id]);
        #pragma omp parallel for schedule(static) reduction(+:total_pos, total_neg) \
          if (!omp_in_parallel())
        for (bst_omp_uint j = gptr[group_id]; j < gptr[group_id + 1]; ++j) {
          const bst_float wt = WeightPolicy::GetWeightOfInstance(info, j, group_id);
          total_pos += wt * h_labels[j];
          total_neg += wt * (1.0f - h_labels[j]);
          rec[j - gptr[group_id]] = {h_preds[j], j};
        }

        if (omp_in_parallel()) {
          std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
        } else {
          XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);
        }

        // we need pos > 0 && neg > 0
        if (total_pos <= 0.0 || total_neg <= 0.0) {
          auc_error += 1;
          continue;
        }
        // calculate AUC
        double tp = 0.0, prevtp = 0.0, fp = 0.0, prevfp = 0.0, h = 0.0, a = 0.0, b = 0.0;
        for (size_t j = 0; j < rec.size(); ++j) {
          const bst_float wt = WeightPolicy::GetWeightOfSortedRecord(info, rec, j, group_id);
          tp += wt * h_labels[rec[j].second];
          fp += wt * (1.0f - h_labels[rec[j].second]);
          if ((j < rec.size() - 1 && rec[j].first != rec[j + 1].first) || j  == rec.size() - 1) {
            if (tp == prevtp) {
              a = 1.0;
              b = 0.0;
            } else {
              h = (fp - prevfp) / (tp - prevtp);
              a = 1.0 + h;
              b = (prevfp - h * prevtp) / total_pos;
            }
            if (0.0 != b) {
              sum_auc += (tp / total_pos - prevtp / total_pos -
                          b / a * (std::log(a * tp / total_pos + b) -
                                   std::log(a * prevtp / total_pos + b))) / a;
            } else {
              sum_auc += (tp / total_pos - prevtp / total_pos) / a;
            }
            prevtp = tp;
            prevfp = fp;
          }
        }
        // sanity check
        if (tp < 0 || prevtp < 0 || fp < 0 || prevfp < 0) {
          CHECK(!auc_error) << "AUC-PR: error in calculation";
        }
      }
    }

    // Report average AUC-PR across all groups
    // In distributed mode, workers which only contains pos or neg samples
    // will be ignored when aggregate AUC-PR.
    bst_float dat[2] = {0.0f, 0.0f};
    if (auc_error < static_cast<int>(ngroups)) {
      dat[0] = static_cast<bst_float>(sum_auc);
      dat[1] = static_cast<bst_float>(static_cast<int>(ngroups) - auc_error);
    }
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    CHECK_GT(dat[1], 0.0f)
      << "AUC-PR: the dataset only contains pos or neg samples";
    CHECK_LE(dat[0], dat[1]) << "AUC-PR: AUC > 1.0";
    return dat[0] / dat[1];
  }

 public:
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    // For ranking task, weights are per-group
    // For binary classification task, weights are per-instance
    const bool is_ranking_task =
      !info.group_ptr_.empty() && info.weights_.Size() != info.num_row_;
    if (is_ranking_task) {
      return Eval<PerGroupWeightPolicy>(preds, info, distributed);
    } else {
      return Eval<PerInstanceWeightPolicy>(preds, info, distributed);
    }
  }

  const char *Name() const override { return "aucpr-cpu"; }
};
}  // end of namespace internal

XGBOOST_REGISTER_METRIC(AucCpu, "auc-cpu")
.describe("Internal AUC metric computation on CPU for classification and rank.")
.set_body([](const char* param) { return new internal::EvalAucCpu(); });

XGBOOST_REGISTER_METRIC(AucPRCpu, "aucpr-cpu")
.describe("Internal Area under PR curve computation on CPU for both classification and rank.")
.set_body([](const char* param) { return new internal::EvalAucPRCpu(); });

XGBOOST_REGISTER_METRIC(AMS, "ams")
.describe("AMS metric for higgs.")
.set_body([](const char* param) { return new EvalAMS(param); });

XGBOOST_REGISTER_METRIC(Cox, "cox-nloglik")
.describe("Negative log partial likelihood of Cox proportioanl hazards model.")
.set_body([](const char* param) { return new EvalCox(); });
}  // end of metric namespace
}  // end of xgboost namespace

#if !defined(XGBOOST_USE_CUDA)
#include "rank_metric.cu"
#endif  // !defined(XGBOOST_USE_CUDA)
