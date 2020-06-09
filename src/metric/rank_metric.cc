/*!
 * Copyright 2020 XGBoost contributors
 */
// When device ordinal is present, we would want to build the metrics on the GPU. It is *not*
// possible for a valid device ordinal to be present for non GPU builds. However, it is possible
// for an invalid device ordinal to be specified in GPU builds - to train/predict and/or compute
// the metrics on CPU. To accommodate these scenarios, the following is done for the metrics
// accelarated on the GPU.
// - An internal GPU registry holds all the GPU metric types (defined in the .cu file)
// - An instance of the appropriate gpu metric type is created when a device ordinal is present
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

#include <rabit/rabit.h>
#include <xgboost/metric.h>
#include <dmlc/registry.h>
#include <cmath>

#include <vector>

#include "xgboost/host_device_vector.h"
#include "../common/math.h"
#include "metric_common.h"

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
                      unsigned instance_id, unsigned group_id) {
    return info.GetWeight(instance_id);
  }
  inline static xgboost::bst_float
  GetWeightOfSortedRecord(const xgboost::MetaInfo& info,
                          const PredIndPairContainer& rec,
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
                          const PredIndPairContainer& rec,
                          unsigned record_id, unsigned group_id) {
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

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK(!distributed) << "metric AMS do not support distributed evaluation";
    using namespace std;  // NOLINT(*)

    const auto ndata = static_cast<bst_omp_uint>(info.labels_.Size());
    PredIndPairContainer rec(ndata);

    const auto &h_preds = preds.ConstHostVector();
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

/*! \brief Area Under Curve, for both classification and rank computed on CPU */
struct EvalAuc : public Metric {
 private:
  // This is used to compute the AUC metrics on the GPU - for ranking tasks and
  // for training jobs that run on the GPU.
  std::unique_ptr<xgboost::Metric> auc_gpu_;

  template <typename WeightPolicy>
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed,
                 const std::vector<unsigned> &gptr) {
    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum of all AUC's across all query groups
    double sum_auc = 0.0;
    int auc_error = 0;
    const auto& labels = info.labels_.ConstHostVector();
    const auto &h_preds = preds.ConstHostVector();

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

        XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);
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
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << "label size predict size not match";
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());

    const auto &gptr = info.group_ptr_.empty() ? tgptr : info.group_ptr_;
    CHECK_EQ(gptr.back(), info.labels_.Size())
        << "EvalAuc: group structure must match number of prediction";

    // For ranking task, weights are per-group
    // For binary classification task, weights are per-instance
    const bool is_ranking_task =
      !info.group_ptr_.empty() && info.weights_.Size() != info.num_row_;

    // Check if we have a GPU assignment; else, revert back to CPU
    if (tparam_->gpu_id >= 0) {
      if (!auc_gpu_) {
        // Check and see if we have the GPU metric registered in the internal registry
        auc_gpu_.reset(GPUMetric::CreateGPUMetric(this->Name(), tparam_));
      }

      if (auc_gpu_) {
        return auc_gpu_->Eval(preds, info, distributed);
      }
    }

    if (is_ranking_task) {
      return Eval<PerGroupWeightPolicy>(preds, info, distributed, gptr);
    } else {
      return Eval<PerInstanceWeightPolicy>(preds, info, distributed, gptr);
    }
  }

  const char *Name() const override { return "auc"; }
};

/*! \brief Evaluate rank list */
struct EvalRank : public Metric, public EvalRankConfig {
 private:
  // This is used to compute the ranking metrics on the GPU - for training jobs that run on the GPU.
  std::unique_ptr<xgboost::Metric> rank_gpu_;

 public:
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_EQ(preds.Size(), info.labels_.Size())
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
        sum_metric = rank_gpu_->Eval(preds, info, distributed);
      }
    }

    if (!rank_gpu_ || tparam_->gpu_id < 0) {
      const auto &labels = info.labels_.ConstHostVector();
      const auto &h_preds = preds.ConstHostVector();

      #pragma omp parallel reduction(+:sum_metric)
      {
        // each thread takes a local rec
        PredIndPairContainer rec;
        #pragma omp for schedule(static)
        for (bst_omp_uint k = 0; k < ngroups; ++k) {
          rec.clear();
          for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
            rec.emplace_back(h_preds[j], static_cast<int>(labels[j]));
          }
          sum_metric += this->EvalGroup(&rec);
        }
      }
    }

    if (distributed) {
      bst_float dat[2];
      dat[0] = static_cast<bst_float>(sum_metric);
      dat[1] = static_cast<bst_float>(ngroups);
      // approximately estimate the metric using mean
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
      return dat[0] / dat[1];
    } else {
      return static_cast<bst_float>(sum_metric) / ngroups;
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
        return 0.0f;
      } else {
        return 1.0f;
      }
    }
  }
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

/*! \brief Area Under PR Curve, for both classification and rank computed on CPU */
struct EvalAucPR : public Metric {
  // implementation of AUC-PR for weighted data
  // translated from PRROC R Package
  // see https://doi.org/10.1371/journal.pone.0092209
 private:
  // This is used to compute the AUCPR metrics on the GPU - for ranking tasks and
  // for training jobs that run on the GPU.
  std::unique_ptr<xgboost::Metric> aucpr_gpu_;

  template <typename WeightPolicy>
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed,
                 const std::vector<unsigned> &gptr) {
    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);

    // sum of all AUC's across all query groups
    double sum_auc = 0.0;
    int auc_error = 0;

    const auto &h_labels = info.labels_.ConstHostVector();
    const auto &h_preds = preds.ConstHostVector();

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
          if (!omp_in_parallel())  // NOLINT
        for (bst_omp_uint j = gptr[group_id]; j < gptr[group_id + 1]; ++j) {
          const bst_float wt = WeightPolicy::GetWeightOfInstance(info, j, group_id);
          total_pos += wt * h_labels[j];
          total_neg += wt * (1.0f - h_labels[j]);
          rec[j - gptr[group_id]] = {h_preds[j], j};
        }

        // we need pos > 0 && neg > 0
        if (total_pos <= 0.0 || total_neg <= 0.0) {
          auc_error += 1;
          continue;
        }

        XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);

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
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << "label size predict size not match";
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());

    const auto &gptr = info.group_ptr_.empty() ? tgptr : info.group_ptr_;
    CHECK_EQ(gptr.back(), info.labels_.Size())
        << "EvalAucPR: group structure must match number of prediction";

    // For ranking task, weights are per-group
    // For binary classification task, weights are per-instance
    const bool is_ranking_task =
      !info.group_ptr_.empty() && info.weights_.Size() != info.num_row_;

    // Check if we have a GPU assignment; else, revert back to CPU
    if (tparam_->gpu_id >= 0 && is_ranking_task) {
      if (!aucpr_gpu_) {
        // Check and see if we have the GPU metric registered in the internal registry
        aucpr_gpu_.reset(GPUMetric::CreateGPUMetric(this->Name(), tparam_));
      }

      if (aucpr_gpu_) {
        return aucpr_gpu_->Eval(preds, info, distributed);
      }
    }

    if (is_ranking_task) {
      return Eval<PerGroupWeightPolicy>(preds, info, distributed, gptr);
    } else {
      return Eval<PerInstanceWeightPolicy>(preds, info, distributed, gptr);
    }
  }

  const char *Name() const override { return "aucpr"; }
};

XGBOOST_REGISTER_METRIC(AMS, "ams")
.describe("AMS metric for higgs.")
.set_body([](const char* param) { return new EvalAMS(param); });

XGBOOST_REGISTER_METRIC(Auc, "auc")
.describe("Area under curve for both classification and rank.")
.set_body([](const char* param) { return new EvalAuc(); });

XGBOOST_REGISTER_METRIC(AucPR, "aucpr")
.describe("Area under PR curve for both classification and rank.")
.set_body([](const char* param) { return new EvalAucPR(); });

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
.describe("Negative log partial likelihood of Cox proportioanl hazards model.")
.set_body([](const char* param) { return new EvalCox(); });
}  // namespace metric
}  // namespace xgboost
