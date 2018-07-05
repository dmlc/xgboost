/*!
 * Copyright 2015 by Contributors
 * \file rank_metric.cc
 * \brief prediction rank based metrics.
 * \author Kailong Chen, Tianqi Chen
 */
#include <xgboost/metric.h>
#include <dmlc/registry.h>
#include <cmath>
#include "../common/sync.h"
#include "../common/math.h"

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
  bst_float Eval(const std::vector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) const override {
    CHECK(!distributed) << "metric AMS do not support distributed evaluation";
    using namespace std;  // NOLINT(*)

    const auto ndata = static_cast<bst_omp_uint>(info.labels_.Size());
    std::vector<std::pair<bst_float, unsigned> > rec(ndata);

    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      rec[i] = std::make_pair(preds[i], i);
    }
    std::sort(rec.begin(), rec.end(), common::CmpFirst);
    auto ntop = static_cast<unsigned>(ratio_ * ndata);
    if (ntop == 0) ntop = ndata;
    const double br = 10.0;
    unsigned thresindex = 0;
    double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
    const auto& labels = info.labels_.HostVector();
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

/*! \brief Area Under Curve, for both classification and rank */
struct EvalAuc : public Metric {
  bst_float Eval(const std::vector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) const override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels_.Size())
        << "label size predict size not match";
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());

    const std::vector<unsigned> &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;
    CHECK_EQ(gptr.back(), info.labels_.Size())
        << "EvalAuc: group structure must match number of prediction";
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    bst_float sum_auc = 0.0f;
    int auc_error = 0;
    // each thread takes a local rec
    std::vector< std::pair<bst_float, unsigned> > rec;
    const auto& labels = info.labels_.HostVector();
    for (bst_omp_uint k = 0; k < ngroup; ++k) {
      rec.clear();
      for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
        rec.emplace_back(preds[j], j);
      }
      XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);
      // calculate AUC
      double sum_pospair = 0.0;
      double sum_npos = 0.0, sum_nneg = 0.0, buf_pos = 0.0, buf_neg = 0.0;
      for (size_t j = 0; j < rec.size(); ++j) {
        const bst_float wt = info.GetWeight(rec[j].second);
        const bst_float ctr = labels[rec[j].second];
        // keep bucketing predictions in same bucket
        if (j != 0 && rec[j].first != rec[j - 1].first) {
          sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
          sum_npos += buf_pos;
          sum_nneg += buf_neg;
          buf_neg = buf_pos = 0.0f;
        }
        buf_pos += ctr * wt;
        buf_neg += (1.0f - ctr) * wt;
      }
      sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
      sum_npos += buf_pos;
      sum_nneg += buf_neg;
      // check weird conditions
      if (sum_npos <= 0.0 || sum_nneg <= 0.0) {
        auc_error = 1;
        continue;
      }
      // this is the AUC
      sum_auc += sum_pospair / (sum_npos*sum_nneg);
    }
    CHECK(!auc_error)
      << "AUC: the dataset only contains pos or neg samples";
    if (distributed) {
      bst_float dat[2];
      dat[0] = static_cast<bst_float>(sum_auc);
      dat[1] = static_cast<bst_float>(ngroup);
      // approximately estimate auc using mean
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
      return dat[0] / dat[1];
    } else {
      return static_cast<bst_float>(sum_auc) / ngroup;
    }
  }
  const char* Name() const override {
    return "auc";
  }
};

/*! \brief Evaluate rank list */
struct EvalRankList : public Metric {
 public:
  bst_float Eval(const std::vector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) const override {
    CHECK_EQ(preds.size(), info.labels_.Size())
        << "label size predict size not match";
    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(preds.size());
    const std::vector<unsigned> &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;
    CHECK_NE(gptr.size(), 0U) << "must specify group when constructing rank file";
    CHECK_EQ(gptr.back(), preds.size())
        << "EvalRanklist: group structure must match number of prediction";
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    double sum_metric = 0.0f;
    const auto& labels = info.labels_.HostVector();
    #pragma omp parallel reduction(+:sum_metric)
    {
      // each thread takes a local rec
      std::vector< std::pair<bst_float, unsigned> > rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        rec.clear();
        for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
          rec.emplace_back(preds[j], static_cast<int>(labels[j]));
        }
        sum_metric += this->EvalMetric(rec);
      }
    }
    if (distributed) {
      bst_float dat[2];
      dat[0] = static_cast<bst_float>(sum_metric);
      dat[1] = static_cast<bst_float>(ngroup);
      // approximately estimate the metric using mean
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
      return dat[0] / dat[1];
    } else {
      return static_cast<bst_float>(sum_metric) / ngroup;
    }
  }
  const char* Name() const override {
    return name_.c_str();
  }

 protected:
  explicit EvalRankList(const char* name, const char* param) {
    using namespace std;  // NOLINT(*)
    minus_ = false;
    if (param != nullptr) {
      std::ostringstream os;
      os << name << '@' << param;
      name_ = os.str();
      if (sscanf(param, "%u[-]?", &topn_) != 1) {
        topn_ = std::numeric_limits<unsigned>::max();
      }
      if (param[strlen(param) - 1] == '-') {
        minus_ = true;
      }
    } else {
      name_ = name;
      topn_ = std::numeric_limits<unsigned>::max();
    }
  }
  /*! \return evaluation metric, given the pair_sort record, (pred,label) */
  virtual bst_float EvalMetric(std::vector<std::pair<bst_float, unsigned> > &pair_sort) const = 0; // NOLINT(*)

 protected:
  unsigned topn_;
  std::string name_;
  bool minus_;
};

/*! \brief Precision at N, for both classification and rank */
struct EvalPrecision : public EvalRankList{
 public:
  explicit EvalPrecision(const char *name) : EvalRankList("pre", name) {}

 protected:
  bst_float EvalMetric(std::vector< std::pair<bst_float, unsigned> > &rec) const override {
    // calculate Precision
    std::sort(rec.begin(), rec.end(), common::CmpFirst);
    unsigned nhit = 0;
    for (size_t j = 0; j < rec.size() && j < this->topn_; ++j) {
      nhit += (rec[j].second != 0);
    }
    return static_cast<bst_float>(nhit) / topn_;
  }
};

/*! \brief NDCG: Normalized Discounted Cumulative Gain at N */
struct EvalNDCG : public EvalRankList{
 public:
  explicit EvalNDCG(const char *name) : EvalRankList("ndcg", name) {}

 protected:
  inline bst_float CalcDCG(const std::vector<std::pair<bst_float, unsigned> > &rec) const {
    double sumdcg = 0.0;
    for (size_t i = 0; i < rec.size() && i < this->topn_; ++i) {
      const unsigned rel = rec[i].second;
      if (rel != 0) {
        sumdcg += ((1 << rel) - 1) / std::log2(i + 2.0);
      }
    }
    return sumdcg;
  }
  virtual bst_float EvalMetric(std::vector<std::pair<bst_float, unsigned> > &rec) const { // NOLINT(*)
    XGBOOST_PARALLEL_STABLE_SORT(rec.begin(), rec.end(), common::CmpFirst);
    bst_float dcg = this->CalcDCG(rec);
    XGBOOST_PARALLEL_STABLE_SORT(rec.begin(), rec.end(), common::CmpSecond);
    bst_float idcg = this->CalcDCG(rec);
    if (idcg == 0.0f) {
      if (minus_) {
        return 0.0f;
      } else {
        return 1.0f;
      }
    }
    return dcg/idcg;
  }
};

/*! \brief Mean Average Precision at N, for both classification and rank */
struct EvalMAP : public EvalRankList {
 public:
  explicit EvalMAP(const char *name) : EvalRankList("map", name) {}

 protected:
  bst_float EvalMetric(std::vector< std::pair<bst_float, unsigned> > &rec) const override {
    std::sort(rec.begin(), rec.end(), common::CmpFirst);
    unsigned nhits = 0;
    double sumap = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
      if (rec[i].second != 0) {
        nhits += 1;
        if (i < this->topn_) {
          sumap += static_cast<bst_float>(nhits) / (i + 1);
        }
      }
    }
    if (nhits != 0) {
      sumap /= nhits;
      return static_cast<bst_float>(sumap);
    } else {
      if (minus_) {
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
  bst_float Eval(const std::vector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) const override {
    CHECK(!distributed) << "Cox metric does not support distributed evaluation";
    using namespace std;  // NOLINT(*)

    const auto ndata = static_cast<bst_omp_uint>(info.labels_.Size());
    const std::vector<size_t> &label_order = info.LabelAbsSort();

    // pre-compute a sum for the denominator
    double exp_p_sum = 0;  // we use double because we might need the precision with large datasets
    for (omp_ulong i = 0; i < ndata; ++i) {
      exp_p_sum += preds[i];
    }

    double out = 0;
    double accumulated_sum = 0;
    bst_omp_uint num_events = 0;
    const auto& labels = info.labels_.HostVector();
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      const size_t ind = label_order[i];
      const auto label = labels[ind];
      if (label > 0) {
        out -= log(preds[ind]) - log(exp_p_sum);
        ++num_events;
      }

      // only update the denominator after we move forward in time (labels are sorted)
      accumulated_sum += preds[ind];
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

/*! \brief Area Under PR Curve, for both classification and rank */
struct EvalAucPR : public Metric {
  // implementation of AUC-PR for weighted data
  // translated from PRROC R Package
  // see https://doi.org/10.1371/journal.pone.0092209

  bst_float Eval(const std::vector<bst_float> &preds, const MetaInfo &info,
                 bool distributed) const override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels_.Size())
        << "label size predict size not match";
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());
    const std::vector<unsigned> &gptr =
        info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;
    CHECK_EQ(gptr.back(), info.labels_.Size())
        << "EvalAucPR: group structure must match number of prediction";
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    double auc = 0.0;
    int auc_error = 0, auc_gt_one = 0;
    // each thread takes a local rec
    std::vector<std::pair<bst_float, unsigned>> rec;
    const auto& labels = info.labels_.HostVector();
    for (bst_omp_uint k = 0; k < ngroup; ++k) {
      double total_pos = 0.0;
      double total_neg = 0.0;
      rec.clear();
      for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
        total_pos += info.GetWeight(j) * labels[j];
        total_neg += info.GetWeight(j) * (1.0f - labels[j]);
        rec.emplace_back(preds[j], j);
      }
      XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);
      // we need pos > 0 && neg > 0
      if (0.0 == total_pos || 0.0 == total_neg) {
        auc_error = 1;
      }
      // calculate AUC
      double tp = 0.0, prevtp = 0.0, fp = 0.0, prevfp = 0.0, h = 0.0, a = 0.0, b = 0.0;
      for (size_t j = 0; j < rec.size(); ++j) {
        tp += info.GetWeight(rec[j].second) * labels[rec[j].second];
        fp += info.GetWeight(rec[j].second) * (1.0f - labels[rec[j].second]);
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
            auc += (tp / total_pos - prevtp / total_pos -
                    b / a * (std::log(a * tp / total_pos + b) -
                             std::log(a * prevtp / total_pos + b))) / a;
          } else {
            auc += (tp / total_pos - prevtp / total_pos) / a;
          }
          if (auc > 1.0) {
            auc_gt_one = 1;
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
    CHECK(!auc_error) << "AUC-PR: the dataset only contains pos or neg samples";
    CHECK(!auc_gt_one) << "AUC-PR: AUC > 1.0";
    if (distributed) {
      bst_float dat[2];
      dat[0] = static_cast<bst_float>(auc);
      dat[1] = static_cast<bst_float>(ngroup);
      // approximately estimate auc using mean
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
      return dat[0] / dat[1];
    } else {
      return static_cast<bst_float>(auc) / ngroup;
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
.set_body([](const char* param) { return new EvalPrecision(param); });

XGBOOST_REGISTER_METRIC(NDCG, "ndcg")
.describe("ndcg@k for rank.")
.set_body([](const char* param) { return new EvalNDCG(param); });

XGBOOST_REGISTER_METRIC(MAP, "map")
.describe("map@k for rank.")
.set_body([](const char* param) { return new EvalMAP(param); });

XGBOOST_REGISTER_METRIC(Cox, "cox-nloglik")
.describe("Negative log partial likelihood of Cox proportioanl hazards model.")
.set_body([](const char* param) { return new EvalCox(); });
}  // namespace metric
}  // namespace xgboost

