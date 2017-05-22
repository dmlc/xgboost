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
    CHECK(param != nullptr)
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

    const bst_omp_uint ndata = static_cast<bst_omp_uint>(info.labels.size());
    std::vector<std::pair<bst_float, unsigned> > rec(ndata);

    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      rec[i] = std::make_pair(preds[i], i);
    }
    std::sort(rec.begin(), rec.end(), common::CmpFirst);
    unsigned ntop = static_cast<unsigned>(ratio_ * ndata);
    if (ntop == 0) ntop = ndata;
    const double br = 10.0;
    unsigned thresindex = 0;
    double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
    for (unsigned i = 0; i < static_cast<unsigned>(ndata-1) && i < ntop; ++i) {
      const unsigned ridx = rec[i].second;
      const bst_float wt = info.GetWeight(ridx);
      if (info.labels[ridx] > 0.5f) {
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
    CHECK_NE(info.labels.size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size())
        << "label size predict size not match";
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels.size());

    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    CHECK_EQ(gptr.back(), info.labels.size())
        << "EvalAuc: group structure must match number of prediction";
    const bst_omp_uint ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    bst_float sum_auc = 0.0f;
    int auc_error = 0;
    // each thread takes a local rec
    std::vector< std::pair<bst_float, unsigned> > rec;
    for (bst_omp_uint k = 0; k < ngroup; ++k) {
      rec.clear();
      for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
        rec.push_back(std::make_pair(preds[j], j));
      }
      XGBOOST_PARALLEL_SORT(rec.begin(), rec.end(), common::CmpFirst);
      // calculate AUC
      double sum_pospair = 0.0;
      double sum_npos = 0.0, sum_nneg = 0.0, buf_pos = 0.0, buf_neg = 0.0;
      for (size_t j = 0; j < rec.size(); ++j) {
        const bst_float wt = info.GetWeight(rec[j].second);
        const bst_float ctr = info.labels[rec[j].second];
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
    CHECK_EQ(preds.size(), info.labels.size())
        << "label size predict size not match";
    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(preds.size());
    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    CHECK_NE(gptr.size(), 0U) << "must specify group when constructing rank file";
    CHECK_EQ(gptr.back(), preds.size())
        << "EvalRanklist: group structure must match number of prediction";
    const bst_omp_uint ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    double sum_metric = 0.0f;
    #pragma omp parallel reduction(+:sum_metric)
    {
      // each thread takes a local rec
      std::vector< std::pair<bst_float, unsigned> > rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        rec.clear();
        for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
          rec.push_back(std::make_pair(preds[j], static_cast<int>(info.labels[j])));
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
  virtual bst_float EvalMetric(std::vector< std::pair<bst_float, unsigned> > &rec) const {
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
  virtual bst_float EvalMetric(std::vector< std::pair<bst_float, unsigned> > &rec) const {
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

XGBOOST_REGISTER_METRIC(AMS, "ams")
.describe("AMS metric for higgs.")
.set_body([](const char* param) { return new EvalAMS(param); });

XGBOOST_REGISTER_METRIC(Auc, "auc")
.describe("Area under curve for both classification and rank.")
.set_body([](const char* param) { return new EvalAuc(); });

XGBOOST_REGISTER_METRIC(Precision, "pre")
.describe("precision@k for rank.")
.set_body([](const char* param) { return new EvalPrecision(param); });

XGBOOST_REGISTER_METRIC(NDCG, "ndcg")
.describe("ndcg@k for rank.")
.set_body([](const char* param) { return new EvalNDCG(param); });

XGBOOST_REGISTER_METRIC(MAP, "map")
.describe("map@k for rank.")
.set_body([](const char* param) { return new EvalMAP(param); });
}  // namespace metric
}  // namespace xgboost
