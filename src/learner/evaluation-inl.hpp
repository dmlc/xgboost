/*!
 * Copyright 2014 by Contributors
 * \file xgboost_evaluation-inl.hpp
 * \brief evaluation metrics for regression and classification and rank
 * \author Kailong Chen, Tianqi Chen
 */
#ifndef XGBOOST_LEARNER_EVALUATION_INL_HPP_
#define XGBOOST_LEARNER_EVALUATION_INL_HPP_

#include <vector>
#include <utility>
#include <string>
#include <cmath>
#include <climits>
#include <algorithm>
#include "../sync/sync.h"
#include "../utils/math.h"
#include "./evaluation.h"
#include "./helper_utils.h"

namespace xgboost {
namespace learner {
/*!
 * \brief base class of elementwise evaluation
 * \tparam Derived the name of subclass
 */
template<typename Derived>
struct EvalEWiseBase : public IEvaluator {
  virtual float Eval(const std::vector<float> &preds,
                     const MetaInfo &info,
                     bool distributed) const {
    utils::Check(info.labels.size() != 0, "label set cannot be empty");
    utils::Check(preds.size() == info.labels.size(),
                 "label and prediction size not match"\
                 "hint: use merror or mlogloss for multi-class classification");

    const bst_omp_uint ndata = static_cast<bst_omp_uint>(info.labels.size());

    float sum = 0.0, wsum = 0.0;
    #pragma omp parallel for reduction(+: sum, wsum) schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      const float wt = info.GetWeight(i);
      sum += Derived::EvalRow(info.labels[i], preds[i]) * wt;
      wsum += wt;
    }
    float dat[2]; dat[0] = sum, dat[1] = wsum;
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return Derived::GetFinal(dat[0], dat[1]);
  }
  /*!
   * \brief to be implemented by subclass,
   *   get evaluation result from one row
   * \param label label of current instance
   * \param pred prediction value of current instance
   */
  inline static float EvalRow(float label, float pred);
  /*!
   * \brief to be overide by subclas, final trasnformation
   * \param esum the sum statistics returned by EvalRow
   * \param wsum sum of weight
   */
  inline static float GetFinal(float esum, float wsum) {
    return esum / wsum;
  }
};

/*! \brief RMSE */
struct EvalRMSE : public EvalEWiseBase<EvalRMSE> {
  virtual const char *Name(void) const {
    return "rmse";
  }
  inline static float EvalRow(float label, float pred) {
    float diff = label - pred;
    return diff * diff;
  }
  inline static float GetFinal(float esum, float wsum) {
    return std::sqrt(esum / wsum);
  }
};

/*! \brief logloss */
struct EvalLogLoss : public EvalEWiseBase<EvalLogLoss> {
  virtual const char *Name(void) const {
    return "logloss";
  }
  inline static float EvalRow(float y, float py) {
    const float eps = 1e-16f;
    const float pneg = 1.0f - py;
    if (py < eps) {
      return -y * std::log(eps) - (1.0f - y)  * std::log(1.0f - eps);
    } else if (pneg < eps) {
      return -y * std::log(1.0f - eps) - (1.0f - y)  * std::log(eps);
    } else {
      return -y * std::log(py) - (1.0f - y) * std::log(pneg);
    }
  }
};

/*! \brief error */
struct EvalError : public EvalEWiseBase<EvalError> {
  virtual const char *Name(void) const {
    return "error";
  }
  inline static float EvalRow(float label, float pred) {
    // assume label is in [0,1]
    return pred > 0.5f ? 1.0f - label : label;
  }
};

/*! \brief loglikelihood of poission distribution */
struct EvalPoissionNegLogLik : public EvalEWiseBase<EvalPoissionNegLogLik> {
  virtual const char *Name(void) const {
    return "poisson-nloglik";
  }
  inline static float EvalRow(float y, float py) {
    const float eps = 1e-16f;
    if (py < eps) py = eps;
    return utils::LogGamma(y + 1.0f) + py - std::log(py) * y;
  }
};

/*!
 * \brief base class of multi-class evaluation
 * \tparam Derived the name of subclass
 */
template<typename Derived>
struct EvalMClassBase : public IEvaluator {
  virtual float Eval(const std::vector<float> &preds,
                     const MetaInfo &info,
                     bool distributed) const {
    utils::Check(info.labels.size() != 0, "label set cannot be empty");
    utils::Check(preds.size() % info.labels.size() == 0,
                 "label and prediction size not match");
    const size_t nclass = preds.size() / info.labels.size();
    utils::Check(nclass > 1,
                 "mlogloss and merror are only used for multi-class classification,"\
                 " use logloss for binary classification");
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(info.labels.size());
    float sum = 0.0, wsum = 0.0;
    int label_error = 0;
    #pragma omp parallel for reduction(+: sum, wsum) schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      const float wt = info.GetWeight(i);
      int label =  static_cast<int>(info.labels[i]);
      if (label >= 0 && label < static_cast<int>(nclass)) {
        sum += Derived::EvalRow(label,
                                BeginPtr(preds) + i * nclass,
                                nclass) * wt;
        wsum += wt;
      } else {
        label_error = label;
      }
    }
    utils::Check(label_error >= 0 && label_error < static_cast<int>(nclass),
                 "MultiClassEvaluation: label must be in [0, num_class)," \
                 " num_class=%d but found %d in label",
                 static_cast<int>(nclass), label_error);
    float dat[2]; dat[0] = sum, dat[1] = wsum;
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    return Derived::GetFinal(dat[0], dat[1]);
  }
  /*!
   * \brief to be implemented by subclass,
   *   get evaluation result from one row
   * \param label label of current instance
   * \param pred prediction value of current instance
   * \param nclass number of class in the prediction
   */
  inline static float EvalRow(int label,
                              const float *pred,
                              size_t nclass);
  /*!
   * \brief to be overide by subclas, final trasnformation
   * \param esum the sum statistics returned by EvalRow
   * \param wsum sum of weight
   */
  inline static float GetFinal(float esum, float wsum) {
    return esum / wsum;
  }
  // used to store error message
  const char *error_msg_;
};
/*! \brief match error */
struct EvalMatchError : public EvalMClassBase<EvalMatchError> {
  virtual const char *Name(void) const {
    return "merror";
  }
  inline static float EvalRow(int label,
                              const float *pred,
                              size_t nclass) {
    return FindMaxIndex(pred, nclass) != static_cast<int>(label);
  }
};
/*! \brief match error */
struct EvalMultiLogLoss : public EvalMClassBase<EvalMultiLogLoss> {
  virtual const char *Name(void) const {
    return "mlogloss";
  }
  inline static float EvalRow(int label,
                              const float *pred,
                              size_t nclass) {
    const float eps = 1e-16f;
    size_t k = static_cast<size_t>(label);
    if (pred[k] > eps) {
      return -std::log(pred[k]);
    } else {
      return -std::log(eps);
    }
  }
};

/*! \brief ctest */
struct EvalCTest: public IEvaluator {
  EvalCTest(IEvaluator *base, const char *name)
      : base_(base), name_(name) {}
  virtual ~EvalCTest(void) {
    delete base_;
  }
  virtual const char *Name(void) const {
    return name_.c_str();
  }
  virtual float Eval(const std::vector<float> &preds,
                     const MetaInfo &info,
                     bool distributed) const {
    utils::Check(!distributed, "metric %s do not support distributed evaluation", name_.c_str());
    utils::Check(preds.size() % info.labels.size() == 0,
                 "label and prediction size not match");
    size_t ngroup = preds.size() / info.labels.size() - 1;
    const unsigned ndata = static_cast<unsigned>(info.labels.size());
    utils::Check(ngroup > 1, "pred size does not meet requirement");
    utils::Check(ndata == info.info.fold_index.size(), "need fold index");
    double wsum = 0.0;
    for (size_t k = 0; k < ngroup; ++k) {
      std::vector<float> tpred;
      MetaInfo tinfo;
      for (unsigned i = 0; i < ndata; ++i) {
        if (info.info.fold_index[i] == k) {
          tpred.push_back(preds[i + (k + 1) * ndata]);
          tinfo.labels.push_back(info.labels[i]);
          tinfo.weights.push_back(info.GetWeight(i));
        }
      }
      wsum += base_->Eval(tpred, tinfo);
    }
    return static_cast<float>(wsum / ngroup);
  }

 private:
  IEvaluator *base_;
  std::string name_;
};

/*! \brief AMS: also records best threshold */
struct EvalAMS : public IEvaluator {
 public:
  explicit EvalAMS(const char *name) {
    name_ = name;
    // note: ams@0 will automatically select which ratio to go
    utils::Check(std::sscanf(name, "ams@%f", &ratio_) == 1, "invalid ams format");
  }
  virtual float Eval(const std::vector<float> &preds,
                     const MetaInfo &info,
                     bool distributed) const {
    utils::Check(!distributed, "metric AMS do not support distributed evaluation");
    using namespace std;
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(info.labels.size());

    utils::Check(info.weights.size() == ndata, "we need weight to evaluate ams");
    std::vector< std::pair<float, unsigned> > rec(ndata);

    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      rec[i] = std::make_pair(preds[i], i);
    }
    std::sort(rec.begin(), rec.end(), CmpFirst);
    unsigned ntop = static_cast<unsigned>(ratio_ * ndata);
    if (ntop == 0) ntop = ndata;
    const double br = 10.0;
    unsigned thresindex = 0;
    double s_tp = 0.0, b_fp = 0.0, tams = 0.0;
    for (unsigned i = 0; i < static_cast<unsigned>(ndata-1) && i < ntop; ++i) {
      const unsigned ridx = rec[i].second;
      const float wt = info.weights[ridx];
      if (info.labels[ridx] > 0.5f) {
        s_tp += wt;
      } else {
        b_fp += wt;
      }
      if (rec[i].first != rec[i+1].first) {
        double ams = sqrt(2*((s_tp+b_fp+br) * log(1.0 + s_tp/(b_fp+br)) - s_tp));
        if (tams < ams) {
          thresindex = i;
          tams = ams;
        }
      }
    }
    if (ntop == ndata) {
      utils::Printf("\tams-ratio=%g", static_cast<float>(thresindex) / ndata);
      return static_cast<float>(tams);
    } else {
      return static_cast<float>(sqrt(2*((s_tp+b_fp+br) * log(1.0 + s_tp/(b_fp+br)) - s_tp)));
    }
  }
  virtual const char *Name(void) const {
    return name_.c_str();
  }

 private:
  std::string name_;
  float ratio_;
};

/*! \brief precision with cut off at top percentile */
struct EvalPrecisionRatio : public IEvaluator{
 public:
  explicit EvalPrecisionRatio(const char *name) : name_(name) {
    using namespace std;
    if (sscanf(name, "apratio@%f", &ratio_) == 1) {
      use_ap = 1;
    } else {
      utils::Assert(sscanf(name, "pratio@%f", &ratio_) == 1, "BUG");
      use_ap = 0;
    }
  }
  virtual float Eval(const std::vector<float> &preds,
                     const MetaInfo &info,
                     bool distributed) const {
    utils::Check(!distributed, "metric %s do not support distributed evaluation", Name());
    utils::Check(info.labels.size() != 0, "label set cannot be empty");
    utils::Assert(preds.size() % info.labels.size() == 0,
                  "label size predict size not match");
    std::vector< std::pair<float, unsigned> > rec;
    for (size_t j = 0; j < info.labels.size(); ++j) {
      rec.push_back(std::make_pair(preds[j], static_cast<unsigned>(j)));
    }
    std::sort(rec.begin(), rec.end(), CmpFirst);
    double pratio = CalcPRatio(rec, info);
    return static_cast<float>(pratio);
  }
  virtual const char *Name(void) const {
    return name_.c_str();
  }

 protected:
  inline double CalcPRatio(const std::vector< std::pair<float, unsigned> >& rec,
                           const MetaInfo &info) const {
    size_t cutoff = static_cast<size_t>(ratio_ * rec.size());
    double wt_hit = 0.0, wsum = 0.0, wt_sum = 0.0;
    for (size_t j = 0; j < cutoff; ++j) {
      const float wt = info.GetWeight(j);
      wt_hit += info.labels[rec[j].second] * wt;
      wt_sum += wt;
      wsum += wt_hit / wt_sum;
    }
    if (use_ap != 0) {
      return wsum / cutoff;
    } else {
      return wt_hit / wt_sum;
    }
  }
  int use_ap;
  float ratio_;
  std::string name_;
};

/*! \brief Area under curve, for both classification and rank */
struct EvalAuc : public IEvaluator {
  virtual float Eval(const std::vector<float> &preds,
                     const MetaInfo &info,
                     bool distributed) const {
    utils::Check(info.labels.size() != 0, "label set cannot be empty");
    utils::Check(preds.size() % info.labels.size() == 0,
                 "label size predict size not match");
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels.size());

    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    utils::Check(gptr.back() == info.labels.size(),
                 "EvalAuc: group structure must match number of prediction");
    const bst_omp_uint ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statictis
    double sum_auc = 0.0f;
    #pragma omp parallel reduction(+:sum_auc)
    {
      // each thread takes a local rec
      std::vector< std::pair<float, unsigned> > rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        rec.clear();
        for (unsigned j = gptr[k]; j < gptr[k + 1]; ++j) {
          rec.push_back(std::make_pair(preds[j], j));
        }
        std::sort(rec.begin(), rec.end(), CmpFirst);
        // calculate AUC
        double sum_pospair = 0.0;
        double sum_npos = 0.0, sum_nneg = 0.0, buf_pos = 0.0, buf_neg = 0.0;
        for (size_t j = 0; j < rec.size(); ++j) {
          const float wt = info.GetWeight(rec[j].second);
          const float ctr = info.labels[rec[j].second];
          // keep bucketing predictions in same bucket
          if (j != 0 && rec[j].first != rec[j - 1].first) {
            sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
            sum_npos += buf_pos; sum_nneg += buf_neg;
            buf_neg = buf_pos = 0.0f;
          }
          buf_pos += ctr * wt; buf_neg += (1.0f - ctr) * wt;
        }
        sum_pospair += buf_neg * (sum_npos + buf_pos *0.5);
        sum_npos += buf_pos; sum_nneg += buf_neg;
        // check weird conditions
        utils::Check(sum_npos > 0.0 && sum_nneg > 0.0,
                     "AUC: the dataset only contains pos or neg samples");
        // this is the AUC
        sum_auc += sum_pospair / (sum_npos*sum_nneg);
      }
    }
    if (distributed) {
      float dat[2];
      dat[0] = static_cast<float>(sum_auc);
      dat[1] = static_cast<float>(ngroup);
      // approximately estimate auc using mean
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
      return dat[0] / dat[1];
    } else {
      return static_cast<float>(sum_auc) / ngroup;
    }
  }
  virtual const char *Name(void) const {
    return "auc";
  }
};

/*! \brief Evaluate rank list */
struct EvalRankList : public IEvaluator {
 public:
  virtual float Eval(const std::vector<float> &preds,
                     const MetaInfo &info,
                     bool distributed) const {
    utils::Check(preds.size() == info.labels.size(),
                  "label size predict size not match");
    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0); tgptr[1] = static_cast<unsigned>(preds.size());
    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    utils::Assert(gptr.size() != 0, "must specify group when constructing rank file");
    utils::Assert(gptr.back() == preds.size(),
                   "EvalRanklist: group structure must match number of prediction");
    const bst_omp_uint ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    double sum_metric = 0.0f;
    #pragma omp parallel reduction(+:sum_metric)
    {
      // each thread takes a local rec
      std::vector< std::pair<float, unsigned> > rec;
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
      float dat[2];
      dat[0] = static_cast<float>(sum_metric);
      dat[1] = static_cast<float>(ngroup);
      // approximately estimate auc using mean
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
      return dat[0] / dat[1];
    } else {
      return static_cast<float>(sum_metric) / ngroup;
    }
  }
  virtual const char *Name(void) const {
    return name_.c_str();
  }

 protected:
  explicit EvalRankList(const char *name) {
    using namespace std;
    name_ = name;
    minus_ = false;
    if (sscanf(name, "%*[^@]@%u[-]?", &topn_) != 1) {
      topn_ = UINT_MAX;
    }
    if (name[strlen(name) - 1] == '-') {
      minus_ = true;
    }
  }
  /*! \return evaluation metric, given the pair_sort record, (pred,label) */
  virtual float EvalMetric(std::vector< std::pair<float, unsigned> > &pair_sort) const = 0; // NOLINT(*)

 protected:
  unsigned topn_;
  std::string name_;
  bool minus_;
};

/*! \brief Precison at N, for both classification and rank */
struct EvalPrecision : public EvalRankList{
 public:
  explicit EvalPrecision(const char *name) : EvalRankList(name) {}

 protected:
  virtual float EvalMetric(std::vector< std::pair<float, unsigned> > &rec) const {
    // calculate Preicsion
    std::sort(rec.begin(), rec.end(), CmpFirst);
    unsigned nhit = 0;
    for (size_t j = 0; j < rec.size() && j < this->topn_; ++j) {
      nhit += (rec[j].second != 0);
    }
    return static_cast<float>(nhit) / topn_;
  }
};

/*! \brief NDCG */
struct EvalNDCG : public EvalRankList{
 public:
  explicit EvalNDCG(const char *name) : EvalRankList(name) {}

 protected:
  inline float CalcDCG(const std::vector< std::pair<float, unsigned> > &rec) const {
    double sumdcg = 0.0;
    for (size_t i = 0; i < rec.size() && i < this->topn_; ++i) {
      const unsigned rel = rec[i].second;
      if (rel != 0) {
        sumdcg += ((1 << rel) - 1) / std::log(i + 2.0);
      }
    }
    return static_cast<float>(sumdcg);
  }
  virtual float EvalMetric(std::vector< std::pair<float, unsigned> > &rec) const { // NOLINT(*)
    std::stable_sort(rec.begin(), rec.end(), CmpFirst);
    float dcg = this->CalcDCG(rec);
    std::stable_sort(rec.begin(), rec.end(), CmpSecond);
    float idcg = this->CalcDCG(rec);
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

/*! \brief Precison at N, for both classification and rank */
struct EvalMAP : public EvalRankList {
 public:
  explicit EvalMAP(const char *name) : EvalRankList(name) {}

 protected:
  virtual float EvalMetric(std::vector< std::pair<float, unsigned> > &rec) const {
    std::sort(rec.begin(), rec.end(), CmpFirst);
    unsigned nhits = 0;
    double sumap = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
      if (rec[i].second != 0) {
        nhits += 1;
        if (i < this->topn_) {
          sumap += static_cast<float>(nhits) / (i+1);
        }
      }
    }
    if (nhits != 0) {
      sumap /= nhits;
      return static_cast<float>(sumap);
    } else {
      if (minus_) {
        return 0.0f;
      } else {
        return 1.0f;
      }
    }
  }
};

}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_EVALUATION_INL_HPP_
