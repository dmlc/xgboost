/*!
 * Copyright 2014 by Contributors
 * \file objective-inl.hpp
 * \brief objective function implementations
 * \author Tianqi Chen, Kailong Chen
 */
#ifndef XGBOOST_LEARNER_OBJECTIVE_INL_HPP_
#define XGBOOST_LEARNER_OBJECTIVE_INL_HPP_

#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
#include <functional>
#include "../data.h"
#include "./objective.h"
#include "./helper_utils.h"
#include "../utils/random.h"
#include "../utils/omp.h"

namespace xgboost {
namespace learner {
/*! \brief defines functions to calculate some commonly used functions */
struct LossType {
  /*! \brief indicate which type we are using */
  int loss_type;
  // list of constants
  static const int kLinearSquare = 0;
  static const int kLogisticNeglik = 1;
  static const int kLogisticClassify = 2;
  static const int kLogisticRaw = 3;
  /*!
   * \brief transform the linear sum to prediction
   * \param x linear sum of boosting ensemble
   * \return transformed prediction
   */
  inline float PredTransform(float x) const {
    switch (loss_type) {
      case kLogisticRaw:
      case kLinearSquare: return x;
      case kLogisticClassify:
      case kLogisticNeglik: return 1.0f / (1.0f + std::exp(-x));
      default: utils::Error("unknown loss_type"); return 0.0f;
    }
  }
  /*!
   * \brief check if label range is valid
   */
  inline bool CheckLabel(float x) const {
    if (loss_type != kLinearSquare) {
      return x >= 0.0f && x <= 1.0f;
    }
    return true;
  }
  /*!
   * \brief error message displayed when check label fail
   */
  inline const char * CheckLabelErrorMsg(void) const {
    if (loss_type != kLinearSquare) {
      return "label must be in [0,1] for logistic regression";
    } else {
      return "";
    }
  }
  /*!
   * \brief calculate first order gradient of loss, given transformed prediction
   * \param predt transformed prediction
   * \param label true label
   * \return first order gradient
   */
  inline float FirstOrderGradient(float predt, float label) const {
    switch (loss_type) {
      case kLinearSquare: return predt - label;
      case kLogisticRaw: predt = 1.0f / (1.0f + std::exp(-predt));
      case kLogisticClassify:
      case kLogisticNeglik: return predt - label;
      default: utils::Error("unknown loss_type"); return 0.0f;
    }
  }
  /*!
   * \brief calculate second order gradient of loss, given transformed prediction
   * \param predt transformed prediction
   * \param label true label
   * \return second order gradient
   */
  inline float SecondOrderGradient(float predt, float label) const {
    // cap second order gradient to postive value
    const float eps = 1e-16f;
    switch (loss_type) {
      case kLinearSquare: return 1.0f;
      case kLogisticRaw: predt = 1.0f / (1.0f + std::exp(-predt));
      case kLogisticClassify:
      case kLogisticNeglik: return std::max(predt * (1.0f - predt), eps);
      default: utils::Error("unknown loss_type"); return 0.0f;
    }
  }
  /*!
   * \brief transform probability value back to margin
   */
  inline float ProbToMargin(float base_score) const {
    if (loss_type == kLogisticRaw ||
        loss_type == kLogisticClassify ||
        loss_type == kLogisticNeglik ) {
      utils::Check(base_score > 0.0f && base_score < 1.0f,
                   "base_score must be in (0,1) for logistic loss");
      base_score = -std::log(1.0f / base_score - 1.0f);
    }
    return base_score;
  }
  /*! \brief get default evaluation metric for the objective */
  inline const char *DefaultEvalMetric(void) const {
    if (loss_type == kLogisticClassify) return "error";
    if (loss_type == kLogisticRaw) return "auc";
    return "rmse";
  }
};

/*! \brief objective function that only need to */
class RegLossObj : public IObjFunction {
 public:
  explicit RegLossObj(int loss_type) {
    loss.loss_type = loss_type;
    scale_pos_weight = 1.0f;
  }
  virtual ~RegLossObj(void) {}
  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
    if (!strcmp("scale_pos_weight", name)) {
      scale_pos_weight = static_cast<float>(atof(val));
    }
  }
  virtual void GetGradient(const std::vector<float> &preds,
                           const MetaInfo &info,
                           int iter,
                           std::vector<bst_gpair> *out_gpair) {
    utils::Check(info.labels.size() != 0, "label set cannot be empty");
    utils::Check(preds.size() % info.labels.size() == 0,
                 "labels are not correctly provided");
    std::vector<bst_gpair> &gpair = *out_gpair;
    gpair.resize(preds.size());
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const unsigned nstep = static_cast<unsigned>(info.labels.size());
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(preds.size());
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ndata; ++i) {
      const unsigned j = i % nstep;
      float p = loss.PredTransform(preds[i]);
      float w = info.GetWeight(j);
      if (info.labels[j] == 1.0f) w *= scale_pos_weight;
      if (!loss.CheckLabel(info.labels[j])) label_correct = false;
      gpair[i] = bst_gpair(loss.FirstOrderGradient(p, info.labels[j]) * w,
                           loss.SecondOrderGradient(p, info.labels[j]) * w);
    }
    utils::Check(label_correct, loss.CheckLabelErrorMsg());
  }
  virtual const char* DefaultEvalMetric(void) const {
    return loss.DefaultEvalMetric();
  }
  virtual void PredTransform(std::vector<float> *io_preds) {
    std::vector<float> &preds = *io_preds;
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(preds.size());
    #pragma omp parallel for schedule(static)
    for (bst_omp_uint j = 0; j < ndata; ++j) {
      preds[j] = loss.PredTransform(preds[j]);
    }
  }
  virtual float ProbToMargin(float base_score) const {
    return loss.ProbToMargin(base_score);
  }

 protected:
  float scale_pos_weight;
  LossType loss;
};

// poisson regression for count
class PoissonRegression : public IObjFunction {
 public:
  PoissonRegression(void) {
    max_delta_step = 0.0f;
  }
  virtual ~PoissonRegression(void) {}

  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
    if (!strcmp("max_delta_step", name)) {
      max_delta_step = static_cast<float>(atof(val));
    }
  }
  virtual void GetGradient(const std::vector<float> &preds,
                           const MetaInfo &info,
                           int iter,
                           std::vector<bst_gpair> *out_gpair) {
    utils::Check(max_delta_step != 0.0f,
                 "PoissonRegression: need to set max_delta_step");
    utils::Check(info.labels.size() != 0, "label set cannot be empty");
    utils::Check(preds.size() == info.labels.size(),
                 "labels are not correctly provided");
    std::vector<bst_gpair> &gpair = *out_gpair;
    gpair.resize(preds.size());
    // check if label in range
    bool label_correct = true;
    // start calculating gradient
    const long ndata = static_cast<bst_omp_uint>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < ndata; ++i) { // NOLINT(*)
      float p = preds[i];
      float w = info.GetWeight(i);
      float y = info.labels[i];
      if (y >= 0.0f) {
        gpair[i] = bst_gpair((std::exp(p) - y) * w,
                             std::exp(p + max_delta_step) * w);
      } else {
        label_correct = false;
      }
    }
    utils::Check(label_correct,
                 "PoissonRegression: label must be nonnegative");
  }
  virtual void PredTransform(std::vector<float> *io_preds) {
    std::vector<float> &preds = *io_preds;
    const long ndata = static_cast<long>(preds.size()); // NOLINT(*)
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < ndata; ++j) {  // NOLINT(*)
      preds[j] = std::exp(preds[j]);
    }
  }
  virtual void EvalTransform(std::vector<float> *io_preds) {
    PredTransform(io_preds);
  }
  virtual float ProbToMargin(float base_score) const {
    return std::log(base_score);
  }
  virtual const char* DefaultEvalMetric(void) const {
    return "poisson-nloglik";
  }

 private:
  float max_delta_step;
};

// softmax multi-class classification
class SoftmaxMultiClassObj : public IObjFunction {
 public:
  explicit SoftmaxMultiClassObj(int output_prob)
      : output_prob(output_prob) {
    nclass = 0;
  }
  virtual ~SoftmaxMultiClassObj(void) {}
  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
    if (!strcmp( "num_class", name )) nclass = atoi(val);
  }
  virtual void GetGradient(const std::vector<float> &preds,
                           const MetaInfo &info,
                           int iter,
                           std::vector<bst_gpair> *out_gpair) {
    utils::Check(nclass != 0, "must set num_class to use softmax");
    utils::Check(info.labels.size() != 0, "label set cannot be empty");
    utils::Check(preds.size() % (static_cast<size_t>(nclass) * info.labels.size()) == 0,
                 "SoftmaxMultiClassObj: label size and pred size does not match");
    std::vector<bst_gpair> &gpair = *out_gpair;
    gpair.resize(preds.size());
    const unsigned nstep = static_cast<unsigned>(info.labels.size() * nclass);
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(preds.size() / nclass);
    int label_error = 0;
    #pragma omp parallel
    {
      std::vector<float> rec(nclass);
      #pragma omp for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[i * nclass + k];
        }
        Softmax(&rec);
        const unsigned j = i % nstep;
        int label = static_cast<int>(info.labels[j]);
        if (label < 0 || label >= nclass)  {
          label_error = label; label = 0;
        }
        const float wt = info.GetWeight(j);
        for (int k = 0; k < nclass; ++k) {
          float p = rec[k];
          const float h = 2.0f * p * (1.0f - p) * wt;
          if (label == k) {
            gpair[i * nclass + k] = bst_gpair((p - 1.0f) * wt, h);
          } else {
            gpair[i * nclass + k] = bst_gpair(p* wt, h);
          }
        }
      }
    }
    utils::Check(label_error >= 0 && label_error < nclass,
                 "SoftmaxMultiClassObj: label must be in [0, num_class),"\
                 " num_class=%d but found %d in label", nclass, label_error);
  }
  virtual void PredTransform(std::vector<float> *io_preds) {
    this->Transform(io_preds, output_prob);
  }
  virtual void EvalTransform(std::vector<float> *io_preds) {
    this->Transform(io_preds, 1);
  }
  virtual const char* DefaultEvalMetric(void) const {
    return "merror";
  }

 private:
  inline void Transform(std::vector<float> *io_preds, int prob) {
    utils::Check(nclass != 0, "must set num_class to use softmax");
    std::vector<float> &preds = *io_preds;
    std::vector<float> tmp;
    const bst_omp_uint ndata = static_cast<bst_omp_uint>(preds.size()/nclass);
    if (prob == 0) tmp.resize(ndata);
    #pragma omp parallel
    {
      std::vector<float> rec(nclass);
      #pragma omp for schedule(static)
      for (bst_omp_uint j = 0; j < ndata; ++j) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[j * nclass + k];
        }
        if (prob == 0) {
          tmp[j] = static_cast<float>(FindMaxIndex(rec));
        } else {
          Softmax(&rec);
          for (int k = 0; k < nclass; ++k) {
            preds[j * nclass + k] = rec[k];
          }
        }
      }
    }
    if (prob == 0) preds = tmp;
  }
  // data field
  int nclass;
  int output_prob;
};

/*! \brief objective for lambda rank */
class LambdaRankObj : public IObjFunction {
 public:
  LambdaRankObj(void) {
    loss.loss_type = LossType::kLogisticRaw;
    fix_list_weight = 0.0f;
    num_pairsample = 1;
  }
  virtual ~LambdaRankObj(void) {}
  virtual void SetParam(const char *name, const char *val) {
    using namespace std;
    if (!strcmp( "loss_type", name )) loss.loss_type = atoi(val);
    if (!strcmp( "fix_list_weight", name)) fix_list_weight = static_cast<float>(atof(val));
    if (!strcmp( "num_pairsample", name)) num_pairsample = atoi(val);
  }
  virtual void GetGradient(const std::vector<float> &preds,
                           const MetaInfo &info,
                           int iter,
                           std::vector<bst_gpair> *out_gpair) {
    utils::Check(preds.size() == info.labels.size(), "label size predict size not match");
    std::vector<bst_gpair> &gpair = *out_gpair;
    gpair.resize(preds.size());
    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0); tgptr[1] = static_cast<unsigned>(info.labels.size());
    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    utils::Check(gptr.size() != 0 && gptr.back() == info.labels.size(),
                 "group structure not consistent with #rows");
    const bst_omp_uint ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    #pragma omp parallel
    {
      // parall construct, declare random number generator here, so that each
      // thread use its own random number generator, seed by thread id and current iteration
      random::Random rnd; rnd.Seed(iter* 1111 + omp_get_thread_num());
      std::vector<LambdaPair> pairs;
      std::vector<ListEntry>  lst;
      std::vector< std::pair<float, unsigned> > rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        lst.clear(); pairs.clear();
        for (unsigned j = gptr[k]; j < gptr[k+1]; ++j) {
          lst.push_back(ListEntry(preds[j], info.labels[j], j));
          gpair[j] = bst_gpair(0.0f, 0.0f);
        }
        std::sort(lst.begin(), lst.end(), ListEntry::CmpPred);
        rec.resize(lst.size());
        for (unsigned i = 0; i < lst.size(); ++i) {
          rec[i] = std::make_pair(lst[i].label, i);
        }
        std::sort(rec.begin(), rec.end(), CmpFirst);
        // enumerate buckets with same label, for each item in the lst, grab another sample randomly
        for (unsigned i = 0; i < rec.size(); ) {
          unsigned j = i + 1;
          while (j < rec.size() && rec[j].first == rec[i].first) ++j;
          // bucket in [i,j), get a sample outside bucket
          unsigned nleft = i, nright = static_cast<unsigned>(rec.size() - j);
          if (nleft + nright != 0) {
            int nsample = num_pairsample;
            while (nsample --) {
              for (unsigned pid = i; pid < j; ++pid) {
                unsigned ridx = static_cast<unsigned>(rnd.RandDouble() * (nleft+nright));
                if (ridx < nleft) {
                  pairs.push_back(LambdaPair(rec[ridx].second, rec[pid].second));
                } else {
                  pairs.push_back(LambdaPair(rec[pid].second, rec[ridx+j-i].second));
                }
              }
            }
          }
          i = j;
        }
        // get lambda weight for the pairs
        this->GetLambdaWeight(lst, &pairs);
        // rescale each gradient and hessian so that the lst have constant weighted
        float scale = 1.0f / num_pairsample;
        if (fix_list_weight != 0.0f) {
          scale *= fix_list_weight / (gptr[k+1] - gptr[k]);
        }
        for (size_t i = 0; i < pairs.size(); ++i) {
          const ListEntry &pos = lst[pairs[i].pos_index];
          const ListEntry &neg = lst[pairs[i].neg_index];
          const float w = pairs[i].weight * scale;
          float p = loss.PredTransform(pos.pred - neg.pred);
          float g = loss.FirstOrderGradient(p, 1.0f);
          float h = loss.SecondOrderGradient(p, 1.0f);
          // accumulate gradient and hessian in both pid, and nid
          gpair[pos.rindex].grad += g * w;
          gpair[pos.rindex].hess += 2.0f * w * h;
          gpair[neg.rindex].grad -= g * w;
          gpair[neg.rindex].hess += 2.0f * w * h;
        }
      }
    }
  }
  virtual const char* DefaultEvalMetric(void) const {
    return "map";
  }

 protected:
  /*! \brief helper information in a list */
  struct ListEntry {
    /*! \brief the predict score we in the data */
    float pred;
    /*! \brief the actual label of the entry */
    float label;
    /*! \brief row index in the data matrix */
    unsigned rindex;
    // constructor
    ListEntry(float pred, float label, unsigned rindex)
        : pred(pred), label(label), rindex(rindex) {}
    // comparator by prediction
    inline static bool CmpPred(const ListEntry &a, const ListEntry &b) {
      return a.pred > b.pred;
    }
    // comparator by label
    inline static bool CmpLabel(const ListEntry &a, const ListEntry &b) {
      return a.label > b.label;
    }
  };
  /*! \brief a pair in the lambda rank */
  struct LambdaPair {
    /*! \brief positive index: this is a position in the list */
    unsigned pos_index;
    /*! \brief negative index: this is a position in the list */
    unsigned neg_index;
    /*! \brief weight to be filled in */
    float weight;
    // constructor
    LambdaPair(unsigned pos_index, unsigned neg_index)
        : pos_index(pos_index), neg_index(neg_index), weight(1.0f) {}
  };
  /*!
   * \brief get lambda weight for existing pairs
   * \param list a list that is sorted by pred score
   * \param io_pairs record of pairs, containing the pairs to fill in weights
   */
  virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                               std::vector<LambdaPair> *io_pairs) = 0;

 private:
  // loss function
  LossType loss;
  // number of samples peformed for each instance
  int num_pairsample;
  // fix weight of each elements in list
  float fix_list_weight;
};

class PairwiseRankObj: public LambdaRankObj{
 public:
  virtual ~PairwiseRankObj(void) {}

 protected:
  virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                               std::vector<LambdaPair> *io_pairs) {}
};

// beta version: NDCG lambda rank
class LambdaRankObjNDCG : public LambdaRankObj {
 public:
  virtual ~LambdaRankObjNDCG(void) {}

 protected:
  virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                               std::vector<LambdaPair> *io_pairs) {
    std::vector<LambdaPair> &pairs = *io_pairs;
    float IDCG;
    {
      std::vector<float> labels(sorted_list.size());
      for (size_t i = 0; i < sorted_list.size(); ++i) {
        labels[i] = sorted_list[i].label;
      }
      std::sort(labels.begin(), labels.end(), std::greater<float>());
      IDCG = CalcDCG(labels);
    }
    if (IDCG == 0.0) {
      for (size_t i = 0; i < pairs.size(); ++i) {
        pairs[i].weight = 0.0f;
      }
    } else {
      IDCG = 1.0f / IDCG;
      for (size_t i = 0; i < pairs.size(); ++i) {
        unsigned pos_idx = pairs[i].pos_index;
        unsigned neg_idx = pairs[i].neg_index;
        float pos_loginv = 1.0f / std::log(pos_idx + 2.0f);
        float neg_loginv = 1.0f / std::log(neg_idx + 2.0f);
        int pos_label = static_cast<int>(sorted_list[pos_idx].label);
        int neg_label = static_cast<int>(sorted_list[neg_idx].label);
        float original =
            ((1 << pos_label) - 1) * pos_loginv + ((1 << neg_label) - 1) * neg_loginv;
        float changed  =
            ((1 << neg_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * neg_loginv;
        float delta = (original - changed) * IDCG;
        if (delta < 0.0f) delta = - delta;
        pairs[i].weight = delta;
      }
    }
  }
  inline static float CalcDCG(const std::vector<float> &labels) {
    double sumdcg = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
      const unsigned rel = static_cast<unsigned>(labels[i]);
      if (rel != 0) {
        sumdcg += ((1 << rel) - 1) / std::log(static_cast<float>(i + 2));
      }
    }
    return static_cast<float>(sumdcg);
  }
};

class LambdaRankObjMAP : public LambdaRankObj {
 public:
  virtual ~LambdaRankObjMAP(void) {}

 protected:
  struct MAPStats {
    /*! \brief the accumulated precision */
    float ap_acc;
    /*!
     * \brief the accumulated precision,
     *   assuming a positive instance is missing
     */
    float ap_acc_miss;
    /*!
     * \brief the accumulated precision,
     * assuming that one more positive instance is inserted ahead
     */
    float ap_acc_add;
    /* \brief the accumulated positive instance count */
    float hits;
    MAPStats(void) {}
    MAPStats(float ap_acc, float ap_acc_miss, float ap_acc_add, float hits)
        : ap_acc(ap_acc), ap_acc_miss(ap_acc_miss), ap_acc_add(ap_acc_add), hits(hits) {}
  };
  /*!
   * \brief Obtain the delta MAP if trying to switch the positions of instances in index1 or index2
   *        in sorted triples
   * \param sorted_list the list containing entry information
   * \param index1,index2 the instances switched
   * \param map_stats a vector containing the accumulated precisions for each position in a list
   */
  inline float GetLambdaMAP(const std::vector<ListEntry> &sorted_list,
                            int index1, int index2,
                            std::vector<MAPStats> *p_map_stats) {
    std::vector<MAPStats> &map_stats = *p_map_stats;
    if (index1 == index2 || map_stats[map_stats.size() - 1].hits == 0) {
      return 0.0f;
    }
    if (index1 > index2) std::swap(index1, index2);
    float original = map_stats[index2].ap_acc;
    if (index1 != 0) original -= map_stats[index1 - 1].ap_acc;
    float changed = 0;
    float label1 = sorted_list[index1].label > 0.0f ? 1.0f : 0.0f;
    float label2 = sorted_list[index2].label > 0.0f ? 1.0f : 0.0f;
    if (label1 == label2) {
      return 0.0;
    } else if (label1 < label2) {
      changed += map_stats[index2 - 1].ap_acc_add - map_stats[index1].ap_acc_add;
      changed += (map_stats[index1].hits + 1.0f) / (index1 + 1);
    } else {
      changed += map_stats[index2 - 1].ap_acc_miss - map_stats[index1].ap_acc_miss;
      changed += map_stats[index2].hits / (index2 + 1);
    }
    float ans = (changed - original) / (map_stats[map_stats.size() - 1].hits);
    if (ans < 0) ans = -ans;
    return ans;
  }
  /*
   * \brief obtain preprocessing results for calculating delta MAP
   * \param sorted_list the list containing entry information
   * \param map_stats a vector containing the accumulated precisions for each position in a list
   */
  inline void GetMAPStats(const std::vector<ListEntry> &sorted_list,
                          std::vector<MAPStats> *p_map_acc) {
    std::vector<MAPStats> &map_acc = *p_map_acc;
    map_acc.resize(sorted_list.size());
    float hit = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (size_t i = 1; i <= sorted_list.size(); ++i) {
      if (sorted_list[i - 1].label > 0.0f) {
        hit++;
        acc1 += hit / i;
        acc2 += (hit - 1) / i;
        acc3 += (hit + 1) / i;
      }
      map_acc[i - 1] = MAPStats(acc1, acc2, acc3, hit);
    }
  }
  virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                               std::vector<LambdaPair> *io_pairs) {
    std::vector<LambdaPair> &pairs = *io_pairs;
    std::vector<MAPStats> map_stats;
    GetMAPStats(sorted_list, &map_stats);
    for (size_t i = 0; i < pairs.size(); ++i) {
      pairs[i].weight =
          GetLambdaMAP(sorted_list, pairs[i].pos_index,
                       pairs[i].neg_index, &map_stats);
    }
  }
};

}  // namespace learner
}  // namespace xgboost
#endif  // XGBOOST_LEARNER_OBJECTIVE_INL_HPP_
