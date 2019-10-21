/*!
 * Copyright 2015-2019 XGBoost contributors
 */
#include <dmlc/omp.h>
#include <dmlc/timer.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>

#include "xgboost/json.h"
#include "xgboost/parameter.h"

#include "../common/math.h"
#include "../common/random.h"

#if defined(__CUDACC__)
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#include <cub/util_allocator.cuh>

#include "../common/device_helpers.cuh"
#endif

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(rank_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

struct LambdaRankParam : public XGBoostParameter<LambdaRankParam> {
  int num_pairsample;
  float fix_list_weight;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LambdaRankParam) {
    DMLC_DECLARE_FIELD(num_pairsample).set_lower_bound(1).set_default(1)
        .describe("Number of pair generated for each instance.");
    DMLC_DECLARE_FIELD(fix_list_weight).set_lower_bound(0.0f).set_default(0.0f)
        .describe("Normalize the weight of each list by this value,"
                  " if equals 0, no effect will happen");
  }
};

/*! \brief helper information in a list */
struct ListEntry {
  /*! \brief the predict score we in the data */
  bst_float pred;
  /*! \brief the actual label of the entry */
  bst_float label;
  /*! \brief row index in the data matrix */
  unsigned rindex;
  // constructor
  ListEntry(bst_float pred, bst_float label, unsigned rindex)
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
  bst_float weight;
  // constructor
  LambdaPair(unsigned pos_index, unsigned neg_index)
    : pos_index(pos_index), neg_index(neg_index), weight(1.0f) {}
  // constructor
  LambdaPair(unsigned pos_index, unsigned neg_index, bst_float weight)
    : pos_index(pos_index), neg_index(neg_index), weight(weight) {}
};

struct PairwiseLambdaWeightComputer {
  /*!
   * \brief get lambda weight for existing pairs - for pairwise objective
   * \param list a list that is sorted by pred score
   * \param io_pairs record of pairs, containing the pairs to fill in weights
   */
  static void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                              std::vector<LambdaPair> *io_pairs) {}

  static char const* Name() {
    return "rank:pairwise";
  }

  // Stopgap method - will be removed when we support other type of ranking - ndcg, map etc.
  // on GPU later
  inline static bool SupportOnGPU() { return true; }
};

// beta version: NDCG lambda rank
struct NDCGLambdaWeightComputer {
  // Stopgap method - will be removed when we support other type of ranking - ndcg, map etc.
  // on GPU later
  inline static bool SupportOnGPU() { return false; }

  static void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                              std::vector<LambdaPair> *io_pairs) {
    std::vector<LambdaPair> &pairs = *io_pairs;
    float IDCG;  // NOLINT
    {
      std::vector<bst_float> labels(sorted_list.size());
      for (size_t i = 0; i < sorted_list.size(); ++i) {
        labels[i] = sorted_list[i].label;
      }
      std::sort(labels.begin(), labels.end(), std::greater<bst_float>());
      IDCG = CalcDCG(labels);
    }
    if (IDCG == 0.0) {
      for (auto & pair : pairs) {
        pair.weight = 0.0f;
      }
    } else {
      IDCG = 1.0f / IDCG;
      for (auto & pair : pairs) {
        unsigned pos_idx = pair.pos_index;
        unsigned neg_idx = pair.neg_index;
        float pos_loginv = 1.0f / std::log2(pos_idx + 2.0f);
        float neg_loginv = 1.0f / std::log2(neg_idx + 2.0f);
        auto pos_label = static_cast<int>(sorted_list[pos_idx].label);
        auto neg_label = static_cast<int>(sorted_list[neg_idx].label);
        bst_float original =
            ((1 << pos_label) - 1) * pos_loginv + ((1 << neg_label) - 1) * neg_loginv;
        float changed  =
            ((1 << neg_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * neg_loginv;
        bst_float delta = (original - changed) * IDCG;
        if (delta < 0.0f) delta = - delta;
        pair.weight *= delta;
      }
    }
  }

  static char const* Name() {
    return "rank:ndcg";
  }

 private:
  inline static bst_float CalcDCG(const std::vector<bst_float> &labels) {
    double sumdcg = 0.0;
    for (size_t i = 0; i < labels.size(); ++i) {
      const auto rel = static_cast<unsigned>(labels[i]);
      if (rel != 0) {
        sumdcg += ((1 << rel) - 1) / std::log2(static_cast<bst_float>(i + 2));
      }
    }
    return static_cast<bst_float>(sumdcg);
  }
};

struct MAPLambdaWeightComputer {
 private:
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
    MAPStats() = default;
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
  inline static bst_float GetLambdaMAP(const std::vector<ListEntry> &sorted_list,
                                       int index1, int index2,
                                       std::vector<MAPStats> *p_map_stats) {
    std::vector<MAPStats> &map_stats = *p_map_stats;
    if (index1 == index2 || map_stats[map_stats.size() - 1].hits == 0) {
      return 0.0f;
    }
    if (index1 > index2) std::swap(index1, index2);
    bst_float original = map_stats[index2].ap_acc;
    if (index1 != 0) original -= map_stats[index1 - 1].ap_acc;
    bst_float changed = 0;
    bst_float label1 = sorted_list[index1].label > 0.0f ? 1.0f : 0.0f;
    bst_float label2 = sorted_list[index2].label > 0.0f ? 1.0f : 0.0f;
    if (label1 == label2) {
      return 0.0;
    } else if (label1 < label2) {
      changed += map_stats[index2 - 1].ap_acc_add - map_stats[index1].ap_acc_add;
      changed += (map_stats[index1].hits + 1.0f) / (index1 + 1);
    } else {
      changed += map_stats[index2 - 1].ap_acc_miss - map_stats[index1].ap_acc_miss;
      changed += map_stats[index2].hits / (index2 + 1);
    }
    bst_float ans = (changed - original) / (map_stats[map_stats.size() - 1].hits);
    if (ans < 0) ans = -ans;
    return ans;
  }

  /*
   * \brief obtain preprocessing results for calculating delta MAP
   * \param sorted_list the list containing entry information
   * \param map_stats a vector containing the accumulated precisions for each position in a list
   */
  inline static void GetMAPStats(const std::vector<ListEntry> &sorted_list,
                                 std::vector<MAPStats> *p_map_acc) {
    std::vector<MAPStats> &map_acc = *p_map_acc;
    map_acc.resize(sorted_list.size());
    bst_float hit = 0, acc1 = 0, acc2 = 0, acc3 = 0;
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

 public:
  // Stopgap method - will be removed when we support other type of ranking - ndcg, map etc.
  // on GPU later
  inline static bool SupportOnGPU() { return false; }

  static char const* Name() {
    return "rank:map";
  }

  static void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                              std::vector<LambdaPair> *io_pairs) {
    std::vector<LambdaPair> &pairs = *io_pairs;
    std::vector<MAPStats> map_stats;
    GetMAPStats(sorted_list, &map_stats);
    for (auto & pair : pairs) {
      pair.weight *=
          GetLambdaMAP(sorted_list, pair.pos_index,
                       pair.neg_index, &map_stats);
    }
  }
};

#if defined(__CUDACC__)
// Helper functions

// Labels of size 'n' are sorted in a descending order
// If left is true,  find the number of elements > v; 0 if nothing is greater
// If left is false, find the number of elements < v; 0 if nothing is lesser
__device__ __forceinline__ int
CountNumLabelsImpl(bool left, const float * __restrict__ labels, int n, float v) {
  const float *labels_begin = labels;
  int num_remaining = n;
  const float *middle_item = nullptr;
  int middle;
  while (num_remaining > 0) {
    middle_item = labels_begin;
    middle = num_remaining / 2;
    middle_item += middle;
    if ((left && *middle_item > v) || (!left && !(v > *middle_item))) {
      labels_begin = ++middle_item;
      num_remaining -= middle + 1;
    } else {
      num_remaining = middle;
    }
  }

  return left ? labels_begin - labels : labels + n - labels_begin;
}

__device__ __forceinline__ int
CountNumLabelsToTheLeftOf(const float * __restrict__ labels, int n, float v) {
  return CountNumLabelsImpl(true, labels, n, v);
}

__device__ __forceinline__ int
CountNumLabelsToTheRightOf(const float * __restrict__ labels, int n, float v) {
  return CountNumLabelsImpl(false, labels, n, v);
}

class SortedLabelList {
 private:
  // Labels sorted within the group
  dh::caching_device_vector<float> dsorted_labels_;

  // Original position of the labels before they are sorted descendingly within its groups
  dh::caching_device_vector<uint32_t> doriginal_pos_;

  // Segments within the original list that delineates the different groups
  dh::caching_device_vector<uint32_t> group_segments_;

  // Need this on the device as it is used in the kernels
  dh::caching_device_vector<uint32_t> dgroups_;       // Group information on device

  int device_id_{-1};                                 // GPU device ID
  const LambdaRankParam &param_;                      // Objective configuration

  dh::XGBCachingDeviceAllocator<char> alloc_;         // Allocator to be used by sort for managing
                                                      // space overhead while sorting

 public:
  SortedLabelList(int dev_id,
                  const LambdaRankParam &param)
    : device_id_(dev_id),
      param_(param) {}

  void InitWithTrainingInfo(const std::vector<uint32_t> &groups) {
    int num_elems = groups.back();

    dsorted_labels_.resize(num_elems);

    doriginal_pos_.resize(num_elems);
    thrust::sequence(doriginal_pos_.begin(), doriginal_pos_.end());

    group_segments_.resize(num_elems);

    dgroups_ = groups;

    // Launch a kernel that populates the segment information for the different groups
    uint32_t *gsegs = group_segments_.data().get();
    const unsigned *dgroups = dgroups_.data().get();
    int ngroups = dgroups_.size();
    dh::LaunchN(device_id_, num_elems, nullptr, [=] __device__(unsigned idx){
      // Find the group first
      int group_idx = dh::UpperBound(dgroups, ngroups, idx);
      gsegs[idx] = group_idx - 1;
    });
  }

  // Sort the groups by labels. We would like to avoid using predicates to perform the sort,
  // as thrust resorts to using a merge sort as opposed to a much much faster radix sort
  // when comparators are used. Hence, the following algorithm is used. This is done so that
  // we can grab the appropriate prediction values from the original list later, after the
  // labels are sorted.
  //
  // Here is the internal representation:
  // dgroups_:          [ 0, 3, 5, 8, 10 ]
  // group_segments_:   0 0 0 | 1 1 | 2 2 2 | 3 3
  // doriginal_pos_:    0 1 2 | 3 4 | 5 6 7 | 8 9
  // dsorted_labels_:   1 0 1 | 2 1 | 1 3 3 | 4 4 (from original labels)
  //
  // Sort the labels first and make a note of the original positions in doriginal_pos_
  // based on the sort
  // dsorted_labels_:   4 4 3 3 2 1 1 1 1 0
  // doriginal_pos_:    8 9 6 7 3 0 2 4 5 1
  // NOTE: This consumes space, but is much faster than some of the other approaches - sorting
  //       in kernel, sorting using predicates etc.
  void Sort(const HostDeviceVector<bst_float> &dlabels) {
    dsorted_labels_.assign(dh::tcbegin(dlabels), dh::tcend(dlabels));
    thrust::stable_sort_by_key(thrust::cuda::par(alloc_),
                               dsorted_labels_.begin(), dsorted_labels_.end(),
                               doriginal_pos_.begin(), thrust::greater<float>());

    // Next, gather the segments based on the doriginal_pos_. This is to reflect the
    // holisitic label sort order on the segments
    // group_segments_c_:   3 3 2 2 1 0 0 1 2 0
    // doriginal_pos_:      8 9 6 7 3 0 2 4 5 1 (stays the same)
    thrust::device_vector<uint32_t> group_segments_c(group_segments_);
    thrust::gather(doriginal_pos_.begin(), doriginal_pos_.end(),
                   group_segments_.begin(), group_segments_c.begin());

    // Now, sort the group segments so that you may bring the labels within the group together,
    // in the process also noting the relative changes to the doriginal_pos_ while that happens
    // group_segments_c_:   0 0 0 1 1 2 2 2 3 3
    // doriginal_pos_:      0 2 1 3 4 6 7 5 8 9
    thrust::stable_sort_by_key(thrust::cuda::par(alloc_),
                               group_segments_c.begin(), group_segments_c.end(),
                               doriginal_pos_.begin(), thrust::less<int>());

    // Finally, gather the original labels based on doriginal_pos_ to sort the input and
    // to store them in dsorted_labels_
    // doriginal_pos_:      0 2 1 3 4 6 7 5 8 9  (stays the same)
    // dsorted_labels_:     1 1 0 2 1 3 3 1 4 4  (from unsorted dlabels - dlabels)
    thrust::gather(doriginal_pos_.begin(), doriginal_pos_.end(),
                   dh::tcbegin(dlabels), dsorted_labels_.begin());
  }

  ~SortedLabelList() {
    dh::safe_cuda(cudaSetDevice(device_id_));
  }

  // This kernel can only run *after* the kernel in sort is completed, as they
  // use the default stream
  void ComputeGradients(const bst_float *dpreds,
                        GradientPair *out_gpair,
                        const HostDeviceVector<bst_float> &weights,
                        float weight_normalization_factor) {
    // Group info on device
    const unsigned *dgroups = dgroups_.data().get();
    int ngroups = dgroups_.size();

    uint32_t total_items = group_segments_.size();
    int niter = param_.num_pairsample * total_items;

    float fix_list_weight = param_.fix_list_weight;

    const uint32_t *original_pos = doriginal_pos_.data().get();

    size_t num_weights = weights.Size();
    auto dweights = num_weights ? weights.ConstDevicePointer() : nullptr;

    const bst_float *sorted_labels = dsorted_labels_.data().get();

    // For each instance in the group, compute the gradient pair concurrently
    dh::LaunchN(device_id_, niter, nullptr, [=] __device__(size_t idx) {
      // First, determine the group 'idx' belongs to
      unsigned item_idx = idx % total_items;
      int group_idx = dh::UpperBound(dgroups, ngroups, item_idx);
      // Span of this group within the larger labels/predictions sorted tuple
      int group_begin = dgroups[group_idx - 1];
      int group_end = dgroups[group_idx];
      int total_group_items = group_end - group_begin;

      // Are the labels diverse enough? If they are all the same, then there is nothing to pick
      // from another group - bail sooner
      if (sorted_labels[group_begin] == sorted_labels[group_end - 1]) return;

      // Find the number of labels less than and greater than the current label
      // at the sorted index position item_idx
      int nleft  = CountNumLabelsToTheLeftOf(
        sorted_labels + group_begin, total_group_items, sorted_labels[item_idx]);
      int nright = CountNumLabelsToTheRightOf(
        sorted_labels + group_begin, total_group_items, sorted_labels[item_idx]);

      // Create a minstd_rand object to act as our source of randomness
      thrust::minstd_rand rng;
      rng.discard(idx);
      // Create a uniform_int_distribution to produce a sample from outside of the
      // present label group
      thrust::uniform_int_distribution<int> dist(0, nleft + nright - 1);

      int sample = dist(rng);
      int pos_idx = -1;  // Bigger label
      int neg_idx = -1;  // Smaller label
      // Are we picking a sample to the left/right of the current group?
      if (sample < nleft) {
        // Go left
        pos_idx = sample + group_begin;
        neg_idx = item_idx;
      } else {
        pos_idx = item_idx;
        int items_in_group = total_group_items - nleft - nright;
        neg_idx = sample + items_in_group + group_begin;
      }

      // Compute and assign the gradients now
      const float eps = 1e-16f;
      bst_float p = common::Sigmoid(dpreds[original_pos[pos_idx]] - dpreds[original_pos[neg_idx]]);
      bst_float g = p - 1.0f;
      bst_float h = thrust::max(p * (1.0f - p), eps);

      // Rescale each gradient and hessian so that the group has a weighted constant
      float scale = 1.0f / (niter / total_items);
      if (fix_list_weight != 0.0f) {
        scale *= fix_list_weight / total_group_items;
      }

      float weight = num_weights ? dweights[group_idx - 1] : 1.0f;
      weight *= weight_normalization_factor;
      weight *= scale;
      // Accumulate gradient and hessian in both positive and negative indices
      const GradientPair in_pos_gpair(g * weight, 2.0f * weight * h);
      dh::AtomicAddGpair(&out_gpair[original_pos[pos_idx]], in_pos_gpair);

      const GradientPair in_neg_gpair(-g * weight, 2.0f * weight * h);
      dh::AtomicAddGpair(&out_gpair[original_pos[neg_idx]], in_neg_gpair);
    });
  }
};
#endif

// objective for lambda rank
template <typename LambdaWeightComputerT>
class LambdaRankObj : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.UpdateAllowUnknown(args);
  }

  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    CHECK_EQ(preds.Size(), info.labels_.Size()) << "label size predict size not match";

    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0); tgptr[1] = static_cast<unsigned>(info.labels_.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;
    CHECK(gptr.size() != 0 && gptr.back() == info.labels_.Size())
          << "group structure not consistent with #rows";

#if defined(__CUDACC__)
    // For now, we only support pairwise ranking computation on GPU.
    // Check if we have a GPU assignment; else, revert back to CPU
    auto device = tparam_->gpu_id;
    if (device >= 0 && LambdaWeightComputerT::SupportOnGPU()) {
      ComputeGradientsOnGPU(preds, info, out_gpair, gptr);
    } else {
      // Revert back to CPU
#endif
      ComputeGradientsOnCPU(preds, info, iter, out_gpair, gptr);
#if defined(__CUDACC__)
    }
#endif
  }

  const char* DefaultEvalMetric() const override {
    return "map";
  }

  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["name"] = String(LambdaWeightComputerT::Name());
    out["lambda_rank_param"] = Object();
    for (auto const& kv : param_.__DICT__()) {
      out["lambda_rank_param"][kv.first] = kv.second;
    }
  }

  void LoadConfig(Json const& in) override {
    fromJson(in["lambda_rank_param"], &param_);
  }

 private:
  bst_float ComputeWeightNormalizationFactor(const MetaInfo& info,
                                             const std::vector<unsigned> &gptr) {
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    bst_float sum_weights = 0;
    for (bst_omp_uint k = 0; k < ngroup; ++k) {
      sum_weights += info.GetWeight(k);
    }
    return ngroup / sum_weights;
  }

  void ComputeGradientsOnCPU(const HostDeviceVector<bst_float>& preds,
                             const MetaInfo& info,
                             int iter,
                             HostDeviceVector<GradientPair>* out_gpair,
                             const std::vector<unsigned> &gptr) {
    LOG(DEBUG) << "Computing pairwise gradients on CPU.";

    bst_float weight_normalization_factor = ComputeWeightNormalizationFactor(info, gptr);
    out_gpair->Resize(preds.Size());
    #pragma omp parallel
    {
      // parallel construct, declare random number generator here, so that each
      // thread use its own random number generator, seed by thread id and current iteration
      common::RandomEngine rnd(iter * 1111 + omp_get_thread_num());

      std::vector<LambdaPair> pairs;
      std::vector<ListEntry>  lst;
      std::vector< std::pair<bst_float, unsigned> > rec;
      const auto& preds_h = preds.HostVector();
      const auto& labels = info.labels_.HostVector();
      std::vector<GradientPair>& gpair = out_gpair->HostVector();
      const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);

      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        lst.clear(); pairs.clear();
        for (unsigned j = gptr[k]; j < gptr[k+1]; ++j) {
          lst.emplace_back(preds_h[j], labels[j], j);
          gpair[j] = GradientPair(0.0f, 0.0f);
        }
        std::sort(lst.begin(), lst.end(), ListEntry::CmpPred);
        rec.resize(lst.size());
        for (unsigned i = 0; i < lst.size(); ++i) {
          rec[i] = std::make_pair(lst[i].label, i);
        }
        std::sort(rec.begin(), rec.end(), common::CmpFirst);
        // enumerate buckets with same label, for each item in the lst, grab another sample randomly
        for (unsigned i = 0; i < rec.size(); ) {
          unsigned j = i + 1;
          while (j < rec.size() && rec[j].first == rec[i].first) ++j;
          // bucket in [i,j), get a sample outside bucket
          unsigned nleft = i, nright = static_cast<unsigned>(rec.size() - j);
          if (nleft + nright != 0) {
            int nsample = param_.num_pairsample;
            while (nsample --) {
              for (unsigned pid = i; pid < j; ++pid) {
                unsigned ridx = std::uniform_int_distribution<unsigned>(0, nleft + nright - 1)(rnd);
                if (ridx < nleft) {
                  pairs.emplace_back(rec[ridx].second, rec[pid].second,
                      info.GetWeight(k) * weight_normalization_factor);
                } else {
                  pairs.emplace_back(rec[pid].second, rec[ridx+j-i].second,
                      info.GetWeight(k) * weight_normalization_factor);
                }
              }
            }
          }
          i = j;
        }
        // get lambda weight for the pairs
        LambdaWeightComputerT::GetLambdaWeight(lst, &pairs);
        // rescale each gradient and hessian so that the lst have constant weighted
        float scale = 1.0f / param_.num_pairsample;
        if (param_.fix_list_weight != 0.0f) {
          scale *= param_.fix_list_weight / (gptr[k + 1] - gptr[k]);
        }
        for (auto & pair : pairs) {
          const ListEntry &pos = lst[pair.pos_index];
          const ListEntry &neg = lst[pair.neg_index];
          const bst_float w = pair.weight * scale;
          const float eps = 1e-16f;
          bst_float p = common::Sigmoid(pos.pred - neg.pred);
          bst_float g = p - 1.0f;
          bst_float h = std::max(p * (1.0f - p), eps);
          // accumulate gradient and hessian in both pid, and nid
          gpair[pos.rindex] += GradientPair(g * w, 2.0f*w*h);
          gpair[neg.rindex] += GradientPair(-g * w, 2.0f*w*h);
        }
      }
    }
  }

#if defined(__CUDACC__)
  void ComputeGradientsOnGPU(const HostDeviceVector<bst_float>& preds,
                             const MetaInfo& info,
                             HostDeviceVector<GradientPair>* out_gpair,
                             const std::vector<unsigned> &gptr) {
    LOG(DEBUG) << "Computing pairwise gradients on GPU.";

    auto device = tparam_->gpu_id;
    dh::safe_cuda(cudaSetDevice(device));

    bst_float weight_normalization_factor = ComputeWeightNormalizationFactor(info, gptr);

    // Set the device ID and copy them to the device
    out_gpair->SetDevice(device);
    info.labels_.SetDevice(device);
    preds.SetDevice(device);
    info.weights_.SetDevice(device);

    out_gpair->Resize(preds.Size());

    auto d_preds = preds.ConstDevicePointer();
    auto d_gpair = out_gpair->DevicePointer();

    if (!slist_) {
      slist_.reset(new SortedLabelList(device, param_));
    }

    // Create segments based on group info
    slist_->InitWithTrainingInfo(gptr);

    // Sort the labels within the groups on the device
    slist_->Sort(info.labels_);

    // Initialize the gradients next
    out_gpair->Fill(GradientPair(0.0f, 0.0f));

    // Finally, compute the gradients
    slist_->ComputeGradients(d_preds, d_gpair, info.weights_, weight_normalization_factor);

    // Wait until the computations done by the kernel is complete
    dh::safe_cuda(cudaStreamSynchronize(nullptr));
  }
#endif

  LambdaRankParam param_;
#if defined(__CUDACC__)
  std::unique_ptr<SortedLabelList> slist_;
#endif
};

// register the objective functions
DMLC_REGISTER_PARAMETER(LambdaRankParam);

XGBOOST_REGISTER_OBJECTIVE(PairwiseRankObj, PairwiseLambdaWeightComputer::Name())
.describe("Pairwise rank objective.")
.set_body([]() { return new LambdaRankObj<PairwiseLambdaWeightComputer>(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankNDCG, NDCGLambdaWeightComputer::Name())
.describe("LambdaRank with NDCG as objective.")
.set_body([]() { return new LambdaRankObj<NDCGLambdaWeightComputer>(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankObjMAP, MAPLambdaWeightComputer::Name())
.describe("LambdaRank with MAP as objective.")
.set_body([]() { return new LambdaRankObj<MAPLambdaWeightComputer>(); });

}  // namespace obj
}  // namespace xgboost
