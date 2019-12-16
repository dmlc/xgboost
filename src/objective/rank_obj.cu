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
#include <thrust/iterator/discard_iterator.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#include <cub/util_allocator.cuh>

#include "../common/device_helpers.cuh"
#endif

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA) && !defined(GTEST_TEST)
DMLC_REGISTRY_FILE_TAG(rank_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

struct LambdaRankParam : public XGBoostParameter<LambdaRankParam> {
  size_t num_pairsample;
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

#if defined(__CUDACC__)
// This type sorts an array which is divided into multiple groups. The sorting is influenced
// by the function object 'Comparator'
template <typename T>
class SegmentSorter {
 private:
  // Items sorted within the group
  dh::caching_device_vector<T> ditems_;

  // Original position of the items before they are sorted descendingly within its groups
  dh::caching_device_vector<uint32_t> doriginal_pos_;

  // Segments within the original list that delineates the different groups
  dh::caching_device_vector<uint32_t> group_segments_;

  // Need this on the device as it is used in the kernels
  dh::caching_device_vector<uint32_t> dgroups_;       // Group information on device

  // Where did the item that was originally present at position 'x' move to after they are sorted
  dh::caching_device_vector<uint32_t> dindexable_sorted_pos_;

  // Initialize everything but the segments
  void Init(uint32_t num_elems) {
    ditems_.resize(num_elems);

    doriginal_pos_.resize(num_elems);
    thrust::sequence(doriginal_pos_.begin(), doriginal_pos_.end());
  }

  // Initialize all with group info
  void Init(const std::vector<uint32_t> &groups) {
    uint32_t num_elems = groups.back();
    this->Init(num_elems);
    this->CreateGroupSegments(groups);
  }

 public:
  // This needs to be public due to device lambda
  void CreateGroupSegments(const std::vector<uint32_t> &groups) {
    uint32_t num_elems = groups.back();
    group_segments_.resize(num_elems);

    dgroups_ = groups;

    // Define the segments by assigning a group ID to each element
    const uint32_t *dgroups = dgroups_.data().get();
    uint32_t ngroups = dgroups_.size();
    auto ComputeGroupIDLambda = [=] __device__(uint32_t idx) {
      return dh::UpperBound(dgroups, ngroups, idx) - 1;
    };  // NOLINT

    thrust::transform(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                      thrust::make_counting_iterator(num_elems),
                      group_segments_.begin(),
                      ComputeGroupIDLambda);
  }

  // Accessors that returns device pointer
  inline const T *GetItemsPtr() const { return ditems_.data().get(); }
  inline uint32_t GetNumItems() const { return ditems_.size(); }
  inline const dh::caching_device_vector<T> &GetItems() const {
    return ditems_;
  }

  inline const uint32_t *GetOriginalPositionsPtr() const { return doriginal_pos_.data().get(); }
  inline const dh::caching_device_vector<uint32_t> &GetOriginalPositions() const {
    return doriginal_pos_;
  }

  inline const dh::caching_device_vector<uint32_t> &GetGroupSegments() const {
    return group_segments_;
  }

  inline uint32_t GetNumGroups() const { return dgroups_.size() - 1; }
  inline const uint32_t *GetGroupsPtr() const { return dgroups_.data().get(); }
  inline const dh::caching_device_vector<uint32_t> &GetGroups() const { return dgroups_; }

  inline const dh::caching_device_vector<uint32_t> &GetIndexableSortedPositions() const {
    return dindexable_sorted_pos_;
  }

  // Sort an array that is divided into multiple groups. The array is sorted within each group.
  // This version provides the group information that is on the host.
  // The array is sorted based on an adaptable binary predicate. By default a stateless predicate
  // is used.
  template <typename Comparator = thrust::greater<T>>
  void SortItems(const T *ditems, uint32_t item_size, const std::vector<uint32_t> &groups,
                 const Comparator &comp = Comparator()) {
    this->Init(groups);
    this->SortItems(ditems, item_size, group_segments_, comp);
  }

  // Sort an array that is divided into multiple groups. The array is sorted within each group.
  // This version provides the group information that is on the device.
  // The array is sorted based on an adaptable binary predicate. By default a stateless predicate
  // is used.
  template <typename Comparator = thrust::greater<T>>
  void SortItems(const T *ditems, uint32_t item_size,
                 const dh::caching_device_vector<uint32_t> &group_segments,
                 const Comparator &comp = Comparator()) {
    this->Init(item_size);

    // Sort the items that are grouped. We would like to avoid using predicates to perform the sort,
    // as thrust resorts to using a merge sort as opposed to a much much faster radix sort
    // when comparators are used. Hence, the following algorithm is used. This is done so that
    // we can grab the appropriate related values from the original list later, after the
    // items are sorted.
    //
    // Here is the internal representation:
    // dgroups_:          [ 0, 3, 5, 8, 10 ]
    // group_segments_:   0 0 0 | 1 1 | 2 2 2 | 3 3
    // doriginal_pos_:    0 1 2 | 3 4 | 5 6 7 | 8 9
    // ditems_:           1 0 1 | 2 1 | 1 3 3 | 4 4 (from original items)
    //
    // Sort the items first and make a note of the original positions in doriginal_pos_
    // based on the sort
    // ditems_:           4 4 3 3 2 1 1 1 1 0
    // doriginal_pos_:    8 9 6 7 3 0 2 4 5 1
    // NOTE: This consumes space, but is much faster than some of the other approaches - sorting
    //       in kernel, sorting using predicates etc.

    ditems_.assign(thrust::device_ptr<const T>(ditems),
                   thrust::device_ptr<const T>(ditems) + item_size);

    // Allocator to be used by sort for managing space overhead while sorting
    dh::XGBCachingDeviceAllocator<char> alloc;

    thrust::stable_sort_by_key(thrust::cuda::par(alloc),
                               ditems_.begin(), ditems_.end(),
                               doriginal_pos_.begin(), comp);

    // Next, gather the segments based on the doriginal_pos_. This is to reflect the
    // holisitic item sort order on the segments
    // group_segments_c_:   3 3 2 2 1 0 0 1 2 0
    // doriginal_pos_:      8 9 6 7 3 0 2 4 5 1 (stays the same)
    dh::caching_device_vector<uint32_t> group_segments_c(group_segments);
    thrust::gather(doriginal_pos_.begin(), doriginal_pos_.end(),
                   group_segments.begin(), group_segments_c.begin());

    // Now, sort the group segments so that you may bring the items within the group together,
    // in the process also noting the relative changes to the doriginal_pos_ while that happens
    // group_segments_c_:   0 0 0 1 1 2 2 2 3 3
    // doriginal_pos_:      0 2 1 3 4 6 7 5 8 9
    thrust::stable_sort_by_key(thrust::cuda::par(alloc),
                               group_segments_c.begin(), group_segments_c.end(),
                               doriginal_pos_.begin(), thrust::less<uint32_t>());

    // Finally, gather the original items based on doriginal_pos_ to sort the input and
    // to store them in ditems_
    // doriginal_pos_:      0 2 1 3 4 6 7 5 8 9  (stays the same)
    // ditems_:             1 1 0 2 1 3 3 1 4 4  (from unsorted items - ditems)
    thrust::gather(doriginal_pos_.begin(), doriginal_pos_.end(),
                   thrust::device_ptr<const T>(ditems), ditems_.begin());
  }

  // Determine where an item that was originally present at position 'x' has been relocated to
  // after a sort. Creation of such an index has to be explicitly requested after a sort
  void CreateIndexableSortedPositions() {
    dindexable_sorted_pos_.resize(GetNumItems());
    thrust::scatter(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                    thrust::make_counting_iterator(GetNumItems()),  // Rearrange indices...
                    // ...based on this map
                    thrust::device_ptr<const uint32_t>(GetOriginalPositionsPtr()),
                    dindexable_sorted_pos_.begin());  // Write results into this
  }
};

// Helper functions

template <typename T>
XGBOOST_DEVICE __forceinline__ uint32_t
CountNumItemsToTheLeftOf(const T * __restrict__ items, uint32_t n, T v) {
  return dh::LowerBound(items, n, v, thrust::greater<T>());
}

template <typename T>
XGBOOST_DEVICE __forceinline__ uint32_t
CountNumItemsToTheRightOf(const T * __restrict__ items, uint32_t n, T v) {
  return n - dh::UpperBound(items, n, v, thrust::greater<T>());
}
#endif

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

class PairwiseLambdaWeightComputer {
 public:
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

#if defined(__CUDACC__)
  PairwiseLambdaWeightComputer(const bst_float *dpreds,
                               const bst_float *dlabels,
                               const SegmentSorter<float> &segment_label_sorter) {}

  class PairwiseLambdaWeightMultiplier {
   public:
    // Adjust the items weight by this value
    __device__ __forceinline__ bst_float GetWeight(uint32_t gidx, int pidx, int nidx) const {
      return 1.0f;
    }
  };

  inline const PairwiseLambdaWeightMultiplier GetWeightMultiplier() const {
    return {};
  }
#endif
};

#if defined(__CUDACC__)
class BaseLambdaWeightMultiplier {
 public:
  BaseLambdaWeightMultiplier(const SegmentSorter<float> &segment_label_sorter,
                             const SegmentSorter<float> &segment_pred_sorter)
    : dsorted_labels_(segment_label_sorter.GetItemsPtr()),
      dorig_pos_(segment_label_sorter.GetOriginalPositionsPtr()),
      dgroups_(segment_label_sorter.GetGroupsPtr()),
      dindexable_sorted_preds_pos_ptr_(
        segment_pred_sorter.GetIndexableSortedPositions().data().get()) {}

 protected:
  const float *dsorted_labels_{nullptr};  // Labels sorted within a group
  const uint32_t *dorig_pos_{nullptr};  // Original indices of the labels before they are sorted
  const uint32_t *dgroups_{nullptr};  // The group indices
  // Where can a prediction for a label be found in the original array, when they are sorted
  const uint32_t *dindexable_sorted_preds_pos_ptr_{nullptr};
};

// While computing the weight that needs to be adjusted by this ranking objective, we need
// to figure out where positive and negative labels chosen earlier exists, if the group
// were to be sorted by its predictions. To accommodate this, we employ the following algorithm.
// For a given group, let's assume the following:
// labels:        1 5 9 2 4 8 0 7 6 3
// predictions:   1 9 0 8 2 7 3 6 5 4
// position:      0 1 2 3 4 5 6 7 8 9
//
// After label sort:
// labels:        9 8 7 6 5 4 3 2 1 0
// position:      2 5 7 8 1 4 9 3 0 6
//
// After prediction sort:
// predictions:   9 8 7 6 5 4 3 2 1 0
// position:      1 3 5 7 8 9 6 4 0 2
//
// If a sorted label at position 'x' is chosen, then we need to find out where the prediction
// for this label 'x' exists, if the group were to be sorted by predictions.
// We first take the sorted prediction positions:
// position:      1 3 5 7 8 9 6 4 0 2
// at indices:    0 1 2 3 4 5 6 7 8 9
//
// We create a sorted prediction positional array, such that value at position 'x' gives
// us the position in the sorted prediction array where its related prediction lies.
// dindexable_sorted_preds_pos_ptr_:  8 0 9 1 7 2 6 3 4 5
// at indices:                        0 1 2 3 4 5 6 7 8 9
// Basically, swap the previous 2 arrays, sort the indices and reorder positions
// for an O(1) lookup using the position where the sorted label exists.
//
// This type does that using the SegmentSorter
class IndexablePredictionSorter {
 public:
  IndexablePredictionSorter(const bst_float *dpreds,
                            const SegmentSorter<float> &segment_label_sorter) {
    // Sort the predictions first
    segment_pred_sorter_.SortItems(dpreds, segment_label_sorter.GetNumItems(),
                                   segment_label_sorter.GetGroupSegments());

    // Create an index for the sorted prediction positions
    segment_pred_sorter_.CreateIndexableSortedPositions();
  }

  inline const SegmentSorter<float> &GetPredictionSorter() const {
    return segment_pred_sorter_;
  }

 private:
  SegmentSorter<float> segment_pred_sorter_;  // For sorting the predictions
};
#endif

// beta version: NDCG lambda rank
class NDCGLambdaWeightComputer
#if defined(__CUDACC__)
  : public IndexablePredictionSorter
#endif
{
 public:
#if defined(__CUDACC__)
  // This function object computes the item's DCG value
  class ComputeItemDCG : public thrust::unary_function<uint32_t, float> {
   public:
    XGBOOST_DEVICE ComputeItemDCG(const float *dsorted_labels,
                                  const uint32_t *dgroups,
                                  const uint32_t *gidxs)
      : dsorted_labels_(dsorted_labels),
        dgroups_(dgroups),
        dgidxs_(gidxs) {}

    // Compute DCG for the item at 'idx'
    __device__ __forceinline__ float operator()(uint32_t idx) const {
      return ComputeItemDCGWeight(dsorted_labels_[idx], idx - dgroups_[dgidxs_[idx]]);
    }

   private:
    const float *dsorted_labels_{nullptr};  // Labels sorted within a group
    const uint32_t *dgroups_{nullptr};  // The group indices - where each group begins and ends
    const uint32_t *dgidxs_{nullptr};  // The group each items belongs to
  };

  // Type containing device pointers that can be cheaply copied on the kernel
  class NDCGLambdaWeightMultiplier : public BaseLambdaWeightMultiplier {
   public:
    NDCGLambdaWeightMultiplier(const SegmentSorter<float> &segment_label_sorter,
                               const NDCGLambdaWeightComputer &lwc)
      : BaseLambdaWeightMultiplier(segment_label_sorter, lwc.GetPredictionSorter()),
        dgroup_dcg_ptr_(lwc.GetGroupDcgs().data().get()) {}

    // Adjust the items weight by this value
    __device__ __forceinline__ bst_float GetWeight(uint32_t gidx, int pidx, int nidx) const {
      if (dgroup_dcg_ptr_[gidx] == 0.0) return 0.0f;

      uint32_t group_begin = dgroups_[gidx];

      auto ppred_idx = dorig_pos_[pidx];
      auto npred_idx = dorig_pos_[nidx];
      KERNEL_CHECK(ppred_idx != npred_idx);

      // Note: the label positive and negative indices are relative to the entire dataset.
      // Hence, scale them back to an index within the group
      ppred_idx = dindexable_sorted_preds_pos_ptr_[ppred_idx] - group_begin;
      npred_idx = dindexable_sorted_preds_pos_ptr_[npred_idx] - group_begin;
      return NDCGLambdaWeightComputer::ComputeDeltaWeight(
        ppred_idx, npred_idx,
        static_cast<int>(dsorted_labels_[pidx]), static_cast<int>(dsorted_labels_[nidx]),
        dgroup_dcg_ptr_[gidx]);
    }

   private:
     const float *dgroup_dcg_ptr_{nullptr};  // Start address of the group DCG values
  };

  NDCGLambdaWeightComputer(const bst_float *dpreds,
                           const bst_float *dlabels,
                           const SegmentSorter<float> &segment_label_sorter)
    : IndexablePredictionSorter(dpreds, segment_label_sorter),
      dgroup_dcg_(segment_label_sorter.GetNumGroups(), 0.0f),
      weight_multiplier_(segment_label_sorter, *this) {
    const auto &group_segments = segment_label_sorter.GetGroupSegments();

    // Compute each elements DCG values and reduce them across groups concurrently.
    auto end_range =
      thrust::reduce_by_key(group_segments.begin(), group_segments.end(),
                            thrust::make_transform_iterator(
                              // The indices need not be sequential within a group, as we care only
                              // about the sum of items DCG values within a group
                              segment_label_sorter.GetOriginalPositions().begin(),
                              ComputeItemDCG(segment_label_sorter.GetItemsPtr(),
                                             segment_label_sorter.GetGroupsPtr(),
                                             group_segments.data().get())),
                            thrust::make_discard_iterator(),  // We don't care for the group indices
                            dgroup_dcg_.begin());  // Sum of the item's DCG values in the group
    CHECK(end_range.second - dgroup_dcg_.begin() == dgroup_dcg_.size());
  }

  inline const dh::caching_device_vector<float> &GetGroupDcgs() const {
    return dgroup_dcg_;
  }

  inline const NDCGLambdaWeightMultiplier GetWeightMultiplier() const {
    return weight_multiplier_;
  }
#endif

  static void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                              std::vector<LambdaPair> *io_pairs) {
    std::vector<LambdaPair> &pairs = *io_pairs;
    float IDCG;  // NOLINT
    {
      std::vector<bst_float> labels(sorted_list.size());
      for (size_t i = 0; i < sorted_list.size(); ++i) {
        labels[i] = sorted_list[i].label;
      }
      std::stable_sort(labels.begin(), labels.end(), std::greater<bst_float>());
      IDCG = ComputeGroupDCGWeight(&labels[0], labels.size());
    }
    if (IDCG == 0.0) {
      for (auto & pair : pairs) {
        pair.weight = 0.0f;
      }
    } else {
      for (auto & pair : pairs) {
        unsigned pos_idx = pair.pos_index;
        unsigned neg_idx = pair.neg_index;
        pair.weight *= ComputeDeltaWeight(pos_idx, neg_idx,
                                          sorted_list[pos_idx].label, sorted_list[neg_idx].label,
                                          IDCG);
      }
    }
  }

  static char const* Name() {
    return "rank:ndcg";
  }

  inline static bst_float ComputeGroupDCGWeight(const float *sorted_labels, uint32_t size) {
    double sumdcg = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
      sumdcg += ComputeItemDCGWeight(sorted_labels[i], i);
    }

    return static_cast<bst_float>(sumdcg);
  }

 private:
  XGBOOST_DEVICE inline static bst_float ComputeItemDCGWeight(unsigned label, uint32_t idx) {
    return (label != 0) ? (((1 << label) - 1) / std::log2(static_cast<bst_float>(idx + 2))) : 0;
  }

  // Compute the weight adjustment for an item within a group:
  // ppred_idx => Where does the positive label live, had the list been sorted by prediction
  // npred_idx => Where does the negative label live, had the list been sorted by prediction
  // pos_label => positive label value from sorted label list
  // neg_label => negative label value from sorted label list
  XGBOOST_DEVICE inline static bst_float ComputeDeltaWeight(uint32_t ppred_idx, uint32_t npred_idx,
                                                            int pos_label, int neg_label,
                                                            float idcg) {
    float pos_loginv = 1.0f / std::log2(ppred_idx + 2.0f);
    float neg_loginv = 1.0f / std::log2(npred_idx + 2.0f);
    bst_float original = ((1 << pos_label) - 1) * pos_loginv + ((1 << neg_label) - 1) * neg_loginv;
    float changed = ((1 << neg_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * neg_loginv;
    bst_float delta = (original - changed) * (1.0f / idcg);
    if (delta < 0.0f) delta = - delta;
    return delta;
  }

#if defined(__CUDACC__)
  dh::caching_device_vector<float> dgroup_dcg_;
  // This computes the adjustment to the weight
  const NDCGLambdaWeightMultiplier weight_multiplier_;
#endif
};

class MAPLambdaWeightComputer
#if defined(__CUDACC__)
  : public IndexablePredictionSorter
#endif
{
 public:
  struct MAPStats {
    /*! \brief the accumulated precision */
    float ap_acc{0.0f};
    /*!
     * \brief the accumulated precision,
     *   assuming a positive instance is missing
     */
    float ap_acc_miss{0.0f};
    /*!
     * \brief the accumulated precision,
     * assuming that one more positive instance is inserted ahead
     */
    float ap_acc_add{0.0f};
    /* \brief the accumulated positive instance count */
    float hits{0.0f};

    XGBOOST_DEVICE MAPStats() {}  // NOLINT
    XGBOOST_DEVICE MAPStats(float ap_acc, float ap_acc_miss, float ap_acc_add, float hits)
      : ap_acc(ap_acc), ap_acc_miss(ap_acc_miss), ap_acc_add(ap_acc_add), hits(hits) {}

    // For prefix scan
    XGBOOST_DEVICE MAPStats operator +(const MAPStats &v1) const {
      return {ap_acc + v1.ap_acc, ap_acc_miss + v1.ap_acc_miss,
              ap_acc_add + v1.ap_acc_add, hits + v1.hits};
    }

    // For test purposes - compare for equality
    XGBOOST_DEVICE bool operator ==(const MAPStats &rhs) const {
      return ap_acc == rhs.ap_acc && ap_acc_miss == rhs.ap_acc_miss &&
             ap_acc_add == rhs.ap_acc_add && hits == rhs.hits;
    }
  };

 private:
  template <typename T>
  XGBOOST_DEVICE inline static void Swap(T &v0, T &v1) {
#if defined(__CUDACC__)
    thrust::swap(v0, v1);
#else
    std::swap(v0, v1);
#endif
  }

  /*!
   * \brief Obtain the delta MAP by trying to switch the positions of labels in ppred_idx or
   *        npred_idx when sorted by predictions
   * \param ppred_idx, npred_idx the position of the chosen labels when sorted by predictions
   * \param pos_label, neg_label the chosen positive and negative labels
   * \param p_map_stats a vector containing the accumulated precisions for each position in a list
   * \param map_stats_size size of the accumulated precisions vector
   */
  XGBOOST_DEVICE inline static bst_float GetLambdaMAP(
    int ppred_idx, int npred_idx,
    bst_float pos_label, bst_float neg_label,
    const MAPStats *p_map_stats, uint32_t map_stats_size) {
    if (ppred_idx == npred_idx || p_map_stats[map_stats_size - 1].hits == 0) {
      return 0.0f;
    }
    if (ppred_idx > npred_idx) {
      Swap(ppred_idx, npred_idx);
      Swap(pos_label, neg_label);
    }
    bst_float original = p_map_stats[npred_idx].ap_acc;
    if (ppred_idx != 0) original -= p_map_stats[ppred_idx - 1].ap_acc;
    bst_float changed = 0;
    bst_float label1 = pos_label > 0.0f ? 1.0f : 0.0f;
    bst_float label2 = neg_label > 0.0f ? 1.0f : 0.0f;
    if (label1 == label2) {
      return 0.0;
    } else if (label1 < label2) {
      changed += p_map_stats[npred_idx - 1].ap_acc_add - p_map_stats[ppred_idx].ap_acc_add;
      changed += (p_map_stats[ppred_idx].hits + 1.0f) / (ppred_idx + 1);
    } else {
      changed += p_map_stats[npred_idx - 1].ap_acc_miss - p_map_stats[ppred_idx].ap_acc_miss;
      changed += p_map_stats[npred_idx].hits / (npred_idx + 1);
    }
    bst_float ans = (changed - original) / (p_map_stats[map_stats_size - 1].hits);
    if (ans < 0) ans = -ans;
    return ans;
  }

 public:
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
        GetLambdaMAP(pair.pos_index, pair.neg_index,
                     sorted_list[pair.pos_index].label, sorted_list[pair.neg_index].label,
                     &map_stats[0], map_stats.size());
    }
  }

#if defined(__CUDACC__)
  MAPLambdaWeightComputer(const bst_float *dpreds,
                          const bst_float *dlabels,
                          const SegmentSorter<float> &segment_label_sorter)
    : IndexablePredictionSorter(dpreds, segment_label_sorter),
      dmap_stats_(segment_label_sorter.GetNumItems(), MAPStats()),
      weight_multiplier_(segment_label_sorter, *this) {
    this->CreateMAPStats(dlabels, segment_label_sorter);
  }

  void CreateMAPStats(const bst_float *dlabels,
                      const SegmentSorter<float> &segment_label_sorter) {
    // For each group, go through the sorted prediction positions, and look up its corresponding
    // label from the unsorted labels (from the original label list)

    // For each item in the group, compute its MAP stats.
    // Interleave the computation of map stats amongst different groups.

    // First, determine postive labels in the dataset individually
    auto nitems = segment_label_sorter.GetNumItems();
    dh::caching_device_vector<uint32_t> dhits(nitems, 0);
    // Original positions of the predictions after they have been sorted
    const uint32_t *pred_original_pos = this->GetPredictionSorter().GetOriginalPositionsPtr();
    // Unsorted labels
    const float *unsorted_labels = dlabels;
    auto DeterminePositiveLabelLambda = [=] __device__(uint32_t idx) {
      return (unsorted_labels[pred_original_pos[idx]] > 0.0f) ? 1 : 0;
    };  // NOLINT

    thrust::transform(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                      thrust::make_counting_iterator(nitems),
                      dhits.begin(),
                      DeterminePositiveLabelLambda);

    // Next, prefix scan the positive labels that are segmented to accumulate them.
    // This is required for computing the accumulated precisions
    const auto &group_segments = segment_label_sorter.GetGroupSegments();
    // Data segmented into different groups...
    thrust::inclusive_scan_by_key(group_segments.begin(), group_segments.end(),
                                  dhits.begin(),  // Input value
                                  dhits.begin());  // In-place scan

    // Compute accumulated precisions for each item, assuming positive and
    // negative instances are missing.
    // But first, compute individual item precisions
    const auto *dgidx_arr = group_segments.data().get();
    const auto *dhits_arr = dhits.data().get();
    // Group info on device
    const uint32_t *dgroups = segment_label_sorter.GetGroupsPtr();
    uint32_t ngroups = segment_label_sorter.GetNumGroups();
    auto ComputeItemPrecisionLambda = [=] __device__(uint32_t idx) {
      if (unsorted_labels[pred_original_pos[idx]] > 0.0f) {
        auto idx_within_group = (idx - dgroups[dgidx_arr[idx]]) + 1;
        return MAPStats(static_cast<float>(dhits_arr[idx]) / idx_within_group,
                        static_cast<float>(dhits_arr[idx] - 1) / idx_within_group,
                        static_cast<float>(dhits_arr[idx] + 1) / idx_within_group,
                        1.0f);
      }
      return MAPStats();
    };  // NOLINT

    thrust::transform(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                      thrust::make_counting_iterator(nitems),
                      this->dmap_stats_.begin(),
                      ComputeItemPrecisionLambda);

    // Lastly, compute the accumulated precisions for all the items segmentd by groups.
    // The precisions are accumulated within each group
    thrust::inclusive_scan_by_key(group_segments.begin(), group_segments.end(),
                                  this->dmap_stats_.begin(),  // Input map stats
                                  this->dmap_stats_.begin());  // In-place scan and output here
  }

  inline const dh::caching_device_vector<MAPStats> &GetMapStats() const {
    return dmap_stats_;
  }

  // Type containing device pointers that can be cheaply copied on the kernel
  class MAPLambdaWeightMultiplier : public BaseLambdaWeightMultiplier {
   public:
    MAPLambdaWeightMultiplier(const SegmentSorter<float> &segment_label_sorter,
                              const MAPLambdaWeightComputer &lwc)
      : BaseLambdaWeightMultiplier(segment_label_sorter, lwc.GetPredictionSorter()),
        dmap_stats_ptr_(lwc.GetMapStats().data().get()) {}

    // Adjust the items weight by this value
    __device__ __forceinline__ bst_float GetWeight(uint32_t gidx, int pidx, int nidx) const {
      uint32_t group_begin = dgroups_[gidx];
      uint32_t group_end = dgroups_[gidx + 1];

      auto ppred_idx = dorig_pos_[pidx];
      auto npred_idx = dorig_pos_[nidx];
      KERNEL_CHECK(ppred_idx != npred_idx);

      // Note: the label positive and negative indices are relative to the entire dataset.
      // Hence, scale them back to an index within the group
      ppred_idx = dindexable_sorted_preds_pos_ptr_[ppred_idx] - group_begin;
      npred_idx = dindexable_sorted_preds_pos_ptr_[npred_idx] - group_begin;
      return MAPLambdaWeightComputer::GetLambdaMAP(
        ppred_idx, npred_idx,
        dsorted_labels_[pidx], dsorted_labels_[nidx],
        &dmap_stats_ptr_[group_begin], group_end - group_begin);
    }

   private:
     const MAPStats *dmap_stats_ptr_{nullptr};  // Start address of the map stats for every sorted
                                                // prediction value
  };

  inline const MAPLambdaWeightMultiplier GetWeightMultiplier() const { return weight_multiplier_; }

 private:
  dh::caching_device_vector<MAPStats> dmap_stats_;
  // This computes the adjustment to the weight
  const MAPLambdaWeightMultiplier weight_multiplier_;
#endif
};

#if defined(__CUDACC__)
class SortedLabelList : SegmentSorter<float> {
 private:
  const LambdaRankParam &param_;                      // Objective configuration

 public:
  explicit SortedLabelList(const LambdaRankParam &param)
    : param_(param) {}

  // Sort the labels that are grouped by 'groups'
  void Sort(const HostDeviceVector<bst_float> &dlabels, const std::vector<uint32_t> &groups) {
    this->SortItems(dlabels.ConstDevicePointer(), dlabels.Size(), groups);
  }

  // This kernel can only run *after* the kernel in sort is completed, as they
  // use the default stream
  template <typename LambdaWeightComputerT>
  void ComputeGradients(const bst_float *dpreds,   // Unsorted predictions
                        const bst_float *dlabels,  // Unsorted labels
                        const HostDeviceVector<bst_float> &weights,
                        int iter,
                        GradientPair *out_gpair,
                        float weight_normalization_factor) {
    // Group info on device
    const uint32_t *dgroups = this->GetGroupsPtr();
    uint32_t ngroups = this->GetNumGroups() + 1;

    uint32_t total_items = this->GetNumItems();
    uint32_t niter = param_.num_pairsample * total_items;

    float fix_list_weight = param_.fix_list_weight;

    const uint32_t *original_pos = this->GetOriginalPositionsPtr();

    uint32_t num_weights = weights.Size();
    auto dweights = num_weights ? weights.ConstDevicePointer() : nullptr;

    const bst_float *sorted_labels = this->GetItemsPtr();

    // This is used to adjust the weight of different elements based on the different ranking
    // objective function policies
    LambdaWeightComputerT weight_computer(dpreds, dlabels, *this);
    auto wmultiplier = weight_computer.GetWeightMultiplier();

    int device_id = -1;
    dh::safe_cuda(cudaGetDevice(&device_id));
    // For each instance in the group, compute the gradient pair concurrently
    dh::LaunchN(device_id, niter, nullptr, [=] __device__(uint32_t idx) {
      // First, determine the group 'idx' belongs to
      uint32_t item_idx = idx % total_items;
      uint32_t group_idx = dh::UpperBound(dgroups, ngroups, item_idx);
      // Span of this group within the larger labels/predictions sorted tuple
      uint32_t group_begin = dgroups[group_idx - 1];
      uint32_t group_end = dgroups[group_idx];
      uint32_t total_group_items = group_end - group_begin;

      // Are the labels diverse enough? If they are all the same, then there is nothing to pick
      // from another group - bail sooner
      if (sorted_labels[group_begin] == sorted_labels[group_end - 1]) return;

      // Find the number of labels less than and greater than the current label
      // at the sorted index position item_idx
      uint32_t nleft  = CountNumItemsToTheLeftOf(
        sorted_labels + group_begin, item_idx - group_begin + 1, sorted_labels[item_idx]);
      uint32_t nright = CountNumItemsToTheRightOf(
        sorted_labels + item_idx, group_end - item_idx, sorted_labels[item_idx]);

      // Create a minstd_rand object to act as our source of randomness
      thrust::minstd_rand rng((iter + 1) * 1111);
      rng.discard(((idx / total_items) * total_group_items) + item_idx - group_begin);
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
        uint32_t items_in_group = total_group_items - nleft - nright;
        neg_idx = sample + items_in_group + group_begin;
      }

      // Compute and assign the gradients now
      const float eps = 1e-16f;
      bst_float p = common::Sigmoid(dpreds[original_pos[pos_idx]] - dpreds[original_pos[neg_idx]]);
      bst_float g = p - 1.0f;
      bst_float h = thrust::max(p * (1.0f - p), eps);

      // Rescale each gradient and hessian so that the group has a weighted constant
      float scale = __frcp_ru(niter / total_items);
      if (fix_list_weight != 0.0f) {
        scale *= fix_list_weight / total_group_items;
      }

      float weight = num_weights ? dweights[group_idx - 1] : 1.0f;
      weight *= weight_normalization_factor;
      weight *= wmultiplier.GetWeight(group_idx - 1, pos_idx, neg_idx);
      weight *= scale;
      // Accumulate gradient and hessian in both positive and negative indices
      const GradientPair in_pos_gpair(g * weight, 2.0f * weight * h);
      dh::AtomicAddGpair(&out_gpair[original_pos[pos_idx]], in_pos_gpair);

      const GradientPair in_neg_gpair(-g * weight, 2.0f * weight * h);
      dh::AtomicAddGpair(&out_gpair[original_pos[neg_idx]], in_neg_gpair);
    });

    // Wait until the computations done by the kernel is complete
    dh::safe_cuda(cudaStreamSynchronize(nullptr));
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
    // Check if we have a GPU assignment; else, revert back to CPU
    auto device = tparam_->gpu_id;
    if (device >= 0) {
      ComputeGradientsOnGPU(preds, info, iter, out_gpair, gptr);
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

    const auto& preds_h = preds.HostVector();
    const auto& labels = info.labels_.HostVector();
    std::vector<GradientPair>& gpair = out_gpair->HostVector();
    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    out_gpair->Resize(preds.Size());

    #pragma omp parallel
    {
      // parallel construct, declare random number generator here, so that each
      // thread use its own random number generator, seed by thread id and current iteration
      std::minstd_rand rnd((iter + 1) * 1111);
      std::vector<LambdaPair> pairs;
      std::vector<ListEntry>  lst;
      std::vector< std::pair<bst_float, unsigned> > rec;

      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        lst.clear(); pairs.clear();
        for (unsigned j = gptr[k]; j < gptr[k+1]; ++j) {
          lst.emplace_back(preds_h[j], labels[j], j);
          gpair[j] = GradientPair(0.0f, 0.0f);
        }
        std::stable_sort(lst.begin(), lst.end(), ListEntry::CmpPred);
        rec.resize(lst.size());
        for (unsigned i = 0; i < lst.size(); ++i) {
          rec[i] = std::make_pair(lst[i].label, i);
        }
        std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
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
                             int iter,
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
    auto d_labels = info.labels_.ConstDevicePointer();

    SortedLabelList slist(param_);

    // Sort the labels within the groups on the device
    slist.Sort(info.labels_, gptr);

    // Initialize the gradients next
    out_gpair->Fill(GradientPair(0.0f, 0.0f));

    // Finally, compute the gradients
    slist.ComputeGradients<LambdaWeightComputerT>
      (d_preds, d_labels, info.weights_, iter, d_gpair, weight_normalization_factor);
  }
#endif

  LambdaRankParam param_;
};

#if !defined(GTEST_TEST)
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
#endif

}  // namespace obj
}  // namespace xgboost
