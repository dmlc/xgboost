/*!
 * Copyright 2019 XGBoost contributors
 */
#include <dmlc/omp.h>
#include <dmlc/timer.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../common/math.h"
#include "../common/random.h"

#if defined(__CUDACC__)
#include <thrust/gather.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#include "../common/device_helpers.cuh"
#endif

namespace xgboost {
namespace obj {

#if defined(XGBOOST_USE_CUDA)
DMLC_REGISTRY_FILE_TAG(rank_obj_gpu);
#endif  // defined(XGBOOST_USE_CUDA)

struct LambdaRankParam : public dmlc::Parameter<LambdaRankParam> {
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
class SortedLabelList {
 private:
  dh::device_vector<bst_float> dpreds;   // Used to store sorted predictions
  dh::device_vector<bst_float> dlabels;  // Used to store sorted labels
  const bst_float *orig_dpreds;          // Original predictions - unsorted
  const bst_float *orig_dlabels;         // Original labels - unsorted

  dh::device_vector<int> dpos;           // Original position of the labels in the dataset
  dh::device_vector<int> dlabels_count;  // Unique label count in CSR format
  cudaStream_t stream{nullptr};
  int begin_group_idx{-1};               // Begining index within the group
  int device_id{-1};                     // GPU device ID

 public:
  SortedLabelList(int dev_id,
                  const bst_float *preds, const bst_float *labels,
                  int begin, int end)
    : dpreds(preds + begin, preds + end),
      dlabels(labels + begin, labels + end),
      orig_dpreds(preds),
      orig_dlabels(labels),
      dpos(end - begin),
      dlabels_count(end - begin + 1, 1),
      begin_group_idx(begin),
      device_id(dev_id) {
    dh::safe_cuda(cudaStreamCreate(&stream));
    thrust::sequence(thrust::cuda::par.on(stream), dpos.begin(), dpos.end());
    dlabels_count[0] = 0;
  }

  ~SortedLabelList() {
    dh::safe_cuda(cudaSetDevice(device_id));
    dh::safe_cuda(cudaStreamDestroy(stream));
  }

  // Sort by predictions first and then sort the labels by predictions next
  void Sort() {
    dh::device_vector<bst_float> cdpreds(dpreds);
    // Sort the predictions first and rearrange its positional indices
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        cdpreds.begin(), cdpreds.end(), dpos.begin(),
                        thrust::greater<bst_float>());
    // Gather the labels based on the sorted indices
    thrust::gather(thrust::cuda::par.on(stream),
                   dpos.begin(), dpos.end(), orig_dlabels + begin_group_idx, dlabels.begin());

    // Sort the labels next and get the final order
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        dlabels.begin(), dlabels.end(), dpos.begin(),
                        thrust::greater<bst_float>());
    // Use the order to then sort the original predictions
    thrust::gather(thrust::cuda::par.on(stream),
                   dpos.begin(), dpos.end(), orig_dpreds + begin_group_idx, dpreds.begin());
  }

  // For all the unique labels, create a number of such labels for those unique label
  // values. Returns the number of such unique labels
  int CreateUniqueLabelCount() {
    dh::device_vector<bst_float> dunique_labels(dlabels.size());
    // Find all unique values first and the number of such labels in that group
    auto itr_pair =
      thrust::reduce_by_key(thrust::cuda::par.on(stream),
                            dlabels.begin(), dlabels.end(), dlabels_count.begin() + 1,
                            dunique_labels.begin(), dlabels_count.begin() + 1);

    dlabels_count.resize(itr_pair.second - dlabels_count.begin());
    dlabels_count.shrink_to_fit();
    // Create a CSR style unique count array that can be used to pick a sample outside
    // the label group while computing the lambda pairs
    thrust::inclusive_scan(thrust::cuda::par.on(stream),
                           dlabels_count.begin(), dlabels_count.end(), dlabels_count.begin());

    return dlabels_count.size() - 1;  // -1 for the first element which is 0 for the CSR format
  }

  void ComputeGradients(GradientPair *out_gpair, float weight, int nsamples) {
    // Unique labels
    int *dlabels_count_arr = thrust::raw_pointer_cast(&dlabels_count[0]);
    int dlabels_count_arr_size = dlabels_count.size();

    // Position within the original dataset
    int *dpos_arr = thrust::raw_pointer_cast(&dpos[0]);

    // Group predictions
    float *dpreds_arr = thrust::raw_pointer_cast(&dpreds[0]);

    int niter = nsamples * dpreds.size();

    // As multiple groups can be processed in parallel, this is the index offset into the
    // out_gpair
    int pos_offset = begin_group_idx;

    // For each instance in the group, compute the gradient pair concurrently
    dh::LaunchN(device_id, niter, stream, [=] __device__(size_t idx) {
      int total_items = dlabels_count_arr[dlabels_count_arr_size-1];
      int item_idx = idx % total_items;

      // Determine the label count index from the item so that we can pick another
      // item from outside its label
      int lidx = dh::UpperBound(dlabels_count_arr, dlabels_count_arr_size, item_idx);
      int items_in_group = dlabels_count_arr[lidx] - dlabels_count_arr[lidx-1];
      int rem_items = total_items - items_in_group;

      // Create a minstd_rand object to act as our source of randomness
      thrust::minstd_rand rng;
      rng.discard(pos_offset + idx);
      // Create a uniform_int_distribution to produce a sample from [0, rem_items-1]
      thrust::uniform_int_distribution<int> dist(0, rem_items-1);

      int sample = dist(rng);
      int pos_idx = -1;  // Bigger label
      int neg_idx = -1;  // Smaller label
      // Are we picking a sample to the left/right of the current group?
      if (sample < dlabels_count_arr[lidx-1]) {
        // Go left
        pos_idx = sample;
        neg_idx = item_idx;
      } else {
        pos_idx = item_idx;
        neg_idx = sample + items_in_group;
      }

      // Compute and assign the gradients now
      const float eps = 1e-16f;
      bst_float p = common::Sigmoid(dpreds_arr[pos_idx] - dpreds_arr[neg_idx]);
      bst_float g = p - 1.0f;
      bst_float h = thrust::max(p * (1.0f - p), eps);

      // Accumulate gradient and hessian in both positive and negative indices
      float *out_pos_gpair = reinterpret_cast<float *>(&out_gpair[dpos_arr[pos_idx] + pos_offset]);
      const GradientPair in_pos_gpair(g * weight, 2.0f * weight * h);
      atomicAdd(out_pos_gpair, in_pos_gpair.GetGrad());
      atomicAdd(out_pos_gpair + 1, in_pos_gpair.GetHess());

      float *out_neg_gpair = reinterpret_cast<float *>(&out_gpair[dpos_arr[neg_idx] + pos_offset]);
      const GradientPair in_neg_gpair(-g * weight, 2.0f * weight * h);
      atomicAdd(out_neg_gpair, in_neg_gpair.GetGrad());
      atomicAdd(out_neg_gpair + 1, in_neg_gpair.GetHess());
    });
  }
};
#endif

// objective for lambda rank
template <typename LambdaWeightComputerT>
class LambdaRankObj : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
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

    const auto ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);
    bst_float sum_weights = 0;
    for (bst_omp_uint k = 0; k < ngroup; ++k) {
      sum_weights += info.GetWeight(k);
    }
    bst_float weight_normalization_factor = ngroup/sum_weights;

#if defined(__CUDACC__)
    // For now, we only support pairwise ranking computation on GPU.
    // Check if we have a GPU assignment; else, revert back to CPU
    auto device = tparam_->gpu_id;
    if (device >= 0 && LambdaWeightComputerT::SupportOnGPU()) {
      LOG(DEBUG) << "Computing pairwise gradients on GPU.";
      dh::safe_cuda(cudaSetDevice(device));

      // Set the device ID and copy them to the device
      out_gpair->SetDevice(device);
      info.labels_.SetDevice(device);
      preds.SetDevice(device);

      out_gpair->Resize(preds.Size());

      auto d_preds = preds.ConstDevicePointer();
      auto d_labels = info.labels_.ConstDevicePointer();
      auto d_gpair = out_gpair->DevicePointer();

      // Initialize the gradients next
      out_gpair->Fill(GradientPair(0.0f, 0.0f));

      // For each group, sort all the labels and the predictions within those labels
      // in a descending order concurrently
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        dh::safe_cuda(cudaSetDevice(device));
        SortedLabelList slist(device, d_preds, d_labels, gptr[k], gptr[k + 1]);

        slist.Sort();

        // There is no label diversity - continue with other groups
        if (slist.CreateUniqueLabelCount() <= 1) continue;

        // Compute the gradients now
        // Rescale each gradient and hessian so that the group has a weighted constant
        float scale = 1.0f / param_.num_pairsample;
        if (param_.fix_list_weight != 0.0f) {
          scale *= param_.fix_list_weight / (gptr[k + 1] - gptr[k]);
        }
        float weight = info.GetWeight(k) * weight_normalization_factor * scale;

        slist.ComputeGradients(d_gpair, weight, param_.num_pairsample);
      }
    } else {
      // Revert back to CPU
#endif
    LOG(DEBUG) << "Computing pairwise gradients on CPU.";
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
#if defined(__CUDACC__)
    }
#endif
  }
  const char* DefaultEvalMetric() const override {
    return "map";
  }

 private:
  LambdaRankParam param_;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(LambdaRankParam);

XGBOOST_REGISTER_OBJECTIVE(PairwiseRankObj, "rank:pairwise")
.describe("Pairwise rank objective.")
.set_body([]() { return new LambdaRankObj<PairwiseLambdaWeightComputer>(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankNDCG, "rank:ndcg")
.describe("LambdaRank with NDCG as objective.")
.set_body([]() { return new LambdaRankObj<NDCGLambdaWeightComputer>(); });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankObjMAP, "rank:map")
.describe("LambdaRank with MAP as objective.")
.set_body([]() { return new LambdaRankObj<MAPLambdaWeightComputer>(); });

}  // namespace obj
}  // namespace xgboost
