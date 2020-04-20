/*!
 * Copyright 2020 by Contributors
 * \file rank_metric.cc
 * \brief prediction rank based metrics.
 * \author Kailong Chen, Tianqi Chen
 */
#include <rabit/rabit.h>
#include <dmlc/registry.h>

#include <xgboost/metric.h>
#include <xgboost/host_device_vector.h>
#include <thrust/iterator/discard_iterator.h>

#include <cmath>
#include <vector>

#include "metric_common.h"

#include "../common/math.h"
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(rank_metric_gpu);

/*! \brief Evaluate rank list on GPU */
template <typename EvalMetricT>
struct EvalRankGpu : public Metric, public EvalRankConfig {
 public:
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    // Sanity check is done by the caller
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(preds.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;

    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);

    auto device = tparam_->gpu_id;
    dh::safe_cuda(cudaSetDevice(device));

    info.labels_.SetDevice(device);
    preds.SetDevice(device);

    auto dpreds = preds.ConstDevicePointer();
    auto dlabels = info.labels_.ConstDevicePointer();

    // Sort all the predictions
    dh::SegmentSorter<float> segment_pred_sorter;
    segment_pred_sorter.SortItems(dpreds, preds.Size(), gptr);

    // Compute individual group metric and sum them up
    return EvalMetricT::EvalMetric(segment_pred_sorter, dlabels, *this);
  }

  const char* Name() const override {
    return name.c_str();
  }

  explicit EvalRankGpu(const char* name, const char* param) {
    using namespace std;  // NOLINT(*)
    if (param != nullptr) {
      std::ostringstream os;
      if (sscanf(param, "%u[-]?", &this->topn) == 1) {
        os << name << '@' << param;
        this->name = os.str();
      } else {
        os << name << param;
        this->name = os.str();
      }
      if (param[strlen(param) - 1] == '-') {
        this->minus = true;
      }
    } else {
      this->name = name;
    }
  }
};

/*! \brief Precision at N, for both classification and rank */
struct EvalPrecisionGpu {
 public:
  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter,
                           const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Group info on device
    const auto &dgroups = pred_sorter.GetGroupsSpan();
    const auto ngroups = pred_sorter.GetNumGroups();
    const auto &dgroup_idx = pred_sorter.GetGroupSegmentsSpan();

    // Original positions of the predictions after they have been sorted
    const auto &dpreds_orig_pos = pred_sorter.GetOriginalPositionsSpan();

    // First, determine non zero labels in the dataset individually
    auto DetermineNonTrivialLabelLambda = [=] __device__(uint32_t idx) {
      return (static_cast<unsigned>(dlabels[dpreds_orig_pos[idx]]) != 0) ? 1 : 0;
    };  // NOLINT

    // Find each group's metric sum
    dh::caching_device_vector<uint32_t> hits(ngroups, 0);
    const auto nitems = pred_sorter.GetNumItems();
    auto *dhits = hits.data().get();

    int device_id = -1;
    dh::safe_cuda(cudaGetDevice(&device_id));
    // For each group item compute the aggregated precision
    dh::LaunchN(device_id, nitems, nullptr, [=] __device__(uint32_t idx) {
      const auto group_idx = dgroup_idx[idx];
      const auto group_begin = dgroups[group_idx];
      const auto ridx = idx - group_begin;
      if (ridx < ecfg.topn && DetermineNonTrivialLabelLambda(idx)) {
        atomicAdd(&dhits[group_idx], 1);
      }
    });

    // Allocator to be used for managing space overhead while performing reductions
    dh::XGBCachingDeviceAllocator<char> alloc;
    return static_cast<double>(thrust::reduce(thrust::cuda::par(alloc),
                                              hits.begin(), hits.end())) / ecfg.topn;
  }
};

/*! \brief NDCG: Normalized Discounted Cumulative Gain at N */
struct EvalNDCGGpu {
 public:
  static void ComputeDCG(const dh::SegmentSorter<float> &pred_sorter,
                         const float *dlabels,
                         const EvalRankConfig &ecfg,
                         // The order in which labels have to be accessed. The order is determined
                         // by sorting the predictions or the labels for the entire dataset
                         const xgboost::common::Span<const uint32_t> &dlabels_sort_order,
                         dh::caching_device_vector<double> *dcgptr) {
    dh::caching_device_vector<double> &dcgs(*dcgptr);
    // Group info on device
    const auto &dgroups = pred_sorter.GetGroupsSpan();
    const auto &dgroup_idx = pred_sorter.GetGroupSegmentsSpan();

    // First, determine non zero labels in the dataset individually
    auto DetermineNonTrivialLabelLambda = [=] __device__(uint32_t idx) {
      return (static_cast<unsigned>(dlabels[dlabels_sort_order[idx]]));
    };  // NOLINT

    // Find each group's DCG value
    const auto nitems = pred_sorter.GetNumItems();
    auto *ddcgs = dcgs.data().get();

    int device_id = -1;
    dh::safe_cuda(cudaGetDevice(&device_id));

    // For each group item compute the aggregated precision
    dh::LaunchN(device_id, nitems, nullptr, [=] __device__(uint32_t idx) {
      const auto group_idx = dgroup_idx[idx];
      const auto group_begin = dgroups[group_idx];
      const auto ridx = idx - group_begin;
      auto label = DetermineNonTrivialLabelLambda(idx);
      if (ridx < ecfg.topn && label) {
        atomicAdd(&ddcgs[group_idx], ((1 << label) - 1) / std::log2(ridx + 2.0));
      }
    });
  }

  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter,
                           const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Sort the labels and compute IDCG
    dh::SegmentSorter<float> segment_label_sorter;
    segment_label_sorter.SortItems(dlabels, pred_sorter.GetNumItems(),
                                   pred_sorter.GetGroupSegmentsSpan());

    uint32_t ngroups = pred_sorter.GetNumGroups();

    dh::caching_device_vector<double> idcg(ngroups, 0);
    ComputeDCG(pred_sorter, dlabels, ecfg, segment_label_sorter.GetOriginalPositionsSpan(), &idcg);

    // Compute the DCG values next
    dh::caching_device_vector<double> dcg(ngroups, 0);
    ComputeDCG(pred_sorter, dlabels, ecfg, pred_sorter.GetOriginalPositionsSpan(), &dcg);

    double *ddcg = dcg.data().get();
    double *didcg = idcg.data().get();

    int device_id = -1;
    dh::safe_cuda(cudaGetDevice(&device_id));
    // Compute the group's DCG and reduce it across all groups
    dh::LaunchN(device_id, ngroups, nullptr, [=] __device__(uint32_t gidx) {
      if (didcg[gidx] == 0.0f) {
        ddcg[gidx] = (ecfg.minus) ? 0.0f : 1.0f;
      } else {
        ddcg[gidx] /= didcg[gidx];
      }
    });

    // Allocator to be used for managing space overhead while performing reductions
    dh::XGBCachingDeviceAllocator<char> alloc;
    return thrust::reduce(thrust::cuda::par(alloc), dcg.begin(), dcg.end());
  }
};

/*! \brief Mean Average Precision at N, for both classification and rank */
struct EvalMAPGpu {
 public:
  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter,
                           const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Group info on device
    const auto &dgroups = pred_sorter.GetGroupsSpan();
    const auto ngroups = pred_sorter.GetNumGroups();
    const auto &dgroup_idx = pred_sorter.GetGroupSegmentsSpan();

    // Original positions of the predictions after they have been sorted
    const auto &dpreds_orig_pos = pred_sorter.GetOriginalPositionsSpan();

    // First, determine non zero labels in the dataset individually
    const auto nitems = pred_sorter.GetNumItems();
    dh::caching_device_vector<uint32_t> hits(nitems, 0);
    auto DetermineNonTrivialLabelLambda = [=] __device__(uint32_t idx) {
      return (static_cast<unsigned>(dlabels[dpreds_orig_pos[idx]]) != 0) ? 1 : 0;
    };  // NOLINT

    thrust::transform(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                      thrust::make_counting_iterator(nitems),
                      hits.begin(),
                      DetermineNonTrivialLabelLambda);

    // Allocator to be used by sort for managing space overhead while performing prefix scans
    dh::XGBCachingDeviceAllocator<char> alloc;

    // Next, prefix scan the nontrivial labels that are segmented to accumulate them.
    // This is required for computing the metric sum
    // Data segmented into different groups...
    thrust::inclusive_scan_by_key(thrust::cuda::par(alloc),
                                  dh::tcbegin(dgroup_idx), dh::tcend(dgroup_idx),
                                  hits.begin(),  // Input value
                                  hits.begin());  // In-place scan

    // Find each group's metric sum
    dh::caching_device_vector<double> sumap(ngroups, 0);
    auto *dsumap = sumap.data().get();
    const auto *dhits = hits.data().get();

    int device_id = -1;
    dh::safe_cuda(cudaGetDevice(&device_id));
    // For each group item compute the aggregated precision
    dh::LaunchN(device_id, nitems, nullptr, [=] __device__(uint32_t idx) {
      if (DetermineNonTrivialLabelLambda(idx)) {
        const auto group_idx = dgroup_idx[idx];
        const auto group_begin = dgroups[group_idx];
        const auto ridx = idx - group_begin;
        if (ridx < ecfg.topn) {
          atomicAdd(&dsumap[group_idx],
                    static_cast<double>(dhits[idx]) / (ridx + 1));
        }
      }
    });

    // Aggregate the group's item precisions
    dh::LaunchN(device_id, ngroups, nullptr, [=] __device__(uint32_t gidx) {
      auto nhits = dgroups[gidx + 1] ? dhits[dgroups[gidx + 1] - 1] : 0;
      if (nhits != 0) {
        dsumap[gidx] /= nhits;
      } else {
        if (ecfg.minus) {
          dsumap[gidx] = 0;
        } else {
          dsumap[gidx] = 1;
        }
      }
    });

    return thrust::reduce(thrust::cuda::par(alloc), sumap.begin(), sumap.end());
  }
};

/*! \brief Area Under Curve metric computation for ranking datasets */
struct EvalAucGpu : public Metric {
 public:
  // This function object computes the positive precision pair for each prediction group
  class ComputePosPair : public thrust::unary_function<uint32_t, double> {
   public:
    XGBOOST_DEVICE ComputePosPair(const double *pred_group_pos_precision,
                                  const double *pred_group_neg_precision,
                                  const double *pred_group_incr_precision)
      : pred_group_pos_precision_(pred_group_pos_precision),
        pred_group_neg_precision_(pred_group_neg_precision),
        pred_group_incr_precision_(pred_group_incr_precision) {}

    // Compute positive precision pair for the prediction group at 'idx'
    __device__ __forceinline__ double operator()(uint32_t idx) const {
        return pred_group_neg_precision_[idx] *
               (pred_group_incr_precision_[idx] + pred_group_pos_precision_[idx] * 0.5);
    }

   private:
    // Accumulated positive precision for the prediction group
    const double *pred_group_pos_precision_{nullptr};
    // Accumulated negative precision for the prediction group
    const double *pred_group_neg_precision_{nullptr};
    // Incremental positive precision for the prediction group
    const double *pred_group_incr_precision_{nullptr};
  };

  template <typename T>
  void ReleaseMemory(dh::caching_device_vector<T> &vec) {  // NOLINT
    dh::caching_device_vector<T>().swap(vec);
  }

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    // Sanity check is done by the caller
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.empty() ? tgptr : info.group_ptr_;

    auto device = tparam_->gpu_id;
    dh::safe_cuda(cudaSetDevice(device));

    info.labels_.SetDevice(device);
    preds.SetDevice(device);
    info.weights_.SetDevice(device);

    auto dpreds = preds.ConstDevicePointer();
    auto dlabels = info.labels_.ConstDevicePointer();
    auto dweights = info.weights_.ConstDevicePointer();

    // Sort all the predictions (from one or more groups)
    dh::SegmentSorter<float> segment_pred_sorter;
    segment_pred_sorter.SortItems(dpreds, preds.Size(), gptr);

    const auto &dsorted_preds = segment_pred_sorter.GetItemsSpan();
    const auto &dpreds_orig_pos = segment_pred_sorter.GetOriginalPositionsSpan();

    // Group info on device
    const auto &dgroups = segment_pred_sorter.GetGroupsSpan();
    uint32_t ngroups = segment_pred_sorter.GetNumGroups();

    // Final values
    double hsum_auc = 0.0;
    unsigned hauc_error = 0;

    int device_id = -1;
    dh::safe_cuda(cudaGetDevice(&device_id));

    // Allocator to be used for managing space overhead while performing reductions
    dh::XGBCachingDeviceAllocator<char> alloc;

    if (ngroups == 1) {
      const auto nitems = segment_pred_sorter.GetNumItems();

      // First, segment all the predictions in the group. This is required so that we can
      // aggregate the positive and negative precisions within that prediction group
      dh::caching_device_vector<unsigned> dpred_segs(nitems, 0);
      auto *pred_seg_arr = dpred_segs.data().get();
      // This is for getting the next segment number
      dh::caching_device_vector<unsigned> seg_idx(1, 0);
      auto *seg_idx_ptr = seg_idx.data().get();

      dh::caching_device_vector<double> dbuf_pos(nitems, 0);
      dh::caching_device_vector<double> dbuf_neg(nitems, 0);
      auto *buf_pos_arr = dbuf_pos.data().get();
      auto *buf_neg_arr = dbuf_neg.data().get();

      dh::LaunchN(device_id, nitems, nullptr, [=] __device__(int idx) {
        auto ctr = dlabels[dpreds_orig_pos[idx]];
        // For ranking task, weights are per-group
        // For binary classification task, weights are per-instance
        const auto wt = dweights == nullptr ? 1.0f : dweights[dpreds_orig_pos[idx]];
        buf_pos_arr[idx] = ctr * wt;
        buf_neg_arr[idx] = (1.0f - ctr) * wt;
        if (idx == nitems - 1 || dsorted_preds[idx] != dsorted_preds[idx + 1]) {
          auto new_seg_idx = atomicAdd(seg_idx_ptr, 1);
          auto pred_val = dsorted_preds[idx];
          do {
            pred_seg_arr[idx] = new_seg_idx;
            idx--;
          } while (idx >= 0 && dsorted_preds[idx] == pred_val);
        }
      });

      auto nunique_preds = seg_idx.back();
      ReleaseMemory(seg_idx);

      // Next, accumulate the positive and negative precisions for every prediction group
      dh::caching_device_vector<double> sum_dbuf_pos(nunique_preds, 0);
      auto itr = thrust::reduce_by_key(thrust::cuda::par(alloc),
                                       dpred_segs.begin(), dpred_segs.end(),  // Segmented by this
                                       dbuf_pos.begin(),  // Individual precisions
                                       thrust::make_discard_iterator(),  // Ignore unique segments
                                       sum_dbuf_pos.begin());  // Write accumulated results here
      ReleaseMemory(dbuf_pos);
      CHECK(itr.second - sum_dbuf_pos.begin() == nunique_preds);

      dh::caching_device_vector<double> sum_dbuf_neg(nunique_preds, 0);
      itr = thrust::reduce_by_key(thrust::cuda::par(alloc),
                                  dpred_segs.begin(), dpred_segs.end(),
                                  dbuf_neg.begin(),
                                  thrust::make_discard_iterator(),
                                  sum_dbuf_neg.begin());
      ReleaseMemory(dbuf_neg);
      ReleaseMemory(dpred_segs);
      CHECK(itr.second - sum_dbuf_neg.begin() == nunique_preds);

      dh::caching_device_vector<double> sum_nneg(nunique_preds, 0);
      thrust::inclusive_scan(thrust::cuda::par(alloc),
                             sum_dbuf_neg.begin(), sum_dbuf_neg.end(),
                             sum_nneg.begin());
      double sum_neg_prec_val = sum_nneg.back();
      ReleaseMemory(sum_nneg);

      // Find incremental sum for the positive precisions that is then used to
      // compute incremental positive precision pair
      dh::caching_device_vector<double> sum_npos(nunique_preds + 1, 0);
      thrust::inclusive_scan(thrust::cuda::par(alloc),
                             sum_dbuf_pos.begin(), sum_dbuf_pos.end(),
                             sum_npos.begin() + 1);
      double sum_pos_prec_val = sum_npos.back();

      if (sum_pos_prec_val <= 0.0 || sum_neg_prec_val <= 0.0) {
        hauc_error = 1;
      } else {
        dh::caching_device_vector<double> sum_pospair(nunique_preds, 0);
        // Finally, compute the positive precision pair
        thrust::transform(thrust::make_counting_iterator(static_cast<uint32_t>(0)),
                          thrust::make_counting_iterator(static_cast<uint32_t>(nunique_preds)),
                          sum_pospair.begin(),
                          ComputePosPair(sum_dbuf_pos.data().get(),
                                         sum_dbuf_neg.data().get(),
                                         sum_npos.data().get()));
        ReleaseMemory(sum_dbuf_pos);
        ReleaseMemory(sum_dbuf_neg);
        ReleaseMemory(sum_npos);
        hsum_auc = thrust::reduce(thrust::cuda::par(alloc),
                                  sum_pospair.begin(), sum_pospair.end())
                     / (sum_pos_prec_val * sum_neg_prec_val);
      }
    } else {
      // AUC sum for each group
      dh::caching_device_vector<double> sum_auc(ngroups, 0);
      // AUC error across all groups
      dh::caching_device_vector<int> auc_error(1, 0);
      auto *dsum_auc = sum_auc.data().get();
      auto *dauc_error = auc_error.data().get();

      // For each group item compute the aggregated precision
      dh::LaunchN<1, 32>(device_id, ngroups, nullptr, [=] __device__(uint32_t gidx) {
        double sum_pospair = 0.0, sum_npos = 0.0, sum_nneg = 0.0, buf_pos = 0.0, buf_neg = 0.0;

        for (auto i = dgroups[gidx]; i < dgroups[gidx + 1]; ++i) {
          const auto ctr = dlabels[dpreds_orig_pos[i]];
          // Keep bucketing predictions in same bucket
          if (i != dgroups[gidx] && dsorted_preds[i] != dsorted_preds[i - 1]) {
            sum_pospair += buf_neg * (sum_npos + buf_pos * 0.5);
            sum_npos += buf_pos;
            sum_nneg += buf_neg;
            buf_neg = buf_pos = 0.0f;
          }
          // For ranking task, weights are per-group
          // For binary classification task, weights are per-instance
          const auto wt = dweights == nullptr ? 1.0f : dweights[gidx];
          buf_pos += ctr * wt;
          buf_neg += (1.0f - ctr) * wt;
        }
        sum_pospair += buf_neg * (sum_npos + buf_pos * 0.5);
        sum_npos += buf_pos;
        sum_nneg += buf_neg;

        // Check weird conditions
        if (sum_npos <= 0.0 || sum_nneg <= 0.0) {
          atomicAdd(dauc_error, 1);
        } else {
          // This is the AUC
          dsum_auc[gidx] = sum_pospair / (sum_npos * sum_nneg);
        }
      });

      hsum_auc = thrust::reduce(thrust::cuda::par(alloc), sum_auc.begin(), sum_auc.end());
      hauc_error = auc_error.back();  // Copy it back to host
    }

    // Report average AUC across all groups
    // In distributed mode, workers which only contains pos or neg samples
    // will be ignored when aggregate AUC.
    bst_float dat[2] = {0.0f, 0.0f};
    if (hauc_error < ngroups) {
      dat[0] = static_cast<bst_float>(hsum_auc);
      dat[1] = static_cast<bst_float>(ngroups - hauc_error);
    }
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    CHECK_GT(dat[1], 0.0f)
      << "AUC: the dataset only contains pos or neg samples";
    return dat[0] / dat[1];
  }

  const char* Name() const override {
    return "auc";
  }
};

/*! \brief Area Under PR Curve metric computation for ranking datasets */
struct EvalAucPRGpu : public Metric {
 public:
  // This function object computes the item's positive/negative precision value
  class ComputeItemPrecision : public thrust::unary_function<uint32_t, float> {
   public:
    // The precision type to be computed
    enum class PrecisionType {
      kPositive,
      kNegative
    };

    XGBOOST_DEVICE ComputeItemPrecision(PrecisionType ptype,
                                        uint32_t ngroups,
                                        const float *dweights,
                                        const xgboost::common::Span<const uint32_t> &dgidxs,
                                        const float *dlabels)
      : ptype_(ptype), ngroups_(ngroups), dweights_(dweights), dgidxs_(dgidxs), dlabels_(dlabels) {}

    // Compute precision value for the prediction that was originally at 'idx'
    __device__ __forceinline__ float operator()(uint32_t idx) const {
      // For ranking task, weights are per-group
      // For binary classification task, weights are per-instance
      const auto wt = dweights_ == nullptr ? 1.0f : dweights_[ngroups_ == 1 ? idx : dgidxs_[idx]];
      return wt * (ptype_ == PrecisionType::kPositive ? dlabels_[idx] : (1.0f - dlabels_[idx]));
    }

   private:
    PrecisionType ptype_;  // Precision type to be computed
    uint32_t ngroups_;  // Number of groups in the dataset
    const float *dweights_;  // Instance/group weights
    const xgboost::common::Span<const uint32_t> dgidxs_;  // The group a given instance belongs to
    const float *dlabels_;  // Unsorted labels in the dataset
  };

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    // Sanity check is done by the caller
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.empty() ? tgptr : info.group_ptr_;

    auto device = tparam_->gpu_id;
    dh::safe_cuda(cudaSetDevice(device));

    info.labels_.SetDevice(device);
    preds.SetDevice(device);
    info.weights_.SetDevice(device);

    auto dpreds = preds.ConstDevicePointer();
    auto dlabels = info.labels_.ConstDevicePointer();
    auto dweights = info.weights_.ConstDevicePointer();

    // Sort all the predictions
    dh::SegmentSorter<float> segment_pred_sorter;
    segment_pred_sorter.SortItems(dpreds, preds.Size(), gptr);

    const auto &dsorted_preds = segment_pred_sorter.GetItemsSpan();
    // Original positions of the predictions after they have been sorted
    const auto &dpreds_orig_pos = segment_pred_sorter.GetOriginalPositionsSpan();

    // Group info on device
    const auto &dgroups = segment_pred_sorter.GetGroupsSpan();
    uint32_t ngroups = segment_pred_sorter.GetNumGroups();
    const auto &dgroup_idx = segment_pred_sorter.GetGroupSegmentsSpan();

    // First, aggregate the positive and negative precision for each group
    dh::caching_device_vector<double> total_pos(ngroups, 0);
    dh::caching_device_vector<double> total_neg(ngroups, 0);

    // Allocator to be used for managing space overhead while performing transformed reductions
    dh::XGBCachingDeviceAllocator<char> alloc;

    // Compute each elements positive precision value and reduce them across groups concurrently.
    ComputeItemPrecision pos_prec_functor(ComputeItemPrecision::PrecisionType::kPositive,
                                          ngroups, dweights, dgroup_idx, dlabels);
    auto end_range =
      thrust::reduce_by_key(thrust::cuda::par(alloc),
                            dh::tcbegin(dgroup_idx), dh::tcend(dgroup_idx),
                            thrust::make_transform_iterator(
                              // The indices need not be sequential within a group, as we care only
                              // about the sum of positive precision values within a group
                              dh::tcbegin(segment_pred_sorter.GetOriginalPositionsSpan()),
                              pos_prec_functor),
                            thrust::make_discard_iterator(),  // We don't care for the group indices
                            total_pos.begin());  // Sum of positive precision values in the group
    CHECK(end_range.second - total_pos.begin() == total_pos.size());

    // Compute each elements negative precision value and reduce them across groups concurrently.
    ComputeItemPrecision neg_prec_functor(ComputeItemPrecision::PrecisionType::kNegative,
                                          ngroups, dweights, dgroup_idx, dlabels);
    end_range =
      thrust::reduce_by_key(thrust::cuda::par(alloc),
                            dh::tcbegin(dgroup_idx), dh::tcend(dgroup_idx),
                            thrust::make_transform_iterator(
                              // The indices need not be sequential within a group, as we care only
                              // about the sum of negative precision values within a group
                              dh::tcbegin(segment_pred_sorter.GetOriginalPositionsSpan()),
                              neg_prec_functor),
                            thrust::make_discard_iterator(),  // We don't care for the group indices
                            total_neg.begin());  // Sum of negative precision values in the group
    CHECK(end_range.second - total_neg.begin() == total_neg.size());

    const auto *dtotal_pos = total_pos.data().get();
    const auto *dtotal_neg = total_neg.data().get();

    // AUC sum for each group
    dh::caching_device_vector<double> sum_auc(ngroups, 0);
    // AUC error across all groups
    dh::caching_device_vector<int> auc_error(1, 0);
    auto *dsum_auc = sum_auc.data().get();
    auto *dauc_error = auc_error.data().get();

    int device_id = -1;
    dh::safe_cuda(cudaGetDevice(&device_id));
    // For each group item compute the aggregated precision
    dh::LaunchN<1, 32>(device_id, ngroups, nullptr, [=] __device__(uint32_t gidx) {
      // We need pos > 0 && neg > 0
      if (dtotal_pos[gidx] <= 0.0 || dtotal_neg[gidx] <= 0.0) {
        atomicAdd(dauc_error, 1);
      } else {
        auto gbegin = dgroups[gidx];
        auto gend = dgroups[gidx + 1];
        // Calculate AUC
        double tp = 0.0, prevtp = 0.0, fp = 0.0, prevfp = 0.0, h = 0.0, a = 0.0, b = 0.0;
        for (auto i = gbegin; i < gend; ++i) {
          const auto wt = dweights == nullptr ? 1.0f
                                              : dweights[ngroups == 1 ? dpreds_orig_pos[i] : gidx];
          tp += wt * dlabels[dpreds_orig_pos[i]];
          fp += wt * (1.0f - dlabels[dpreds_orig_pos[i]]);
          if ((i < gend - 1 && dsorted_preds[i] != dsorted_preds[i + 1]) || (i == gend - 1)) {
            if (tp == prevtp) {
              a = 1.0;
              b = 0.0;
            } else {
              h = (fp - prevfp) / (tp - prevtp);
              a = 1.0 + h;
              b = (prevfp - h * prevtp) / dtotal_pos[gidx];
            }
            if (0.0 != b) {
              dsum_auc[gidx] += (tp / dtotal_pos[gidx] - prevtp / dtotal_pos[gidx] -
                                 b / a * (std::log(a * tp / dtotal_pos[gidx] + b) -
                                          std::log(a * prevtp / dtotal_pos[gidx] + b))) / a;
            } else {
              dsum_auc[gidx] += (tp / dtotal_pos[gidx] - prevtp / dtotal_pos[gidx]) / a;
            }
            prevtp = tp;
            prevfp = fp;
          }
        }

        // Sanity check
        if (tp < 0 || prevtp < 0 || fp < 0 || prevfp < 0) {
          // Check if we have any metric error thus far
          auto current_auc_error = atomicAdd(dauc_error, 0);
          KERNEL_CHECK(!current_auc_error);
        }
      }
    });

    const auto hsum_auc = thrust::reduce(thrust::cuda::par(alloc), sum_auc.begin(), sum_auc.end());
    const auto hauc_error = auc_error.back();  // Copy it back to host

    // Report average AUC-PR across all groups
    // In distributed mode, workers which only contains pos or neg samples
    // will be ignored when aggregate AUC-PR.
    bst_float dat[2] = {0.0f, 0.0f};
    if (hauc_error < static_cast<int>(ngroups)) {
      dat[0] = static_cast<bst_float>(hsum_auc);
      dat[1] = static_cast<bst_float>(static_cast<int>(ngroups) - hauc_error);
    }
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    CHECK_GT(dat[1], 0.0f)
      << "AUC-PR: the dataset only contains pos or neg samples";
    CHECK_LE(dat[0], dat[1]) << "AUC-PR: AUC > 1.0";
    return dat[0] / dat[1];
  }

  const char* Name() const override {
    return "aucpr";
  }
};

XGBOOST_REGISTER_GPU_METRIC(AucGpu, "auc")
.describe("Area under curve for rank computed on GPU.")
.set_body([](const char* param) { return new EvalAucGpu(); });

XGBOOST_REGISTER_GPU_METRIC(AucPRGpu, "aucpr")
.describe("Area under PR curve for rank computed on GPU.")
.set_body([](const char* param) { return new EvalAucPRGpu(); });

XGBOOST_REGISTER_GPU_METRIC(PrecisionGpu, "pre")
.describe("precision@k for rank computed on GPU.")
.set_body([](const char* param) { return new EvalRankGpu<EvalPrecisionGpu>("pre", param); });

XGBOOST_REGISTER_GPU_METRIC(NDCGGpu, "ndcg")
.describe("ndcg@k for rank computed on GPU.")
.set_body([](const char* param) { return new EvalRankGpu<EvalNDCGGpu>("ndcg", param); });

XGBOOST_REGISTER_GPU_METRIC(MAPGpu, "map")
.describe("map@k for rank computed on GPU.")
.set_body([](const char* param) { return new EvalRankGpu<EvalMAPGpu>("map", param); });
}  // namespace metric
}  // namespace xgboost
