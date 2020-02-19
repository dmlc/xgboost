/*!
 * Copyright 2015 by Contributors
 * \file rank_metric.cc
 * \brief prediction rank based metrics.
 * \author Kailong Chen, Tianqi Chen
 */
#include <rabit/rabit.h>
#include <dmlc/registry.h>

#include <cmath>
#include <vector>

#include <xgboost/metric.h>
#include <xgboost/host_device_vector.h>
#include "../common/math.h"
#include "metric_common.h"

#if defined(__CUDACC__)
#include <thrust/iterator/discard_iterator.h>
#include "../common/device_helpers.cuh"
#endif

namespace xgboost {
namespace metric {
#ifdef XGBOOST_USE_CUDA
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(rank_metric_gpu);
#endif

class EvalRankConfig {
 public:
  unsigned topn_{std::numeric_limits<unsigned>::max()};
  std::string name_;
  bool minus_{false};
};

/*! \brief Evaluate rank list */
template <typename EvalMetricT>
struct EvalRankList : public Metric, public EvalRankConfig {
 private:
#if defined(__CUDACC__)
  bst_float EvalRankOnGPU(const HostDeviceVector<bst_float> &preds,
                          const MetaInfo &info,
                          const std::vector<unsigned> &gptr) {
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
#endif
  bst_float EvalRankOnCPU(const HostDeviceVector<bst_float> &preds,
                          const MetaInfo &info,
                          const std::vector<unsigned> &gptr) {
    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);

    const auto& labels = info.labels_.ConstHostVector();
    const std::vector<bst_float>& h_preds = preds.ConstHostVector();
    // sum statistics
    double sum_metric = 0.0f;

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
        sum_metric += EvalMetricT::EvalMetric(&rec, *this);
      }
    }

    return sum_metric;
  }

 public:
  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << "label size predict size not match";
    // quick consistency when group is not available
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(preds.Size());
    const std::vector<unsigned> &gptr = info.group_ptr_.size() == 0 ? tgptr : info.group_ptr_;
    CHECK_NE(gptr.size(), 0U) << "must specify group when constructing rank file";
    CHECK_EQ(gptr.back(), preds.Size())
        << "EvalRanklist: group structure must match number of prediction";
    const auto ngroups = static_cast<bst_omp_uint>(gptr.size() - 1);
    // sum statistics
    double sum_metric = 0.0f;

#if defined(__CUDACC__)
    // Check if we have a GPU assignment; else, revert back to CPU
    auto device = tparam_->gpu_id;
    if (device >= 0) {
      sum_metric = this->EvalRankOnGPU(preds, info, gptr);
    } else {
#endif
      sum_metric = this->EvalRankOnCPU(preds, info, gptr);
#if defined(__CUDACC__)
    }
#endif

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
    return name_.c_str();
  }

  explicit EvalRankList(const char* name, const char* param) {
    using namespace std;  // NOLINT(*)
    if (param != nullptr) {
      std::ostringstream os;
      if (sscanf(param, "%u[-]?", &topn_) == 1) {
        os << name << '@' << param;
        name_ = os.str();
      } else {
        os << name << param;
        name_ = os.str();
      }
      if (param[strlen(param) - 1] == '-') {
        minus_ = true;
      }
    } else {
      name_ = name;
    }
  }
};

/*! \brief Precision at N, for both classification and rank */
struct EvalPrecision {
 public:
#if defined(__CUDACC__)
  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter, const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Group info on device
    const auto *dgroups = pred_sorter.GetGroupsPtr();
    const auto ngroups = pred_sorter.GetNumGroups();
    const auto *dgroup_idx = pred_sorter.GetGroupSegments().data().get();

    // Original positions of the predictions after they have been sorted
    const auto *dpreds_orig_pos = pred_sorter.GetOriginalPositionsPtr();

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
      if (ridx < ecfg.topn_ && DetermineNonTrivialLabelLambda(idx)) {
        atomicAdd(&dhits[group_idx], 1);
      }
    });

    // Allocator to be used for managing space overhead while performing reductions
    dh::XGBCachingDeviceAllocator<char> alloc;
    return static_cast<bst_float>(thrust::reduce(thrust::cuda::par(alloc),
                                                 hits.begin(), hits.end())) / ecfg.topn_;
  }
#endif

  static bst_float EvalMetric(PredIndPairContainer *recptr,
                              const EvalRankConfig &ecfg) {
    PredIndPairContainer &rec(*recptr);
    // calculate Precision
    std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
    unsigned nhit = 0;
    for (size_t j = 0; j < rec.size() && j < ecfg.topn_; ++j) {
      nhit += (rec[j].second != 0);
    }
    return static_cast<bst_float>(nhit) / ecfg.topn_;
  }
};

/*! \brief NDCG: Normalized Discounted Cumulative Gain at N */
struct EvalNDCG {
 private:
  static bst_float CalcDCG(const PredIndPairContainer &rec,
                           const EvalRankConfig &ecfg) {
    double sumdcg = 0.0;
    for (size_t i = 0; i < rec.size() && i < ecfg.topn_; ++i) {
      const unsigned rel = rec[i].second;
      if (rel != 0) {
        sumdcg += ((1 << rel) - 1) / std::log2(i + 2.0);
      }
    }
    return sumdcg;
  }

 public:
#if defined(__CUDACC__)
  static void ComputeDCG(const dh::SegmentSorter<float> &pred_sorter,
                         const float *dlabels,
                         const EvalRankConfig &ecfg,
                         // The order in which labels have to be accessed. The order is determined
                         // by sorting the predictions or the labels for the entire dataset
                         const uint32_t *dlabels_sort_order,
                         dh::caching_device_vector<float> *dcgptr) {
    dh::caching_device_vector<float> &dcgs(*dcgptr);
    // Group info on device
    const auto *dgroups = pred_sorter.GetGroupsPtr();
    const auto ngroups = pred_sorter.GetNumGroups();
    const auto *dgroup_idx = pred_sorter.GetGroupSegments().data().get();

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
      if (ridx < ecfg.topn_ && label) {
        atomicAdd(&ddcgs[group_idx], ((1 << label) - 1) / std::log2(ridx + 2.0));
      }
    });
  }

  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter, const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Sort the labels and compute IDCG
    dh::SegmentSorter<float> segment_label_sorter;
    segment_label_sorter.SortItems(dlabels, pred_sorter.GetNumItems(),
                                   pred_sorter.GetGroupSegments());

    uint32_t ngroups = pred_sorter.GetNumGroups();

    dh::caching_device_vector<float> idcg(ngroups, 0);
    ComputeDCG(pred_sorter, dlabels, ecfg, segment_label_sorter.GetOriginalPositionsPtr(), &idcg);

    // Compute the DCG values next
    dh::caching_device_vector<float> dcg(ngroups, 0);
    ComputeDCG(pred_sorter, dlabels, ecfg, pred_sorter.GetOriginalPositionsPtr(), &dcg);

    float *ddcg = dcg.data().get();
    float *didcg = idcg.data().get();

    int device_id = -1;
    dh::safe_cuda(cudaGetDevice(&device_id));
    // Compute the group's DCG and reduce it across all groups
    dh::LaunchN(device_id, ngroups, nullptr, [=] __device__(uint32_t gidx) {
      if (didcg[gidx] == 0.0f) {
        ddcg[gidx] = (ecfg.minus_) ? 0.0f : 1.0f;
      } else {
        ddcg[gidx] /= didcg[gidx];
      }
    });

    // Allocator to be used for managing space overhead while performing reductions
    dh::XGBCachingDeviceAllocator<char> alloc;
    return thrust::reduce(thrust::cuda::par(alloc), dcg.begin(), dcg.end());
  }
#endif
   static bst_float EvalMetric(PredIndPairContainer *recptr,
                               const EvalRankConfig &ecfg) { // NOLINT(*)
    PredIndPairContainer &rec(*recptr);
    std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
    bst_float dcg = CalcDCG(rec, ecfg);
    std::stable_sort(rec.begin(), rec.end(), common::CmpSecond);
    bst_float idcg = CalcDCG(rec, ecfg);
    if (idcg == 0.0f) {
      if (ecfg.minus_) {
        return 0.0f;
      } else {
        return 1.0f;
      }
    }
    return dcg/idcg;
  }
};

/*! \brief Mean Average Precision at N, for both classification and rank */
struct EvalMAP {
 public:
#if defined(__CUDACC__)
  static double EvalMetric(const dh::SegmentSorter<float> &pred_sorter, const float *dlabels,
                           const EvalRankConfig &ecfg) {
    // Group info on device
    const auto *dgroups = pred_sorter.GetGroupsPtr();
    const auto ngroups = pred_sorter.GetNumGroups();
    const auto *dgroup_idx = pred_sorter.GetGroupSegments().data().get();

    // Original positions of the predictions after they have been sorted
    const auto *dpreds_orig_pos = pred_sorter.GetOriginalPositionsPtr();

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
    const auto &group_segments = pred_sorter.GetGroupSegments();
    // Data segmented into different groups...
    thrust::inclusive_scan_by_key(thrust::cuda::par(alloc),
                                  group_segments.begin(), group_segments.end(),
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
        if (ridx < ecfg.topn_) {
          atomicAdd(&dsumap[group_idx],
                    static_cast<bst_float>(dhits[idx]) / (ridx + 1));
        }
      }
    });

    // Aggregate the group's item precisions
    dh::LaunchN(device_id, ngroups, nullptr, [=] __device__(uint32_t gidx) {
      auto nhits = dgroups[gidx + 1] ? dhits[dgroups[gidx + 1] - 1] : 0;
      if (nhits != 0) {
        dsumap[gidx] /= nhits;
      } else {
        if (ecfg.minus_) {
          dsumap[gidx] = 0;
        } else {
          dsumap[gidx] = 1;
        }
      }
    });

    return thrust::reduce(thrust::cuda::par(alloc), sumap.begin(), sumap.end());
  }
#endif
   static bst_float EvalMetric(PredIndPairContainer *recptr,
                               const EvalRankConfig &ecfg) {
    PredIndPairContainer &rec(*recptr);
    std::stable_sort(rec.begin(), rec.end(), common::CmpFirst);
    unsigned nhits = 0;
    double sumap = 0.0;
    for (size_t i = 0; i < rec.size(); ++i) {
      if (rec[i].second != 0) {
        nhits += 1;
        if (i < ecfg.topn_) {
          sumap += static_cast<bst_float>(nhits) / (i + 1);
        }
      }
    }
    if (nhits != 0) {
      sumap /= nhits;
      return static_cast<bst_float>(sumap);
    } else {
      if (ecfg.minus_) {
        return 0.0f;
      } else {
        return 1.0f;
      }
    }
  }
};

/*! \brief Area Under Curve, for both classification and rank */
struct EvalAuc : public Metric {
 private:
  // This is used to compute the AUC metrics on the CPU - for non-ranking tasks and
  // for training jobs that are run on the CPU. See rank_metric.cc for the need to split
  // them in two places.
  std::unique_ptr<xgboost::Metric> auc_cpu_;

 public:
#if defined(__CUDACC__)
  bst_float EvalAucOnGPU(const HostDeviceVector<bst_float> &preds,
                         const MetaInfo &info,
                         const std::vector<unsigned> &gptr,
                         bool distributed) {
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

    const auto *dsorted_preds = segment_pred_sorter.GetItemsPtr();
    const auto *dpreds_orig_pos = segment_pred_sorter.GetOriginalPositionsPtr();

    // Group info on device
    const uint32_t *dgroups = segment_pred_sorter.GetGroupsPtr();
    uint32_t ngroups = segment_pred_sorter.GetNumGroups();

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
        const auto wt = dweights == nullptr ? 1.0f
                                            : dweights[ngroups == 1 ? dpreds_orig_pos[i] : gidx];
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

    // Allocator to be used for managing space overhead while performing reductions
    dh::XGBCachingDeviceAllocator<char> alloc;
    const auto hsum_auc = thrust::reduce(thrust::cuda::par(alloc), sum_auc.begin(), sum_auc.end());
    const auto hauc_error = auc_error.back();  // Copy it back to host

    // Report average AUC across all groups
    // In distributed mode, workers which only contains pos or neg samples
    // will be ignored when aggregate AUC.
    bst_float dat[2] = {0.0f, 0.0f};
    if (hauc_error < static_cast<int>(ngroups)) {
      dat[0] = static_cast<bst_float>(hsum_auc);
      dat[1] = static_cast<bst_float>(static_cast<int>(ngroups) - hauc_error);
    }
    if (distributed) {
      rabit::Allreduce<rabit::op::Sum>(dat, 2);
    }
    CHECK_GT(dat[1], 0.0f)
      << "AUC: the dataset only contains pos or neg samples";
    return dat[0] / dat[1];
  }
#endif

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << "label size predict size not match";
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());

    const std::vector<unsigned> &gptr = info.group_ptr_.empty() ? tgptr : info.group_ptr_;
    CHECK_EQ(gptr.back(), info.labels_.Size())
        << "EvalAuc: group structure must match number of prediction";

    // For ranking task, weights are per-group
    // For binary classification task, weights are per-instance
    const bool is_ranking_task =
      !info.group_ptr_.empty() && info.weights_.Size() != info.num_row_;

#if defined(__CUDACC__)
    // Check if we have a GPU assignment; else, revert back to CPU
    auto device = tparam_->gpu_id;
    if (device >= 0 && is_ranking_task) {
      return this->EvalAucOnGPU(preds, info, gptr, distributed);
    } else {
#endif
      if (!auc_cpu_) {
        auc_cpu_.reset(xgboost::Metric::Create("auc-cpu", nullptr));
      }
      return auc_cpu_->Eval(preds, info, distributed);
#if defined(__CUDACC__)
    }
#endif
  }

  const char* Name() const override {
    return "auc";
  }
};

/*! \brief Area Under PR Curve, for both classification and rank */
struct EvalAucPR : public Metric {
 private:
  // This is used to compute the AUC PR metrics on the CPU - for non-ranking tasks and
  // for training jobs that are run on the CPU. See rank_metric.cc for the need to split
  // them in two places.
  std::unique_ptr<xgboost::Metric> aucpr_cpu_;

 public:
#if defined(__CUDACC__)
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
                                        const uint32_t *dgidxs,
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
    const uint32_t *dgidxs_;  // The group a given instance belongs to
    const float *dlabels_;  // Unsorted labels in the dataset
  };

  bst_float EvalAucPROnGPU(const HostDeviceVector<bst_float> &preds,
                           const MetaInfo &info,
                           const std::vector<unsigned> &gptr,
                           bool distributed) {
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

    const auto *dsorted_preds = segment_pred_sorter.GetItemsPtr();
    // Original positions of the predictions after they have been sorted
    const auto *dpreds_orig_pos = segment_pred_sorter.GetOriginalPositionsPtr();

    // Group info on device
    const uint32_t *dgroups = segment_pred_sorter.GetGroupsPtr();
    uint32_t ngroups = segment_pred_sorter.GetNumGroups();
    const auto *dgroup_idx = segment_pred_sorter.GetGroupSegments().data().get();
    const auto &group_segments = segment_pred_sorter.GetGroupSegments();

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
                            group_segments.begin(), group_segments.end(),
                            thrust::make_transform_iterator(
                              // The indices need not be sequential within a group, as we care only
                              // about the sum of positive precision values within a group
                              segment_pred_sorter.GetOriginalPositions().begin(),
                              pos_prec_functor),
                            thrust::make_discard_iterator(),  // We don't care for the group indices
                            total_pos.begin());  // Sum of positive precision values in the group
    CHECK(end_range.second - total_pos.begin() == total_pos.size());

    // Compute each elements negative precision value and reduce them across groups concurrently.
    ComputeItemPrecision neg_prec_functor(ComputeItemPrecision::PrecisionType::kNegative,
                                          ngroups, dweights, dgroup_idx, dlabels);
    end_range =
      thrust::reduce_by_key(thrust::cuda::par(alloc),
                            group_segments.begin(), group_segments.end(),
                            thrust::make_transform_iterator(
                              // The indices need not be sequential within a group, as we care only
                              // about the sum of negative precision values within a group
                              segment_pred_sorter.GetOriginalPositions().begin(),
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
          if ((i < gend - 1 && dsorted_preds[i] != dsorted_preds[i + 1]) || (i  == gend - 1)) {
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
#endif

  bst_float Eval(const HostDeviceVector<bst_float> &preds,
                 const MetaInfo &info,
                 bool distributed) override {
    CHECK_NE(info.labels_.Size(), 0U) << "label set cannot be empty";
    CHECK_EQ(preds.Size(), info.labels_.Size())
        << "label size predict size not match";
    std::vector<unsigned> tgptr(2, 0);
    tgptr[1] = static_cast<unsigned>(info.labels_.Size());

    const std::vector<unsigned> &gptr = info.group_ptr_.empty() ? tgptr : info.group_ptr_;
    CHECK_EQ(gptr.back(), info.labels_.Size())
        << "EvalAucPR: group structure must match number of prediction";

    // For ranking task, weights are per-group
    // For binary classification task, weights are per-instance
    const bool is_ranking_task =
      !info.group_ptr_.empty() && info.weights_.Size() != info.num_row_;

#if defined(__CUDACC__)
    // Check if we have a GPU assignment; else, revert back to CPU
    auto device = tparam_->gpu_id;
    if (device >= 0 && is_ranking_task) {
      return this->EvalAucPROnGPU(preds, info, gptr, distributed);
    } else {
#endif
      if (!aucpr_cpu_) {
        aucpr_cpu_.reset(xgboost::Metric::Create("aucpr-cpu", nullptr));
      }
      return aucpr_cpu_->Eval(preds, info, distributed);
#if defined(__CUDACC__)
    }
#endif
  }

  const char* Name() const override {
    return "aucpr";
  }
};

XGBOOST_REGISTER_METRIC(Auc, "auc")
.describe("Area under curve for both classification and rank.")
.set_body([](const char* param) { return new EvalAuc(); });

XGBOOST_REGISTER_METRIC(AucPR, "aucpr")
.describe("Area under PR curve for both classification and rank.")
.set_body([](const char* param) { return new EvalAucPR(); });

XGBOOST_REGISTER_METRIC(Precision, "pre")
.describe("precision@k for rank.")
.set_body([](const char* param) { return new EvalRankList<EvalPrecision>("pre", param); });

XGBOOST_REGISTER_METRIC(NDCG, "ndcg")
.describe("ndcg@k for rank.")
.set_body([](const char* param) { return new EvalRankList<EvalNDCG>("ndcg", param); });

XGBOOST_REGISTER_METRIC(MAP, "map")
.describe("map@k for rank.")
.set_body([](const char* param) { return new EvalRankList<EvalMAP>("map", param); });
}  // namespace metric
}  // namespace xgboost
