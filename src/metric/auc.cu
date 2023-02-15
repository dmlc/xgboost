/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#include <thrust/scan.h>

#include <algorithm>
#include <cassert>
#include <cub/cub.cuh>  // NOLINT
#include <limits>
#include <memory>
#include <tuple>
#include <utility>

#include "../collective/device_communicator.cuh"
#include "../common/algorithm.cuh"        // SegmentedArgSort
#include "../common/optional_weight.h"    // OptionalWeights
#include "../common/threading_utils.cuh"  // UnravelTrapeziodIdx,SegmentedTrapezoidThreads
#include "auc.h"
#include "xgboost/data.h"
#include "xgboost/span.h"

namespace xgboost {
namespace metric {
// tag the this file, used by force static link later.
DMLC_REGISTRY_FILE_TAG(auc_gpu);

namespace {
// Pair of FP/TP
using Pair = thrust::pair<double, double>;

template <typename T, typename U, typename P = thrust::pair<T, U>>
struct PairPlus : public thrust::binary_function<P, P, P> {
  XGBOOST_DEVICE P operator()(P const& l, P const& r) const {
    return thrust::make_pair(l.first + r.first, l.second + r.second);
  }
};
}  // namespace

/**
 * A cache to GPU data to avoid reallocating memory.
 */
struct DeviceAUCCache {
  // index sorted by prediction value
  dh::device_vector<size_t> sorted_idx;
  // track FP/TP for computation on trapezoid area
  dh::device_vector<Pair> fptp;
  // track FP_PREV/TP_PREV for computation on trapezoid area
  dh::device_vector<Pair> neg_pos;
  // index of unique prediction values.
  dh::device_vector<size_t> unique_idx;
  // p^T: transposed prediction matrix, used by MultiClassAUC
  dh::device_vector<float> predts_t;

  void Init(common::Span<float const> predts, bool is_multi) {
    if (sorted_idx.size() != predts.size()) {
      sorted_idx.resize(predts.size());
      fptp.resize(sorted_idx.size());
      unique_idx.resize(sorted_idx.size());
      neg_pos.resize(sorted_idx.size());
      if (is_multi) {
        predts_t.resize(sorted_idx.size());
      }
    }
  }
};

template <bool is_multi>
void InitCacheOnce(common::Span<float const> predts, std::shared_ptr<DeviceAUCCache> *p_cache) {
  auto& cache = *p_cache;
  if (!cache) {
    cache.reset(new DeviceAUCCache);
  }
  cache->Init(predts, is_multi);
}

/**
 * The GPU implementation uses same calculation as CPU with a few more steps to distribute
 * work across threads:
 *
 * - Run scan to obtain TP/FP values, which are right coordinates of trapezoid.
 * - Find distinct prediction values and get the corresponding FP_PREV/TP_PREV value,
 *   which are left coordinates of trapezoids.
 * - Reduce the scan array into 1 AUC value.
 */
template <typename Fn>
std::tuple<double, double, double>
GPUBinaryAUC(common::Span<float const> predts, MetaInfo const &info,
             int32_t device, common::Span<size_t const> d_sorted_idx,
             Fn area_fn, std::shared_ptr<DeviceAUCCache> cache) {
  auto labels = info.labels.View(device);
  auto weights = info.weights_.ConstDeviceSpan();
  dh::safe_cuda(cudaSetDevice(device));

  CHECK_NE(labels.Size(), 0);
  CHECK_EQ(labels.Size(), predts.size());

  /**
   * Linear scan
   */
  auto get_weight = common::OptionalWeights{weights};
  auto get_fp_tp = [=]XGBOOST_DEVICE(size_t i) {
    size_t idx = d_sorted_idx[i];

    float label = labels(idx);
    float w = get_weight[d_sorted_idx[i]];

    float fp = (1.0 - label) * w;
    float tp = label * w;

    return thrust::make_pair(fp, tp);
  };  // NOLINT
  auto d_fptp = dh::ToSpan(cache->fptp);
  dh::LaunchN(d_sorted_idx.size(),
              [=] XGBOOST_DEVICE(size_t i) { d_fptp[i] = get_fp_tp(i); });

  dh::XGBDeviceAllocator<char> alloc;
  auto d_unique_idx = dh::ToSpan(cache->unique_idx);
  dh::Iota(d_unique_idx);

  auto uni_key = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0),
      [=] XGBOOST_DEVICE(size_t i) { return predts[d_sorted_idx[i]]; });
  auto end_unique = thrust::unique_by_key_copy(
      thrust::cuda::par(alloc), uni_key, uni_key + d_sorted_idx.size(),
      dh::tbegin(d_unique_idx), thrust::make_discard_iterator(),
      dh::tbegin(d_unique_idx));
  d_unique_idx = d_unique_idx.subspan(0, end_unique.second - dh::tbegin(d_unique_idx));

  dh::InclusiveScan(dh::tbegin(d_fptp), dh::tbegin(d_fptp),
                    PairPlus<double, double>{}, d_fptp.size());

  auto d_neg_pos = dh::ToSpan(cache->neg_pos);
  // scatter unique negaive/positive values
  // shift to right by 1 with initial value being 0
  dh::LaunchN(d_unique_idx.size(), [=] XGBOOST_DEVICE(size_t i) {
    if (d_unique_idx[i] == 0) {  // first unique index is 0
      assert(i == 0);
      d_neg_pos[0] = {0, 0};
      return;
    }
    d_neg_pos[d_unique_idx[i]] = d_fptp[d_unique_idx[i] - 1];
    if (i == d_unique_idx.size() - 1) {
      // last one needs to be included, may override above assignment if the last
      // prediction value is distinct from previous one.
      d_neg_pos.back() = d_fptp[d_unique_idx[i] - 1];
      return;
    }
  });

  auto in = dh::MakeTransformIterator<double>(
      thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(size_t i) {
        double fp, tp;
        double fp_prev, tp_prev;
        if (i == 0) {
          // handle the last element
          thrust::tie(fp, tp) = d_fptp.back();
          thrust::tie(fp_prev, tp_prev) = d_neg_pos[d_unique_idx.back()];
        } else {
          thrust::tie(fp, tp) = d_fptp[d_unique_idx[i] - 1];
          thrust::tie(fp_prev, tp_prev) = d_neg_pos[d_unique_idx[i - 1]];
        }
        return area_fn(fp_prev, fp, tp_prev, tp);
      });

  Pair last = cache->fptp.back();
  double auc = thrust::reduce(thrust::cuda::par(alloc), in, in + d_unique_idx.size());
  return std::make_tuple(last.first, last.second, auc);
}

std::tuple<double, double, double> GPUBinaryROCAUC(common::Span<float const> predts,
                                                   MetaInfo const &info, std::int32_t device,
                                                   std::shared_ptr<DeviceAUCCache> *p_cache) {
  auto &cache = *p_cache;
  InitCacheOnce<false>(predts, p_cache);

  /**
   * Create sorted index for each class
   */
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);
  dh::ArgSort<false>(predts, d_sorted_idx);
  // Create lambda to avoid pass function pointer.
  return GPUBinaryAUC(
      predts, info, device, d_sorted_idx,
      [] XGBOOST_DEVICE(double x0, double x1, double y0, double y1) -> double {
        return TrapezoidArea(x0, x1, y0, y1);
      },
      cache);
}

void Transpose(common::Span<float const> in, common::Span<float> out, size_t m,
               size_t n) {
  CHECK_EQ(in.size(), out.size());
  CHECK_EQ(in.size(), m * n);
  dh::LaunchN(in.size(), [=] XGBOOST_DEVICE(size_t i) {
    size_t col = i / m;
    size_t row = i % m;
    size_t idx = row * n + col;
    out[i] = in[idx];
  });
}

double ScaleClasses(common::Span<double> results, common::Span<double> local_area,
                    common::Span<double> tp, common::Span<double> auc, size_t n_classes) {
  dh::XGBDeviceAllocator<char> alloc;
  if (collective::IsDistributed()) {
    int32_t device = dh::CurrentDevice();
    CHECK_EQ(dh::CudaGetPointerDevice(results.data()), device);
    auto* communicator = collective::Communicator::GetDevice(device);
    communicator->AllReduceSum(results.data(), results.size());
  }
  auto reduce_in = dh::MakeTransformIterator<Pair>(
      thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(size_t i) {
        if (local_area[i] > 0) {
          return thrust::make_pair(auc[i] / local_area[i] * tp[i], tp[i]);
        }
        return thrust::make_pair(std::numeric_limits<double>::quiet_NaN(), 0.0);
      });

  double tp_sum;
  double auc_sum;
  thrust::tie(auc_sum, tp_sum) =
      thrust::reduce(thrust::cuda::par(alloc), reduce_in, reduce_in + n_classes,
                     Pair{0.0, 0.0}, PairPlus<double, double>{});
  if (tp_sum != 0 && !std::isnan(auc_sum)) {
    auc_sum /= tp_sum;
  } else {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return auc_sum;
}

/**
 * Calculate FP/TP for multi-class and PR-AUC ranking. `segment_id` is a function for
 * getting class id or group id given scan index.
 */
template <typename Fn>
void SegmentedFPTP(common::Span<Pair> d_fptp, Fn segment_id) {
  using Triple = thrust::tuple<uint32_t, double, double>;
  // expand to tuple to include idx
  auto fptp_it_in = dh::MakeTransformIterator<Triple>(
      thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(size_t i) {
        return thrust::make_tuple(i, d_fptp[i].first, d_fptp[i].second);
      });
  // shrink down to pair
  auto fptp_it_out = thrust::make_transform_output_iterator(
      dh::TypedDiscard<Triple>{}, [d_fptp] XGBOOST_DEVICE(Triple const &t) {
        d_fptp[thrust::get<0>(t)] =
            thrust::make_pair(thrust::get<1>(t), thrust::get<2>(t));
        return t;
      });
  dh::InclusiveScan(
      fptp_it_in, fptp_it_out,
      [=] XGBOOST_DEVICE(Triple const &l, Triple const &r) {
        uint32_t l_gid = segment_id(thrust::get<0>(l));
        uint32_t r_gid = segment_id(thrust::get<0>(r));
        if (l_gid != r_gid) {
          return r;
        }

        return Triple(thrust::get<0>(r),
                      thrust::get<1>(l) + thrust::get<1>(r),   // fp
                      thrust::get<2>(l) + thrust::get<2>(r));  // tp
      },
      d_fptp.size());
}

/**
 * Reduce the values of AUC for each group/class.
 */
template <typename Area, typename Seg>
void SegmentedReduceAUC(common::Span<size_t const> d_unique_idx,
                        common::Span<uint32_t const> d_class_ptr,
                        common::Span<uint32_t const> d_unique_class_ptr,
                        std::shared_ptr<DeviceAUCCache> cache,
                        Area area_fn,
                        Seg segment_id,
                        common::Span<double> d_auc) {
  auto d_fptp = dh::ToSpan(cache->fptp);
  auto d_neg_pos = dh::ToSpan(cache->neg_pos);
  dh::XGBDeviceAllocator<char> alloc;
  auto key_in = dh::MakeTransformIterator<uint32_t>(
      thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(size_t i) {
        size_t class_id = segment_id(d_unique_idx[i]);
        return class_id;
      });
  auto val_in = dh::MakeTransformIterator<double>(
      thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(size_t i) {
        size_t class_id = segment_id(d_unique_idx[i]);

        double fp, tp, fp_prev, tp_prev;
        if (i == d_unique_class_ptr[class_id]) {
          // first item is ignored, we use this thread to calculate the last item
          thrust::tie(fp, tp) = d_fptp[common::LastOf(class_id, d_class_ptr)];
          thrust::tie(fp_prev, tp_prev) =
              d_neg_pos[d_unique_idx[common::LastOf(class_id, d_unique_class_ptr)]];
        } else {
          thrust::tie(fp, tp) = d_fptp[d_unique_idx[i] - 1];
          thrust::tie(fp_prev, tp_prev) = d_neg_pos[d_unique_idx[i - 1]];
        }
        double auc = area_fn(fp_prev, fp, tp_prev, tp, class_id);
        return auc;
      });
  thrust::reduce_by_key(thrust::cuda::par(alloc), key_in,
                        key_in + d_unique_idx.size(), val_in,
                        thrust::make_discard_iterator(), dh::tbegin(d_auc));
}

/**
 * MultiClass implementation is similar to binary classification, except we need to split
 * up each class in all kernels.
 */
template <bool scale, typename Fn>
double GPUMultiClassAUCOVR(MetaInfo const &info, int32_t device, common::Span<uint32_t> d_class_ptr,
                           size_t n_classes, std::shared_ptr<DeviceAUCCache> cache, Fn area_fn) {
  dh::safe_cuda(cudaSetDevice(device));
  /**
   * Sorted idx
   */
  auto d_predts_t = dh::ToSpan(cache->predts_t);
  // Index is sorted within class.
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);

  auto labels = info.labels.View(device);
  auto weights = info.weights_.ConstDeviceSpan();

  size_t n_samples = labels.Shape(0);

  if (n_samples == 0) {
    dh::TemporaryArray<double> resutls(n_classes * 4, 0.0f);
    auto d_results = dh::ToSpan(resutls);
    dh::LaunchN(n_classes * 4,
                [=] XGBOOST_DEVICE(size_t i) { d_results[i] = 0.0f; });
    auto local_area = d_results.subspan(0, n_classes);
    auto tp = d_results.subspan(2 * n_classes, n_classes);
    auto auc = d_results.subspan(3 * n_classes, n_classes);
    return ScaleClasses(d_results, local_area, tp, auc, n_classes);
  }

  /**
   * Linear scan
   */
  dh::caching_device_vector<double> d_auc(n_classes, 0);
  auto get_weight = common::OptionalWeights{weights};
  auto d_fptp = dh::ToSpan(cache->fptp);
  auto get_fp_tp = [=]XGBOOST_DEVICE(size_t i) {
    size_t idx = d_sorted_idx[i];

    size_t class_id = i / n_samples;
    // labels is a vector of size n_samples.
    float label = labels(idx % n_samples) == class_id;

    float w = get_weight[d_sorted_idx[i] % n_samples];
    float fp = (1.0 - label) * w;
    float tp = label * w;
    return thrust::make_pair(fp, tp);
  };  // NOLINT
  dh::LaunchN(d_sorted_idx.size(),
              [=] XGBOOST_DEVICE(size_t i) { d_fptp[i] = get_fp_tp(i); });

  /**
   *  Handle duplicated predictions
   */
  dh::XGBDeviceAllocator<char> alloc;
  auto d_unique_idx = dh::ToSpan(cache->unique_idx);
  dh::Iota(d_unique_idx);
  auto uni_key = dh::MakeTransformIterator<thrust::pair<uint32_t, float>>(
      thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(size_t i) {
        uint32_t class_id = i / n_samples;
        float predt = d_predts_t[d_sorted_idx[i]];
        return thrust::make_pair(class_id, predt);
      });

  // unique values are sparse, so we need a CSR style indptr
  dh::TemporaryArray<uint32_t> unique_class_ptr(d_class_ptr.size());
  auto d_unique_class_ptr = dh::ToSpan(unique_class_ptr);
  auto n_uniques = dh::SegmentedUniqueByKey(
      thrust::cuda::par(alloc),
      dh::tbegin(d_class_ptr),
      dh::tend(d_class_ptr),
      uni_key,
      uni_key + d_sorted_idx.size(),
      dh::tbegin(d_unique_idx),
      d_unique_class_ptr.data(),
      dh::tbegin(d_unique_idx),
      thrust::equal_to<thrust::pair<uint32_t, float>>{});
  d_unique_idx = d_unique_idx.subspan(0, n_uniques);

  auto get_class_id = [=] XGBOOST_DEVICE(size_t idx) { return idx / n_samples; };
  SegmentedFPTP(d_fptp, get_class_id);

  // scatter unique FP_PREV/TP_PREV values
  auto d_neg_pos = dh::ToSpan(cache->neg_pos);
  // When dataset is not empty, each class must have at least 1 (unique) sample
  // prediction, so no need to handle special case.
  dh::LaunchN(d_unique_idx.size(), [=] XGBOOST_DEVICE(size_t i) {
    if (d_unique_idx[i] % n_samples == 0) {  // first unique index is 0
      assert(d_unique_idx[i] % n_samples == 0);
      d_neg_pos[d_unique_idx[i]] = {0, 0};   // class_id * n_samples = i
      return;
    }
    uint32_t class_id = d_unique_idx[i] / n_samples;
    d_neg_pos[d_unique_idx[i]] = d_fptp[d_unique_idx[i] - 1];
    if (i == common::LastOf(class_id, d_unique_class_ptr)) {
      // last one needs to be included.
      size_t last = d_unique_idx[common::LastOf(class_id, d_unique_class_ptr)];
      d_neg_pos[common::LastOf(class_id, d_class_ptr)] = d_fptp[last - 1];
      return;
    }
  });

  /**
   * Reduce the result for each class
   */
  auto s_d_auc = dh::ToSpan(d_auc);
  SegmentedReduceAUC(d_unique_idx, d_class_ptr, d_unique_class_ptr, cache,
                     area_fn, get_class_id, s_d_auc);

  /**
   * Scale the classes with number of samples for each class.
   */
  dh::TemporaryArray<double> resutls(n_classes * 4);
  auto d_results = dh::ToSpan(resutls);
  auto local_area = d_results.subspan(0, n_classes);
  auto fp = d_results.subspan(n_classes, n_classes);
  auto tp = d_results.subspan(2 * n_classes, n_classes);
  auto auc = d_results.subspan(3 * n_classes, n_classes);

  dh::LaunchN(n_classes, [=] XGBOOST_DEVICE(size_t c) {
    auc[c] = s_d_auc[c];
    auto last = d_fptp[n_samples * c + (n_samples - 1)];
    fp[c] = last.first;
    if (scale) {
      local_area[c] = last.first * last.second;
      tp[c] = last.second;
    } else {
      local_area[c] = 1.0f;
      tp[c] = 1.0f;
    }
  });
  return ScaleClasses(d_results, local_area, tp, auc, n_classes);
}

void MultiClassSortedIdx(Context const *ctx, common::Span<float const> predts,
                         common::Span<uint32_t> d_class_ptr,
                         std::shared_ptr<DeviceAUCCache> cache) {
  size_t n_classes = d_class_ptr.size() - 1;
  auto d_predts_t = dh::ToSpan(cache->predts_t);
  auto n_samples = d_predts_t.size() / n_classes;
  if (n_samples == 0) {
    return;
  }
  Transpose(predts, d_predts_t, n_samples, n_classes);
  dh::LaunchN(n_classes + 1,
              [=] XGBOOST_DEVICE(size_t i) { d_class_ptr[i] = i * n_samples; });
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);
  common::SegmentedArgSort<false, false>(ctx, d_predts_t, d_class_ptr, d_sorted_idx);
}

double GPUMultiClassROCAUC(Context const *ctx, common::Span<float const> predts,
                           MetaInfo const &info, std::shared_ptr<DeviceAUCCache> *p_cache,
                           std::size_t n_classes) {
  auto& cache = *p_cache;
  InitCacheOnce<true>(predts, p_cache);

  /**
   * Create sorted index for each class
   */
  dh::TemporaryArray<uint32_t> class_ptr(n_classes + 1, 0);
  MultiClassSortedIdx(ctx, predts, dh::ToSpan(class_ptr), cache);

  auto fn = [] XGBOOST_DEVICE(double fp_prev, double fp, double tp_prev,
                              double tp, size_t /*class_id*/) {
    return TrapezoidArea(fp_prev, fp, tp_prev, tp);
  };
  return GPUMultiClassAUCOVR<true>(info, ctx->gpu_id, dh::ToSpan(class_ptr), n_classes, cache, fn);
}

namespace {
struct RankScanItem {
  size_t idx;
  double predt;
  double w;
  bst_group_t group_id;
};
}  // anonymous namespace

std::pair<double, std::uint32_t> GPURankingAUC(Context const *ctx, common::Span<float const> predts,
                                               MetaInfo const &info,
                                               std::shared_ptr<DeviceAUCCache> *p_cache) {
  auto& cache = *p_cache;
  InitCacheOnce<false>(predts, p_cache);

  dh::caching_device_vector<bst_group_t> group_ptr(info.group_ptr_);
  dh::XGBCachingDeviceAllocator<char> alloc;

  auto d_group_ptr = dh::ToSpan(group_ptr);
  /**
   * Validate the dataset
   */
  auto check_it = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0),
      [=] XGBOOST_DEVICE(size_t i) { return d_group_ptr[i + 1] - d_group_ptr[i]; });
  size_t n_valid = thrust::count_if(
      thrust::cuda::par(alloc), check_it, check_it + group_ptr.size() - 1,
      [=] XGBOOST_DEVICE(size_t len) { return len >= 3; });
  if (n_valid < info.group_ptr_.size() - 1) {
    InvalidGroupAUC();
  }
  if (n_valid == 0) {
    return std::make_pair(0.0, 0);
  }

  /**
   * Sort the labels
   */
  auto d_labels = info.labels.View(ctx->gpu_id);

  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);
  common::SegmentedArgSort<false, false>(ctx, d_labels.Values(), d_group_ptr, d_sorted_idx);

  auto d_weights = info.weights_.ConstDeviceSpan();

  dh::caching_device_vector<size_t> threads_group_ptr(group_ptr.size(), 0);
  auto d_threads_group_ptr = dh::ToSpan(threads_group_ptr);
  // Use max to represent triangle
  auto n_threads = common::SegmentedTrapezoidThreads(
      d_group_ptr, d_threads_group_ptr, std::numeric_limits<size_t>::max());
  CHECK_LT(n_threads, std::numeric_limits<int32_t>::max());
  // get the coordinate in nested summation
  auto get_i_j = [=]XGBOOST_DEVICE(size_t idx, size_t query_group_idx) {
    auto data_group_begin = d_group_ptr[query_group_idx];
    size_t n_samples = d_group_ptr[query_group_idx + 1] - data_group_begin;
    auto thread_group_begin = d_threads_group_ptr[query_group_idx];
    auto idx_in_thread_group = idx - thread_group_begin;

    size_t i, j;
    common::UnravelTrapeziodIdx(idx_in_thread_group, n_samples, &i, &j);
    // we use global index among all groups for sorted idx, so i, j should also be global
    // index.
    i += data_group_begin;
    j += data_group_begin;
    return thrust::make_pair(i, j);
  };  // NOLINT
  auto in = dh::MakeTransformIterator<RankScanItem>(
      thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(size_t idx) {
        bst_group_t query_group_idx = dh::SegmentId(d_threads_group_ptr, idx);
        auto data_group_begin = d_group_ptr[query_group_idx];
        size_t n_samples = d_group_ptr[query_group_idx + 1] - data_group_begin;
        if (n_samples < 3) {
          // at least 3 documents are required.
          return RankScanItem{idx, 0, 0, query_group_idx};
        }

        size_t i, j;
        thrust::tie(i, j) = get_i_j(idx, query_group_idx);

        float predt = predts[d_sorted_idx[i]] - predts[d_sorted_idx[j]];
        float w = common::Sqr(d_weights.empty() ? 1.0f : d_weights[query_group_idx]);
        if (predt > 0) {
          predt = 1.0;
        } else if (predt == 0) {
          predt = 0.5;
        } else {
          predt = 0;
        }
        predt *= w;
        return RankScanItem{idx, predt, w, query_group_idx};
      });

  dh::TemporaryArray<double> d_auc(group_ptr.size() - 1);
  auto s_d_auc = dh::ToSpan(d_auc);
  auto out = thrust::make_transform_output_iterator(
      dh::TypedDiscard<RankScanItem>{},
      [=] XGBOOST_DEVICE(RankScanItem const &item) -> RankScanItem {
        auto group_id = item.group_id;
        assert(group_id < d_group_ptr.size());
        auto data_group_begin = d_group_ptr[group_id];
        size_t n_samples = d_group_ptr[group_id + 1] - data_group_begin;
        // last item of current group
        if (item.idx == common::LastOf(group_id, d_threads_group_ptr)) {
          if (item.w > 0) {
            s_d_auc[group_id] = item.predt / item.w;
          } else {
            s_d_auc[group_id] = 0;
          }
        }
        return {};  // discard
      });
  dh::InclusiveScan(
      in, out,
      [] XGBOOST_DEVICE(RankScanItem const &l, RankScanItem const &r) {
        if (l.group_id != r.group_id) {
          return r;
        }
        return RankScanItem{r.idx, l.predt + r.predt, l.w + r.w, l.group_id};
      },
      n_threads);

  /**
   * Scale the AUC with number of items in each group.
   */
  double auc = thrust::reduce(thrust::cuda::par(alloc), dh::tbegin(s_d_auc),
                              dh::tend(s_d_auc), 0.0);
  return std::make_pair(auc, n_valid);
}

std::tuple<double, double, double> GPUBinaryPRAUC(common::Span<float const> predts,
                                                  MetaInfo const &info, std::int32_t device,
                                                  std::shared_ptr<DeviceAUCCache> *p_cache) {
  auto& cache = *p_cache;
  InitCacheOnce<false>(predts, p_cache);

  /**
   * Create sorted index for each class
   */
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);
  dh::ArgSort<false>(predts, d_sorted_idx);

  auto labels = info.labels.View(device);
  auto d_weights = info.weights_.ConstDeviceSpan();
  auto get_weight = common::OptionalWeights{d_weights};
  auto it = dh::MakeTransformIterator<Pair>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) {
        auto w = get_weight[d_sorted_idx[i]];
        return thrust::make_pair(labels(d_sorted_idx[i]) * w,
                                 (1.0f - labels(d_sorted_idx[i])) * w);
      });
  dh::XGBCachingDeviceAllocator<char> alloc;
  double total_pos, total_neg;
  thrust::tie(total_pos, total_neg) =
      thrust::reduce(thrust::cuda::par(alloc), it, it + labels.Size(),
                     Pair{0.0, 0.0}, PairPlus<double, double>{});

  if (total_pos <= 0.0 || total_neg <= 0.0) {
    return {0.0f, 0.0f, 0.0f};
  }

  auto fn = [total_pos] XGBOOST_DEVICE(double fp_prev, double fp, double tp_prev,
                                       double tp) {
    return detail::CalcDeltaPRAUC(fp_prev, fp, tp_prev, tp, total_pos);
  };
  double fp, tp, auc;
  std::tie(fp, tp, auc) = GPUBinaryAUC(predts, info, device, d_sorted_idx, fn, cache);
  return std::make_tuple(1.0, 1.0, auc);
}

double GPUMultiClassPRAUC(Context const *ctx, common::Span<float const> predts,
                          MetaInfo const &info, std::shared_ptr<DeviceAUCCache> *p_cache,
                          std::size_t n_classes) {
  auto& cache = *p_cache;
  InitCacheOnce<true>(predts, p_cache);

  /**
   * Create sorted index for each class
   */
  dh::TemporaryArray<uint32_t> class_ptr(n_classes + 1, 0);
  auto d_class_ptr = dh::ToSpan(class_ptr);
  MultiClassSortedIdx(ctx, predts, d_class_ptr, cache);
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);

  auto d_weights = info.weights_.ConstDeviceSpan();

  /**
   * Get total positive/negative
   */
  auto labels = info.labels.View(ctx->gpu_id);
  auto n_samples = info.num_row_;
  dh::caching_device_vector<Pair> totals(n_classes);
  auto key_it =
      dh::MakeTransformIterator<size_t>(thrust::make_counting_iterator(0ul),
                                        [n_samples] XGBOOST_DEVICE(size_t i) {
                                          return i / n_samples;  // class id
                                        });
  auto get_weight = common::OptionalWeights{d_weights};
  auto val_it = dh::MakeTransformIterator<thrust::pair<double, double>>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) {
        auto idx = d_sorted_idx[i] % n_samples;
        auto w = get_weight[idx];
        auto class_id = i / n_samples;
        auto y = labels(idx) == class_id;
        return thrust::make_pair(y * w, (1.0f - y) * w);
      });
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::reduce_by_key(thrust::cuda::par(alloc), key_it,
                        key_it + predts.size(), val_it,
                        thrust::make_discard_iterator(), totals.begin(),
                        thrust::equal_to<size_t>{}, PairPlus<double, double>{});

  /**
   * Calculate AUC
   */
  auto d_totals = dh::ToSpan(totals);
  auto fn = [d_totals] XGBOOST_DEVICE(double fp_prev, double fp, double tp_prev,
                                      double tp, size_t class_id) {
    auto total_pos = d_totals[class_id].first;
    return detail::CalcDeltaPRAUC(fp_prev, fp, tp_prev, tp,
                                  d_totals[class_id].first);
  };
  return GPUMultiClassAUCOVR<false>(info, ctx->gpu_id, d_class_ptr, n_classes, cache, fn);
}

template <typename Fn>
std::pair<double, uint32_t>
GPURankingPRAUCImpl(common::Span<float const> predts, MetaInfo const &info,
                    common::Span<uint32_t> d_group_ptr, int32_t device,
                    std::shared_ptr<DeviceAUCCache> cache, Fn area_fn) {
  /**
   * Sorted idx
   */
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);

  auto labels = info.labels.View(device);
  auto weights = info.weights_.ConstDeviceSpan();

  uint32_t n_groups = static_cast<uint32_t>(info.group_ptr_.size() - 1);

  /**
   * Linear scan
   */
  size_t n_samples = labels.Shape(0);
  dh::caching_device_vector<double> d_auc(n_groups, 0);
  auto get_weight = common::OptionalWeights{weights};
  auto d_fptp = dh::ToSpan(cache->fptp);
  auto get_fp_tp = [=] XGBOOST_DEVICE(size_t i) {
    size_t idx = d_sorted_idx[i];

    size_t group_id = dh::SegmentId(d_group_ptr, idx);
    float label = labels(idx);

    float w = get_weight[group_id];
    float fp = (1.0 - label) * w;
    float tp = label * w;
    return thrust::make_pair(fp, tp);
  };  // NOLINT
  dh::LaunchN(d_sorted_idx.size(),
              [=] XGBOOST_DEVICE(size_t i) { d_fptp[i] = get_fp_tp(i); });

  /**
   *  Handle duplicated predictions
   */
  dh::XGBDeviceAllocator<char> alloc;
  auto d_unique_idx = dh::ToSpan(cache->unique_idx);
  dh::Iota(d_unique_idx);
  auto uni_key = dh::MakeTransformIterator<thrust::pair<uint32_t, float>>(
      thrust::make_counting_iterator(0), [=] XGBOOST_DEVICE(size_t i) {
        auto idx = d_sorted_idx[i];
        bst_group_t group_id = dh::SegmentId(d_group_ptr, idx);
        float predt = predts[idx];
        return thrust::make_pair(group_id, predt);
      });

  // unique values are sparse, so we need a CSR style indptr
  dh::TemporaryArray<uint32_t> unique_class_ptr(d_group_ptr.size());
  auto d_unique_class_ptr = dh::ToSpan(unique_class_ptr);
  auto n_uniques = dh::SegmentedUniqueByKey(
      thrust::cuda::par(alloc),
      dh::tbegin(d_group_ptr),
      dh::tend(d_group_ptr),
      uni_key,
      uni_key + d_sorted_idx.size(),
      dh::tbegin(d_unique_idx),
      d_unique_class_ptr.data(),
      dh::tbegin(d_unique_idx),
      thrust::equal_to<thrust::pair<uint32_t, float>>{});
  d_unique_idx = d_unique_idx.subspan(0, n_uniques);

  auto get_group_id = [=] XGBOOST_DEVICE(size_t idx) {
    return dh::SegmentId(d_group_ptr, idx);
  };
  SegmentedFPTP(d_fptp, get_group_id);

  // scatter unique FP_PREV/TP_PREV values
  auto d_neg_pos = dh::ToSpan(cache->neg_pos);
  dh::LaunchN(d_unique_idx.size(), [=] XGBOOST_DEVICE(size_t i) {
    if (thrust::binary_search(thrust::seq, d_unique_class_ptr.cbegin(),
                              d_unique_class_ptr.cend(),
                              i)) {  // first unique index is 0
      d_neg_pos[d_unique_idx[i]] = {0, 0};
      return;
    }
    auto group_idx = dh::SegmentId(d_group_ptr, d_unique_idx[i]);
    d_neg_pos[d_unique_idx[i]] = d_fptp[d_unique_idx[i] - 1];
    if (i == common::LastOf(group_idx, d_unique_class_ptr)) {
      // last one needs to be included.
      size_t last = d_unique_idx[common::LastOf(group_idx, d_unique_class_ptr)];
      d_neg_pos[common::LastOf(group_idx, d_group_ptr)] = d_fptp[last - 1];
      return;
    }
  });

  /**
   * Reduce the result for each group
   */
  auto s_d_auc = dh::ToSpan(d_auc);
  SegmentedReduceAUC(d_unique_idx, d_group_ptr, d_unique_class_ptr, cache,
                     area_fn, get_group_id, s_d_auc);

  /**
   * Scale the groups with number of samples for each group.
   */
  double auc;
  uint32_t invalid_groups;
  {
    auto it = dh::MakeTransformIterator<thrust::pair<double, uint32_t>>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t g) {
          double fp, tp;
          thrust::tie(fp, tp) = d_fptp[common::LastOf(g, d_group_ptr)];
          double area = fp * tp;
          auto n_documents = d_group_ptr[g + 1] - d_group_ptr[g];
          if (area > 0 && n_documents >= 2) {
            return thrust::make_pair(s_d_auc[g], static_cast<uint32_t>(0));
          }
          return thrust::make_pair(0.0, static_cast<uint32_t>(1));
        });
    thrust::tie(auc, invalid_groups) = thrust::reduce(
        thrust::cuda::par(alloc), it, it + n_groups,
        thrust::pair<double, uint32_t>(0.0, 0), PairPlus<double, uint32_t>{});
  }
  return std::make_pair(auc, n_groups - invalid_groups);
}

std::pair<double, std::uint32_t> GPURankingPRAUC(Context const *ctx,
                                                 common::Span<float const> predts,
                                                 MetaInfo const &info,
                                                 std::shared_ptr<DeviceAUCCache> *p_cache) {
  dh::safe_cuda(cudaSetDevice(ctx->gpu_id));
  if (predts.empty()) {
    return std::make_pair(0.0, static_cast<uint32_t>(0));
  }

  auto &cache = *p_cache;
  InitCacheOnce<false>(predts, p_cache);

  dh::device_vector<bst_group_t> group_ptr(info.group_ptr_.size());
  thrust::copy(info.group_ptr_.begin(), info.group_ptr_.end(), group_ptr.begin());
  auto d_group_ptr = dh::ToSpan(group_ptr);
  CHECK_GE(info.group_ptr_.size(), 1) << "Must have at least 1 query group for LTR.";
  size_t n_groups = info.group_ptr_.size() - 1;

  /**
   * Create sorted index for each group
   */
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);
  common::SegmentedArgSort<false, false>(ctx, predts, d_group_ptr, d_sorted_idx);

  dh::XGBDeviceAllocator<char> alloc;
  auto labels = info.labels.View(ctx->gpu_id);
  if (thrust::any_of(thrust::cuda::par(alloc), dh::tbegin(labels.Values()),
                     dh::tend(labels.Values()), PRAUCLabelInvalid{})) {
    InvalidLabels();
  }
  /**
   * Get total positive/negative for each group.
   */
  auto d_weights = info.weights_.ConstDeviceSpan();
  dh::caching_device_vector<thrust::pair<double, double>> totals(n_groups);
  auto key_it = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0ul),
      [=] XGBOOST_DEVICE(size_t i) { return dh::SegmentId(d_group_ptr, i); });
  auto val_it = dh::MakeTransformIterator<Pair>(
      thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) {
        float w = 1.0f;
        if (!d_weights.empty()) {
          // Avoid a binary search if the groups are not weighted.
          auto g = dh::SegmentId(d_group_ptr, i);
          w = d_weights[g];
        }
        auto y = labels(i);
        return thrust::make_pair(y * w, (1.0 - y) * w);
      });
  thrust::reduce_by_key(thrust::cuda::par(alloc), key_it,
                        key_it + predts.size(), val_it,
                        thrust::make_discard_iterator(), totals.begin(),
                        thrust::equal_to<size_t>{}, PairPlus<double, double>{});

  /**
   * Calculate AUC
   */
  auto d_totals = dh::ToSpan(totals);
  auto fn = [d_totals] XGBOOST_DEVICE(double fp_prev, double fp, double tp_prev,
                                      double tp, size_t group_id) {
    auto total_pos = d_totals[group_id].first;
    return detail::CalcDeltaPRAUC(fp_prev, fp, tp_prev, tp,
                                  d_totals[group_id].first);
  };
  return GPURankingPRAUCImpl(predts, info, d_group_ptr, ctx->gpu_id, cache, fn);
}
}  // namespace metric
}  // namespace xgboost
