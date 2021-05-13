/*!
 * Copyright 2021 by XGBoost Contributors
 */
#include <thrust/scan.h>
#include <cub/cub.cuh>
#include <cassert>
#include <limits>
#include <memory>
#include <utility>
#include <tuple>

#include "rabit/rabit.h"
#include "xgboost/span.h"
#include "xgboost/data.h"
#include "auc.h"
#include "../common/device_helpers.cuh"
#include "../common/ranking_utils.cuh"

namespace xgboost {
namespace metric {
namespace {
template <typename T>
class Discard : public thrust::discard_iterator<T>  {
 public:
  using value_type = T;  // NOLINT
};

struct GetWeightOp {
  common::Span<float const> weights;
  common::Span<size_t const> sorted_idx;

  __device__ float operator()(size_t i) const {
    return weights.empty() ? 1.0f : weights[sorted_idx[i]];
  }
};
}  // namespace

/**
 * A cache to GPU data to avoid reallocating memory.
 */
struct DeviceAUCCache {
  // Pair of FP/TP
  using Pair = thrust::pair<float, float>;
  // index sorted by prediction value
  dh::device_vector<size_t> sorted_idx;
  // track FP/TP for computation on trapesoid area
  dh::device_vector<Pair> fptp;
  // track FP_PREV/TP_PREV for computation on trapesoid area
  dh::device_vector<Pair> neg_pos;
  // index of unique prediction values.
  dh::device_vector<size_t> unique_idx;
  // p^T: transposed prediction matrix, used by MultiClassAUC
  dh::device_vector<float> predts_t;
  std::unique_ptr<dh::AllReducer> reducer;

  void Init(common::Span<float const> predts, bool is_multi, int32_t device) {
    if (sorted_idx.size() != predts.size()) {
      sorted_idx.resize(predts.size());
      fptp.resize(sorted_idx.size());
      unique_idx.resize(sorted_idx.size());
      neg_pos.resize(sorted_idx.size());
      if (is_multi) {
        predts_t.resize(sorted_idx.size());
      }
    }
    if (is_multi && !reducer) {
      reducer.reset(new dh::AllReducer);
      reducer->Init(device);
    }
  }
};

/**
 * The GPU implementation uses same calculation as CPU with a few more steps to distribute
 * work across threads:
 *
 * - Run scan to obtain TP/FP values, which are right coordinates of trapesoid.
 * - Find distinct prediction values and get the corresponding FP_PREV/TP_PREV value,
 *   which are left coordinates of trapesoid.
 * - Reduce the scan array into 1 AUC value.
 */
std::tuple<float, float, float>
GPUBinaryAUC(common::Span<float const> predts, MetaInfo const &info,
             int32_t device, std::shared_ptr<DeviceAUCCache> *p_cache) {
  auto& cache = *p_cache;
  if (!cache) {
    cache.reset(new DeviceAUCCache);
  }
  cache->Init(predts, false, device);

  auto labels = info.labels_.ConstDeviceSpan();
  auto weights = info.weights_.ConstDeviceSpan();
  dh::safe_cuda(cudaSetDevice(device));

  CHECK(!labels.empty());
  CHECK_EQ(labels.size(), predts.size());

  /**
   * Create sorted index for each class
   */
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);
  dh::ArgSort<false>(predts, d_sorted_idx);

  /**
   * Linear scan
   */
  auto get_weight = GetWeightOp{weights, d_sorted_idx};
  using Pair = thrust::pair<float, float>;
  auto get_fp_tp = [=]__device__(size_t i) {
    size_t idx = d_sorted_idx[i];

    float label = labels[idx];
    float w = get_weight(i);

    float fp = (1.0 - label) * w;
    float tp = label * w;

    return thrust::make_pair(fp, tp);
  };  // NOLINT
  auto d_fptp = dh::ToSpan(cache->fptp);
  dh::LaunchN(device, d_sorted_idx.size(),
              [=] __device__(size_t i) { d_fptp[i] = get_fp_tp(i); });

  dh::XGBDeviceAllocator<char> alloc;
  auto d_unique_idx = dh::ToSpan(cache->unique_idx);
  dh::Iota(d_unique_idx, device);

  auto uni_key = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0),
      [=] __device__(size_t i) { return predts[d_sorted_idx[i]]; });
  auto end_unique = thrust::unique_by_key_copy(
      thrust::cuda::par(alloc), uni_key, uni_key + d_sorted_idx.size(),
      dh::tbegin(d_unique_idx), thrust::make_discard_iterator(),
      dh::tbegin(d_unique_idx));
  d_unique_idx = d_unique_idx.subspan(0, end_unique.second - dh::tbegin(d_unique_idx));

  dh::InclusiveScan(
      dh::tbegin(d_fptp), dh::tbegin(d_fptp),
      [=] __device__(Pair const &l, Pair const &r) {
        return thrust::make_pair(l.first + r.first, l.second + r.second);
      },
      d_fptp.size());

  auto d_neg_pos = dh::ToSpan(cache->neg_pos);
  // scatter unique negaive/positive values
  // shift to right by 1 with initial value being 0
  dh::LaunchN(device, d_unique_idx.size(), [=] __device__(size_t i) {
    if (d_unique_idx[i] == 0) {  // first unique index is 0
      assert(i == 0);
      d_neg_pos[0] = {0, 0};
      return;
    }
    d_neg_pos[d_unique_idx[i]] = d_fptp[d_unique_idx[i] - 1];
    if (i == d_unique_idx.size() - 1) {
      // last one needs to be included, may override above assignment if the last
      // prediction value is district from previous one.
      d_neg_pos.back() = d_fptp[d_unique_idx[i] - 1];
      return;
    }
  });

  auto in = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0), [=] __device__(size_t i) {
        float fp, tp;
        float fp_prev, tp_prev;
        if (i == 0) {
          // handle the last element
          thrust::tie(fp, tp) = d_fptp.back();
          thrust::tie(fp_prev, tp_prev) = d_neg_pos[d_unique_idx.back()];
        } else {
          thrust::tie(fp, tp) = d_fptp[d_unique_idx[i] - 1];
          thrust::tie(fp_prev, tp_prev) = d_neg_pos[d_unique_idx[i - 1]];
        }
        return TrapesoidArea(fp_prev, fp, tp_prev, tp);
      });

  Pair last = cache->fptp.back();
  float auc = thrust::reduce(thrust::cuda::par(alloc), in, in + d_unique_idx.size());
  return std::make_tuple(last.first, last.second, auc);
}

void Transpose(common::Span<float const> in, common::Span<float> out, size_t m,
               size_t n, int32_t device) {
  CHECK_EQ(in.size(), out.size());
  CHECK_EQ(in.size(), m * n);
  dh::LaunchN(device, in.size(), [=] __device__(size_t i) {
    size_t col = i / m;
    size_t row = i % m;
    size_t idx = row * n + col;
    out[i] = in[idx];
  });
}

/**
 * Last index of a group in a CSR style of index pointer.
 */
template <typename Idx>
XGBOOST_DEVICE size_t LastOf(size_t group, common::Span<Idx> indptr) {
  return indptr[group + 1] - 1;
}


float ScaleClasses(common::Span<float> results, common::Span<float> local_area,
                   common::Span<float> fp, common::Span<float> tp,
                   common::Span<float> auc, std::shared_ptr<DeviceAUCCache> cache,
                   size_t n_classes) {
  dh::XGBDeviceAllocator<char> alloc;
  if (rabit::IsDistributed()) {
    CHECK_EQ(dh::CudaGetPointerDevice(results.data()), dh::CurrentDevice());
    cache->reducer->AllReduceSum(results.data(), results.data(), results.size());
  }
  auto reduce_in = dh::MakeTransformIterator<thrust::pair<float, float>>(
      thrust::make_counting_iterator(0), [=] __device__(size_t i) {
        if (local_area[i] > 0) {
          return thrust::make_pair(auc[i] / local_area[i] * tp[i], tp[i]);
        }
        return thrust::make_pair(std::numeric_limits<float>::quiet_NaN(), 0.0f);
      });

  float tp_sum;
  float auc_sum;
  thrust::tie(auc_sum, tp_sum) = thrust::reduce(
      thrust::cuda::par(alloc), reduce_in, reduce_in + n_classes,
      thrust::make_pair(0.0f, 0.0f),
      [=] __device__(auto const &l, auto const &r) {
        return thrust::make_pair(l.first + r.first, l.second + r.second);
      });
  if (tp_sum != 0 && !std::isnan(auc_sum)) {
    auc_sum /= tp_sum;
  } else {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return auc_sum;
}

/**
 * MultiClass implementation is similar to binary classification, except we need to split
 * up each class in all kernels.
 */
float GPUMultiClassAUCOVR(common::Span<float const> predts, MetaInfo const &info,
                          int32_t device, std::shared_ptr<DeviceAUCCache>* p_cache,
                          size_t n_classes) {
  dh::safe_cuda(cudaSetDevice(device));
  auto& cache = *p_cache;
  if (!cache) {
    cache.reset(new DeviceAUCCache);
  }
  cache->Init(predts, true, device);

  auto labels = info.labels_.ConstDeviceSpan();
  auto weights = info.weights_.ConstDeviceSpan();

  size_t n_samples = labels.size();

  if (n_samples == 0) {
    dh::TemporaryArray<float> resutls(n_classes * 4, 0.0f);
    auto d_results = dh::ToSpan(resutls);
    dh::LaunchN(device, n_classes * 4, [=]__device__(size_t i) {
      d_results[i] = 0.0f;
    });
    auto local_area = d_results.subspan(0, n_classes);
    auto fp = d_results.subspan(n_classes, n_classes);
    auto tp = d_results.subspan(2 * n_classes, n_classes);
    auto auc = d_results.subspan(3 * n_classes, n_classes);
    return ScaleClasses(d_results, local_area, fp, tp, auc, cache, n_classes);
  }

  /**
   * Create sorted index for each class
   */
  auto d_predts_t = dh::ToSpan(cache->predts_t);
  Transpose(predts, d_predts_t, n_samples, n_classes, device);

  dh::TemporaryArray<uint32_t> class_ptr(n_classes + 1, 0);
  auto d_class_ptr = dh::ToSpan(class_ptr);
  dh::LaunchN(device, n_classes + 1, [=]__device__(size_t i) {
    d_class_ptr[i] = i * n_samples;
  });
  // no out-of-place sort for thrust, cub sort doesn't accept general iterator. So can't
  // use transform iterator in sorting.
  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);
  dh::SegmentedArgSort<false>(d_predts_t, d_class_ptr, d_sorted_idx);

  /**
   * Linear scan
   */
  dh::caching_device_vector<float> d_auc(n_classes, 0);
  auto s_d_auc = dh::ToSpan(d_auc);
  auto get_weight = GetWeightOp{weights, d_sorted_idx};
  using Pair = thrust::pair<float, float>;
  auto d_fptp = dh::ToSpan(cache->fptp);
  auto get_fp_tp = [=]__device__(size_t i) {
    size_t idx = d_sorted_idx[i];

    size_t class_id = i / n_samples;
    // labels is a vector of size n_samples.
    float label = labels[idx % n_samples] == class_id;

    float w = get_weight(i % n_samples);
    float fp = (1.0 - label) * w;
    float tp = label * w;
    return thrust::make_pair(fp, tp);
  };  // NOLINT
  dh::LaunchN(device, d_sorted_idx.size(),
              [=] __device__(size_t i) { d_fptp[i] = get_fp_tp(i); });

  /**
   *  Handle duplicated predictions
   */
  dh::XGBDeviceAllocator<char> alloc;
  auto d_unique_idx = dh::ToSpan(cache->unique_idx);
  dh::Iota(d_unique_idx, device);
  auto uni_key = dh::MakeTransformIterator<thrust::pair<uint32_t, float>>(
      thrust::make_counting_iterator(0), [=] __device__(size_t i) {
        uint32_t class_id = i / n_samples;
        float predt = d_predts_t[d_sorted_idx[i]];
        return thrust::make_pair(class_id, predt);
      });

  // unique values are sparse, so we need a CSR style indptr
  dh::TemporaryArray<uint32_t> unique_class_ptr(class_ptr.size());
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

  using Triple = thrust::tuple<uint32_t, float, float>;
  // expand to tuple to include class id
  auto fptp_it_in = dh::MakeTransformIterator<Triple>(
      thrust::make_counting_iterator(0), [=] __device__(size_t i) {
        uint32_t class_id = i / n_samples;
        return thrust::make_tuple(class_id, d_fptp[i].first, d_fptp[i].second);
      });
  // shrink down to pair
  auto fptp_it_out = thrust::make_transform_output_iterator(
      dh::tbegin(d_fptp), [=] __device__(Triple const &t) {
        return thrust::make_pair(thrust::get<1>(t), thrust::get<2>(t));
      });
  dh::InclusiveScan(
      fptp_it_in, fptp_it_out,
      [=] __device__(Triple const &l, Triple const &r) {
        uint32_t l_cid = thrust::get<0>(l);
        uint32_t r_cid = thrust::get<0>(r);
        if (l_cid != r_cid) {
          return r;
        }

        return Triple(r_cid,                                   // class_id
                      thrust::get<1>(l) + thrust::get<1>(r),   // fp
                      thrust::get<2>(l) + thrust::get<2>(r));  // tp
      },
      d_fptp.size());

  // scatter unique FP_PREV/TP_PREV values
  auto d_neg_pos = dh::ToSpan(cache->neg_pos);
  // When dataset is not empty, each class must have at least 1 (unique) sample
  // prediction, so no need to handle special case.
  dh::LaunchN(device, d_unique_idx.size(), [=]__device__(size_t i) {
    if (d_unique_idx[i] % n_samples == 0) {  // first unique index is 0
      assert(d_unique_idx[i] % n_samples == 0);
      d_neg_pos[d_unique_idx[i]] = {0, 0};   // class_id * n_samples = i
      return;
    }
    uint32_t class_id = d_unique_idx[i] / n_samples;
    d_neg_pos[d_unique_idx[i]] = d_fptp[d_unique_idx[i] - 1];
    if (i == LastOf(class_id, d_unique_class_ptr)) {
      // last one needs to be included.
      size_t last = d_unique_idx[LastOf(class_id, d_unique_class_ptr)];
      d_neg_pos[LastOf(class_id, d_class_ptr)] = d_fptp[last - 1];
      return;
    }
  });

  /**
   * Reduce the result for each class
   */
  auto key_in = dh::MakeTransformIterator<uint32_t>(
      thrust::make_counting_iterator(0), [=] __device__(size_t i) {
        size_t class_id = d_unique_idx[i] / n_samples;
        return class_id;
      });
  auto val_in = dh::MakeTransformIterator<float>(
      thrust::make_counting_iterator(0), [=] __device__(size_t i) {
        size_t class_id = d_unique_idx[i] / n_samples;
        float fp, tp;
        float fp_prev, tp_prev;
        if (i == d_unique_class_ptr[class_id]) {
          // first item is ignored, we use this thread to calculate the last item
          thrust::tie(fp, tp) = d_fptp[class_id * n_samples + (n_samples - 1)];
          thrust::tie(fp_prev, tp_prev) =
              d_neg_pos[d_unique_idx[LastOf(class_id, d_unique_class_ptr)]];
        } else {
          thrust::tie(fp, tp) = d_fptp[d_unique_idx[i] - 1];
          thrust::tie(fp_prev, tp_prev) = d_neg_pos[d_unique_idx[i - 1]];
        }
        float auc = TrapesoidArea(fp_prev, fp, tp_prev, tp);
        return auc;
      });

  thrust::reduce_by_key(thrust::cuda::par(alloc), key_in,
                        key_in + d_unique_idx.size(), val_in,
                        thrust::make_discard_iterator(), d_auc.begin());

  /**
   * Scale the classes with number of samples for each class.
   */
  dh::TemporaryArray<float> resutls(n_classes * 4);
  auto d_results = dh::ToSpan(resutls);
  auto local_area = d_results.subspan(0, n_classes);
  auto fp = d_results.subspan(n_classes, n_classes);
  auto tp = d_results.subspan(2 * n_classes, n_classes);
  auto auc = d_results.subspan(3 * n_classes, n_classes);

  dh::LaunchN(device, n_classes, [=] __device__(size_t c) {
    auc[c] = s_d_auc[c];
    auto last = d_fptp[n_samples * c + (n_samples - 1)];
    fp[c] = last.first;
    tp[c] = last.second;
    local_area[c] = last.first * last.second;
  });
  return ScaleClasses(d_results, local_area, fp, tp, auc, cache, n_classes);
}

namespace {
struct RankScanItem {
  size_t idx;
  float predt;
  float w;
  bst_group_t group_id;
};
}  // anonymous namespace

std::pair<float, uint32_t>
GPURankingAUC(common::Span<float const> predts, MetaInfo const &info,
              int32_t device, std::shared_ptr<DeviceAUCCache> *p_cache) {
  auto& cache = *p_cache;
  if (!cache) {
    cache.reset(new DeviceAUCCache);
  }
  cache->Init(predts, false, device);

  dh::caching_device_vector<bst_group_t> group_ptr(info.group_ptr_);
  dh::XGBCachingDeviceAllocator<char> alloc;

  auto d_group_ptr = dh::ToSpan(group_ptr);
  /**
   * Validate the dataset
   */
  auto check_it = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0),
      [=] __device__(size_t i) { return d_group_ptr[i + 1] - d_group_ptr[i]; });
  size_t n_valid = thrust::count_if(
      thrust::cuda::par(alloc), check_it, check_it + group_ptr.size() - 1,
      [=] __device__(size_t len) { return len >= 3; });
  if (n_valid < info.group_ptr_.size() - 1) {
    InvalidGroupAUC();
  }
  if (n_valid == 0) {
    return std::make_pair(0.0f, 0);
  }

  /**
   * Sort the labels
   */
  auto d_labels = info.labels_.ConstDeviceSpan();

  auto d_sorted_idx = dh::ToSpan(cache->sorted_idx);
  dh::SegmentedArgSort<false>(d_labels, d_group_ptr, d_sorted_idx);

  auto d_weights = info.weights_.ConstDeviceSpan();

  dh::caching_device_vector<size_t> threads_group_ptr(group_ptr.size(), 0);
  auto d_threads_group_ptr = dh::ToSpan(threads_group_ptr);
  // Use max to represent triangle
  auto n_threads = common::SegmentedTrapezoidThreads(
      d_group_ptr, d_threads_group_ptr, std::numeric_limits<size_t>::max());
  // get the coordinate in nested summation
  auto get_i_j = [=]__device__(size_t idx, size_t query_group_idx) {
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
      thrust::make_counting_iterator(0), [=] __device__(size_t idx) {
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

  dh::TemporaryArray<float> d_auc(group_ptr.size() - 1);
  auto s_d_auc = dh::ToSpan(d_auc);
  auto out = thrust::make_transform_output_iterator(
      Discard<RankScanItem>(), [=] __device__(RankScanItem const &item) -> RankScanItem {
        auto group_id = item.group_id;
        assert(group_id < d_group_ptr.size());
        auto data_group_begin = d_group_ptr[group_id];
        size_t n_samples = d_group_ptr[group_id + 1] - data_group_begin;
        // last item of current group
        if (item.idx == LastOf(group_id, d_threads_group_ptr)) {
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
      [] __device__(RankScanItem const &l, RankScanItem const &r) {
        if (l.group_id != r.group_id) {
          return r;
        }
        return RankScanItem{r.idx, l.predt + r.predt, l.w + r.w, l.group_id};
      },
      n_threads);

  /**
   * Scale the AUC with number of items in each group.
   */
  float auc = thrust::reduce(thrust::cuda::par(alloc), dh::tbegin(s_d_auc),
                             dh::tend(s_d_auc), 0.0f);
  return std::make_pair(auc, n_valid);
}
}  // namespace metric
}  // namespace xgboost
