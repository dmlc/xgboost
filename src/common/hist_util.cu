/*!
 * Copyright 2018~2020 XGBoost contributors
 */

#include <xgboost/logging.h>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "device_helpers.cuh"
#include "hist_util.h"
#include "hist_util.cuh"
#include "math.h"  // NOLINT
#include "quantile.h"
#include "categorical.h"
#include "xgboost/host_device_vector.h"


namespace xgboost {
namespace common {

constexpr float SketchContainer::kFactor;

namespace detail {
size_t RequiredSampleCutsPerColumn(int max_bins, size_t num_rows) {
  double eps = 1.0 / (WQSketch::kFactor * max_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
      num_rows, eps, &dummy_nlevel, &num_cuts);
  return std::min(num_cuts, num_rows);
}

size_t RequiredSampleCuts(bst_row_t num_rows, bst_feature_t num_columns,
                          size_t max_bins, size_t nnz) {
  auto per_column = RequiredSampleCutsPerColumn(max_bins, num_rows);
  auto if_dense = num_columns * per_column;
  auto result = std::min(nnz, if_dense);
  return result;
}

size_t SketchBatchNumElements(size_t sketch_batch_num_elements, size_t nnz) {
  // Use total memory if compiled with rmm.
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  auto total = dh::TotalMemory(dh::CurrentDevice());
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  double total = dh::AvailableMemory(dh::CurrentDevice());
  double constexpr kGB{1024 * 1024 * 1024};
  double constexpr kRatio = 0.8;
  size_t up;
  auto factor = [](int32_t f) { return (1u << f) * kRatio; };
  if (total < kGB / 4) {                                   // 256 MB available mem
    up = std::numeric_limits<int32_t>::max() / factor(7);  // 102 MB for sorting
  } else if (total < kGB / 2) {
    up = std::numeric_limits<int32_t>::max() / factor(6);
  } else if (total < kGB) {
    up = std::numeric_limits<int32_t>::max() / factor(5);
  } else if (total < 2 * kGB) {
    up = std::numeric_limits<int32_t>::max() / factor(4);
  } else if (total < 4 * kGB) {
    up = std::numeric_limits<int32_t>::max() / factor(3);
  } else if (total < 8 * kGB) {
    up = std::numeric_limits<int32_t>::max() / factor(2);
  } else if (total < 16 * kGB) {
    up = std::numeric_limits<int32_t>::max() / factor(1);
  } else if (total < 32 * kGB) {
    up = std::numeric_limits<int32_t>::max() * kRatio;
  } else {
    up = std::numeric_limits<uint32_t>::max() * kRatio;
  }
  if (sketch_batch_num_elements == 0) {
    return std::min(nnz, up);
  } else {
    return std::min(sketch_batch_num_elements, std::min(nnz, up));
  }
}

template <typename Batch>
void SortWithWeight(Batch batch, Span<uint32_t> sorted_idx, Span<float> weights) {
  dh::XGBDeviceAllocator<char> alloc;
  thrust::sort_by_key(thrust::cuda::par(alloc), dh::tbegin(sorted_idx), dh::tend(sorted_idx),
                      dh::tbegin(weights), SortIdxOp<Batch>{batch});

  // Scan weights
  dh::XGBCachingDeviceAllocator<char> caching;
  thrust::inclusive_scan_by_key(thrust::cuda::par(caching), dh::tcbegin(sorted_idx),
                                dh::tcend(sorted_idx), dh::tbegin(weights), dh::tbegin(weights),
                                [=] __device__(uint32_t l, uint32_t r) {
                                  auto le = batch.GetElement(l);
                                  auto re = batch.GetElement(r);
                                  return le.column_idx == re.column_idx;
                                });
}

template void SortWithWeight(data::CupyAdapterBatch, Span<uint32_t>, Span<float>);
template void SortWithWeight(data::CudfAdapterBatch, Span<uint32_t>, Span<float>);
template void SortWithWeight(EntryBatch, Span<uint32_t>, Span<float>);
}  // namespace detail

void ProcessBatch(int device, MetaInfo const& info, const SparsePage& page, size_t begin,
                  size_t end, SketchContainer* sketch_container, int num_cuts_per_feature,
                  size_t num_columns) {
  dh::device_vector<Entry> sorted_entries_storage;
  Span<Entry const> sorted_entries;
  if (page.data.DeviceCanRead()) {
    const auto& device_data = page.data.ConstDevicePointer();
    sorted_entries = Span<Entry const>(device_data + begin, end - begin);
  } else {
    const auto& host_data = page.data.ConstHostVector();
    sorted_entries_storage =
        dh::device_vector<Entry>(host_data.begin() + begin, host_data.begin() + end);
    sorted_entries = dh::ToSpan(sorted_entries_storage);
  }

  dh::device_vector<uint32_t> sorted_idx(sorted_entries.size());
  dh::Iota(dh::ToSpan(sorted_idx));
  detail::EntryBatch adapter{sorted_entries};
  dh::XGBDeviceAllocator<char> alloc;
  thrust::sort(thrust::cuda::par(alloc), sorted_idx.begin(), sorted_idx.end(),
               detail::SortIdxOp<detail::EntryBatch>{adapter});

  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  data::IsValidFunctor dummy_is_valid(std::numeric_limits<float>::quiet_NaN());
  auto batch_it = dh::MakeTransformIterator<data::COOTuple>(
      sorted_entries.data(), [] __device__(Entry const& e) -> data::COOTuple {
        return {0, e.index, e.fvalue};  // row_idx is not needed for scanning column size.
      });
  detail::GetColumnSizesScan(device, num_columns, num_cuts_per_feature, batch_it, dummy_is_valid, 0,
                             sorted_entries.size(), &cuts_ptr, &column_sizes_scan);

  if (sketch_container->HasCategorical()) {
    auto d_cuts_ptr = cuts_ptr.DeviceSpan();
    detail::RemoveDuplicatedCategories(adapter, info, d_cuts_ptr, &sorted_idx, &column_sizes_scan);
  }

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  CHECK_EQ(d_cuts_ptr.size(), column_sizes_scan.size());

  // add cuts into sketches
  sketch_container->Push(adapter, dh::ToSpan(sorted_idx), dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back());
  CHECK_NE(cuts_ptr.Size(), 0);
}

void ProcessWeightedBatch(int device, const SparsePage& page,
                          MetaInfo const& info, size_t begin, size_t end,
                          SketchContainer* sketch_container, int num_cuts_per_feature,
                          size_t num_columns,
                          bool is_ranking, Span<bst_group_t const> d_group_ptr) {
  auto weights = info.weights_.ConstDeviceSpan();

  dh::device_vector<Entry> sorted_entries_storage;
  Span<Entry const> sorted_entries;
  if (page.data.DeviceCanRead()) {
    const auto& device_data = page.data.ConstDevicePointer();
    sorted_entries = Span<Entry const>(device_data + begin, end - begin);
  } else {
    const auto& host_data = page.data.ConstHostVector();
    sorted_entries_storage =
        dh::device_vector<Entry>(host_data.begin() + begin, host_data.begin() + end);
    sorted_entries = dh::ToSpan(sorted_entries_storage);
  }

  // Binary search to assign weights to each element
  dh::device_vector<float> temp_weights(sorted_entries.size());
  auto d_temp_weights = temp_weights.data().get();
  page.offset.SetDevice(device);
  auto row_ptrs = page.offset.ConstDeviceSpan();
  size_t base_rowid = page.base_rowid;
  if (is_ranking) {
    CHECK_GE(d_group_ptr.size(), 2)
        << "Must have at least 1 group for ranking.";
    CHECK_EQ(weights.size(), d_group_ptr.size() - 1)
        << "Weight size should equal to number of groups.";
    dh::LaunchN(temp_weights.size(), [=] __device__(size_t idx) {
        size_t element_idx = idx + begin;
        size_t ridx = dh::SegmentId(row_ptrs, element_idx);
        bst_group_t group_idx = dh::SegmentId(d_group_ptr, ridx + base_rowid);
        d_temp_weights[idx] = weights[group_idx];
      });
  } else {
    dh::LaunchN(temp_weights.size(), [=] __device__(size_t idx) {
        size_t element_idx = idx + begin;
        size_t ridx = dh::SegmentId(row_ptrs, element_idx);
        d_temp_weights[idx] = weights[ridx + base_rowid];
      });
  }

  dh::device_vector<uint32_t> sorted_idx(sorted_entries.size());
  detail::EntryBatch adapter{sorted_entries};
  dh::Iota(dh::ToSpan(sorted_idx));
  detail::SortWithWeight(adapter, dh::ToSpan(sorted_idx), dh::ToSpan(temp_weights));

  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  data::IsValidFunctor dummy_is_valid(std::numeric_limits<float>::quiet_NaN());
  auto batch_it = dh::MakeTransformIterator<data::COOTuple>(
      sorted_entries.data(), [] __device__(Entry const& e) -> data::COOTuple {
        return {0, e.index, e.fvalue};  // row_idx is not needed for scaning column size.
      });
  detail::GetColumnSizesScan(device, num_columns, num_cuts_per_feature,
                             batch_it, dummy_is_valid,
                             0, sorted_entries.size(),
                             &cuts_ptr, &column_sizes_scan);
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  if (sketch_container->HasCategorical()) {
    info.feature_types.SetDevice(device);
    detail::RemoveDuplicatedCategories(adapter, info, d_cuts_ptr, &sorted_idx, &column_sizes_scan);
  }
  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();

  // Extract cuts
  sketch_container->Push(adapter, dh::ToSpan(sorted_idx), dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back(), dh::ToSpan(temp_weights));
}

HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements) {
  dh::safe_cuda(cudaSetDevice(device));
  dmat->Info().feature_types.SetDevice(device);
  dmat->Info().feature_types.ConstDevicePointer();  // pull to device early
  // Configure batch size based on available memory
  bool has_weights = dmat->Info().weights_.Size() > 0;
  size_t num_cuts_per_feature =
      detail::RequiredSampleCutsPerColumn(max_bins, dmat->Info().num_row_);
  sketch_batch_num_elements =
      detail::SketchBatchNumElements(sketch_batch_num_elements, dmat->Info().num_nonzero_);

  HistogramCuts cuts;
  SketchContainer sketch_container(dmat->Info().feature_types, max_bins, dmat->Info().num_col_,
                                   dmat->Info().num_row_, device);

  dmat->Info().weights_.SetDevice(device);
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    size_t batch_nnz = batch.data.Size();
    auto const& info = dmat->Info();
    for (auto begin = 0ull; begin < batch_nnz; begin += sketch_batch_num_elements) {
      size_t end = std::min(batch_nnz, size_t(begin + sketch_batch_num_elements));
      if (has_weights) {
        bool is_ranking = HostSketchContainer::UseGroup(dmat->Info());
        dh::caching_device_vector<uint32_t> groups(info.group_ptr_.cbegin(),
                                                   info.group_ptr_.cend());
        ProcessWeightedBatch(
            device, batch, dmat->Info(), begin, end,
            &sketch_container,
            num_cuts_per_feature,
            dmat->Info().num_col_,
            is_ranking, dh::ToSpan(groups));
      } else {
        ProcessBatch(device, dmat->Info(), batch, begin, end, &sketch_container,
                     num_cuts_per_feature, dmat->Info().num_col_);
      }
    }
  }
  sketch_container.MakeCuts(&cuts);
  return cuts;
}
}  // namespace common
}  // namespace xgboost
