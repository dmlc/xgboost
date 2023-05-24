/**
 * Copyright 2018~2023 by XGBoost contributors
 */
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <xgboost/logging.h>
#include <xgboost/span.h>

#include <cstddef>  // for size_t
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "categorical.h"
#include "device_helpers.cuh"
#include "hist_util.cuh"
#include "hist_util.h"
#include "math.h"  // NOLINT
#include "quantile.h"
#include "xgboost/host_device_vector.h"

namespace xgboost::common {
constexpr float SketchContainer::kFactor;
namespace detail {
size_t RequiredSampleCutsPerColumn(int max_bins, size_t num_rows) {
  double eps = 1.0 / (WQSketch::kFactor * max_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(num_rows, eps, &dummy_nlevel, &num_cuts);
  return std::min(num_cuts, num_rows);
}

size_t RequiredSampleCuts(bst_row_t num_rows, bst_feature_t num_columns, size_t max_bins,
                          size_t nnz) {
  auto per_column = RequiredSampleCutsPerColumn(max_bins, num_rows);
  auto if_dense = num_columns * per_column;
  auto result = std::min(nnz, if_dense);
  return result;
}

std::size_t ArgSortEntryMemUsage(std::size_t n) {
  std::size_t bytes{0};
  cub_argsort::DeviceRadixSort<cub_argsort::EntryExtractor>::Argsort(
      nullptr, bytes, Span<Entry const>::const_iterator{}, static_cast<std::uint32_t*>(nullptr), n);
  return bytes;
}

std::size_t RequiredMemory(bst_row_t num_rows, bst_feature_t num_columns, size_t nnz,
                           bst_bin_t num_bins, bool with_weights, bool d_can_read) {
  std::size_t peak = 0;     // peak memory consumption
  std::size_t running = 0;  // running memory consumption

  std::size_t n_entries = std::min(nnz, num_rows * num_columns);

  if (!d_can_read) {
    // Pull data from host to device
    running += sizeof(Entry) * n_entries;
    if (with_weights) {
      // Row offset.
      running += num_rows + 1;
    }
  }
  // Allocate sorted idx
  running += sizeof(SortedIdxT) * n_entries;
  // Extra memory used by sort.
  running += ArgSortEntryMemUsage(n_entries);
  peak = std::max(peak, running);
  // Deallocate memory used by sort
  running -= ArgSortEntryMemUsage(std::min(nnz, num_rows * num_columns));
  if (with_weights) {
    // temp weight
    running += n_entries * sizeof(float);
  }
  peak = std::max(peak, running);

  // Allocate cut pointer in quantile container by increasing: n_columns + 1
  running += (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // Allocate colomn size scan by increasing: n_columns + 1
  running += (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // Allocate cuts: assuming rows is greater than bins: n_columns * limit_size
  running += RequiredSampleCuts(num_rows, num_columns, num_bins, nnz) * sizeof(SketchEntry);
  peak = std::max(peak, running);
  return peak;
}

std::size_t SketchBatchNumElements(size_t sketch_batch_num_elements, bst_row_t num_rows,
                                   bst_feature_t columns, size_t nnz, int device, size_t num_cuts,
                                   bool has_weight, bool d_can_read) {
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  // Device available memory is not accurate when rmm is used.
  return std::min(nnz, static_cast<decltype(sketch_batch_num_elements)>(
                           std::numeric_limits<std::uint32_t>::max()));
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1

  if (sketch_batch_num_elements == 0) {
    // Use up to 80% of available space
    auto avail = dh::AvailableMemory(device) * 0.8;
    nnz = std::min(num_rows * static_cast<size_t>(columns), nnz);
    std::size_t required_memory{0ul};

    if (nnz <= 2) {
      // short cut
      return kMaxNumEntrySort;
    }

    do {
      required_memory = RequiredMemory(num_rows, columns, nnz, num_cuts, has_weight, d_can_read);
      if (required_memory > avail) {
        LOG(WARNING) << "Insufficient memory, dividing the data into smaller batches.";
      }
      sketch_batch_num_elements = nnz;
      if (required_memory > avail) {
        nnz = nnz / 2;
      }
    } while (required_memory > avail && nnz >= 2);

    if (nnz <= 2) {
      LOG(WARNING) << "Unable to finish sketching due to memory limit.";
      // let it OOM.
      return kMaxNumEntrySort;
    }
  }

  return std::min(sketch_batch_num_elements, kMaxNumEntrySort);
}
}  // namespace detail

void ProcessBatch(std::int32_t device, MetaInfo const& info, const SparsePage& page,
                  std::size_t begin, std::size_t end, SketchContainer* sketch_container,
                  bst_bin_t num_cuts_per_feature, std::size_t num_columns) {
  std::size_t n = end - begin;
  dh::device_vector<Entry> tmp_entries;
  Span<Entry const> entries_view;
  if (page.data.DeviceCanRead()) {
    entries_view = page.data.ConstDeviceSpan().subspan(begin, n);
  } else {
    const auto& host_data = page.data.ConstHostVector();
    tmp_entries = dh::device_vector<Entry>(host_data.begin() + begin, host_data.begin() + end);
    entries_view = dh::ToSpan(tmp_entries);
  }

  dh::device_vector<detail::SortedIdxT> sorted_idx(n);
  detail::ArgSortEntry(std::as_const(entries_view).data(), &sorted_idx);
  auto d_sorted_idx = dh::ToSpan(sorted_idx);

  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  data::IsValidFunctor dummy_is_valid(std::numeric_limits<float>::quiet_NaN());
  auto d_sorted_entry_it =
      thrust::make_permutation_iterator(entries_view.data(), dh::tcbegin(d_sorted_idx));
  auto sorted_batch_it = dh::MakeTransformIterator<data::COOTuple>(
      d_sorted_entry_it, [=] __device__(Entry const& e) -> data::COOTuple {
        return {0, e.index, e.fvalue};  // row_idx is not needed for scaning column size.
      });

  detail::GetColumnSizesScan(device, num_columns, num_cuts_per_feature,
                             IterSpan{sorted_batch_it, sorted_idx.size()}, dummy_is_valid,
                             &cuts_ptr, &column_sizes_scan);
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();

  if (sketch_container->HasCategorical()) {
    detail::RemoveDuplicatedCategories(device, info, d_cuts_ptr, sorted_batch_it, &sorted_idx,
                                       &column_sizes_scan);
  }

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  CHECK_EQ(d_cuts_ptr.size(), column_sizes_scan.size());

  // add cuts into sketches
  sketch_container->Push(d_sorted_entry_it, dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back());
}

void ProcessWeightedBatch(int device, MetaInfo const& info, const SparsePage& page,
                          std::size_t begin, std::size_t end, SketchContainer* sketch_container,
                          bst_bin_t num_cuts_per_feature, bst_feature_t num_columns,
                          bool is_ranking, Span<bst_group_t const> d_group_ptr) {
  auto weights = info.weights_.ConstDeviceSpan();
  std::size_t n = end - begin;

  dh::device_vector<Entry> tmp_entries;
  common::Span<Entry const> entries_view;
  if (page.data.DeviceCanRead()) {
    entries_view = page.data.ConstDeviceSpan().subspan(begin, n);
  } else {
    const auto& host_data = page.data.ConstHostVector();
    tmp_entries = dh::device_vector<Entry>(host_data.begin() + begin, host_data.begin() + end);
    entries_view = dh::ToSpan(tmp_entries);
  }

  dh::device_vector<detail::SortedIdxT> sorted_idx(n);
  // Binary search to assign weights to each element
  dh::device_vector<float> temp_weights(sorted_idx.size());
  auto d_temp_weights = dh::ToSpan(temp_weights);
  page.offset.SetDevice(device);
  auto row_ptrs = page.offset.ConstDeviceSpan();
  size_t base_rowid = page.base_rowid;
  if (is_ranking) {
    CHECK_GE(d_group_ptr.size(), 2) << "Must have at least 1 group for ranking.";
    CHECK_EQ(weights.size(), d_group_ptr.size() - 1)
        << "Weight size should equal to number of groups.";
    dh::LaunchN(temp_weights.size(), [=] __device__(size_t idx) {
      std::size_t element_idx = idx + begin;
      std::size_t ridx = dh::SegmentId(row_ptrs, element_idx);
      bst_group_t group_idx = dh::SegmentId(d_group_ptr, ridx + base_rowid);
      d_temp_weights[idx] = weights[group_idx];
    });
  } else {
    dh::LaunchN(temp_weights.size(), [=] __device__(size_t idx) {
      std::size_t element_idx = idx + begin;
      std::size_t ridx = dh::SegmentId(row_ptrs, element_idx);
      d_temp_weights[idx] = weights[ridx + base_rowid];
    });
  }

  detail::ArgSortEntry(std::as_const(entries_view).data(), &sorted_idx);
  auto d_sorted_entry_it =
      thrust::make_permutation_iterator(entries_view.data(), sorted_idx.cbegin());
  auto d_sorted_weight_it =
      thrust::make_permutation_iterator(dh::tbegin(d_temp_weights), sorted_idx.cbegin());

  dh::XGBCachingDeviceAllocator<char> caching;
  thrust::inclusive_scan_by_key(
      thrust::cuda::par(caching), d_sorted_entry_it, d_sorted_entry_it + sorted_idx.size(),
      d_sorted_weight_it, d_sorted_weight_it,
      [=] __device__(const Entry& a, const Entry& b) { return a.index == b.index; });

  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  data::IsValidFunctor dummy_is_valid(std::numeric_limits<float>::quiet_NaN());
  auto sorted_batch_it = dh::MakeTransformIterator<data::COOTuple>(
      d_sorted_entry_it, [=] __device__(Entry const& e) -> data::COOTuple {
        return {0, e.index, e.fvalue};  // row_idx is not needed for scaning column size.
      });
  detail::GetColumnSizesScan(device, num_columns, num_cuts_per_feature,
                             IterSpan{sorted_batch_it, sorted_idx.size()}, dummy_is_valid,
                             &cuts_ptr, &column_sizes_scan);
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  if (sketch_container->HasCategorical()) {
    detail::RemoveDuplicatedCategories(device, info, d_cuts_ptr, sorted_batch_it, &sorted_idx,
                                       &column_sizes_scan);
  }

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();

  // Extract cuts
  sketch_container->Push(d_sorted_entry_it, dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back(), IterSpan{d_sorted_weight_it, sorted_idx.size()});
}

HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements) {
  dmat->Info().feature_types.SetDevice(device);
  dmat->Info().feature_types.ConstDevicePointer();  // pull to device early
  // Configure batch size based on available memory
  bool has_weights = dmat->Info().weights_.Size() > 0;
  size_t num_cuts_per_feature =
      detail::RequiredSampleCutsPerColumn(max_bins, dmat->Info().num_row_);

  HistogramCuts cuts;
  SketchContainer sketch_container(dmat->Info().feature_types, max_bins, dmat->Info().num_col_,
                                   dmat->Info().num_row_, device);

  dmat->Info().weights_.SetDevice(device);
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    size_t batch_nnz = batch.data.Size();
    auto const& info = dmat->Info();

    sketch_batch_num_elements = detail::SketchBatchNumElements(
        sketch_batch_num_elements, dmat->Info().num_row_, dmat->Info().num_col_,
        dmat->Info().num_nonzero_, device, num_cuts_per_feature, has_weights,
        batch.data.DeviceCanRead());

    for (auto begin = 0ull; begin < batch_nnz; begin += sketch_batch_num_elements) {
      size_t end = std::min(batch_nnz, static_cast<std::size_t>(begin + sketch_batch_num_elements));
      if (has_weights) {
        bool is_ranking = HostSketchContainer::UseGroup(dmat->Info());
        dh::caching_device_vector<uint32_t> groups(info.group_ptr_.cbegin(),
                                                   info.group_ptr_.cend());
        ProcessWeightedBatch(device, dmat->Info(), batch, begin, end, &sketch_container,
                             num_cuts_per_feature, dmat->Info().num_col_, is_ranking,
                             dh::ToSpan(groups));
      } else {
        ProcessBatch(device, dmat->Info(), batch, begin, end, &sketch_container,
                     num_cuts_per_feature, dmat->Info().num_col_);
      }
    }
  }
  sketch_container.MakeCuts(&cuts);
  return cuts;
}
}  // namespace xgboost::common
