/*!
 * Copyright 2018 XGBoost contributors
 */

#include <xgboost/logging.h>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "../data/adapter.h"
#include "../data/device_adapter.cuh"
#include "../tree/param.h"
#include "device_helpers.cuh"
#include "hist_util.h"
#include "math.h"  // NOLINT
#include "quantile.h"
#include "xgboost/host_device_vector.h"


namespace xgboost {
namespace common {

using WQSketch = DenseCuts::WQSketch;
using SketchEntry = WQSketch::Entry;

/*!
 * \brief A container that holds the device sketches across all
 *  sparse page batches which are distributed to different devices.
 *  As sketches are aggregated by column, the mutex guards
 *  multiple devices pushing sketch summary for the same column
 *  across distinct rows.
 */
struct SketchContainer {
  std::vector<DenseCuts::WQSketch> sketches_;  // NOLINT
  static constexpr int kOmpNumColsParallelizeLimit = 1000;

  SketchContainer(int max_bin, size_t num_columns, size_t num_rows) {
    // Initialize Sketches for this dmatrix
    sketches_.resize(num_columns);
#pragma omp parallel for schedule(static) if (num_columns > kOmpNumColsParallelizeLimit)  // NOLINT
    for (int icol = 0; icol < num_columns; ++icol) {                 // NOLINT
      sketches_[icol].Init(num_rows, 1.0 / (8 * max_bin));
    }
  }

  /**
   * \brief Pushes cuts to the sketches.
   *
   * \param entries_per_column  The entries per column.
   * \param entries             Vector of cuts from all columns, length
   * entries_per_column * num_columns. \param column_scan         Exclusive scan
   * of column sizes. Used to detect cases where there are fewer entries than we
   * have storage for.
   */
  void Push(size_t entries_per_column,
            const thrust::host_vector<SketchEntry>& entries,
            const thrust::host_vector<size_t>& column_scan) {
#pragma omp parallel for schedule(static) if (sketches_.size() > SketchContainer::kOmpNumColsParallelizeLimit)  // NOLINT
    for (int icol = 0; icol < sketches_.size(); ++icol) {
      size_t column_size = column_scan[icol + 1] - column_scan[icol];
      if (column_size == 0) continue;
      WQuantileSketch<bst_float, bst_float>::SummaryContainer summary;
      size_t num_available_cuts =
          std::min(size_t(entries_per_column), column_size);
      summary.Reserve(num_available_cuts);
      summary.MakeFromSorted(&entries[entries_per_column * icol],
                             num_available_cuts);

      sketches_[icol].PushSummary(summary);
    }
  }

  // Prevent copying/assigning/moving this as its internals can't be
  // assigned/copied/moved
  SketchContainer(const SketchContainer&) = delete;
  SketchContainer(const SketchContainer&&) = delete;
  SketchContainer& operator=(const SketchContainer&) = delete;
  SketchContainer& operator=(const SketchContainer&&) = delete;
};

struct EntryCompareOp {
  __device__ bool operator()(const Entry& a, const Entry& b) {
    if (a.index == b.index) {
      return a.fvalue < b.fvalue;
    }
    return a.index < b.index;
  }
};

// Count the entries in each column and exclusive scan
void GetColumnSizesScan(int device,
                        dh::caching_device_vector<size_t>* column_sizes_scan,
                        Span<const Entry> entries, size_t num_columns) {
  column_sizes_scan->resize(num_columns + 1, 0);
  auto d_column_sizes_scan = column_sizes_scan->data().get();
  auto d_entries = entries.data();
  dh::LaunchN(device, entries.size(), [=] __device__(size_t idx) {
    auto& e = d_entries[idx];
    atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                  &d_column_sizes_scan[e.index]),
              static_cast<unsigned long long>(1));  // NOLINT
  });
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan->begin(),
                         column_sizes_scan->end(), column_sizes_scan->begin());
}

/**
 * \brief Extracts the cuts from sorted data.
 *
 * \param device                The device.
 * \param cuts                  Output cuts
 * \param num_cuts_per_feature  Number of cuts per feature.
 * \param sorted_data           Sorted entries in segments of columns
 * \param column_sizes_scan     Describes the boundaries of column segments in
 * sorted data
 */
void ExtractCuts(int device, Span<SketchEntry> cuts,
                 size_t num_cuts_per_feature, Span<Entry> sorted_data,
                 Span<size_t> column_sizes_scan) {
  dh::LaunchN(device, cuts.size(), [=] __device__(size_t idx) {
    // Each thread is responsible for obtaining one cut from the sorted input
    size_t column_idx = idx / num_cuts_per_feature;
    size_t column_size =
        column_sizes_scan[column_idx + 1] - column_sizes_scan[column_idx];
    size_t num_available_cuts =
        min(size_t(num_cuts_per_feature), column_size);
    size_t cut_idx = idx % num_cuts_per_feature;
    if (cut_idx >= num_available_cuts) return;

    Span<Entry> column_entries =
        sorted_data.subspan(column_sizes_scan[column_idx], column_size);

    size_t rank = (column_entries.size() * cut_idx) / num_available_cuts;
    auto value = column_entries[rank].fvalue;
    cuts[idx] = SketchEntry(rank, rank + 1, 1, value);
  });
}

/**
* \brief Extracts the cuts from sorted data, considering weights.
*
* \param device                The device.
* \param cuts                  Output cuts
* \param num_cuts_per_feature  Number of cuts per feature.
* \param sorted_data           Sorted entries in segments of columns
* \param column_sizes_scan     Describes the boundaries of column segments in
* sorted data
*/
void ExtractWeightedCuts(int device, Span<SketchEntry> cuts,
                         size_t num_cuts_per_feature, Span<Entry> sorted_data,
                         Span<float> weights_scan,
                         Span<size_t> column_sizes_scan) {
  dh::LaunchN(device, cuts.size(), [=] __device__(size_t idx) {
    // Each thread is responsible for obtaining one cut from the sorted input
    size_t column_idx = idx / num_cuts_per_feature;
    size_t column_size =
        column_sizes_scan[column_idx + 1] - column_sizes_scan[column_idx];
    size_t num_available_cuts =
        min(size_t(num_cuts_per_feature), column_size);
    size_t cut_idx = idx % num_cuts_per_feature;
    if (cut_idx >= num_available_cuts) return;

    Span<Entry> column_entries =
        sorted_data.subspan(column_sizes_scan[column_idx], column_size);
    Span<float> column_weights =
        weights_scan.subspan(column_sizes_scan[column_idx], column_size);

    float total_column_weight = column_weights.back();
    size_t sample_idx = 0;
    if (cut_idx == 0) {
      // First cut
      sample_idx = 0;
    } else if (cut_idx == num_available_cuts - 1) {
      // Last cut
      sample_idx = column_entries.size() - 1;
    } else if (num_available_cuts == column_size) {
      // There are less samples available than our buffer
      // Take every available sample
      sample_idx = cut_idx;
    } else {
      bst_float rank = (total_column_weight * cut_idx) /
                       static_cast<float>(num_available_cuts);
      sample_idx = thrust::upper_bound(thrust::seq, column_weights.begin(),
                                       column_weights.end(), rank) -
                   column_weights.begin() - 1;
      sample_idx =
          max(size_t(0), min(sample_idx, column_entries.size() - 1));
    }
    // repeated values will be filtered out on the CPU
    bst_float rmin = sample_idx > 0 ? column_weights[sample_idx - 1] : 0;
    bst_float rmax = column_weights[sample_idx];
    cuts[idx] = WQSketch::Entry(rmin, rmax, rmax - rmin,
                                column_entries[sample_idx].fvalue);
  });
}

void ProcessBatch(int device, const SparsePage& page, size_t begin, size_t end,
                  SketchContainer* sketch_container, int num_cuts,
                  size_t num_columns) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  const auto& host_data = page.data.ConstHostVector();
  dh::device_vector<Entry> sorted_entries(host_data.begin() + begin,
                                          host_data.begin() + end);
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
               sorted_entries.end(), EntryCompareOp());

  dh::caching_device_vector<size_t> column_sizes_scan;
  GetColumnSizesScan(device, &column_sizes_scan,
                     {sorted_entries.data().get(), sorted_entries.size()},
                     num_columns);
  thrust::host_vector<size_t> host_column_sizes_scan(column_sizes_scan);

  dh::caching_device_vector<SketchEntry> cuts(num_columns * num_cuts);
  ExtractCuts(device, {cuts.data().get(), cuts.size()}, num_cuts,
              {sorted_entries.data().get(), sorted_entries.size()},
              {column_sizes_scan.data().get(), column_sizes_scan.size()});

  // add cuts into sketches
  thrust::host_vector<SketchEntry> host_cuts(cuts);
  sketch_container->Push(num_cuts, host_cuts, host_column_sizes_scan);
}

void ProcessWeightedBatch(int device, const SparsePage& page,
                          Span<const float> weights, size_t begin, size_t end,
                          SketchContainer* sketch_container, int num_cuts,
                          size_t num_columns) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  const auto& host_data = page.data.ConstHostVector();
  dh::device_vector<Entry> sorted_entries(host_data.begin() + begin,
                                          host_data.begin() + end);

  // Binary search to assign weights to each element
  dh::device_vector<float> temp_weights(sorted_entries.size());
  auto d_temp_weights = temp_weights.data().get();
  page.offset.SetDevice(device);
  auto row_ptrs = page.offset.ConstDeviceSpan();
  size_t base_rowid = page.base_rowid;
  dh::LaunchN(device, temp_weights.size(), [=] __device__(size_t idx) {
    size_t element_idx = idx + begin;
    size_t ridx = thrust::upper_bound(thrust::seq, row_ptrs.begin(),
                                      row_ptrs.end(), element_idx) -
                  row_ptrs.begin() - 1;
    d_temp_weights[idx] = weights[ridx + base_rowid];
  });

  // Sort
  thrust::sort_by_key(thrust::cuda::par(alloc), sorted_entries.begin(),
                      sorted_entries.end(), temp_weights.begin(),
                      EntryCompareOp());
  std::vector<Entry> entries_t(sorted_entries.begin(), sorted_entries.end());

  // Scan weights
  thrust::inclusive_scan_by_key(thrust::cuda::par(alloc),
                                sorted_entries.begin(), sorted_entries.end(),
                                temp_weights.begin(), temp_weights.begin(),
                                [=] __device__(const Entry& a, const Entry& b) {
                                  return a.index == b.index;
                                });
  std::vector<float> weights_t(temp_weights.begin(), temp_weights.end());

  dh::caching_device_vector<size_t> column_sizes_scan;
  GetColumnSizesScan(device, &column_sizes_scan,
                     {sorted_entries.data().get(), sorted_entries.size()},
                     num_columns);
  thrust::host_vector<size_t> host_column_sizes_scan(column_sizes_scan);

  // Extract cuts
  dh::caching_device_vector<SketchEntry> cuts(num_columns * num_cuts);
  ExtractWeightedCuts(
      device, {cuts.data().get(), cuts.size()}, num_cuts,
      {sorted_entries.data().get(), sorted_entries.size()},
      {temp_weights.data().get(), temp_weights.size()},
      {column_sizes_scan.data().get(), column_sizes_scan.size()});

  // add cuts into sketches
  thrust::host_vector<SketchEntry> host_cuts(cuts);
  sketch_container->Push(num_cuts, host_cuts, host_column_sizes_scan);
}

HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements) {
  HistogramCuts cuts;
  DenseCuts dense_cuts(&cuts);
  SketchContainer sketch_container(max_bins, dmat->Info().num_col_,
                                   dmat->Info().num_row_);

  constexpr int kFactor = 8;
  double eps = 1.0 / (kFactor * max_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
      dmat->Info().num_row_, eps, &dummy_nlevel, &num_cuts);
  num_cuts = std::min(num_cuts, dmat->Info().num_row_);
  if (sketch_batch_num_elements == 0) {
    sketch_batch_num_elements = dmat->Info().num_nonzero_;
  }
  dmat->Info().weights_.SetDevice(device);
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    size_t batch_nnz = batch.data.ConstHostVector().size();
    for (auto begin = 0ull; begin < batch_nnz;
         begin += sketch_batch_num_elements) {
      size_t end = std::min(batch_nnz, size_t(begin + sketch_batch_num_elements));
      if (dmat->Info().weights_.Size() > 0) {
        ProcessWeightedBatch(
            device, batch, dmat->Info().weights_.ConstDeviceSpan(), begin, end,
            &sketch_container, num_cuts, dmat->Info().num_col_);
      } else {
        ProcessBatch(device, batch, begin, end, &sketch_container, num_cuts,
                     dmat->Info().num_col_);
      }
    }
  }

  dense_cuts.Init(&sketch_container.sketches_, max_bins, dmat->Info().num_row_);
  return cuts;
}

struct IsValidFunctor : public thrust::unary_function<Entry, bool> {
  explicit IsValidFunctor(float missing) : missing(missing) {}

  float missing;
  __device__ bool operator()(const data::COOTuple& e) const {
    if (common::CheckNAN(e.value) || e.value == missing) {
      return false;
    }
    return true;
  }
  __device__ bool operator()(const Entry& e) const {
    if (common::CheckNAN(e.fvalue) || e.fvalue == missing) {
      return false;
    }
    return true;
  }
};

template <typename ReturnT, typename IterT, typename FuncT>
thrust::transform_iterator<FuncT, IterT, ReturnT> MakeTransformIterator(
    IterT iter, FuncT func) {
  return thrust::transform_iterator<FuncT, IterT, ReturnT>(iter, func);
}

template <typename AdapterT>
void ProcessBatch(AdapterT* adapter, size_t begin, size_t end, float missing,
                  SketchContainer* sketch_container, int num_cuts) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  adapter->BeforeFirst();
  adapter->Next();
  auto& batch = adapter->Value();
  // Enforce single batch
  CHECK(!adapter->Next());

  auto batch_iter = MakeTransformIterator<data::COOTuple>(
      thrust::make_counting_iterator(0llu),
      [=] __device__(size_t idx) { return batch.GetElement(idx); });
  auto entry_iter = MakeTransformIterator<Entry>(
      thrust::make_counting_iterator(0llu), [=] __device__(size_t idx) {
        return Entry(batch.GetElement(idx).column_idx,
                     batch.GetElement(idx).value);
      });

  // Work out how many valid entries we have in each column
  dh::caching_device_vector<size_t> column_sizes_scan(adapter->NumColumns() + 1,
                                                      0);
  auto d_column_sizes_scan = column_sizes_scan.data().get();
  IsValidFunctor is_valid(missing);
  dh::LaunchN(adapter->DeviceIdx(), end - begin, [=] __device__(size_t idx) {
    auto e = batch_iter[begin + idx];
    if (is_valid(e)) {
      atomicAdd(reinterpret_cast<unsigned long long*>(  // NOLINT
                    &d_column_sizes_scan[e.column_idx]),
                static_cast<unsigned long long>(1));  // NOLINT
    }
  });
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan.begin(),
                         column_sizes_scan.end(), column_sizes_scan.begin());
  thrust::host_vector<size_t> host_column_sizes_scan(column_sizes_scan);
  size_t num_valid = host_column_sizes_scan.back();

  // Copy current subset of valid elements into temporary storage and sort
  thrust::device_vector<Entry> sorted_entries(num_valid);
  thrust::copy_if(thrust::cuda::par(alloc), entry_iter + begin,
                  entry_iter + end, sorted_entries.begin(), is_valid);
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
               sorted_entries.end(), EntryCompareOp());

  // Extract the cuts from all columns concurrently
  dh::caching_device_vector<SketchEntry> cuts(adapter->NumColumns() * num_cuts);
  ExtractCuts(adapter->DeviceIdx(), {cuts.data().get(), cuts.size()}, num_cuts,
              {sorted_entries.data().get(), sorted_entries.size()},
              {column_sizes_scan.data().get(), column_sizes_scan.size()});

  // Push cuts into sketches stored in host memory
  thrust::host_vector<SketchEntry> host_cuts(cuts);
  sketch_container->Push(num_cuts, host_cuts, host_column_sizes_scan);
}

template <typename AdapterT>
HistogramCuts AdapterDeviceSketch(AdapterT* adapter, int num_bins,
                                  float missing,
                                  size_t sketch_batch_num_elements) {
  CHECK(adapter->NumRows() != data::kAdapterUnknownSize);
  CHECK(adapter->NumColumns() != data::kAdapterUnknownSize);

  adapter->BeforeFirst();
  adapter->Next();
  auto& batch = adapter->Value();

  // Enforce single batch
  CHECK(!adapter->Next());

  HistogramCuts cuts;
  DenseCuts dense_cuts(&cuts);
  SketchContainer sketch_container(num_bins, adapter->NumColumns(),
                                   adapter->NumRows());

  constexpr int kFactor = 8;
  double eps = 1.0 / (kFactor * num_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
      adapter->NumRows(), eps, &dummy_nlevel, &num_cuts);
  num_cuts = std::min(num_cuts, adapter->NumRows());
  if (sketch_batch_num_elements == 0) {
    sketch_batch_num_elements = batch.Size();
  }
  for (auto begin = 0ull; begin < batch.Size();
       begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
    ProcessBatch(adapter, begin, end, missing, &sketch_container, num_cuts);
  }

  dense_cuts.Init(&sketch_container.sketches_, num_bins, adapter->NumRows());
  return cuts;
}

template HistogramCuts AdapterDeviceSketch(data::CudfAdapter* adapter,
                                           int num_bins, float missing,
                                           size_t sketch_batch_size);
template HistogramCuts AdapterDeviceSketch(data::CupyAdapter* adapter,
                                           int num_bins, float missing,
                                           size_t sketch_batch_size);
}  // namespace common
}  // namespace xgboost
