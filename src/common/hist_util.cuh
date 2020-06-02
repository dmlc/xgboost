#ifndef COMMON_HIST_UTIL_CUH_
#define COMMON_HIST_UTIL_CUH_

#include <thrust/host_vector.h>
#include "hist_util.h"
#include "device_helpers.cuh"
#include "../data/device_adapter.cuh"

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
void ExtractCuts(int device,
                 size_t num_cuts_per_feature,
                 Span<Entry const> sorted_data,
                 Span<size_t const> column_sizes_scan,
                 Span<SketchEntry> out_cuts);

// Compute number of sample cuts needed on local node to maintain accuracy
// We take more cuts than needed and then reduce them later
inline size_t RequiredSampleCuts(int max_bins, size_t num_rows) {
  constexpr int kFactor = 8;
  double eps = 1.0 / (kFactor * max_bins);
  size_t dummy_nlevel;
  size_t num_cuts;
  WQuantileSketch<bst_float, bst_float>::LimitSizeLevel(
      num_rows, eps, &dummy_nlevel, &num_cuts);
  return std::min(num_cuts, num_rows);
}

// sketch_batch_num_elements 0 means autodetect. Only modify this for testing.
HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements = 0);

template <typename AdapterT>
void ProcessBatch(AdapterT* adapter, size_t begin, size_t end, float missing,
                  SketchContainer* sketch_container, int num_cuts) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  adapter->BeforeFirst();
  adapter->Next();
  auto &batch = adapter->Value();
  // Enforce single batch
  CHECK(!adapter->Next());
  auto batch_iter = dh::MakeTransformIterator<data::COOTuple>(
    thrust::make_counting_iterator(0llu),
    [=] __device__(size_t idx) { return batch.GetElement(idx); });
  auto entry_iter = dh::MakeTransformIterator<Entry>(
      thrust::make_counting_iterator(0llu), [=] __device__(size_t idx) {
        return Entry(batch.GetElement(idx).column_idx,
                     batch.GetElement(idx).value);
      });
  // Work out how many valid entries we have in each column
  dh::caching_device_vector<size_t> column_sizes_scan(adapter->NumColumns() + 1,
                                                      0);

  auto d_column_sizes_scan = column_sizes_scan.data().get();
  data::IsValidFunctor is_valid(missing);
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
  dh::caching_device_vector<Entry> sorted_entries(num_valid);
  thrust::copy_if(thrust::cuda::par(alloc), entry_iter + begin,
                  entry_iter + end, sorted_entries.begin(), is_valid);
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
               sorted_entries.end(), EntryCompareOp());

  // Extract the cuts from all columns concurrently
  dh::caching_device_vector<SketchEntry> cuts(adapter->NumColumns() * num_cuts);
  ExtractCuts(adapter->DeviceIdx(), num_cuts,
              dh::ToSpan(sorted_entries),
              dh::ToSpan(column_sizes_scan),
              dh::ToSpan(cuts));

  // Push cuts into sketches stored in host memory
  thrust::host_vector<SketchEntry> host_cuts(cuts);
  sketch_container->Push(num_cuts, host_cuts, host_column_sizes_scan);
}

template <typename AdapterT>
HistogramCuts AdapterDeviceSketch(AdapterT* adapter, int num_bins,
                                  float missing,
                                  size_t sketch_batch_num_elements = 0) {
  size_t num_cuts = RequiredSampleCuts(num_bins, adapter->NumRows());
  if (sketch_batch_num_elements == 0) {
    int bytes_per_element = 16;
    size_t bytes_cuts = num_cuts * adapter->NumColumns() * sizeof(SketchEntry);
    size_t bytes_num_columns = (adapter->NumColumns() + 1) * sizeof(size_t);
    // use up to 80% of available space
    sketch_batch_num_elements = (dh::AvailableMemory(adapter->DeviceIdx()) -
                                 bytes_cuts - bytes_num_columns) *
                                0.8 / bytes_per_element;
  }

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

  for (auto begin = 0ull; begin < batch.Size();
       begin += sketch_batch_num_elements) {
    size_t end = std::min(batch.Size(), size_t(begin + sketch_batch_num_elements));
    ProcessBatch(adapter, begin, end, missing, &sketch_container, num_cuts);
  }

  dense_cuts.Init(&sketch_container.sketches_, num_bins, adapter->NumRows());
  return cuts;
}
}      // namespace common
}      // namespace xgboost

#endif  // COMMON_HIST_UTIL_CUH_