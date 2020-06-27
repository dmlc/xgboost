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
#include "xgboost/host_device_vector.h"


namespace xgboost {
namespace common {

constexpr float SketchContainer::kFactor;

// Count the entries in each column and exclusive scan
void ExtractCutsSparse(int device, common::Span<size_t const> cuts_ptr,
                       Span<Entry const> sorted_data,
                       Span<size_t const> column_sizes_scan,
                       Span<SketchEntry> out_cuts) {
  dh::LaunchN(device, out_cuts.size(), [=] __device__(size_t idx) {
    // Each thread is responsible for obtaining one cut from the sorted input
    // size_t column_idx = idx / num_cuts_per_feature;
    size_t column_idx = dh::SegmentId(cuts_ptr, idx);
    size_t column_size =
        column_sizes_scan[column_idx + 1] - column_sizes_scan[column_idx];
    size_t num_available_cuts = cuts_ptr[column_idx + 1] - cuts_ptr[column_idx];
    size_t cut_idx = idx - cuts_ptr[column_idx];
    Span<Entry const> column_entries =
        sorted_data.subspan(column_sizes_scan[column_idx], column_size);
    size_t rank = (column_entries.size() * cut_idx) /
                  static_cast<float>(num_available_cuts);
    out_cuts[idx] = WQSketch::Entry(rank, rank + 1, 1,
                                    column_entries[rank].fvalue);
  });
}

void ExtractWeightedCutsSparse(int device,
                               common::Span<size_t const> cuts_ptr,
                               Span<Entry> sorted_data,
                               Span<float> weights_scan,
                               Span<size_t> column_sizes_scan,
                               Span<SketchEntry> cuts) {
  dh::LaunchN(device, cuts.size(), [=] __device__(size_t idx) {
    // Each thread is responsible for obtaining one cut from the sorted input
    size_t column_idx = dh::SegmentId(cuts_ptr, idx);
    size_t column_size =
        column_sizes_scan[column_idx + 1] - column_sizes_scan[column_idx];
    size_t num_available_cuts = cuts_ptr[column_idx + 1] - cuts_ptr[column_idx];
    size_t cut_idx = idx - cuts_ptr[column_idx];

    Span<Entry> column_entries =
        sorted_data.subspan(column_sizes_scan[column_idx], column_size);

    Span<float> column_weights_scan =
        weights_scan.subspan(column_sizes_scan[column_idx], column_size);
    float total_column_weight = column_weights_scan.back();
    size_t sample_idx = 0;
    if (cut_idx == 0) {
      // First cut
      sample_idx = 0;
    } else if (cut_idx == num_available_cuts) {
      // Last cut
      sample_idx = column_entries.size() - 1;
    } else if (num_available_cuts == column_size) {
      // There are less samples available than our buffer
      // Take every available sample
      sample_idx = cut_idx;
    } else {
      bst_float rank = (total_column_weight * cut_idx) /
                       static_cast<float>(num_available_cuts);
      sample_idx = thrust::upper_bound(thrust::seq,
                                       column_weights_scan.begin(),
                                       column_weights_scan.end(),
                                       rank) -
                   column_weights_scan.begin();
      sample_idx =
          max(static_cast<size_t>(0),
              min(sample_idx, column_entries.size() - 1));
    }
    // repeated values will be filtered out later.
    bst_float rmin = sample_idx > 0 ? column_weights_scan[sample_idx - 1] : 0.0f;
    bst_float rmax = column_weights_scan[sample_idx];
    cuts[idx] = WQSketch::Entry(rmin, rmax, rmax - rmin,
                                column_entries[sample_idx].fvalue);
  });
}

void GetColumnSizesScan(int device, size_t num_cuts_per_feature,
                        HostDeviceVector<size_t> *cuts_ptr,
                        dh::caching_device_vector<size_t> *column_sizes_scan,
                        Span<const Entry> entries, size_t num_columns) {
  column_sizes_scan->resize(num_columns + 1, 0);
  cuts_ptr->SetDevice(device);
  cuts_ptr->Resize(num_columns + 1, 0);

  auto d_column_sizes_scan = column_sizes_scan->data().get();
  auto d_entries = entries.data();
  dh::LaunchN(device, entries.size(), [=] __device__(size_t idx) {
    auto &e = d_entries[idx];
    atomicAdd(reinterpret_cast<unsigned long long *>( // NOLINT
                  &d_column_sizes_scan[e.index]),
              static_cast<unsigned long long>(1)); // NOLINT
  });
  // Calculate cuts CSC pointer
  auto cut_ptr_it = dh::MakeTransformIterator<size_t>(
      column_sizes_scan->begin(), [=] __device__(size_t column_size) {
        return thrust::min(num_cuts_per_feature, column_size);
      });
  dh::XGBCachingDeviceAllocator<char> alloc;
  thrust::exclusive_scan(thrust::cuda::par(alloc), cut_ptr_it,
                         cut_ptr_it + column_sizes_scan->size(),
                         cuts_ptr->DevicePointer());
  // Calculate entries CSC pointer
  thrust::exclusive_scan(thrust::cuda::par(alloc), column_sizes_scan->begin(),
                         column_sizes_scan->end(), column_sizes_scan->begin());
}

HostDeviceVector<size_t>
MakeCutsPtr(int32_t device,
            thrust::host_vector<size_t> const &host_column_sizes_scan,
            size_t cuts_per_feature) {
  HostDeviceVector<size_t> cuts_ptr;
  cuts_ptr.SetDevice(device);
  auto &h_cuts_ptr = cuts_ptr.HostVector();
  size_t offset = 0;
  h_cuts_ptr.push_back(offset);
  for (size_t i = 1; i < host_column_sizes_scan.size(); ++i) {
    offset +=
        std::min(host_column_sizes_scan[i] - host_column_sizes_scan[i - 1],
                 static_cast<size_t>(cuts_per_feature));
    h_cuts_ptr.push_back(offset);
  }
  return cuts_ptr;
}

void ProcessBatch(int device, const SparsePage& page, size_t begin, size_t end,
                  SketchContainer* sketch_container, int num_cuts,
                  size_t num_columns) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  const auto& host_data = page.data.ConstHostVector();
  dh::caching_device_vector<Entry> sorted_entries(host_data.begin() + begin,
                                                  host_data.begin() + end);
  thrust::sort(thrust::cuda::par(alloc), sorted_entries.begin(),
               sorted_entries.end(), EntryCompareOp());

  HostDeviceVector<size_t> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  GetColumnSizesScan(device, num_cuts, &cuts_ptr, &column_sizes_scan,
                     {sorted_entries.data().get(), sorted_entries.size()},
                     num_columns);

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  dh::caching_device_vector<SketchEntry> cuts(h_cuts_ptr.back());
  auto d_cuts_ptr = cuts_ptr.ConstDeviceSpan();

  CHECK_EQ(d_cuts_ptr.size(), column_sizes_scan.size());
  ExtractCutsSparse(device, d_cuts_ptr, dh::ToSpan(sorted_entries),
                    dh::ToSpan(column_sizes_scan), dh::ToSpan(cuts));

  // add cuts into sketches
  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
  CHECK_EQ(sorted_entries.capacity(), 0);
  CHECK_NE(cuts_ptr.Size(), 0);
  sketch_container->Push(cuts_ptr.ConstDeviceSpan(), &cuts);
}

void SortByWeight(dh::XGBCachingDeviceAllocator<char>* alloc,
                  dh::caching_device_vector<float>* weights,
                  dh::caching_device_vector<Entry>* sorted_entries) {
  // Sort both entries and wegihts.
  thrust::sort_by_key(thrust::cuda::par(*alloc), sorted_entries->begin(),
                      sorted_entries->end(), weights->begin(),
                      EntryCompareOp());

  // Scan weights
  thrust::inclusive_scan_by_key(thrust::cuda::par(*alloc),
                                sorted_entries->begin(), sorted_entries->end(),
                                weights->begin(), weights->begin(),
                                [=] __device__(const Entry& a, const Entry& b) {
                                  return a.index == b.index;
                                });
}

void ProcessWeightedBatch(int device, const SparsePage& page,
                          Span<const float> weights, size_t begin, size_t end,
                          SketchContainer* sketch_container, int num_cuts_per_feature,
                          size_t num_columns,
                          bool is_ranking, Span<bst_group_t const> d_group_ptr) {
  dh::XGBCachingDeviceAllocator<char> alloc;
  const auto& host_data = page.data.ConstHostVector();
  dh::caching_device_vector<Entry> sorted_entries(host_data.begin() + begin,
                                                  host_data.begin() + end);

  // Binary search to assign weights to each element
  dh::caching_device_vector<float> temp_weights(sorted_entries.size());
  auto d_temp_weights = temp_weights.data().get();
  page.offset.SetDevice(device);
  auto row_ptrs = page.offset.ConstDeviceSpan();
  size_t base_rowid = page.base_rowid;
  if (is_ranking) {
    CHECK_GE(d_group_ptr.size(), 2)
        << "Must have at least 1 group for ranking.";
    CHECK_EQ(weights.size(), d_group_ptr.size() - 1)
        << "Weight size should equal to number of groups.";
    dh::LaunchN(device, temp_weights.size(), [=] __device__(size_t idx) {
        size_t element_idx = idx + begin;
        size_t ridx = thrust::upper_bound(thrust::seq, row_ptrs.begin(),
                                          row_ptrs.end(), element_idx) -
                      row_ptrs.begin() - 1;
        auto it =
            thrust::upper_bound(thrust::seq,
                                d_group_ptr.cbegin(), d_group_ptr.cend(),
                                ridx + base_rowid) - 1;
        bst_group_t group = thrust::distance(d_group_ptr.cbegin(), it);
        d_temp_weights[idx] = weights[group];
      });
  } else {
    CHECK_EQ(weights.size(), page.offset.Size() - 1);
    dh::LaunchN(device, temp_weights.size(), [=] __device__(size_t idx) {
        size_t element_idx = idx + begin;
        size_t ridx = thrust::upper_bound(thrust::seq, row_ptrs.begin(),
                                          row_ptrs.end(), element_idx) -
                      row_ptrs.begin() - 1;
        d_temp_weights[idx] = weights[ridx + base_rowid];
      });
  }
  SortByWeight(&alloc, &temp_weights, &sorted_entries);

  HostDeviceVector<size_t> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  GetColumnSizesScan(device, num_cuts_per_feature, &cuts_ptr, &column_sizes_scan,
                     {sorted_entries.data().get(), sorted_entries.size()},
                     num_columns);
  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  dh::caching_device_vector<SketchEntry> cuts(h_cuts_ptr.back());
  auto d_cuts_ptr = cuts_ptr.ConstDeviceSpan();

  // Extract cuts
  ExtractWeightedCutsSparse(device, d_cuts_ptr,
                            dh::ToSpan(sorted_entries),
                            dh::ToSpan(temp_weights),
                            dh::ToSpan(column_sizes_scan),
                            dh::ToSpan(cuts));

  // add cuts into sketches
  sketch_container->Push(cuts_ptr.ConstDeviceSpan(), &cuts);
}

HistogramCuts DeviceSketch(int device, DMatrix* dmat, int max_bins,
                           size_t sketch_batch_num_elements) {
  // Configure batch size based on available memory
  bool has_weights = dmat->Info().weights_.Size() > 0;
  size_t num_cuts_per_feature = RequiredSampleCuts(max_bins, dmat->Info().num_row_);
  sketch_batch_num_elements = SketchBatchNumElements(
      sketch_batch_num_elements,
      dmat->Info().num_col_, device, num_cuts_per_feature, has_weights);

  HistogramCuts cuts;
  DenseCuts dense_cuts(&cuts);
  SketchContainer sketch_container(max_bins, dmat->Info().num_col_,
                                   dmat->Info().num_row_, device);

  dmat->Info().weights_.SetDevice(device);
  for (const auto& batch : dmat->GetBatches<SparsePage>()) {
    size_t batch_nnz = batch.data.Size();
    auto const& info = dmat->Info();
    for (auto begin = 0ull; begin < batch_nnz; begin += sketch_batch_num_elements) {
      size_t end = std::min(batch_nnz, size_t(begin + sketch_batch_num_elements));
      if (has_weights) {
        bool is_ranking = CutsBuilder::UseGroup(dmat);
        dh::caching_device_vector<uint32_t> groups(info.group_ptr_.cbegin(),
                                                   info.group_ptr_.cend());
        ProcessWeightedBatch(
            device, batch, dmat->Info().weights_.ConstDeviceSpan(), begin, end,
            &sketch_container,
            num_cuts_per_feature,
            dmat->Info().num_col_,
            is_ranking, dh::ToSpan(groups));
      } else {
        ProcessBatch(device, batch, begin, end, &sketch_container, num_cuts_per_feature,
                     dmat->Info().num_col_);
      }
    }
  }
  sketch_container.MakeCuts(&cuts);
  return cuts;
}
}  // namespace common
}  // namespace xgboost
