/**
 * Copyright 2018~2024, XGBoost contributors
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

#include <cstddef>  // for size_t
#include <utility>
#include <vector>

#include "categorical.h"
#include "cuda_context.cuh"  // for CUDAContext
#include "device_helpers.cuh"
#include "hist_util.cuh"
#include "hist_util.h"
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

size_t RequiredSampleCuts(bst_idx_t num_rows, bst_feature_t num_columns, size_t max_bins,
                          bst_idx_t nnz) {
  auto per_column = RequiredSampleCutsPerColumn(max_bins, num_rows);
  auto if_dense = num_columns * per_column;
  auto result = std::min(nnz, if_dense);
  return result;
}

size_t RequiredMemory(bst_idx_t num_rows, bst_feature_t num_columns, size_t nnz,
                      size_t num_bins, bool with_weights) {
  size_t peak = 0;
  // 0. Allocate cut pointer in quantile container by increasing: n_columns + 1
  size_t total = (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // 1. Copy and sort: 2 * bytes_per_element * shape
  total += BytesPerElement(with_weights) * num_rows * num_columns;
  peak = std::max(peak, total);
  // 2. Deallocate bytes_per_element * shape due to reusing memory in sort.
  total -= BytesPerElement(with_weights) * num_rows * num_columns / 2;
  // 3. Allocate colomn size scan by increasing: n_columns + 1
  total += (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // 4. Allocate cut pointer by increasing: n_columns + 1
  total += (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // 5. Allocate cuts: assuming rows is greater than bins: n_columns * limit_size
  total += RequiredSampleCuts(num_rows, num_bins, num_bins, nnz) * sizeof(SketchEntry);
  // 6. Deallocate copied entries by reducing: bytes_per_element * shape.
  peak = std::max(peak, total);
  total -= (BytesPerElement(with_weights) * num_rows * num_columns) / 2;
  // 7. Deallocate column size scan.
  peak = std::max(peak, total);
  total -= (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // 8. Deallocate cut size scan.
  total -= (num_columns + 1) * sizeof(SketchContainer::OffsetT);
  // 9. Allocate final cut values, min values, cut ptrs: std::min(rows, bins + 1) *
  //    n_columns + n_columns + n_columns + 1
  total += std::min(num_rows, num_bins) * num_columns * sizeof(float);
  total += num_columns *
           sizeof(std::remove_reference_t<decltype(
                      std::declval<HistogramCuts>().MinValues())>::value_type);
  total += (num_columns + 1) *
           sizeof(std::remove_reference_t<decltype(
                      std::declval<HistogramCuts>().Ptrs())>::value_type);
  peak = std::max(peak, total);

  return peak;
}

bst_idx_t SketchBatchNumElements(bst_idx_t sketch_batch_num_elements, SketchShape shape, int device,
                                 size_t num_cuts, bool has_weight, std::size_t container_bytes) {
  auto constexpr kIntMax = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
#if defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  (void)device;
  // Device available memory is not accurate when rmm is used.
  double total_mem = curt::TotalMemory() - container_bytes;
  double total_f32 = total_mem / sizeof(float);
  double n_max_used_f32 = std::max(total_f32 / 16.0, 1.0);  // a quarter
  if (shape.nnz > shape.Size()) {
    // Unknown nnz
    shape.nnz = shape.Size();
  }
  return std::min(static_cast<bst_idx_t>(n_max_used_f32), shape.nnz);
#endif  // defined(XGBOOST_USE_RMM) && XGBOOST_USE_RMM == 1
  (void)container_bytes;  // We known the remaining size when RMM is not used.
  if (sketch_batch_num_elements == detail::UnknownSketchNumElements()) {
    auto required_memory =
        RequiredMemory(shape.n_samples, shape.n_features, shape.nnz, num_cuts, has_weight);
    // use up to 80% of available space
    auto avail = dh::AvailableMemory(device) * 0.8;
    CHECK_GT(avail, 0) << error::ZeroCudaMemory();
    if (required_memory > avail) {
      sketch_batch_num_elements = avail / BytesPerElement(has_weight);
    } else {
      sketch_batch_num_elements = std::min(shape.Size(), shape.nnz);
    }
  }

  return std::min(sketch_batch_num_elements, kIntMax);
}

void SortByWeight(Context const* ctx, dh::device_vector<float>* weights,
                  dh::device_vector<Entry>* sorted_entries) {
  // Sort both entries and wegihts.
  auto cuctx = ctx->CUDACtx();
  CHECK_EQ(weights->size(), sorted_entries->size());
  thrust::sort_by_key(cuctx->TP(), sorted_entries->begin(), sorted_entries->end(), weights->begin(),
                      detail::EntryCompareOp());

  // Scan weights
  thrust::inclusive_scan_by_key(
      cuctx->CTP(), sorted_entries->begin(), sorted_entries->end(), weights->begin(),
      weights->begin(),
      [=] __device__(const Entry& a, const Entry& b) { return a.index == b.index; });
}

void RemoveDuplicatedCategories(Context const* ctx, MetaInfo const& info,
                                Span<bst_idx_t> d_cuts_ptr,
                                dh::device_vector<Entry>* p_sorted_entries,
                                dh::device_vector<float>* p_sorted_weights,
                                dh::caching_device_vector<size_t>* p_column_sizes_scan) {
  info.feature_types.SetDevice(ctx->Device());
  auto d_feature_types = info.feature_types.ConstDeviceSpan();
  CHECK(!d_feature_types.empty());
  auto& column_sizes_scan = *p_column_sizes_scan;
  auto& sorted_entries = *p_sorted_entries;
  // Removing duplicated entries in categorical features.

  // We don't need to accumulate weight for duplicated entries as there's no weighted
  // sketching for categorical features, the categories are the cut values.
  dh::caching_device_vector<size_t> new_column_scan(column_sizes_scan.size());
  std::size_t n_uniques{0};
  if (p_sorted_weights) {
    using Pair = thrust::tuple<Entry, float>;
    auto d_sorted_entries = dh::ToSpan(sorted_entries);
    auto d_sorted_weights = dh::ToSpan(*p_sorted_weights);
    auto val_in_it = thrust::make_zip_iterator(d_sorted_entries.data(), d_sorted_weights.data());
    auto val_out_it = thrust::make_zip_iterator(d_sorted_entries.data(), d_sorted_weights.data());
    n_uniques =
        dh::SegmentedUnique(ctx->CUDACtx()->CTP(), column_sizes_scan.data().get(),
                            column_sizes_scan.data().get() + column_sizes_scan.size(), val_in_it,
                            val_in_it + sorted_entries.size(), new_column_scan.data().get(),
                            val_out_it, [=] __device__(Pair const& l, Pair const& r) {
                              Entry const& le = thrust::get<0>(l);
                              Entry const& re = thrust::get<0>(r);
                              if (le.index == re.index && IsCat(d_feature_types, le.index)) {
                                return le.fvalue == re.fvalue;
                              }
                              return false;
                            });
    p_sorted_weights->resize(n_uniques);
  } else {
    n_uniques = dh::SegmentedUnique(ctx->CUDACtx()->CTP(), column_sizes_scan.data().get(),
                                    column_sizes_scan.data().get() + column_sizes_scan.size(),
                                    sorted_entries.begin(), sorted_entries.end(),
                                    new_column_scan.data().get(), sorted_entries.begin(),
                                    [=] __device__(Entry const& l, Entry const& r) {
                                      if (l.index == r.index) {
                                        if (IsCat(d_feature_types, l.index)) {
                                          return l.fvalue == r.fvalue;
                                        }
                                      }
                                      return false;
                                    });
  }
  sorted_entries.resize(n_uniques);

  // Renew the column scan and cut scan based on categorical data.
  dh::caching_device_vector<SketchContainer::OffsetT> new_cuts_size(info.num_col_ + 1);
  CHECK_EQ(new_column_scan.size(), new_cuts_size.size());
  dh::LaunchN(new_column_scan.size(), ctx->CUDACtx()->Stream(),
              [=, d_new_cuts_size = dh::ToSpan(new_cuts_size),
               d_old_column_sizes_scan = dh::ToSpan(column_sizes_scan),
               d_new_columns_ptr = dh::ToSpan(new_column_scan)] __device__(size_t idx) {
                d_old_column_sizes_scan[idx] = d_new_columns_ptr[idx];
                if (idx == d_new_columns_ptr.size() - 1) {
                  return;
                }
                if (IsCat(d_feature_types, idx)) {
                  // Cut size is the same as number of categories in input.
                  d_new_cuts_size[idx] = d_new_columns_ptr[idx + 1] - d_new_columns_ptr[idx];
                } else {
                  d_new_cuts_size[idx] = d_cuts_ptr[idx + 1] - d_cuts_ptr[idx];
                }
              });
  // Turn size into ptr.
  thrust::exclusive_scan(ctx->CUDACtx()->CTP(), new_cuts_size.cbegin(), new_cuts_size.cend(),
                         d_cuts_ptr.data());
}
}  // namespace detail

void ProcessWeightedBatch(Context const* ctx, const SparsePage& page, MetaInfo const& info,
                          std::size_t begin, std::size_t end,
                          SketchContainer* sketch_container,  // <- output sketch
                          int num_cuts_per_feature, common::Span<float const> sample_weight) {
  dh::device_vector<Entry> sorted_entries;
  if (page.data.DeviceCanRead()) {
    // direct copy if data is already on device
    auto const& d_data = page.data.ConstDevicePointer();
    sorted_entries = dh::device_vector<Entry>(d_data + begin, d_data + end);
  } else {
    const auto& h_data = page.data.ConstHostVector();
    sorted_entries = dh::device_vector<Entry>(h_data.begin() + begin, h_data.begin() + end);
  }

  bst_idx_t base_rowid = page.base_rowid;

  dh::device_vector<float> entry_weight;
  auto cuctx = ctx->CUDACtx();
  if (!sample_weight.empty()) {
    // Expand sample weight into entry weight.
    CHECK_EQ(sample_weight.size(), info.num_row_);
    entry_weight.resize(sorted_entries.size());
    auto d_temp_weight = dh::ToSpan(entry_weight);
    page.offset.SetDevice(ctx->Device());
    auto row_ptrs = page.offset.ConstDeviceSpan();
    thrust::for_each_n(cuctx->CTP(), thrust::make_counting_iterator(0ul), entry_weight.size(),
                       [=] __device__(std::size_t idx) {
                         std::size_t element_idx = idx + begin;
                         std::size_t ridx = dh::SegmentId(row_ptrs, element_idx);
                         d_temp_weight[idx] = sample_weight[ridx + base_rowid];
                       });
    detail::SortByWeight(ctx, &entry_weight, &sorted_entries);
  } else {
    thrust::sort(cuctx->TP(), sorted_entries.begin(), sorted_entries.end(),
                 detail::EntryCompareOp());
  }

  HostDeviceVector<SketchContainer::OffsetT> cuts_ptr;
  dh::caching_device_vector<size_t> column_sizes_scan;
  data::IsValidFunctor dummy_is_valid(std::numeric_limits<float>::quiet_NaN());
  auto batch_it = dh::MakeTransformIterator<data::COOTuple>(
      sorted_entries.data().get(), [] __device__(Entry const& e) -> data::COOTuple {
        return {0, e.index, e.fvalue};  // row_idx is not needed for scaning column size.
      });
  detail::GetColumnSizesScan(ctx->CUDACtx(), ctx->Device(), info.num_col_, num_cuts_per_feature,
                             IterSpan{batch_it, sorted_entries.size()}, dummy_is_valid, &cuts_ptr,
                             &column_sizes_scan);
  auto d_cuts_ptr = cuts_ptr.DeviceSpan();
  if (sketch_container->HasCategorical()) {
    auto p_weight = entry_weight.empty() ? nullptr : &entry_weight;
    detail::RemoveDuplicatedCategories(ctx, info, d_cuts_ptr, &sorted_entries, p_weight,
                                       &column_sizes_scan);
  }

  auto const& h_cuts_ptr = cuts_ptr.ConstHostVector();
  CHECK_EQ(d_cuts_ptr.size(), column_sizes_scan.size());

  // Add cuts into sketches
  sketch_container->Push(ctx, dh::ToSpan(sorted_entries), dh::ToSpan(column_sizes_scan), d_cuts_ptr,
                         h_cuts_ptr.back(), dh::ToSpan(entry_weight));

  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
  CHECK_EQ(sorted_entries.capacity(), 0);
  CHECK_NE(cuts_ptr.Size(), 0);
}

// Unify group weight, Hessian, and sample weight into sample weight.
[[nodiscard]] Span<float const> UnifyWeight(CUDAContext const* cuctx, MetaInfo const& info,
                                            common::Span<float const> hessian,
                                            HostDeviceVector<float>* p_out_weight) {
  if (hessian.empty()) {
    if (info.IsRanking() && !info.weights_.Empty()) {
      dh::device_vector<bst_group_t> group_ptr(info.group_ptr_);
      auto d_group_ptr = dh::ToSpan(group_ptr);
      CHECK_GE(d_group_ptr.size(), 2) << "Must have at least 1 group for ranking.";
      auto d_weight = info.weights_.ConstDeviceSpan();
      CHECK_EQ(d_weight.size(), d_group_ptr.size() - 1)
          << "Weight size should equal to number of groups.";
      p_out_weight->Resize(info.num_row_);
      auto d_weight_out = p_out_weight->DeviceSpan();

      thrust::for_each_n(cuctx->CTP(), thrust::make_counting_iterator(0ul), d_weight_out.size(),
                         [=] XGBOOST_DEVICE(std::size_t i) {
                           auto gidx = dh::SegmentId(d_group_ptr, i);
                           d_weight_out[i] = d_weight[gidx];
                         });
      return p_out_weight->ConstDeviceSpan();
    } else {
      return info.weights_.ConstDeviceSpan();
    }
  }

  // sketch with hessian as weight
  p_out_weight->Resize(info.num_row_);
  auto d_weight_out = p_out_weight->DeviceSpan();
  if (!info.weights_.Empty()) {
    // merge sample weight with hessian
    auto d_weight = info.weights_.ConstDeviceSpan();
    if (info.IsRanking()) {
      dh::device_vector<bst_group_t> group_ptr(info.group_ptr_);
      CHECK_EQ(hessian.size(), d_weight_out.size());
      auto d_group_ptr = dh::ToSpan(group_ptr);
      CHECK_GE(d_group_ptr.size(), 2) << "Must have at least 1 group for ranking.";
      CHECK_EQ(d_weight.size(), d_group_ptr.size() - 1)
          << "Weight size should equal to number of groups.";
      thrust::for_each_n(cuctx->CTP(), thrust::make_counting_iterator(0ul), hessian.size(),
                         [=] XGBOOST_DEVICE(std::size_t i) {
                           d_weight_out[i] = d_weight[dh::SegmentId(d_group_ptr, i)] * hessian(i);
                         });
    } else {
      CHECK_EQ(hessian.size(), info.num_row_);
      CHECK_EQ(hessian.size(), d_weight.size());
      CHECK_EQ(hessian.size(), d_weight_out.size());
      thrust::for_each_n(
          cuctx->CTP(), thrust::make_counting_iterator(0ul), hessian.size(),
          [=] XGBOOST_DEVICE(std::size_t i) { d_weight_out[i] = d_weight[i] * hessian(i); });
    }
  } else {
    // copy hessian as weight
    CHECK_EQ(d_weight_out.size(), hessian.size());
    dh::safe_cuda(cudaMemcpyAsync(d_weight_out.data(), hessian.data(), hessian.size_bytes(),
                                  cudaMemcpyDefault));
  }
  return d_weight_out;
}

HistogramCuts DeviceSketchWithHessian(Context const* ctx, DMatrix* p_fmat, bst_bin_t max_bin,
                                      Span<float const> hessian,
                                      std::size_t sketch_batch_num_elements) {
  auto const& info = p_fmat->Info();
  bool has_weight = !info.weights_.Empty();
  info.feature_types.SetDevice(ctx->Device());

  HostDeviceVector<float> weight;
  weight.SetDevice(ctx->Device());

  // Configure batch size based on available memory
  std::size_t num_cuts_per_feature = detail::RequiredSampleCutsPerColumn(max_bin, info.num_row_);
  sketch_batch_num_elements = detail::SketchBatchNumElements(
      sketch_batch_num_elements,
      detail::SketchShape{info.num_row_, info.num_col_, info.num_nonzero_}, ctx->Ordinal(),
      num_cuts_per_feature, has_weight, 0);

  CUDAContext const* cuctx = ctx->CUDACtx();

  info.weights_.SetDevice(ctx->Device());
  auto d_weight = UnifyWeight(cuctx, info, hessian, &weight);

  HistogramCuts cuts;
  SketchContainer sketch_container(info.feature_types, max_bin, info.num_col_, info.num_row_,
                                   ctx->Device());
  CHECK_EQ(has_weight || !hessian.empty(), !d_weight.empty());
  for (const auto& page : p_fmat->GetBatches<SparsePage>()) {
    std::size_t page_nnz = page.data.Size();
    for (auto begin = 0ull; begin < page_nnz; begin += sketch_batch_num_elements) {
      std::size_t end =
          std::min(page_nnz, static_cast<std::size_t>(begin + sketch_batch_num_elements));
      ProcessWeightedBatch(ctx, page, info, begin, end, &sketch_container, num_cuts_per_feature,
                           d_weight);
    }
  }

  sketch_container.MakeCuts(ctx, &cuts, p_fmat->Info().IsColumnSplit());
  return cuts;
}
}  // namespace xgboost::common
