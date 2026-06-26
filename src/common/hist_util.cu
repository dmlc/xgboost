/**
 * Copyright 2018~2026, XGBoost contributors
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
#include <thrust/tuple.h>  // for tuple
#include <xgboost/logging.h>

#include <algorithm>
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
namespace detail {
size_t RequiredSampleCutsPerColumn(int max_bins, size_t num_rows) {
  double eps = SketchEpsilon(max_bins, num_rows);
  size_t num_cuts = WQuantileSketch::LimitSizeLevel(num_rows, eps);
  return std::min(num_cuts, num_rows);
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

  // Renew the column scan based on categorical data. Numerical columns preserve their original
  // span, while categorical columns shrink to their unique category count.
  column_sizes_scan = std::move(new_column_scan);
}
}  // namespace detail

namespace {
[[nodiscard]] bst_idx_t RowsInEntrySpan(SparsePage const& page, std::size_t begin,
                                        std::size_t end) {
  CHECK_LT(begin, end);
  auto const& h_offset = page.offset.ConstHostVector();
  auto row_begin_it = std::upper_bound(h_offset.cbegin(), h_offset.cend(), begin);
  auto row_end_it = std::lower_bound(h_offset.cbegin(), h_offset.cend(), end);
  auto row_begin = std::distance(h_offset.cbegin(), row_begin_it) - 1;
  auto row_end = std::distance(h_offset.cbegin(), row_end_it);
  CHECK_LE(row_begin, row_end);
  return std::max<bst_idx_t>(1, row_end - row_begin);
}
}  // namespace

void ProcessWeightedBatch(Context const* ctx, const SparsePage& page, MetaInfo const& info,
                          std::size_t begin, std::size_t end,
                          SketchContainer* sketch_container,  // <- output sketch
                          common::Span<float const> sample_weight) {
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

  dh::caching_device_vector<size_t> column_sizes_scan;
  data::IsValidFunctor dummy_is_valid(std::numeric_limits<float>::quiet_NaN());
  auto batch_it = dh::MakeTransformIterator<data::COOTuple>(
      sorted_entries.data().get(), [] __device__(Entry const& e) -> data::COOTuple {
        return {0, e.index, e.fvalue};  // row_idx is not needed for scaning column size.
      });
  detail::GetColumnSizesScan(ctx->CUDACtx(), ctx->Device(), info.num_col_,
                             IterSpan{batch_it, sorted_entries.size()}, dummy_is_valid,
                             &column_sizes_scan);
  if (sketch_container->HasCategorical()) {
    auto p_weight = entry_weight.empty() ? nullptr : &entry_weight;
    detail::RemoveDuplicatedCategories(ctx, info, &sorted_entries, p_weight, &column_sizes_scan);
  }

  // Add cuts into sketches
  auto n_rows_in_batch = RowsInEntrySpan(page, begin, end);
  sketch_container->Push(ctx, dh::ToSpan(sorted_entries), dh::ToSpan(column_sizes_scan),
                         n_rows_in_batch, dh::ToSpan(entry_weight));

  sorted_entries.clear();
  sorted_entries.shrink_to_fit();
  CHECK_EQ(sorted_entries.capacity(), 0);
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
                                      Span<float const> hessian) {
  auto const& info = p_fmat->Info();
  bool has_weight = !info.weights_.Empty();
  info.feature_types.SetDevice(ctx->Device());

  HostDeviceVector<float> weight;
  weight.SetDevice(ctx->Device());

  auto sketch_batch_num_elements = detail::kSketchBatchNumElements;

  CUDAContext const* cuctx = ctx->CUDACtx();

  info.weights_.SetDevice(ctx->Device());
  auto d_weight = UnifyWeight(cuctx, info, hessian, &weight);

  SketchContainer sketch_container(info.feature_types, max_bin, info.num_col_, ctx->Device());
  CHECK_EQ(has_weight || !hessian.empty(), !d_weight.empty());
  for (const auto& page : p_fmat->GetBatches<SparsePage>()) {
    std::size_t page_nnz = page.data.Size();
    for (auto begin = 0ull; begin < page_nnz; begin += sketch_batch_num_elements) {
      std::size_t end =
          std::min(page_nnz, static_cast<std::size_t>(begin + sketch_batch_num_elements));
      ProcessWeightedBatch(ctx, page, info, begin, end, &sketch_container, d_weight);
    }
  }

  return sketch_container.MakeCuts(ctx, p_fmat->Info().IsColumnSplit());
}
}  // namespace xgboost::common
