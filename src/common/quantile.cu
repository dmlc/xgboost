/**
 * Copyright 2020-2026, XGBoost Contributors
 */
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/tuple.h>  // for make_tuple
#include <thrust/unique.h>

#include <cstdint>      // for uintptr_t
#include <limits>       // for numeric_limits
#include <numeric>      // for partial_sum
#include <type_traits>  // for is_same_v
#include <utility>

#include "../collective/allgather.h"
#include "../collective/allreduce.h"
#include "../collective/communicator-inl.h"  // for GetWorldSize, GetRank
#include "categorical.h"
#include "common.h"
#include "cuda_context.cuh"  // for CUDAContext
#include "cuda_rt_utils.h"   // for SetDevice
#include "device_helpers.cuh"
#include "hist_util.h"
#include "quantile.cuh"
#include "quantile.h"
#include "transform_iterator.h"  // MakeIndexTransformIter
#include "xgboost/span.h"

namespace xgboost::common {
using WQSketch = WQuantileSketch;
using SketchEntry = WQSketch::Entry;

// Algorithm 4 in XGBoost's paper, using binary search to find i.
template <typename EntryIter>
__device__ size_t BinarySearchQueryIndex(EntryIter beg, EntryIter end, float rank) {
  assert(end - beg >= 2);
  rank *= 2;
  auto front = *beg;
  if (rank < front.rmin + front.rmax) {
    return 0;
  }
  auto back = *(end - 1);
  if (rank >= back.rmin + back.rmax) {
    return end - beg - 1;
  }

  auto search_begin = dh::MakeTransformIterator<float>(
      beg, [=] __device__(SketchEntry const &entry) { return entry.rmin + entry.rmax; });
  auto search_end = search_begin + (end - beg);
  auto i =
      thrust::upper_bound(thrust::seq, search_begin + 1, search_end - 1, rank) - search_begin - 1;
  if (rank < (*(beg + i)).RMinNext() + (*(beg + i + 1)).RMaxPrev()) {
    return i;
  } else {
    return i + 1;
  }
}

template <typename EntryFromIndex>
// Select source indices for the pruned summary without materializing output entries.
void SelectPruneIndices(common::Span<SketchContainer::OffsetT const> cuts_ptr,
                        Span<SketchContainer::OffsetT const> columns_ptr_in,
                        Span<FeatureType const> feature_types, Span<size_t> selected_idx,
                        EntryFromIndex entry_from_index, cudaStream_t stream) {
  dh::LaunchN(selected_idx.size(), stream, [=] __device__(size_t idx) {
    size_t column_id = dh::SegmentId(cuts_ptr, idx);
    auto in_begin = columns_ptr_in[column_id];
    auto in_size = columns_ptr_in[column_id + 1] - columns_ptr_in[column_id];
    auto to = cuts_ptr[column_id + 1] - cuts_ptr[column_id];
    idx -= cuts_ptr[column_id];

    auto is_cat = IsCat(feature_types, column_id);
    if (in_size <= to || is_cat) {
      selected_idx[cuts_ptr[column_id] + idx] = in_begin + idx;
      return;
    }
    if (idx == 0) {
      selected_idx[cuts_ptr[column_id]] = in_begin;
      return;
    }
    if (idx == to - 1) {
      selected_idx[cuts_ptr[column_id] + idx] = in_begin + in_size - 1;
      return;
    }

    auto front = entry_from_index(in_begin);
    auto back = entry_from_index(in_begin + in_size - 1);
    float w = back.rmin - front.rmax;
    auto q = ((static_cast<float>(idx) * w) / (static_cast<float>(to) - 1.0f) + front.rmax);
    auto it = dh::MakeTransformIterator<SketchEntry>(
        thrust::make_counting_iterator(in_begin),
        [=] __device__(size_t abs_idx) { return entry_from_index(abs_idx); });
    selected_idx[cuts_ptr[column_id] + idx] =
        in_begin + BinarySearchQueryIndex(it, it + in_size, q);
  });
}

template <typename EntryFromIndex>
void GatherPruneEntries(Span<size_t const> selected_idx, Span<SketchEntry> out_cuts,
                        EntryFromIndex entry_from_index, cudaStream_t stream) {
  dh::LaunchN(selected_idx.size(), stream,
              [=] __device__(size_t idx) { out_cuts[idx] = entry_from_index(selected_idx[idx]); });
}

template <typename InEntry, typename ToSketchEntry>
void PruneImpl(common::Span<SketchContainer::OffsetT const> cuts_ptr,
               Span<InEntry const> sorted_data,
               Span<size_t const> columns_ptr_in,  // could be ptr for data or cuts
               Span<FeatureType const> feature_types, Span<SketchEntry> out_cuts,
               ToSketchEntry to_sketch_entry) {
  dh::LaunchN(out_cuts.size(), [=] __device__(size_t idx) {
    size_t column_id = dh::SegmentId(cuts_ptr, idx);
    auto out_column =
        out_cuts.subspan(cuts_ptr[column_id], cuts_ptr[column_id + 1] - cuts_ptr[column_id]);
    auto in_column = sorted_data.subspan(columns_ptr_in[column_id],
                                         columns_ptr_in[column_id + 1] - columns_ptr_in[column_id]);
    auto to = cuts_ptr[column_id + 1] - cuts_ptr[column_id];
    idx -= cuts_ptr[column_id];
    auto front = to_sketch_entry(0ul, in_column, column_id);
    auto back = to_sketch_entry(in_column.size() - 1, in_column, column_id);

    auto is_cat = IsCat(feature_types, column_id);
    if (in_column.size() <= to || is_cat) {
      // cut idx equals sample idx
      out_column[idx] = to_sketch_entry(idx, in_column, column_id);
      return;
    }
    // 1 thread for each output.  See A.4 for detail.
    auto d_out = out_column;
    if (idx == 0) {
      d_out.front() = front;
      return;
    }
    if (idx == to - 1) {
      d_out.back() = back;
      return;
    }

    float w = back.rmin - front.rmax;
    auto budget = static_cast<float>(d_out.size());
    assert(budget != 0);
    auto q = ((static_cast<float>(idx) * w) / (static_cast<float>(to) - 1.0f) + front.rmax);
    auto it = dh::MakeTransformIterator<SketchEntry>(
        thrust::make_counting_iterator(0ul), [=] __device__(size_t idx) {
          auto e = to_sketch_entry(idx, in_column, column_id);
          return e;
        });
    d_out[idx] = *(it + BinarySearchQueryIndex(it, it + in_column.size(), q));
  });
}

template <typename T, typename U>
void CopyTo(Span<T> out, Span<U> src) {
  CHECK_EQ(out.size(), src.size());
  static_assert(std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<T>>);
  dh::safe_cuda(cudaMemcpyAsync(out.data(), src.data(), out.size_bytes(), cudaMemcpyDefault));
}

XGBOOST_DEVICE thrust::tuple<uint64_t, uint64_t> MergePartition(Span<SketchEntry const> x,
                                                                Span<SketchEntry const> y,
                                                                uint64_t k) {
  // Find the merge partition for the k-th output within one column.  The merged prefix of
  // length k contains i entries from x and j entries from y, where k = i + j.
  auto m = static_cast<uint64_t>(x.size());
  auto n = static_cast<uint64_t>(y.size());
  // Search for i inside the valid merge-partition range.  low/high clamp the partition so
  // j = k - i always stays within [0, n].
  auto low = k > n ? k - n : 0ul;
  auto high = std::min(k, m);
  auto candidate_it = thrust::make_counting_iterator<uint64_t>(low);
  auto need_more_x = dh::MakeTransformIterator<bool>(candidate_it, [=] __device__(uint64_t i) {
    // j is the number of elements taken from y when the partition takes i from x.
    auto j = k - i;
    // Move the boundary right while the last candidate from y still sorts ahead of the
    // next candidate from x.  The first false value is the first valid merge boundary.
    // j > 0: there is a left-hand candidate in y.
    // i < m: there is a right-hand candidate in x.
    return j > 0 && i < m && y[j - 1].value >= x[i].value;
  });
  auto partition_it = thrust::lower_bound(thrust::seq, need_more_x, need_more_x + (high - low + 1),
                                          false, thrust::greater<bool>{});
  auto a_ind = low + (partition_it - need_more_x);
  return thrust::make_tuple(a_ind, k - a_ind);
}

void SketchContainer::SetCurrentColumns(Span<OffsetT const> columns_ptr) {
  CHECK_EQ(columns_ptr.size(), num_columns_ + 1);
  CHECK_EQ(columns_ptr_tmp_.Size(), num_columns_ + 1);
  columns_ptr_.Resize(columns_ptr.size());
  CopyTo(columns_ptr_.DeviceSpan(), columns_ptr);
}

// Merge d_x and d_y into out.  Because the final output depends on predicate (which
// summary does the output element come from) result by definition of merged rank.  So we
// compute the partition for each output directly and customize the standard merge
// algorithm without storing a merge path buffer.
void MergeImpl(Context const *ctx, Span<SketchEntry const> const &d_x,
               Span<bst_idx_t const> const &x_ptr, Span<SketchEntry const> const &d_y,
               Span<bst_idx_t const> const &y_ptr, Span<SketchEntry> d_out,
               Span<bst_idx_t> out_ptr) {
  CHECK_EQ(d_x.size() + d_y.size(), d_out.size());
  CHECK_EQ(x_ptr.size(), out_ptr.size());
  CHECK_EQ(y_ptr.size(), out_ptr.size());

  dh::LaunchN(out_ptr.size(), ctx->CUDACtx()->Stream(),
              [=] __device__(size_t i) { out_ptr[i] = x_ptr[i] + y_ptr[i]; });

  auto merge_entry_at = [=] __device__(Span<SketchEntry const> d_x_column,
                                       Span<SketchEntry const> d_y_column, uint64_t idx) {
    // Materialize one merged entry for a single column and output position.
    // Handle empty column. If both columns are empty, we should not get this column as
    // result of binary search.
    assert((d_x_column.size() != 0) || (d_y_column.size() != 0));
    if (d_x_column.size() == 0) {
      return d_y_column[idx];
    }
    if (d_y_column.size() == 0) {
      return d_x_column[idx];
    }

    uint64_t a_ind, b_ind;
    thrust::tie(a_ind, b_ind) = MergePartition(d_x_column, d_y_column, idx);

    assert(b_ind <= d_y_column.size());
    assert(a_ind <= d_x_column.size());

    // Rank contribution from the opposite summary at the merge boundary.  `ind` is the
    // insertion point of the current element into the other summary.
    auto other_rmin = [] __device__(Span<SketchEntry const> d_column, uint64_t ind) {
      if (ind == 0) {
        return 0.0f;
      }
      if (ind == d_column.size()) {
        return d_column.back().RMinNext();
      }
      return d_column[ind - 1].RMinNext();
    };  // NOLINT
    auto other_rmax = [] __device__(Span<SketchEntry const> d_column, uint64_t ind) {
      if (ind == d_column.size()) {
        return d_column.back().rmax;
      }
      return d_column[ind].RMaxPrev();
    };  // NOLINT
    // Apply the merge equations when the output element comes from x or y.
    auto merge_from_x = [=] __device__(SketchEntry x_elem, uint64_t y_ind) {
      return SketchEntry{x_elem.rmin + other_rmin(d_y_column, y_ind),
                         x_elem.rmax + other_rmax(d_y_column, y_ind), x_elem.wmin, x_elem.value};
    };  // NOLINT
    auto merge_from_y = [=] __device__(SketchEntry y_elem, uint64_t x_ind) {
      return SketchEntry{other_rmin(d_x_column, x_ind) + y_elem.rmin,
                         other_rmax(d_x_column, x_ind) + y_elem.rmax, y_elem.wmin, y_elem.value};
    };  // NOLINT

    // Once one side is exhausted, all remaining outputs come from the other side with
    // boundary ranks taken at the end of the exhausted summary.
    if (a_ind == d_x_column.size()) {
      return merge_from_y(d_y_column[b_ind], a_ind);
    }
    auto x_elem = d_x_column[a_ind];
    if (b_ind == d_y_column.size()) {
      return merge_from_x(x_elem, b_ind);
    }
    auto y_elem = d_y_column[b_ind];

    /* Merge procedure.  See A.3 merge operation eq (26) ~ (28).  The trick to interpret
       it is rewriting the symbols on both side of equality.  Take eq (26) as an example:
       Expand it according to definition of extended rank then rewrite it into:

       If $k_i$ is the $i$ element in output and \textbf{comes from $D_1$}:

         r_\bar{D}(k_i) = r_{\bar{D_1}}(k_i) + w_{\bar{{D_1}}}(k_i) +
                                          [r_{\bar{D_2}}(x_i) + w_{\bar{D_2}}(x_i)]

       Where $x_i$ is the largest element in $D_2$ that's less than $k_i$.  $k_i$ can be
       used in $D_1$ as it's since $k_i \in D_1$.  Other 2 equations can be applied
       similarly with $k_i$ comes from different $D$.  just use different symbol on
       different source of summary.
    */
    // General merge case: combine equal values, otherwise land the smaller value and add
    // the rank contribution from the opposite summary at the partition boundary.
    if (x_elem.value == y_elem.value) {
      return SketchEntry{x_elem.rmin + y_elem.rmin, x_elem.rmax + y_elem.rmax,
                         x_elem.wmin + y_elem.wmin, x_elem.value};
    }
    if (x_elem.value < y_elem.value) {
      return merge_from_x(x_elem, b_ind);
    }

    return merge_from_y(y_elem, a_ind);
  };  // NOLINT

  dh::LaunchN(d_out.size(), ctx->CUDACtx()->Stream(), [=] __device__(size_t idx) {
    // Merge one output element after locating its column segment and per-column partition.
    auto column_id = dh::SegmentId(out_ptr, idx);
    auto out_begin = out_ptr[column_id];
    auto out_idx = idx - out_begin;

    auto d_x_column = d_x.subspan(x_ptr[column_id], x_ptr[column_id + 1] - x_ptr[column_id]);
    auto d_y_column = d_y.subspan(y_ptr[column_id], y_ptr[column_id + 1] - y_ptr[column_id]);
    d_out[idx] = merge_entry_at(d_x_column, d_y_column, out_idx);
  });
}

// Convert one sorted batch into a temporary pruned summary in `prune_buffer_`, normalize
// duplicated raw values in place, then merge that summary into the resident sketch in
// `entries_`. Out-of-place merge/prune results use `entries_tmp_` as scratch before being
// committed back into `entries_`.
void SketchContainer::Push(Context const *ctx, Span<Entry const> entries, Span<size_t> columns_ptr,
                           common::Span<OffsetT> cuts_ptr, size_t total_cuts,
                           bst_idx_t n_rows_in_batch, Span<float> weights) {
  curt::SetDevice(ctx->Ordinal());
  rows_seen_ += n_rows_in_batch;
  this->prune_buffer_.resize(total_cuts);
  auto out = dh::ToSpan(this->prune_buffer_);
  auto ft = this->feature_types_.ConstDeviceSpan();
  if (weights.empty()) {
    auto to_sketch_entry = [] __device__(size_t sample_idx, Span<Entry const> const &column,
                                         size_t) {
      float rmin = sample_idx;
      float rmax = sample_idx + 1;
      return SketchEntry{rmin, rmax, 1, column[sample_idx].fvalue};
    };  // NOLINT
    PruneImpl<Entry>(cuts_ptr, entries, columns_ptr, ft, out, to_sketch_entry);
  } else {
    auto to_sketch_entry = [weights, columns_ptr] __device__(size_t sample_idx,
                                                             Span<Entry const> const &column,
                                                             size_t column_id) {
      Span<float const> column_weights_scan =
          weights.subspan(columns_ptr[column_id], column.size());
      float rmin = sample_idx > 0 ? column_weights_scan[sample_idx - 1] : 0.0f;
      float rmax = column_weights_scan[sample_idx];
      float wmin = rmax - rmin;
      wmin = wmin < 0 ? kRtEps : wmin;  // GPU scan can generate floating error.
      return SketchEntry{rmin, rmax, wmin, column[sample_idx].fvalue};
    };  // NOLINT
    PruneImpl<Entry>(cuts_ptr, entries, columns_ptr, ft, out, to_sketch_entry);
  }
  auto n_uniques = this->ScanInput(ctx, out, cuts_ptr);
  CHECK_EQ(this->columns_ptr_.Size(), cuts_ptr.size());
  if (n_uniques == 0) {
    return;
  }
  this->Merge(ctx, cuts_ptr, out.subspan(0, n_uniques));
  auto intermediate_num_cuts = static_cast<bst_idx_t>(this->IntermediateNumCuts());
  this->Prune(ctx, intermediate_num_cuts);
}

size_t SketchContainer::ScanInput(Context const *ctx, Span<SketchEntry> entries,
                                  Span<OffsetT> d_columns_ptr_in) {
  /* There are 2 types of duplication.  First is duplicated feature values, which comes
   * from user input data.  Second is duplicated sketching entries, which is generated by
   * pruning or merging. We preserve the first type and remove the second type.
   */
  timer_.Start(__func__);
  curt::SetDevice(ctx->Ordinal());
  CHECK_EQ(d_columns_ptr_in.size(), num_columns_ + 1);

  auto key_it = dh::MakeTransformIterator<size_t>(
      thrust::make_reverse_iterator(thrust::make_counting_iterator(entries.size())),
      [=] __device__(size_t idx) { return dh::SegmentId(d_columns_ptr_in, idx); });
  // Reverse scan to accumulate weights into first duplicated element on left.
  auto val_it = thrust::make_reverse_iterator(dh::tend(entries));
  thrust::inclusive_scan_by_key(ctx->CUDACtx()->CTP(), key_it, key_it + entries.size(), val_it,
                                val_it, thrust::equal_to<size_t>{},
                                [] __device__(SketchEntry const &r, SketchEntry const &l) {
                                  // Only accumulate for the first type of duplication.
                                  if (l.value - r.value == 0 && l.rmin - r.rmin != 0) {
                                    auto w = l.wmin + r.wmin;
                                    SketchEntry v{l.rmin, l.rmin + w, w, l.value};
                                    return v;
                                  }
                                  return l;
                                });

  auto d_columns_ptr_out = this->columns_ptr_tmp_.DeviceSpan();
  // thrust unique_by_key preserves the first element.
  auto n_uniques =
      dh::SegmentedUnique(ctx->CUDACtx()->CTP(), d_columns_ptr_in.data(),
                          d_columns_ptr_in.data() + d_columns_ptr_in.size(), entries.data(),
                          entries.data() + entries.size(), d_columns_ptr_out.data(), entries.data(),
                          detail::SketchUnique{});
  CopyTo(d_columns_ptr_in, d_columns_ptr_out);

  timer_.Stop(__func__);
  return n_uniques;
}

void SketchContainer::Prune(Context const *ctx, std::size_t to) {
  timer_.Start(__func__);
  curt::SetDevice(ctx->Ordinal());
  auto &entries = this->entries_;
  auto &scratch = this->entries_tmp_;
  auto &columns_ptr = this->columns_ptr_;
  auto &columns_ptr_tmp = this->columns_ptr_tmp_;
  auto const &feature_types = this->feature_types_;

  if (entries.size() <= to * num_columns_) {
    timer_.Stop(__func__);
    return;
  }

  OffsetT to_total = 0;
  auto &h_columns_ptr = columns_ptr_tmp.HostVector();
  h_columns_ptr[0] = to_total;
  auto const &h_feature_types = feature_types.ConstHostSpan();
  for (bst_feature_t i = 0; i < num_columns_; ++i) {
    size_t length = this->Column(i).size();
    length = std::min(length, to);
    if (IsCat(h_feature_types, i)) {
      length = this->Column(i).size();
    }
    to_total += length;
    h_columns_ptr[i + 1] = to_total;
  }
  scratch.resize(to_total);

  auto d_columns_ptr_in = columns_ptr.ConstDeviceSpan();
  auto d_columns_ptr_out = columns_ptr_tmp.ConstDeviceSpan();
  auto out = dh::ToSpan(scratch);
  auto in = dh::ToSpan(entries);
  auto ft = feature_types.ConstDeviceSpan();
  dh::device_vector<size_t> selected_idx(out.size());
  auto d_selected_idx = dh::ToSpan(selected_idx);
  HostDeviceVector<OffsetT> selected_columns_ptr(columns_ptr_tmp.Size());
  selected_columns_ptr.SetDevice(ctx->Device());
  auto entry_from_index = [=] __device__(size_t abs_idx) {
    return in[abs_idx];
  };  // NOLINT
  auto stream = ctx->CUDACtx()->Stream();
  SelectPruneIndices(d_columns_ptr_out, d_columns_ptr_in, ft, d_selected_idx, entry_from_index,
                     stream);
  auto n_selected = dh::SegmentedUnique(
      ctx->CUDACtx()->CTP(), d_columns_ptr_out.data(),
      d_columns_ptr_out.data() + d_columns_ptr_out.size(), d_selected_idx.data(),
      d_selected_idx.data() + d_selected_idx.size(), selected_columns_ptr.DeviceSpan().data(),
      d_selected_idx.data(), thrust::equal_to<size_t>{});
  GatherPruneEntries(Span<size_t const>{d_selected_idx.data(), n_selected}, out, entry_from_index,
                     stream);
  entries.swap(scratch);
  columns_ptr.Copy(selected_columns_ptr);
  entries.resize(n_selected);
  auto d_column_scan = columns_ptr.DeviceSpan();
  HostDeviceVector<OffsetT> scan_out(d_column_scan.size());
  scan_out.SetDevice(ctx->Device());
  auto n_uniques = dh::SegmentedUnique(ctx->CUDACtx()->CTP(), d_column_scan.data(),
                                       d_column_scan.data() + d_column_scan.size(), out.data(),
                                       out.data() + n_selected, scan_out.DevicePointer(),
                                       out.data(), detail::SketchUnique{});
  columns_ptr.Copy(scan_out);
  CHECK(!columns_ptr.HostCanRead());
  entries.resize(n_uniques);
  timer_.Stop(__func__);
}

void SketchContainer::Merge(Context const *ctx, Span<OffsetT const> d_that_columns_ptr,
                            Span<SketchEntry const> that) {
  curt::SetDevice(ctx->Ordinal());
  auto &entries = this->entries_;
  auto &scratch = this->entries_tmp_;
  auto &columns_ptr = this->columns_ptr_;
  auto &columns_ptr_tmp = this->columns_ptr_tmp_;
  auto self = dh::ToSpan(entries);
  LOG(DEBUG) << "Merge: self:" << HumanMemUnit(self.size_bytes()) << ". "
             << "That:" << HumanMemUnit(that.size_bytes()) << ". "
             << "This capacity:" << HumanMemUnit(this->MemCapacityBytes()) << "." << std::endl;

  timer_.Start(__func__);
  auto normalize_merged = [&] {
    if (this->HasCategorical()) {
      // Numerical summaries are normalized during prune.  Categorical features can still
      // produce repeated category values, so compact those here before exposing the sketch.
      auto d_feature_types = this->FeatureTypes().ConstDeviceSpan();
      auto d_column_scan = columns_ptr.DeviceSpan();
      auto merged_entries = dh::ToSpan(entries);
      HostDeviceVector<OffsetT> scan_out(d_column_scan.size());
      scan_out.SetDevice(ctx->Device());
      auto n_uniques = dh::SegmentedUnique(
          ctx->CUDACtx()->CTP(), d_column_scan.data(), d_column_scan.data() + d_column_scan.size(),
          merged_entries.data(), merged_entries.data() + merged_entries.size(),
          scan_out.DevicePointer(), merged_entries.data(), detail::SketchUnique{},
          [d_feature_types] __device__(size_t l_fidx, size_t r_fidx) {
            return l_fidx == r_fidx && IsCat(d_feature_types, l_fidx);
          });
      columns_ptr.Copy(scan_out);
      entries.resize(n_uniques);
    }
    this->FixError();
  };
  if (entries.empty()) {
    CHECK_EQ(columns_ptr.HostVector().back(), 0);
    CHECK_EQ(columns_ptr.HostVector().size(), d_that_columns_ptr.size());
    CHECK_EQ(columns_ptr.Size(), num_columns_ + 1);
    thrust::copy(ctx->CUDACtx()->CTP(), d_that_columns_ptr.data(),
                 d_that_columns_ptr.data() + d_that_columns_ptr.size(),
                 columns_ptr.DevicePointer());
    auto total = columns_ptr.HostVector().back();
    entries.resize(total);
    CopyTo(dh::ToSpan(entries), that);
    normalize_merged();
    timer_.Stop(__func__);
    return;
  }

  std::size_t new_size = entries.size() + that.size();
  try {
    scratch.resize(new_size);
  } catch (dmlc::Error const &) {
    // Retry
    scratch.clear();
    scratch.shrink_to_fit();
    scratch.resize(new_size);
  }

  CHECK_EQ(d_that_columns_ptr.size(), columns_ptr.Size());

  MergeImpl(ctx, {entries.data().get(), entries.size()}, columns_ptr.ConstDeviceSpan(), that,
            d_that_columns_ptr, dh::ToSpan(scratch), columns_ptr_tmp.DeviceSpan());
  this->CommitScratch(new_size);
  CHECK_EQ(columns_ptr.Size(), num_columns_ + 1);
  normalize_merged();
  timer_.Stop(__func__);
}

void SketchContainer::FixError() {
  auto d_columns_ptr = this->columns_ptr_.ConstDeviceSpan();
  auto in = dh::ToSpan(this->entries_);
  dh::LaunchN(in.size(), [=] __device__(size_t idx) {
    auto column_id = dh::SegmentId(d_columns_ptr, idx);
    auto in_column = in.subspan(d_columns_ptr[column_id],
                                d_columns_ptr[column_id + 1] - d_columns_ptr[column_id]);
    idx -= d_columns_ptr[column_id];
    float prev_rmin = idx == 0 ? 0.0f : in_column[idx - 1].rmin;
    if (in_column[idx].rmin < prev_rmin) {
      in_column[idx].rmin = prev_rmin;
    }
    float prev_rmax = idx == 0 ? 0.0f : in_column[idx - 1].rmax;
    if (in_column[idx].rmax < prev_rmax) {
      in_column[idx].rmax = prev_rmax;
    }
    float rmin_next = in_column[idx].RMinNext();
    if (in_column[idx].rmax < rmin_next) {
      in_column[idx].rmax = rmin_next;
    }
  });
}

void SketchContainer::AllReduce(Context const *ctx, bool is_column_split) {
  curt::SetDevice(ctx->Ordinal());
  auto world = collective::GetWorldSize();
  if (world == 1 || is_column_split) {
    return;
  }

  timer_.Start(__func__);
  // Bound local sketch size before exchanging data across workers.
  auto intermediate_num_cuts = static_cast<bst_idx_t>(num_bins_ * kFactor);
  this->Prune(ctx, intermediate_num_cuts);

  auto d_columns_ptr = this->columns_ptr_.ConstDeviceSpan();
  CHECK_EQ(d_columns_ptr.size(), num_columns_ + 1);
  size_t n = d_columns_ptr.size();
  auto rc = collective::Allreduce(ctx, linalg::MakeVec(&n, 1), collective::Op::kMax);
  SafeColl(rc);
  CHECK_EQ(n, d_columns_ptr.size()) << "Number of columns differs across workers";

  // Get the columns ptr from all workers
  dh::device_vector<SketchContainer::OffsetT> gathered_ptrs;
  gathered_ptrs.resize(d_columns_ptr.size() * world, 0);
  size_t rank = collective::GetRank();
  auto offset = rank * d_columns_ptr.size();
  thrust::copy(thrust::device, d_columns_ptr.data(), d_columns_ptr.data() + d_columns_ptr.size(),
               gathered_ptrs.begin() + offset);
  rc = collective::Allreduce(
      ctx, linalg::MakeVec(gathered_ptrs.data().get(), gathered_ptrs.size(), ctx->Device()),
      collective::Op::kSum);
  SafeColl(rc);

  // Get the data from all workers.
  std::vector<std::int64_t> recv_lengths;
  HostDeviceVector<std::int8_t> recvbuf;
  rc = collective::AllgatherV(
      ctx, linalg::MakeVec(this->entries_.data().get(), this->entries_.size(), ctx->Device()),
      &recv_lengths, &recvbuf);
  collective::SafeColl(rc);
  for (std::size_t i = 0; i < recv_lengths.size() - 1; ++i) {
    recv_lengths[i] = recv_lengths[i + 1] - recv_lengths[i];
  }
  recv_lengths.resize(recv_lengths.size() - 1);

  // Segment the received data.
  auto s_recvbuf = recvbuf.DeviceSpan();
  std::vector<Span<SketchEntry>> allworkers;
  offset = 0;
  for (int32_t i = 0; i < world; ++i) {
    size_t length_as_bytes = recv_lengths.at(i);
    auto raw = s_recvbuf.subspan(offset, length_as_bytes);
    CHECK_EQ(length_as_bytes % sizeof(SketchEntry), 0)
        << "Allgathered GPU sketch buffer has invalid size.";
    auto ptr = reinterpret_cast<std::uintptr_t>(raw.data());
    CHECK_EQ(ptr % alignof(SketchEntry), 0) << "Allgathered GPU sketch buffer is misaligned.";
    auto sketch = Span<SketchEntry>(reinterpret_cast<SketchEntry *>(raw.data()),
                                    length_as_bytes / sizeof(SketchEntry));
    allworkers.emplace_back(sketch);
    offset += length_as_bytes;
  }
  // Stop the timer early to avoid interference from the new sketch container.
  timer_.Stop(__func__);

  // Merge them into a new sketch.
  SketchContainer new_sketch(this->feature_types_, num_bins_, this->num_columns_, ctx->Device());
  for (size_t i = 0; i < allworkers.size(); ++i) {
    auto worker = allworkers[i];
    auto worker_ptr =
        dh::ToSpan(gathered_ptrs).subspan(i * d_columns_ptr.size(), d_columns_ptr.size());
    new_sketch.Merge(ctx, worker_ptr, worker);
  }

  *this = std::move(new_sketch);
}

namespace {
struct InvalidCatOp {
  Span<SketchEntry const> values;
  Span<size_t const> ptrs;
  Span<FeatureType const> ft;

  XGBOOST_DEVICE bool operator()(size_t i) const {
    auto fidx = dh::SegmentId(ptrs, i);
    return IsCat(ft, fidx) && InvalidCat(values[i].value);
  }
};
}  // anonymous namespace

HistogramCuts SketchContainer::MakeCuts(Context const *ctx, bool is_column_split) {
  curt::SetDevice(ctx->Ordinal());
  HistogramCuts cuts{num_columns_};
  auto *p_cuts = &cuts;

  // Sync between workers.
  this->AllReduce(ctx, is_column_split);

  timer_.Start(__func__);
  // Prune to final number of bins.
  this->Prune(ctx, num_bins_ + 1);

  // Set up inputs
  auto d_in_columns_ptr = this->columns_ptr_.ConstDeviceSpan();

  auto const in_cut_values = dh::ToSpan(this->entries_);

  // Set up output ptr
  p_cuts->cut_ptrs_.SetDevice(ctx->Device());
  auto &h_out_columns_ptr = p_cuts->cut_ptrs_.HostVector();
  h_out_columns_ptr.front() = 0;
  auto const &h_feature_types = this->feature_types_.ConstHostSpan();

  auto d_ft = feature_types_.ConstDeviceSpan();

  std::vector<SketchEntry> max_values;
  float max_cat{-1.f};
  if (has_categorical_) {
    auto key_it = dh::MakeTransformIterator<bst_feature_t>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) -> bst_feature_t {
          return dh::SegmentId(d_in_columns_ptr, i);
        });
    auto invalid_op = InvalidCatOp{in_cut_values, d_in_columns_ptr, d_ft};
    auto val_it = dh::MakeTransformIterator<SketchEntry>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) {
          auto fidx = dh::SegmentId(d_in_columns_ptr, i);
          auto v = in_cut_values[i];
          if (IsCat(d_ft, fidx)) {
            if (invalid_op(i)) {
              // use inf to indicate invalid value, this way we can keep it as in
              // indicator in the reduce operation as it's always the greatest value.
              v.value = std::numeric_limits<float>::infinity();
            }
          }
          return v;
        });
    CHECK_EQ(num_columns_, d_in_columns_ptr.size() - 1);
    max_values.resize(d_in_columns_ptr.size() - 1);

    // In some cases (e.g. column-wise data split), we may have empty columns, so we need to keep
    // track of the unique keys (feature indices) after the thrust::reduce_by_key` call.
    dh::caching_device_vector<size_t> d_max_keys(d_in_columns_ptr.size() - 1);
    dh::caching_device_vector<SketchEntry> d_max_values(d_in_columns_ptr.size() - 1);
    auto new_end = thrust::reduce_by_key(
        ctx->CUDACtx()->CTP(), key_it, key_it + in_cut_values.size(), val_it, d_max_keys.begin(),
        d_max_values.begin(), thrust::equal_to<bst_feature_t>{},
        [] __device__(auto l, auto r) { return l.value > r.value ? l : r; });
    d_max_keys.erase(new_end.first, d_max_keys.end());
    d_max_values.erase(new_end.second, d_max_values.end());

    // The device vector needs to be initialized explicitly since we may have some missing columns.
    SketchEntry default_entry{};
    dh::caching_device_vector<SketchEntry> d_max_results(d_in_columns_ptr.size() - 1,
                                                         default_entry);
    thrust::scatter(ctx->CUDACtx()->CTP(), d_max_values.begin(), d_max_values.end(),
                    d_max_keys.begin(), d_max_results.begin());
    dh::CopyDeviceSpanToVector(&max_values, dh::ToSpan(d_max_results));
    auto max_it = MakeIndexTransformIter([&](auto i) {
      if (IsCat(h_feature_types, i)) {
        return max_values[i].value;
      }
      return -1.f;
    });
    max_cat = *std::max_element(max_it, max_it + max_values.size());
    if (std::isinf(max_cat)) {
      InvalidCategory();
    }
  }

  // Set up output cuts
  for (bst_feature_t i = 0; i < num_columns_; ++i) {
    size_t column_size = std::max(static_cast<size_t>(1ul), this->Column(i).size());
    if (IsCat(h_feature_types, i)) {
      // column_size is the number of unique values in that feature.
      CheckMaxCat(max_values[i].value, column_size);
      h_out_columns_ptr[i + 1] = max_values[i].value + 1;  // includes both max_cat and 0.
    } else {
      h_out_columns_ptr[i + 1] =
          std::min(static_cast<size_t>(column_size), static_cast<size_t>(num_bins_));
    }
  }
  std::partial_sum(h_out_columns_ptr.begin(), h_out_columns_ptr.end(), h_out_columns_ptr.begin());
  auto d_out_columns_ptr = p_cuts->cut_ptrs_.ConstDeviceSpan();

  size_t total_bins = h_out_columns_ptr.back();
  p_cuts->cut_values_.SetDevice(ctx->Device());
  p_cuts->cut_values_.Resize(total_bins);
  auto out_cut_values = p_cuts->cut_values_.DeviceSpan();

  dh::LaunchN(total_bins, [=] __device__(size_t idx) {
    auto column_id = dh::SegmentId(d_out_columns_ptr, idx);
    auto in_column = in_cut_values.subspan(
        d_in_columns_ptr[column_id], d_in_columns_ptr[column_id + 1] - d_in_columns_ptr[column_id]);
    auto out_column =
        out_cut_values.subspan(d_out_columns_ptr[column_id],
                               d_out_columns_ptr[column_id + 1] - d_out_columns_ptr[column_id]);
    idx -= d_out_columns_ptr[column_id];
    if (in_column.size() == 0) {
      // If the column is empty, we push a dummy value.  It won't affect training as the
      // column is empty, trees cannot split on it.  This is just to be consistent with
      // rest of the library.
      if (idx == 0) {
        out_column[0] = kRtEps;
        assert(out_column.size() == 1);
      }
      return;
    }

    if (IsCat(d_ft, column_id)) {
      out_column[idx] = idx;
      return;
    }

    // Last thread is responsible for setting a value that's greater than other cuts.
    if (idx == out_column.size() - 1) {
      const bst_float cpt = in_column.back().value;
      // this must be bigger than last value in a scale
      const bst_float last = cpt + (fabs(cpt) + 1e-5);
      out_column[idx] = last;
      return;
    }
    assert(idx + 1 < in_column.size());
    out_column[idx] = in_column[idx + 1].value;
  });

  p_cuts->SetCategorical(this->has_categorical_, max_cat);
  timer_.Stop(__func__);
  return cuts;
}
}  // namespace xgboost::common
