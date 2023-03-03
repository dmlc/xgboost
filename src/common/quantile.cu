/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

#include <limits>  // std::numeric_limits
#include <memory>
#include <utility>

#include "../collective/communicator.h"
#include "../collective/device_communicator.cuh"
#include "categorical.h"
#include "common.h"
#include "device_helpers.cuh"
#include "hist_util.h"
#include "quantile.cuh"
#include "quantile.h"
#include "transform_iterator.h"  // MakeIndexTransformIter
#include "xgboost/span.h"

namespace xgboost {
namespace common {

using WQSketch = HostSketchContainer::WQSketch;
using SketchEntry = WQSketch::Entry;

// Algorithm 4 in XGBoost's paper, using binary search to find i.
template <typename EntryIter>
__device__ SketchEntry BinarySearchQuery(EntryIter beg, EntryIter end, float rank) {
  assert(end - beg >= 2);
  rank *= 2;
  auto front = *beg;
  if (rank < front.rmin + front.rmax) {
    return *beg;
  }
  auto back = *(end - 1);
  if (rank >= back.rmin + back.rmax) {
    return back;
  }

  auto search_begin = dh::MakeTransformIterator<float>(
      beg, [=] __device__(SketchEntry const &entry) {
        return entry.rmin + entry.rmax;
      });
  auto search_end = search_begin + (end - beg);
  auto i =
      thrust::upper_bound(thrust::seq, search_begin + 1, search_end - 1, rank) -
      search_begin - 1;
  if (rank < (*(beg + i)).RMinNext() + (*(beg + i + 1)).RMaxPrev()) {
    return *(beg + i);
  } else {
    return *(beg + i + 1);
  }
}

template <typename InEntry, typename ToSketchEntry>
void PruneImpl(common::Span<SketchContainer::OffsetT const> cuts_ptr,
               Span<InEntry const> sorted_data,
               Span<size_t const> columns_ptr_in,  // could be ptr for data or cuts
               Span<FeatureType const> feature_types, Span<SketchEntry> out_cuts,
               ToSketchEntry to_sketch_entry) {
  dh::LaunchN(out_cuts.size(), [=] __device__(size_t idx) {
    size_t column_id = dh::SegmentId(cuts_ptr, idx);
    auto out_column = out_cuts.subspan(
        cuts_ptr[column_id], cuts_ptr[column_id + 1] - cuts_ptr[column_id]);
    auto in_column = sorted_data.subspan(columns_ptr_in[column_id],
                                         columns_ptr_in[column_id + 1] -
                                             columns_ptr_in[column_id]);
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
    d_out[idx] = BinarySearchQuery(it, it + in_column.size(), q);
  });
}

template <typename T, typename U>
void CopyTo(Span<T> out, Span<U> src) {
  CHECK_EQ(out.size(), src.size());
  static_assert(std::is_same<std::remove_cv_t<T>, std::remove_cv_t<T>>::value);
  dh::safe_cuda(cudaMemcpyAsync(out.data(), src.data(),
                                out.size_bytes(),
                                cudaMemcpyDefault));
}

// Compute the merge path.
common::Span<thrust::tuple<uint64_t, uint64_t>> MergePath(
    Span<SketchEntry const> const &d_x, Span<bst_row_t const> const &x_ptr,
    Span<SketchEntry const> const &d_y, Span<bst_row_t const> const &y_ptr,
    Span<SketchEntry> out, Span<bst_row_t> out_ptr) {
  auto x_merge_key_it = thrust::make_zip_iterator(thrust::make_tuple(
      dh::MakeTransformIterator<bst_row_t>(
          thrust::make_counting_iterator(0ul),
          [=] __device__(size_t idx) { return dh::SegmentId(x_ptr, idx); }),
      d_x.data()));
  auto y_merge_key_it = thrust::make_zip_iterator(thrust::make_tuple(
      dh::MakeTransformIterator<bst_row_t>(
          thrust::make_counting_iterator(0ul),
          [=] __device__(size_t idx) { return dh::SegmentId(y_ptr, idx); }),
      d_y.data()));

  using Tuple = thrust::tuple<uint64_t, uint64_t>;

  thrust::constant_iterator<uint64_t> a_ind_iter(0ul);
  thrust::constant_iterator<uint64_t> b_ind_iter(1ul);

  auto place_holder = thrust::make_constant_iterator<uint64_t>(0u);
  auto x_merge_val_it =
      thrust::make_zip_iterator(thrust::make_tuple(a_ind_iter, place_holder));
  auto y_merge_val_it =
      thrust::make_zip_iterator(thrust::make_tuple(b_ind_iter, place_holder));

  dh::XGBCachingDeviceAllocator<Tuple> alloc;
  static_assert(sizeof(Tuple) == sizeof(SketchEntry));
  // We reuse the memory for storing merge path.
  common::Span<Tuple> merge_path{reinterpret_cast<Tuple *>(out.data()), out.size()};
  // Determine the merge path, 0 if element is from x, 1 if it's from y.
  thrust::merge_by_key(
      thrust::cuda::par(alloc), x_merge_key_it, x_merge_key_it + d_x.size(),
      y_merge_key_it, y_merge_key_it + d_y.size(), x_merge_val_it,
      y_merge_val_it, thrust::make_discard_iterator(), merge_path.data(),
      [=] __device__(auto const &l, auto const &r) -> bool {
        auto l_column_id = thrust::get<0>(l);
        auto r_column_id = thrust::get<0>(r);
        if (l_column_id == r_column_id) {
          return thrust::get<1>(l).value < thrust::get<1>(r).value;
        }
        return l_column_id < r_column_id;
      });

  // Compute output ptr
  auto transform_it =
      thrust::make_zip_iterator(thrust::make_tuple(x_ptr.data(), y_ptr.data()));
  thrust::transform(
      thrust::cuda::par(alloc), transform_it, transform_it + x_ptr.size(),
      out_ptr.data(),
      [] __device__(auto const& t) { return thrust::get<0>(t) + thrust::get<1>(t); });

  // 0^th is the indicator, 1^th is placeholder
  auto get_ind = []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<0>(t); };
  // 0^th is the counter for x, 1^th for y.
  auto get_x =   []XGBOOST_DEVICE(Tuple const &t) { return thrust::get<0>(t); };
  auto get_y =   []XGBOOST_DEVICE(Tuple const &t) { return thrust::get<1>(t); };

  auto scan_key_it = dh::MakeTransformIterator<size_t>(
      thrust::make_counting_iterator(0ul),
      [=] __device__(size_t idx) { return dh::SegmentId(out_ptr, idx); });

  auto scan_val_it = dh::MakeTransformIterator<Tuple>(
      merge_path.data(), [=] __device__(Tuple const &t) -> Tuple {
        auto ind = get_ind(t);  // == 0 if element is from x
        // x_counter, y_counter
        return thrust::make_tuple<uint64_t, uint64_t>(!ind, ind);
      });

  // Compute the index for both x and y (which of the element in a and b are used in each
  // comparison) by scanning the binary merge path.  Take output [(x_0, y_0), (x_0, y_1),
  // ...] as an example, the comparison between (x_0, y_0) adds 1 step in the merge path.
  // Assuming y_0 is less than x_0 so this step is toward the end of y.  After the
  // comparison, index of y is incremented by 1 from y_0 to y_1, and at the same time, y_0
  // is landed into output as the first element in merge result.  The scan result is the
  // subscript of x and y.
  thrust::exclusive_scan_by_key(
      thrust::cuda::par(alloc), scan_key_it, scan_key_it + merge_path.size(),
      scan_val_it, merge_path.data(),
      thrust::make_tuple<uint64_t, uint64_t>(0ul, 0ul),
      thrust::equal_to<size_t>{},
      [=] __device__(Tuple const &l, Tuple const &r) -> Tuple {
        return thrust::make_tuple(get_x(l) + get_x(r), get_y(l) + get_y(r));
      });

  return merge_path;
}

// Merge d_x and d_y into out.  Because the final output depends on predicate (which
// summary does the output element come from) result by definition of merged rank.  So we
// run it in 2 passes to obtain the merge path and then customize the standard merge
// algorithm.
void MergeImpl(int32_t device, Span<SketchEntry const> const &d_x,
               Span<bst_row_t const> const &x_ptr, Span<SketchEntry const> const &d_y,
               Span<bst_row_t const> const &y_ptr, Span<SketchEntry> out, Span<bst_row_t> out_ptr) {
  dh::safe_cuda(cudaSetDevice(device));
  CHECK_EQ(d_x.size() + d_y.size(), out.size());
  CHECK_EQ(x_ptr.size(), out_ptr.size());
  CHECK_EQ(y_ptr.size(), out_ptr.size());

  auto d_merge_path = MergePath(d_x, x_ptr, d_y, y_ptr, out, out_ptr);
  auto d_out = out;

  dh::LaunchN(d_out.size(), [=] __device__(size_t idx) {
    auto column_id = dh::SegmentId(out_ptr, idx);
    idx -= out_ptr[column_id];

    auto d_x_column =
        d_x.subspan(x_ptr[column_id], x_ptr[column_id + 1] - x_ptr[column_id]);
    auto d_y_column =
        d_y.subspan(y_ptr[column_id], y_ptr[column_id + 1] - y_ptr[column_id]);
    auto d_out_column = d_out.subspan(
        out_ptr[column_id], out_ptr[column_id + 1] - out_ptr[column_id]);
    auto d_path_column = d_merge_path.subspan(
        out_ptr[column_id], out_ptr[column_id + 1] - out_ptr[column_id]);

    uint64_t a_ind, b_ind;
    thrust::tie(a_ind, b_ind) = d_path_column[idx];

    // Handle empty column.  If both columns are empty, we should not get this column_id
    // as result of binary search.
    assert((d_x_column.size() != 0) || (d_y_column.size() != 0));
    if (d_x_column.size() == 0) {
      d_out_column[idx] = d_y_column[b_ind];
      return;
    }
    if (d_y_column.size() == 0) {
      d_out_column[idx] = d_x_column[a_ind];
      return;
    }

    // Handle trailing elements.
    assert(a_ind <= d_x_column.size());
    if (a_ind == d_x_column.size()) {
      // Trailing elements are from y because there's no more x to land.
      auto y_elem = d_y_column[b_ind];
      d_out_column[idx] = SketchEntry(y_elem.rmin + d_x_column.back().RMinNext(),
                                      y_elem.rmax + d_x_column.back().rmax,
                                      y_elem.wmin, y_elem.value);
      return;
    }
    auto x_elem = d_x_column[a_ind];
    assert(b_ind <= d_y_column.size());
    if (b_ind == d_y_column.size()) {
      d_out_column[idx] = SketchEntry(x_elem.rmin + d_y_column.back().RMinNext(),
                                      x_elem.rmax + d_y_column.back().rmax,
                                      x_elem.wmin, x_elem.value);
      return;
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
    assert(idx < d_out_column.size());
    if (x_elem.value == y_elem.value) {
      d_out_column[idx] =
          SketchEntry{x_elem.rmin + y_elem.rmin, x_elem.rmax + y_elem.rmax,
                      x_elem.wmin + y_elem.wmin, x_elem.value};
    } else if (x_elem.value < y_elem.value) {
      // elem from x is landed. yprev_min is the element in D_2 that's 1 rank less than
      // x_elem if we put x_elem in D_2.
      float yprev_min = b_ind == 0 ? 0.0f : d_y_column[b_ind - 1].RMinNext();
      // rmin should be equal to x_elem.rmin + x_elem.wmin + yprev_min.  But for
      // implementation, the weight is stored in a separated field and we compute the
      // extended definition on the fly when needed.
      d_out_column[idx] =
          SketchEntry{x_elem.rmin + yprev_min, x_elem.rmax + y_elem.RMaxPrev(),
                      x_elem.wmin, x_elem.value};
    } else {
      // elem from y is landed.
      float xprev_min = a_ind == 0 ? 0.0f : d_x_column[a_ind - 1].RMinNext();
      d_out_column[idx] =
          SketchEntry{xprev_min + y_elem.rmin, x_elem.RMaxPrev() + y_elem.rmax,
                      y_elem.wmin, y_elem.value};
    }
  });
}

void SketchContainer::Push(Span<Entry const> entries, Span<size_t> columns_ptr,
                           common::Span<OffsetT> cuts_ptr,
                           size_t total_cuts, Span<float> weights) {
  dh::safe_cuda(cudaSetDevice(device_));
  Span<SketchEntry> out;
  dh::device_vector<SketchEntry> cuts;
  bool first_window = this->Current().empty();
  if (!first_window) {
    cuts.resize(total_cuts);
    out = dh::ToSpan(cuts);
  } else {
    this->Current().resize(total_cuts);
    out = dh::ToSpan(this->Current());
  }
  auto ft = this->feature_types_.ConstDeviceSpan();
  if (weights.empty()) {
    auto to_sketch_entry = [] __device__(size_t sample_idx,
                                         Span<Entry const> const &column,
                                         size_t) {
      float rmin = sample_idx;
      float rmax = sample_idx + 1;
      return SketchEntry{rmin, rmax, 1, column[sample_idx].fvalue};
    }; // NOLINT
    PruneImpl<Entry>(cuts_ptr, entries, columns_ptr, ft, out, to_sketch_entry);
  } else {
    auto to_sketch_entry = [weights, columns_ptr] __device__(
                               size_t sample_idx,
                               Span<Entry const> const &column,
                               size_t column_id) {
      Span<float const> column_weights_scan =
          weights.subspan(columns_ptr[column_id], column.size());
      float rmin = sample_idx > 0 ? column_weights_scan[sample_idx - 1] : 0.0f;
      float rmax = column_weights_scan[sample_idx];
      float wmin = rmax - rmin;
      wmin = wmin < 0 ? kRtEps : wmin;  // GPU scan can generate floating error.
      return SketchEntry{rmin, rmax, wmin, column[sample_idx].fvalue};
    }; // NOLINT
    PruneImpl<Entry>(cuts_ptr, entries, columns_ptr, ft, out, to_sketch_entry);
  }
  auto n_uniques = this->ScanInput(out, cuts_ptr);

  if (!first_window) {
    CHECK_EQ(this->columns_ptr_.Size(), cuts_ptr.size());
    out = out.subspan(0, n_uniques);
    this->Merge(cuts_ptr, out);
    this->FixError();
  } else {
    this->Current().resize(n_uniques);
    this->columns_ptr_.SetDevice(device_);
    this->columns_ptr_.Resize(cuts_ptr.size());

    auto d_cuts_ptr = this->columns_ptr_.DeviceSpan();
    CopyTo(d_cuts_ptr, cuts_ptr);
  }
}

size_t SketchContainer::ScanInput(Span<SketchEntry> entries, Span<OffsetT> d_columns_ptr_in) {
  /* There are 2 types of duplication.  First is duplicated feature values, which comes
   * from user input data.  Second is duplicated sketching entries, which is generated by
   * pruning or merging. We preserve the first type and remove the second type.
   */
  timer_.Start(__func__);
  dh::safe_cuda(cudaSetDevice(device_));
  CHECK_EQ(d_columns_ptr_in.size(), num_columns_ + 1);
  dh::XGBCachingDeviceAllocator<char> alloc;

  auto key_it = dh::MakeTransformIterator<size_t>(
      thrust::make_reverse_iterator(thrust::make_counting_iterator(entries.size())),
      [=] __device__(size_t idx) {
        return dh::SegmentId(d_columns_ptr_in, idx);
      });
  // Reverse scan to accumulate weights into first duplicated element on left.
  auto val_it = thrust::make_reverse_iterator(dh::tend(entries));
  thrust::inclusive_scan_by_key(
      thrust::cuda::par(alloc), key_it, key_it + entries.size(),
      val_it, val_it,
      thrust::equal_to<size_t>{},
      [] __device__(SketchEntry const &r, SketchEntry const &l) {
        // Only accumulate for the first type of duplication.
        if (l.value - r.value == 0 && l.rmin - r.rmin != 0) {
          auto w = l.wmin + r.wmin;
          SketchEntry v{l.rmin, l.rmin + w, w, l.value};
          return v;
        }
        return l;
      });

  auto d_columns_ptr_out = columns_ptr_b_.DeviceSpan();
  // thrust unique_by_key preserves the first element.
  auto n_uniques = dh::SegmentedUnique(
      d_columns_ptr_in.data(),
      d_columns_ptr_in.data() + d_columns_ptr_in.size(), entries.data(),
      entries.data() + entries.size(), d_columns_ptr_out.data(), entries.data(),
      detail::SketchUnique{});
  CopyTo(d_columns_ptr_in, d_columns_ptr_out);

  timer_.Stop(__func__);
  return n_uniques;
}

void SketchContainer::Prune(size_t to) {
  timer_.Start(__func__);
  dh::safe_cuda(cudaSetDevice(device_));

  OffsetT to_total = 0;
  auto& h_columns_ptr = columns_ptr_b_.HostVector();
  h_columns_ptr[0] = to_total;
  auto const& h_feature_types = feature_types_.ConstHostSpan();
  for (bst_feature_t i = 0; i < num_columns_; ++i) {
    size_t length = this->Column(i).size();
    length = std::min(length, to);
    if (IsCat(h_feature_types, i)) {
      length = this->Column(i).size();
    }
    to_total += length;
    h_columns_ptr[i+1] = to_total;
  }
  this->Other().resize(to_total);

  auto d_columns_ptr_in = this->columns_ptr_.ConstDeviceSpan();
  auto d_columns_ptr_out = columns_ptr_b_.ConstDeviceSpan();
  auto out = dh::ToSpan(this->Other());
  auto in = dh::ToSpan(this->Current());
  auto no_op = [] __device__(size_t sample_idx,
                             Span<SketchEntry const> const &entries,
                             size_t) { return entries[sample_idx]; }; // NOLINT
  auto ft = this->feature_types_.ConstDeviceSpan();
  PruneImpl<SketchEntry>(d_columns_ptr_out, in, d_columns_ptr_in, ft, out, no_op);
  this->columns_ptr_.Copy(columns_ptr_b_);
  this->Alternate();

  this->Unique();
  timer_.Stop(__func__);
}

void SketchContainer::Merge(Span<OffsetT const> d_that_columns_ptr,
                            Span<SketchEntry const> that) {
  dh::safe_cuda(cudaSetDevice(device_));
  timer_.Start(__func__);
  if (this->Current().size() == 0) {
    CHECK_EQ(this->columns_ptr_.HostVector().back(), 0);
    CHECK_EQ(this->columns_ptr_.HostVector().size(), d_that_columns_ptr.size());
    CHECK_EQ(columns_ptr_.Size(), num_columns_ + 1);
    thrust::copy(thrust::device, d_that_columns_ptr.data(),
                 d_that_columns_ptr.data() + d_that_columns_ptr.size(),
                 this->columns_ptr_.DevicePointer());
    auto total = this->columns_ptr_.HostVector().back();
    this->Current().resize(total);
    CopyTo(dh::ToSpan(this->Current()), that);
    timer_.Stop(__func__);
    return;
  }

  this->Other().resize(this->Current().size() + that.size());
  CHECK_EQ(d_that_columns_ptr.size(), this->columns_ptr_.Size());

  MergeImpl(device_, this->Data(), this->ColumnsPtr(), that, d_that_columns_ptr,
            dh::ToSpan(this->Other()), columns_ptr_b_.DeviceSpan());
  this->columns_ptr_.Copy(columns_ptr_b_);
  CHECK_EQ(this->columns_ptr_.Size(), num_columns_ + 1);
  this->Alternate();

  if (this->HasCategorical()) {
    auto d_feature_types = this->FeatureTypes().ConstDeviceSpan();
    this->Unique([d_feature_types] __device__(size_t l_fidx, size_t r_fidx) {
      return l_fidx == r_fidx && IsCat(d_feature_types, l_fidx);
    });
  }
  timer_.Stop(__func__);
}

void SketchContainer::FixError() {
  dh::safe_cuda(cudaSetDevice(device_));
  auto d_columns_ptr = this->columns_ptr_.ConstDeviceSpan();
  auto in = dh::ToSpan(this->Current());
  dh::LaunchN(in.size(), [=] __device__(size_t idx) {
    auto column_id = dh::SegmentId(d_columns_ptr, idx);
    auto in_column = in.subspan(d_columns_ptr[column_id],
                                d_columns_ptr[column_id + 1] -
                                    d_columns_ptr[column_id]);
    idx -= d_columns_ptr[column_id];
    float prev_rmin = idx == 0 ? 0.0f : in_column[idx-1].rmin;
    if (in_column[idx].rmin < prev_rmin) {
      in_column[idx].rmin = prev_rmin;
    }
    float prev_rmax = idx == 0 ? 0.0f : in_column[idx-1].rmax;
    if (in_column[idx].rmax < prev_rmax) {
      in_column[idx].rmax = prev_rmax;
    }
    float rmin_next = in_column[idx].RMinNext();
    if (in_column[idx].rmax < rmin_next) {
      in_column[idx].rmax = rmin_next;
    }
  });
}

void SketchContainer::AllReduce() {
  dh::safe_cuda(cudaSetDevice(device_));
  auto world = collective::GetWorldSize();
  if (world == 1) {
    return;
  }

  timer_.Start(__func__);
  auto* communicator = collective::Communicator::GetDevice(device_);
  // Reduce the overhead on syncing.
  size_t global_sum_rows = num_rows_;
  collective::Allreduce<collective::Operation::kSum>(&global_sum_rows, 1);
  size_t intermediate_num_cuts =
      std::min(global_sum_rows, static_cast<size_t>(num_bins_ * kFactor));
  this->Prune(intermediate_num_cuts);

  auto d_columns_ptr = this->columns_ptr_.ConstDeviceSpan();
  CHECK_EQ(d_columns_ptr.size(), num_columns_ + 1);
  size_t n = d_columns_ptr.size();
  collective::Allreduce<collective::Operation::kMax>(&n, 1);
  CHECK_EQ(n, d_columns_ptr.size()) << "Number of columns differs across workers";

  // Get the columns ptr from all workers
  dh::device_vector<SketchContainer::OffsetT> gathered_ptrs;
  gathered_ptrs.resize(d_columns_ptr.size() * world, 0);
  size_t rank = collective::GetRank();
  auto offset = rank * d_columns_ptr.size();
  thrust::copy(thrust::device, d_columns_ptr.data(), d_columns_ptr.data() + d_columns_ptr.size(),
               gathered_ptrs.begin() + offset);
  communicator->AllReduceSum(gathered_ptrs.data().get(), gathered_ptrs.size());

  // Get the data from all workers.
  std::vector<size_t> recv_lengths;
  dh::caching_device_vector<char> recvbuf;
  communicator->AllGatherV(this->Current().data().get(), dh::ToSpan(this->Current()).size_bytes(),
                            &recv_lengths, &recvbuf);
  communicator->Synchronize();

  // Segment the received data.
  auto s_recvbuf = dh::ToSpan(recvbuf);
  std::vector<Span<SketchEntry>> allworkers;
  offset = 0;
  for (int32_t i = 0; i < world; ++i) {
    size_t length_as_bytes = recv_lengths.at(i);
    auto raw = s_recvbuf.subspan(offset, length_as_bytes);
    auto sketch = Span<SketchEntry>(reinterpret_cast<SketchEntry *>(raw.data()),
                                    length_as_bytes / sizeof(SketchEntry));
    allworkers.emplace_back(sketch);
    offset += length_as_bytes;
  }

  // Merge them into a new sketch.
  SketchContainer new_sketch(this->feature_types_, num_bins_,
                             this->num_columns_, global_sum_rows,
                             this->device_);
  for (size_t i = 0; i < allworkers.size(); ++i) {
    auto worker = allworkers[i];
    auto worker_ptr =
        dh::ToSpan(gathered_ptrs)
            .subspan(i * d_columns_ptr.size(), d_columns_ptr.size());
    new_sketch.Merge(worker_ptr, worker);
    new_sketch.FixError();
  }

  *this = std::move(new_sketch);
  timer_.Stop(__func__);
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

void SketchContainer::MakeCuts(HistogramCuts* p_cuts) {
  timer_.Start(__func__);
  dh::safe_cuda(cudaSetDevice(device_));
  p_cuts->min_vals_.Resize(num_columns_);

  // Sync between workers.
  this->AllReduce();

  // Prune to final number of bins.
  this->Prune(num_bins_ + 1);
  this->FixError();

  // Set up inputs
  auto d_in_columns_ptr = this->columns_ptr_.ConstDeviceSpan();

  p_cuts->min_vals_.SetDevice(device_);
  auto d_min_values = p_cuts->min_vals_.DeviceSpan();
  auto const in_cut_values = dh::ToSpan(this->Current());

  // Set up output ptr
  p_cuts->cut_ptrs_.SetDevice(device_);
  auto& h_out_columns_ptr = p_cuts->cut_ptrs_.HostVector();
  h_out_columns_ptr.clear();
  h_out_columns_ptr.push_back(0);
  auto const& h_feature_types = this->feature_types_.ConstHostSpan();

  auto d_ft = feature_types_.ConstDeviceSpan();

  std::vector<SketchEntry> max_values;
  float max_cat{-1.f};
  if (has_categorical_) {
    dh::XGBCachingDeviceAllocator<char> alloc;
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
    dh::caching_device_vector<SketchEntry> d_max_values(d_in_columns_ptr.size() - 1);
    thrust::reduce_by_key(thrust::cuda::par(alloc), key_it, key_it + in_cut_values.size(), val_it,
                          thrust::make_discard_iterator(), d_max_values.begin(),
                          thrust::equal_to<bst_feature_t>{},
                          [] __device__(auto l, auto r) { return l.value > r.value ? l : r; });
    dh::CopyDeviceSpanToVector(&max_values, dh::ToSpan(d_max_values));
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
      h_out_columns_ptr.push_back(max_values[i].value + 1);  // includes both max_cat and 0.
    } else {
      h_out_columns_ptr.push_back(
          std::min(static_cast<size_t>(column_size), static_cast<size_t>(num_bins_)));
    }
  }
  std::partial_sum(h_out_columns_ptr.begin(), h_out_columns_ptr.end(), h_out_columns_ptr.begin());
  auto d_out_columns_ptr = p_cuts->cut_ptrs_.ConstDeviceSpan();

  size_t total_bins = h_out_columns_ptr.back();
  p_cuts->cut_values_.SetDevice(device_);
  p_cuts->cut_values_.Resize(total_bins);
  auto out_cut_values = p_cuts->cut_values_.DeviceSpan();

  dh::LaunchN(total_bins, [=] __device__(size_t idx) {
    auto column_id = dh::SegmentId(d_out_columns_ptr, idx);
    auto in_column = in_cut_values.subspan(d_in_columns_ptr[column_id],
                                           d_in_columns_ptr[column_id + 1] -
                                               d_in_columns_ptr[column_id]);
    auto out_column = out_cut_values.subspan(d_out_columns_ptr[column_id],
                                             d_out_columns_ptr[column_id + 1] -
                                                 d_out_columns_ptr[column_id]);
    idx -= d_out_columns_ptr[column_id];
    if (in_column.size() == 0) {
      // If the column is empty, we push a dummy value.  It won't affect training as the
      // column is empty, trees cannot split on it.  This is just to be consistent with
      // rest of the library.
      if (idx == 0) {
        d_min_values[column_id] = kRtEps;
        out_column[0] = kRtEps;
        assert(out_column.size() == 1);
      }
      return;
    }

    if (idx == 0 && !IsCat(d_ft, column_id)) {
      auto mval = in_column[idx].value;
      d_min_values[column_id] = mval - (fabs(mval) + 1e-5);
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
    assert(idx+1 < in_column.size());
    out_column[idx] = in_column[idx+1].value;
  });

  p_cuts->SetCategorical(this->has_categorical_, max_cat);
  timer_.Stop(__func__);
}
}  // namespace common
}  // namespace xgboost
