/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

#include <utility>

#include "xgboost/span.h"
#include "quantile.h"
#include "quantile.cuh"
#include "hist_util.h"
#include "device_helpers.cuh"
#include "common.h"

namespace xgboost {
namespace common {

using WQSketch = DenseCuts::WQSketch;
using SketchEntry = WQSketch::Entry;

// Algorithm 4 in XGBoost's paper, using binary search to find i.
__device__ SketchEntry BinarySearchQuery(Span<SketchEntry const> const& entries, float rank) {
  assert(entries.size() >= 2);
  rank *= 2;
  if (rank < entries.front().rmin + entries.front().rmax) {
    return entries.front();
  }
  if (rank >= entries.back().rmin + entries.back().rmax) {
    return entries.back();
  }

  auto begin = dh::MakeTransformIterator<float>(
      entries.begin(), [=] __device__(SketchEntry const &entry) {
        return entry.rmin + entry.rmax;
      });
  auto end = begin + entries.size();
  auto i = thrust::upper_bound(thrust::seq, begin + 1, end - 1, rank) - begin - 1;
  if (rank < entries[i].RMinNext() + entries[i+1].RMaxPrev()) {
    return entries[i];
  } else {
    return entries[i+1];
  }
}

template <typename T>
void CopyTo(Span<T> out, Span<T const> src) {
  CHECK_EQ(out.size(), src.size());
  dh::safe_cuda(cudaMemcpyAsync(out.data(), src.data(),
                                out.size_bytes(),
                                cudaMemcpyDefault));
}

// Compute the merge path.
auto MergePath(Span<SketchEntry const> const &d_x,
               Span<bst_row_t const> const &x_ptr,
               Span<SketchEntry const> const &d_y,
               Span<bst_row_t const> const &y_ptr, Span<SketchEntry> out,
               Span<bst_row_t> out_ptr)
    -> common::Span<thrust::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> {
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

  using Tuple = thrust::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;

  thrust::constant_iterator<uint32_t> a_ind_iter(0);
  thrust::constant_iterator<uint32_t> b_ind_iter(1);

  auto place_holder = thrust::make_constant_iterator(0u);
  auto x_merge_val_it = thrust::make_zip_iterator(
      thrust::make_tuple(place_holder, a_ind_iter, place_holder, place_holder));
  auto y_merge_val_it = thrust::make_zip_iterator(
      thrust::make_tuple(place_holder, b_ind_iter, place_holder, place_holder));

  dh::XGBCachingDeviceAllocator<Tuple> alloc;
  static_assert(sizeof(Tuple) == sizeof(SketchEntry), "");
  // We reuse the memory for storing merge path.
  common::Span<Tuple> merge_path{reinterpret_cast<Tuple *>(out.data()), out.size()};
  // Determine the merge path
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

  auto get_ind = []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<1>(t); };
  auto get_a =   []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<2>(t); };
  auto get_b =   []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<3>(t); };

  auto scan_key_it = dh::MakeTransformIterator<bst_feature_t>(
      thrust::make_counting_iterator(0ul),
      [=] __device__(size_t idx) { return dh::SegmentId(out_ptr, idx); });

  auto scan_val_it = dh::MakeTransformIterator<Tuple>(
      merge_path.data(), [=] __device__(Tuple const &t) -> Tuple {
        auto ind = get_ind(t);  // == 0 if element is from a
        // place_holder, place_holder, a_counter, b_counter
        return thrust::make_tuple(0u, 0u, !ind, ind);
      });

  // Compute the index for both a and b (which of the element in a and b are used in each
  // comparison) by scaning the binary merge path.  Take output [(a_0, b_0), (a_0, b_1),
  // ...] as an example, the comparison between (a_0, b_0) adds 1 step in the merge path.
  // Because b_0 is less than a_0 so this step is torward the end of b.  After the
  // comparison, index of b is incremented by 1 from b_0 to b_1, and at the same time, b_0
  // is landed into output as the first element in merge result.  The scan result is the
  // subscript of a and b.
  thrust::exclusive_scan_by_key(
      thrust::cuda::par(alloc), scan_key_it, scan_key_it + merge_path.size(),
      scan_val_it,
      merge_path.data(), thrust::make_tuple(0u, 0u, 0u, 0u),
      thrust::equal_to<bst_row_t>{},
      [=] __device__(Tuple const &l, Tuple const &r) -> Tuple {
        return thrust::make_tuple(0, 0, get_a(l) + get_a(r), get_b(l) + get_b(r));
      });

  return merge_path;
}

// Merge d_x and d_y into out.  Because the final output depends on predicate (which
// summary does the output element come from) result by definition of merged rank.  So we
// run it in 2 passes to obtain the merge path and then customize the standard merge
// algorithm.
void MergeImpl(int32_t device, Span<SketchEntry const> const &d_x,
               Span<bst_row_t const> const &x_ptr,
               Span<SketchEntry const> const &d_y,
               Span<bst_row_t const> const &y_ptr,
               Span<SketchEntry> out,
               Span<bst_row_t> out_ptr) {
  dh::safe_cuda(cudaSetDevice(device));
  CHECK_EQ(d_x.size() + d_y.size(), out.size());
  CHECK_EQ(x_ptr.size(), out_ptr.size());
  CHECK_EQ(y_ptr.size(), out_ptr.size());

  auto d_merge_path = MergePath(d_x, x_ptr, d_y, y_ptr, out, out_ptr);
  auto d_out = out;

  dh::LaunchN(device, d_out.size(), [=] __device__(size_t idx) {
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

    uint32_t a_ind, b_ind, _0, _1;
    thrust::tie(_0, _1, a_ind, b_ind) = d_path_column[idx];

    // Handle empty column.  When both columns are empty, we should not get this column_id
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

void SketchContainer::Push(common::Span<OffsetT const> cuts_ptr,
                           dh::caching_device_vector<SketchEntry>* entries) {
  timer_.Start(__func__);
  dh::safe_cuda(cudaSetDevice(device_));
  // Copy or merge the new cuts, pruning is performed during `MakeCuts`.
  if (this->Current().size() == 0) {
    CHECK_EQ(this->columns_ptr_.Size(), cuts_ptr.size());
    std::swap(this->Current(), *entries);
    CHECK_EQ(entries->size(), 0);
    auto d_cuts_ptr = this->columns_ptr_.DevicePointer();
    thrust::copy(thrust::device, cuts_ptr.data(),
                 cuts_ptr.data() + cuts_ptr.size(), d_cuts_ptr);
  } else {
    auto d_entries = dh::ToSpan(*entries);
    this->Merge(cuts_ptr, d_entries);
    this->FixError();
  }
  CHECK_NE(this->columns_ptr_.Size(), 0);
  timer_.Stop(__func__);
}

size_t SketchContainer::Unique() {
  timer_.Start(__func__);
  dh::safe_cuda(cudaSetDevice(device_));
  this->columns_ptr_.SetDevice(device_);
  Span<OffsetT> d_column_scan = this->columns_ptr_.DeviceSpan();
  CHECK_EQ(d_column_scan.size(), num_columns_ + 1);
  Span<SketchEntry> entries = dh::ToSpan(this->Current());
  HostDeviceVector<OffsetT> scan_out(d_column_scan.size());
  scan_out.SetDevice(device_);
  auto d_scan_out = scan_out.DeviceSpan();

  d_column_scan = this->columns_ptr_.DeviceSpan();
  size_t n_uniques = dh::SegmentedUnique(
      d_column_scan.data(), d_column_scan.data() + d_column_scan.size(),
      entries.data(), entries.data() + entries.size(), scan_out.DevicePointer(),
      entries.data(),
      detail::SketchUnique{});
  this->columns_ptr_.Copy(scan_out);
  CHECK(!this->columns_ptr_.HostCanRead());

  this->Current().resize(n_uniques);
  timer_.Stop(__func__);
  return n_uniques;
}

void SketchContainer::Prune(size_t to) {
  timer_.Start(__func__);
  dh::safe_cuda(cudaSetDevice(device_));

  this->Unique();
  OffsetT to_total = 0;
  HostDeviceVector<OffsetT> new_columns_ptr{to_total};
  for (bst_feature_t i = 0; i < num_columns_; ++i) {
    size_t length = this->Column(i).size();
    length = std::min(length, to);
    to_total += length;
    new_columns_ptr.HostVector().emplace_back(to_total);
  }
  new_columns_ptr.SetDevice(device_);
  this->Other().resize(to_total);

  auto d_columns_ptr_in = this->columns_ptr_.ConstDeviceSpan();
  auto d_columns_ptr_out = new_columns_ptr.ConstDeviceSpan();
  auto out = dh::ToSpan(this->Other());
  auto in = dh::ToSpan(this->Current());
  dh::LaunchN(0, to_total, [=] __device__(size_t idx) {
    size_t column_id = dh::SegmentId(d_columns_ptr_out, idx);
    auto out_column = out.subspan(d_columns_ptr_out[column_id],
                                  d_columns_ptr_out[column_id + 1] -
                                      d_columns_ptr_out[column_id]);
    auto in_column = in.subspan(d_columns_ptr_in[column_id],
                                d_columns_ptr_in[column_id + 1] -
                                    d_columns_ptr_in[column_id]);
    idx -= d_columns_ptr_out[column_id];
    // Input has lesser columns than `to`, just copy them to the output.  This is correct
    // as the new output size is calculated based on both the size of `to` and current
    // column.
    if (in_column.size() <= to) {
      out_column[idx] = in_column[idx];
      return;
    }
    // 1 thread for each output.  See A.4 for detail.
    auto entries = in_column;
    auto d_out = out_column;
    if (idx == 0) {
      d_out.front() = entries.front();
      return;
    }
    if (idx == to - 1) {
      d_out.back() = entries.back();
      return;
    }

    float w = entries.back().rmin - entries.front().rmax;
    assert(w != 0);
    auto budget = static_cast<float>(d_out.size());
    assert(budget != 0);
    auto q = ((idx * w) / (to - 1) + entries.front().rmax);
    d_out[idx] = BinarySearchQuery(entries, q);
  });
  this->columns_ptr_.HostVector() = new_columns_ptr.HostVector();
  this->Alternate();
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

  HostDeviceVector<OffsetT> new_columns_ptr;
  new_columns_ptr.SetDevice(device_);
  new_columns_ptr.Resize(this->ColumnsPtr().size());
  MergeImpl(device_, this->Data(), this->ColumnsPtr(),
            that, d_that_columns_ptr,
            dh::ToSpan(this->Other()), new_columns_ptr.DeviceSpan());
  this->columns_ptr_ = std::move(new_columns_ptr);
  CHECK_EQ(this->columns_ptr_.Size(), num_columns_ + 1);
  CHECK_EQ(new_columns_ptr.Size(), 0);
  this->Alternate();
  timer_.Stop(__func__);
}

void SketchContainer::FixError() {
  dh::safe_cuda(cudaSetDevice(device_));
  auto d_columns_ptr = this->columns_ptr_.ConstDeviceSpan();
  auto in = dh::ToSpan(this->Current());
  dh::LaunchN(device_, in.size(), [=] __device__(size_t idx) {
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
  auto world = rabit::GetWorldSize();
  if (world == 1) {
    return;
  }

  timer_.Start(__func__);
  if (!reducer_) {
    reducer_ = std::make_unique<dh::AllReducer>();
    reducer_->Init(device_);
  }
  auto d_columns_ptr = this->columns_ptr_.ConstDeviceSpan();
  dh::device_vector<SketchContainer::OffsetT> gathered_ptrs;

  CHECK_EQ(d_columns_ptr.size(), num_columns_ + 1);
  size_t n = d_columns_ptr.size();
  rabit::Allreduce<rabit::op::Max>(&n, 1);
  CHECK_EQ(n, d_columns_ptr.size()) << "Number of columns differs across workers";

  // Get the columns ptr from all workers
  gathered_ptrs.resize(d_columns_ptr.size() * world, 0);
  size_t rank = rabit::GetRank();
  auto offset = rank * d_columns_ptr.size();
  thrust::copy(thrust::device, d_columns_ptr.data(), d_columns_ptr.data() + d_columns_ptr.size(),
               gathered_ptrs.begin() + offset);
  reducer_->AllReduceSum(gathered_ptrs.data().get(), gathered_ptrs.data().get(),
                         gathered_ptrs.size());

  // Get the data from all workers.
  std::vector<size_t> recv_lengths;
  dh::caching_device_vector<char> recvbuf;
  reducer_->AllGather(this->Current().data().get(),
                      dh::ToSpan(this->Current()).size_bytes(), &recv_lengths,
                      &recvbuf);
  reducer_->Synchronize();

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

  // Merge them into current sketch.
  for (size_t i = 0; i < allworkers.size(); ++i) {
    if (i == rank) {
      continue;
    }
    auto worker = allworkers[i];
    auto worker_ptr =
        dh::ToSpan(gathered_ptrs)
            .subspan(i * d_columns_ptr.size(), d_columns_ptr.size());
    this->Merge(worker_ptr, worker);
    this->FixError();
  }
  timer_.Stop(__func__);
}

void SketchContainer::MakeCuts(HistogramCuts* p_cuts) {
  timer_.Start(__func__);
  dh::safe_cuda(cudaSetDevice(device_));
  p_cuts->min_vals_.Resize(num_columns_);
  size_t global_max_rows = num_rows_;
  rabit::Allreduce<rabit::op::Sum>(&global_max_rows, 1);

  // Sync between workers.
  size_t intermediate_num_cuts =
      std::min(global_max_rows, static_cast<size_t>(num_bins_ * kFactor));
  this->Prune(intermediate_num_cuts);
  this->AllReduce();

  // Prune to final number of bins.
  this->Prune(num_bins_ + 1);
  this->Unique();
  this->FixError();

  // Set up inputs
  auto d_in_columns_ptr = this->columns_ptr_.ConstDeviceSpan();

  p_cuts->min_vals_.SetDevice(device_);
  auto d_min_values = p_cuts->min_vals_.DeviceSpan();
  auto in_cut_values = dh::ToSpan(this->Current());

  // Set up output ptr
  p_cuts->cut_ptrs_.SetDevice(device_);
  auto& h_out_columns_ptr = p_cuts->cut_ptrs_.HostVector();
  h_out_columns_ptr.clear();
  h_out_columns_ptr.push_back(0);
  for (bst_feature_t i = 0; i < num_columns_; ++i) {
    h_out_columns_ptr.push_back(
        std::min(static_cast<size_t>(std::max(static_cast<size_t>(1ul),
                                              this->Column(i).size())),
                 static_cast<size_t>(num_bins_)));
  }
  std::partial_sum(h_out_columns_ptr.begin(), h_out_columns_ptr.end(),
                   h_out_columns_ptr.begin());
  auto d_out_columns_ptr = p_cuts->cut_ptrs_.ConstDeviceSpan();

  // Set up output cuts
  size_t total_bins = h_out_columns_ptr.back();
  p_cuts->cut_values_.SetDevice(device_);
  p_cuts->cut_values_.Resize(total_bins);
  auto out_cut_values = p_cuts->cut_values_.DeviceSpan();

  dh::LaunchN(0, total_bins, [=] __device__(size_t idx) {
    auto column_id = dh::SegmentId(d_out_columns_ptr, idx);
    auto in_column = in_cut_values.subspan(d_in_columns_ptr[column_id],
                                           d_in_columns_ptr[column_id + 1] -
                                               d_in_columns_ptr[column_id]);
    auto out_column = out_cut_values.subspan(d_out_columns_ptr[column_id],
                                             d_out_columns_ptr[column_id + 1] -
                                                 d_out_columns_ptr[column_id]);
    idx -= d_out_columns_ptr[column_id];
    if (in_column.size() == 0) {
      if (idx == 0) {
        d_min_values[column_id] = kRtEps;
        out_column[0] = kRtEps;
        assert(out_column.size() == 1);
      }
      return;
    }

    if (idx == 0) {
      auto mval = in_column[idx].value;
      d_min_values[column_id] = mval - (fabs(mval) + 1e-5);
    }
    // idx >= 1 && idx < out.size
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
  timer_.Stop(__func__);
}
}  // namespace common
}  // namespace xgboost
