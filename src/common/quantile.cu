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
#include "segmented_uniques.cuh"
#include "quantile.cuh"
#include "hist_util.h"
#include "device_helpers.cuh"
#include "common.h"

namespace xgboost {
namespace common {

using WQSketch = DenseCuts::WQSketch;
using SketchEntry = WQSketch::Entry;

// Algorithm 4 in XGBoost's paper, using binary search to find i.
__device__ SketchEntry BinarySearchQuery(Span<SketchEntry const> entries, float rank) {
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
namespace {
struct SketchUnique {
  XGBOOST_DEVICE bool operator()(SketchEntry const& a, SketchEntry const& b) const {
    return a.value - b.value == 0;
  }
};
}  // anonymous namespace

template <typename T>
void CopyTo(Span<T> out, Span<T const> src) {
  CHECK_EQ(out.size(), src.size());
  dh::safe_cuda(cudaMemcpyAsync(out.data(), src.data(),
                                out.size_bytes(),
                                cudaMemcpyDefault));
}

// Merge d_x and d_y into out.  Because the final output depends on predicate (which
// summary does the output element come from) result by definition of merged rank.  So we
// run it in 2 passes to obtain the merge path and then customize the standard merge
// algorithm.
void MergeImpl(Span<SketchEntry const> d_x, Span<SketchEntry const> d_y,
               Span<SketchEntry> out, cudaStream_t stream = nullptr) {
  if (d_x.size() == 0) {
    CopyTo(out, d_y);
    return;
  }
  if (d_y.size() == 0) {
    CopyTo(out, d_x);
    return;
  }

  auto a_key_it = dh::MakeTransformIterator<float>(
      d_x.data(), []__device__(SketchEntry const &e) { return e.value; });
  auto b_key_it = dh::MakeTransformIterator<float>(
      d_y.data(), []__device__(SketchEntry const &e) { return e.value; });

  thrust::constant_iterator<int32_t> a_ind_iter(0);
  thrust::constant_iterator<int32_t> b_ind_iter(1);

  // allocate memory for later use in scan
  auto place_holder = thrust::make_constant_iterator(-1);
  auto x_val_it = thrust::make_zip_iterator(
      thrust::make_tuple(place_holder, a_ind_iter, place_holder, place_holder));
  auto y_val_it = thrust::make_zip_iterator(
      thrust::make_tuple(place_holder, b_ind_iter, place_holder, place_holder));

  using Tuple = thrust::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
  auto get_ind = []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<1>(t); };
  auto get_a =   []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<0>(t); };
  auto get_b =   []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<1>(t); };

  dh::XGBCachingDeviceAllocator<Tuple> alloc;
  static_assert(sizeof(Tuple) == sizeof(SketchEntry), "");
  // We reuse the memory for storing merge path.
  common::Span<Tuple> merge_path{reinterpret_cast<Tuple *>(out.data()), out.size()};
  // Determine the merge path
  thrust::merge_by_key(thrust::cuda::par(alloc).on(stream),
                       a_key_it, a_key_it + d_x.size(), b_key_it,
                       b_key_it + d_y.size(), x_val_it, y_val_it,
                       thrust::make_discard_iterator(), merge_path.data());
  // Compute the index for both a and b (which of the element in a and b are used in each
  // comparison) by scaning the binary merge path.  Take output [(a_0, b_0), (a_0, b_1),
  // ...] as an example, the comparison between (a_0, b_0) adds 1 step in the merge path.
  // Because b_0 is less than a_0 so this step is torward the end of b.  After the
  // comparison, index of b is incremented by 1 from b_0 to b_1, and at the same time, b_0
  // is landed into output as the first element in merge result.  The scan result is the
  // subscript of a and b.
  thrust::transform_exclusive_scan(
      thrust::cuda::par(alloc).on(stream), merge_path.data(), merge_path.data() + merge_path.size(),
      merge_path.data(),
      [=] __device__(Tuple const &t) {
        auto ind = get_ind(t);  // == 0 if element is from a
        // a_counter, b_counter
        return thrust::make_tuple(!ind, ind, 0, 0);
      },
      thrust::make_tuple(0, 0, 0, 0),
      [=] __device__(Tuple const &l, Tuple const &r) {
        return thrust::make_tuple(get_a(l) + get_a(r), get_b(l) + get_b(r), 0, 0);
      });

  auto d_merge_path = merge_path;
  auto d_out = Span<SketchEntry>{out.data(), d_x.size() + d_y.size()};

  dh::LaunchN(0, d_out.size(), stream, [=] __device__(size_t idx) {
    int32_t a_ind, b_ind, p_0, p_1;
    thrust::tie(a_ind, b_ind, p_0, p_1) = d_merge_path[idx];
    // Handle trailing elements.
    assert(a_ind <= d_x.size());
    if (a_ind == d_x.size()) {
      // Trailing elements are from y because there's no more x to land.
      auto y_elem = d_y[b_ind];
      d_out[idx] = SketchEntry(y_elem.rmin + d_x.back().RMinNext(),
                               y_elem.rmax + d_x.back().rmax,
                               y_elem.wmin, y_elem.value);
      return;
    }
    auto x_elem = d_x[a_ind];
    assert(b_ind <= d_y.size());
    if (b_ind == d_y.size()) {
      d_out[idx] = SketchEntry(x_elem.rmin + d_y.back().RMinNext(),
                               x_elem.rmax + d_y.back().rmax,
                               x_elem.wmin, x_elem.value);
      return;
    }
    auto y_elem = d_y[b_ind];

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
    assert(idx < d_out.size());
    if (x_elem.value == y_elem.value) {
      d_out[idx] =
          SketchEntry{x_elem.rmin + y_elem.rmin, x_elem.rmax + y_elem.rmax,
                      x_elem.wmin + y_elem.wmin, x_elem.value};
    } else if (x_elem.value < y_elem.value) {
      // elem from x is landed. yprev_min is the element in D_2 that's 1 rank less than
      // x_elem if we put x_elem in D_2.
      float yprev_min = b_ind == 0 ? 0.0f : d_y[b_ind - 1].RMinNext();
      // rmin should be equal to x_elem.rmin + x_elem.wmin + yprev_min.  But for
      // implementation, the weight is stored in a separated field and we compute the
      // extended definition on the fly when needed.
      d_out[idx] =
          SketchEntry{x_elem.rmin + yprev_min, x_elem.rmax + y_elem.RMaxPrev(),
                      x_elem.wmin, x_elem.value};
    } else {
      // elem from y is landed.
      float xprev_min = a_ind == 0 ? 0.0f : d_x[a_ind - 1].RMinNext();
      d_out[idx] =
          SketchEntry{xprev_min + y_elem.rmin, x_elem.RMaxPrev() + y_elem.rmax,
                      y_elem.wmin, y_elem.value};
    }
  });
}

void SketchContainer::Push(common::Span<OffsetT const> cuts_ptr,
                           dh::caching_device_vector<SketchEntry>* entries) {
  timer_.Start(__func__);
  // Copy or merge the new cuts, pruning is performed during `MakeCuts`.
  if (this->Current().size() == 0) {
    CHECK_EQ(this->columns_ptr_.Size(), cuts_ptr.size());
    std::swap(this->Current(), *entries);
    CHECK_EQ(entries->size(), 0);
    auto d_cuts_ptr = this->columns_ptr_.DevicePointer();
    thrust::copy(thrust::device, cuts_ptr.data(),
                 cuts_ptr.data() + cuts_ptr.size(), d_cuts_ptr);
  } else {
    std::vector<size_t> h_cuts_ptr(cuts_ptr.size());
    auto d_entries = dh::ToSpan(*entries);
    this->Merge(cuts_ptr, d_entries);
  }
  CHECK_NE(this->columns_ptr_.Size(), 0);
  timer_.Stop(__func__);
}

size_t SketchContainer::Unique() {
  timer_.Start(__func__);
  this->columns_ptr_.SetDevice(device_);
  Span<OffsetT> d_column_scan = this->columns_ptr_.DeviceSpan();
  CHECK_EQ(d_column_scan.size(), num_columns_ + 1);
  Span<SketchEntry> entries = dh::ToSpan(this->Current());
  HostDeviceVector<OffsetT> scan_out(d_column_scan.size());
  scan_out.SetDevice(device_);
  auto d_scan_out = scan_out.DeviceSpan();

  d_column_scan = this->columns_ptr_.DeviceSpan();
  size_t n_uniques = SegmentedUnique(
      d_column_scan.data(), d_column_scan.data() + d_column_scan.size(),
      entries.data(), entries.data() + entries.size(), scan_out.DevicePointer(),
      entries.data(),
      SketchUnique{});
  this->columns_ptr_.Copy(scan_out);
  CHECK(!this->columns_ptr_.HostCanRead());

  this->Current().resize(n_uniques, SketchEntry{0, 0, 0, 0});
  timer_.Stop(__func__);
  return n_uniques;
}

void SketchContainer::Prune(size_t to) {
  timer_.Start(__func__);
  this->Unique();
  OffsetT to_total = 0;
  HostDeviceVector<OffsetT> new_columns_ptr{to_total};
  for (size_t i = 0; i < num_columns_; ++i) {
    size_t length = this->Column(i).size();
    length = std::min(length, to);
    to_total += length;
    new_columns_ptr.HostVector().emplace_back(to_total);
  }
  new_columns_ptr.SetDevice(device_);
  this->Other().resize(to_total, SketchEntry{0, 0, 0, 0});

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
    // Input has lesser columns than to, just copy them to the output.  This is correct as
    // the new output size is calculated based on both to and current column size.
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
  timer_.Start(__func__);
  if (this->Current().size() == 0) {
    CHECK_EQ(this->columns_ptr_.HostVector().back(), 0);
    CHECK_EQ(this->columns_ptr_.HostVector().size(), d_that_columns_ptr.size());
    CHECK_EQ(columns_ptr_.Size(), num_columns_ + 1);
    thrust::copy(thrust::device, d_that_columns_ptr.data(),
                 d_that_columns_ptr.data() + d_that_columns_ptr.size(),
                 this->columns_ptr_.DevicePointer());
    auto total = this->columns_ptr_.HostVector().back();
    this->Current().resize(total, SketchEntry{0, 0, 0, 0});
    CopyTo(dh::ToSpan(this->Current()), that);
    timer_.Stop(__func__);
    return;
  }

  std::vector<OffsetT> that_columns_ptr(d_that_columns_ptr.size());
  dh::CopyDeviceSpanToVector(&that_columns_ptr, d_that_columns_ptr);
  size_t total = that_columns_ptr.back();
  this->Other().resize(this->Current().size() + total, SketchEntry{0, 0, 0, 0});
  CHECK_EQ(that_columns_ptr.size(), this->columns_ptr_.Size());
  OffsetT out_offset = 0;
  std::vector<OffsetT> new_columns_ptr{out_offset};
  for (size_t i = 1; i < that_columns_ptr.size(); ++i) {
    auto self_column = this->Column(i-1);
    auto that_column = that.subspan(
        that_columns_ptr[i - 1], that_columns_ptr[i] - that_columns_ptr[i - 1]);
    auto out_size = self_column.size() + that_column.size();
    auto out = dh::ToSpan(this->Other()).subspan(out_offset, out_size);
    MergeImpl(self_column, that_column, out);
    out_offset += out_size;
    new_columns_ptr.emplace_back(out_offset);
  }
  CHECK_EQ(this->columns_ptr_.Size(), new_columns_ptr.size());
  this->columns_ptr_.HostVector() = std::move(new_columns_ptr);
  CHECK_EQ(this->columns_ptr_.Size(), num_columns_ + 1);
  CHECK_EQ(this->columns_ptr_.HostVector().back(), this->Other().size());
  this->Alternate();
  this->FixError();
  timer_.Stop(__func__);
}

void SketchContainer::FixError() {
  auto d_columns_ptr = this->columns_ptr_.ConstDeviceSpan();
  auto in = dh::ToSpan(this->Current());
  dh::LaunchN(device_, this->Current().size(), [=] __device__(size_t idx) {
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

  // FIXME: Uneven number of columns.
  CHECK_NE(d_columns_ptr.size(), 0);
  size_t n = d_columns_ptr.size();
  rabit::Allreduce<rabit::op::Max>(&n, 1);
  CHECK_EQ(n, d_columns_ptr.size());

  gathered_ptrs.resize(d_columns_ptr.size() * world, 0);
  size_t rank = rabit::GetRank();
  auto offset = rank * d_columns_ptr.size();
  thrust::copy(thrust::device, d_columns_ptr.data(), d_columns_ptr.data() + d_columns_ptr.size(),
               gathered_ptrs.begin() + offset);
  reducer_->AllReduceSum(gathered_ptrs.data().get(), gathered_ptrs.data().get(),
                         gathered_ptrs.size());

  std::vector<size_t> recv_lengths;
  dh::caching_device_vector<char> recvbuf;
  reducer_->AllGather(this->Current().data().get(),
                      dh::ToSpan(this->Current()).size_bytes(), &recv_lengths,
                      &recvbuf);
  reducer_->Synchronize();

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

  for (size_t i = 0; i < allworkers.size(); ++i) {
    auto worker = allworkers[i];
    auto worker_ptr =
        dh::ToSpan(gathered_ptrs)
            .subspan(i * d_columns_ptr.size(), d_columns_ptr.size());
    this->Merge(worker_ptr, worker);
  }
  timer_.Stop(__func__);
}

void SketchContainer::MakeCuts(HistogramCuts* p_cuts) {
  timer_.Start(__func__);
  p_cuts->min_vals_.Resize(num_columns_);
  size_t global_max_rows = num_rows_;
  rabit::Allreduce<rabit::op::Sum>(&global_max_rows, 1);
  size_t intermediate_num_cuts =
      std::min(global_max_rows, static_cast<size_t>(num_bins_ * kFactor));
  this->Prune(intermediate_num_cuts);
  this->AllReduce();
  this->Prune(num_bins_ + 1);
  this->Unique();

  // Set up inputs
  auto h_in_columns_ptr = this->columns_ptr_.ConstHostSpan();
  auto d_in_columns_ptr = this->columns_ptr_.ConstDeviceSpan();

  p_cuts->min_vals_.SetDevice(device_);
  auto d_min_values = p_cuts->min_vals_.DeviceSpan();
  auto in_cut_values = dh::ToSpan(this->Current());

  // Set up output ptr
  p_cuts->cut_ptrs_.SetDevice(device_);
  auto& h_out_columns_ptr = p_cuts->cut_ptrs_.HostVector();
  h_out_columns_ptr.clear();
  h_out_columns_ptr.push_back(0);
  for (size_t i = 0; i < num_columns_; ++i) {
    h_out_columns_ptr.push_back(
        std::min(static_cast<size_t>(std::max(static_cast<size_t>(1ul),
                                              this->Column(i).size())),
                 static_cast<size_t>(num_bins_)));
  }
  CHECK_EQ(h_out_columns_ptr.size(), h_in_columns_ptr.size());
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
