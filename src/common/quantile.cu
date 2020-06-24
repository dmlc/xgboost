/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

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


struct IsSorted {
  bool XGBOOST_DEVICE operator()(SketchEntry const& a, SketchEntry const& b) const {
    return a.value < b.value;
  }
};
}  // anonymous namespace

void CopyTo(Span<SketchEntry> out, Span<SketchEntry const> src) {
  dh::safe_cuda(cudaMemcpyAsync(out.data(), src.data(),
                                out.size_bytes(),
                                cudaMemcpyDefault));
}

void PrintSorted(Span<SketchEntry const> d_y) {
  return;
  if (!thrust::is_sorted(thrust::device, d_y.data(), d_y.data() + d_y.size(), IsSorted{})) {
    auto it = thrust::is_sorted_until(thrust::device, d_y.data(), d_y.data() + d_y.size(), IsSorted{});
    std::vector<SketchEntry> copied(d_y.size());
    dh::CopyDeviceSpanToVector(&copied, d_y);
    auto pos = it - d_y.data();
    for (size_t i = std::max(0l, pos - 5); i < std::min(pos + 5, decltype(pos)(d_y.size())); ++i) {
      std::cout << std::setprecision(30) << copied[i] << std::endl;
    }
    std::cout << std::setprecision(30) << "Until: "
              << " Pos: " << pos << ", " << copied.at(pos - 1) << " vs "
              << copied.at(pos) << ", IsSorted{}:"
              << IsSorted{}(copied.at(pos - 1), copied.at(pos)) << ", "
              << "SketchUnique:" << SketchUnique{}(copied.at(pos - 1), copied.at(pos)) << std::endl;
    LOG(FATAL) << "Unique not sorted, std "
               << std::is_sorted(copied.begin(), copied.end(), IsSorted{})
               << ", until: "
               << std::distance(copied.cbegin(),
                                std::is_sorted_until(copied.cbegin(),
                                                     copied.cend(),
                                                     IsSorted{}));
  }
}

template <typename T>
void PrintSpan(common::Span<T> x, std::string name) {
  return;
  std::cout << name << std::endl;
  std::vector<T> copied(x.size());
  dh::CopyDeviceSpanToVector(&copied, x);
  for (auto v : copied) {
    std::cout << v << "; ";
  }
  std::cout << std::endl;
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
  auto x_val_it =
      thrust::make_zip_iterator(thrust::make_tuple(place_holder, a_ind_iter, place_holder, place_holder));
  auto y_val_it =
      thrust::make_zip_iterator(thrust::make_tuple(place_holder, b_ind_iter, place_holder, place_holder));

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
    if (a_ind == d_x.size()){
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

void MergeImpl(Span<SketchEntry const> d_x, Span<SketchEntry const> d_y,
               dh::device_vector<SketchEntry> *out, cudaStream_t stream = nullptr) {
  out->resize(d_x.size() + d_y.size());
  MergeImpl(d_x, d_y, Span<SketchEntry>(out->data().get(), out->size()), stream);
}

void DeviceQuantile::SetMerge(std::vector<Span<SketchEntry const>> const& others) {
  dh::safe_cuda(cudaSetDevice(device_));
  auto x = others.front();
  dh::device_vector<SketchEntry> buffer;
  // We don't have k way merging, so iterate it through.
  for (size_t i = 1; i < others.size(); ++i) {
    auto const y = others[i];
    MergeImpl(x, y, &buffer);
    std::swap(this->Current(), buffer);  // move the result into data_.
    x = dh::ToSpan(this->Current());     // update x to the latest sketch.
  }
}

void DeviceQuantile::AllReduce() {
  dh::safe_cuda(cudaSetDevice(device_));
  auto world = rabit::GetWorldSize();
  if (world == 1) {
    return;
  }
  if (!comm_) {
    comm_ = std::make_unique<dh::AllReducer>();
    comm_->Init(device_, false);
  }

  dh::caching_device_vector<char> recvbuf;
  std::vector<size_t> global_size;
  comm_->AllGather(this->Data().data(), this->Data().size_bytes(), &global_size, &recvbuf);;
  auto s_recvbuf = dh::ToSpan(recvbuf);

  std::vector<Span<SketchEntry const>> allworkers;

  size_t offset = 0;
  for (int32_t i = 0; i < world; ++i) {
    size_t length_as_bytes = global_size[i];
    auto raw = s_recvbuf.subspan(offset, length_as_bytes);
    auto sketch = Span<SketchEntry>(reinterpret_cast<SketchEntry *>(raw.data()),
                                    length_as_bytes / sizeof(SketchEntry));
    allworkers.emplace_back(sketch);
    offset += length_as_bytes;
  }
  this->SetMerge(allworkers);
}

void SketchContainer::Push(size_t entries_per_column,
                           const common::Span<SketchEntry>& entries,
                           const thrust::host_vector<size_t>& column_scan) {
  timer.Start(__func__);
  std::vector<Span<SketchEntry>> columns;
  std::vector<bst_feature_t> new_columns_ptr{0};
  for (size_t icol = 0; icol < num_columns_; ++icol) {
    size_t column_size = column_scan[icol + 1] - column_scan[icol];
    size_t num_available_cuts =
        std::min(size_t(entries_per_column), column_size);
    CHECK_GT(column_size, 0);  // FIXME
    // if (column_size == 0) continue;
    columns.emplace_back(entries.subspan(entries_per_column * icol, num_available_cuts));
    new_columns_ptr.emplace_back(num_available_cuts);
  }
  CHECK_EQ(entries.size(), entries_per_column * num_columns_);
  timer.Start("CopyIntoQuantile");
  if (this->Current().size() == 0) {
    CHECK_EQ(this->columns_ptr_.Size(), 0);
    dh::device_vector<size_t> d_columns_scan(column_scan.size());
    thrust::copy(column_scan.begin(), column_scan.end(),
                 d_columns_scan.begin());
    auto s_columns_scan = dh::ToSpan(d_columns_scan);
    std::partial_sum(new_columns_ptr.begin(), new_columns_ptr.end(), new_columns_ptr.begin());
    this->Current().resize(new_columns_ptr.back());
    dh::XGBCachingDeviceAllocator<char> alloc;
    size_t n =
        thrust::copy_if(
            thrust::cuda::par(alloc), entries.data(), entries.data() + entries.size(),
            thrust::make_counting_iterator(0ul), this->Current().begin(),
            [=] __device__(size_t i) {
              auto column_id = i / entries_per_column;
              size_t column_size =
                  s_columns_scan[column_id + 1] - s_columns_scan[column_id];
              size_t num_available_cuts =
                  thrust::min(size_t(entries_per_column), column_size);
              size_t pos = i - entries_per_column * column_id;
              if (pos < num_available_cuts) {
                return true;
              } else {
                return false;
              }
            }) -
        this->Current().begin();
    CHECK_EQ(n, new_columns_ptr.back());
    this->columns_ptr_.HostVector() = new_columns_ptr;
  } else {
    CHECK_EQ(columns.size(), num_columns_);
    this->Merge(columns);
  }
  timer.Stop("CopyIntoQuantile");

  this->Prune(limit_size_);
  timer.Stop(__func__);
}

void SketchContainer::Push(common::Span<size_t const> cuts_ptr,
                           const common::Span<SketchEntry>& entries) {
  timer.Start(__func__);
  if(this->Current().size() == 0) {
    this->Current().resize(entries.size());
    dh::safe_cuda(cudaMemcpyAsync(this->Current().data().get(),
                                  entries.data(), entries.size_bytes(),
                                  cudaMemcpyDeviceToHost));
    auto& h_columns_ptr = this->columns_ptr_.HostVector();
    h_columns_ptr.resize(cuts_ptr.size());
    std::copy(cuts_ptr.cbegin(), cuts_ptr.cend(), h_columns_ptr.begin());
  } else {
    LOG(FATAL) << "Not implemented";
  }
  this->Prune(limit_size_);
  timer.Stop(__func__);
}

size_t SketchContainer::Unique() {
  timer.Start(__func__);
  this->columns_ptr_.SetDevice(device_);
  Span<bst_feature_t> d_column_scan = this->columns_ptr_.DeviceSpan();
  CHECK_EQ(d_column_scan.size(), num_columns_ + 1);

  Span<SketchEntry> entries = dh::ToSpan(this->Current());
  HostDeviceVector<bst_feature_t> scan_out(d_column_scan.size());
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
  timer.Stop(__func__);
  return n_uniques;
}

void SketchContainer::Prune(size_t to) {
  timer.Start(__func__);
  auto n_uniques = this->Unique();

  bst_feature_t to_total = 0;
  HostDeviceVector<bst_feature_t> new_columns_ptr{to_total};
  new_columns_ptr.SetDevice(device_);
  for (size_t i = 0; i < num_columns_; ++i) {
    size_t length = this->Column(i).size();
    length = std::min(length, to);
    to_total += length;
    new_columns_ptr.HostVector().emplace_back(to_total);
  }
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
    // Just copy the output.
    if (d_columns_ptr_in[column_id + 1] - d_columns_ptr_in[column_id] <= to) {
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
    auto budget = static_cast<float>(d_out.size());
    assert(w != 0);
    assert(budget != 0);
    auto q = ((idx * w) / (to - 1) + entries.front().rmax);
    assert(idx < d_out.size());
    d_out[idx] = BinarySearchQuery(entries, q);
  });
  this->columns_ptr_.HostVector() = new_columns_ptr.HostVector();
  this->Alternate();
  timer.Stop(__func__);
}

void SketchContainer::Merge(std::vector< Span<SketchEntry> > that) {
  timer.Start(__func__);
  CHECK_EQ(that.size(), this->num_columns_);
  size_t total = 0;
  for (auto s : that) {
    total += s.size();
  }

  if (this->Current().size() == 0) {
    CHECK_EQ(this->columns_ptr_.Size(), 0);
    this->Current().resize(total, SketchEntry{0, 0, 0, 0});
    size_t offset = 0;
    columns_ptr_.HostVector().emplace_back(offset);
    for (auto c : that) {
      dh::safe_cuda(cudaMemcpyAsync(this->Current().data().get() + offset,
                                    c.data(), c.size_bytes(),
                                    cudaMemcpyDeviceToDevice));
      offset += c.size();
      columns_ptr_.HostVector().emplace_back(offset);
    }

    CHECK_EQ(columns_ptr_.Size(), num_columns_ + 1);
    timer.Stop(__func__);
    return;
  }

  this->Other().resize(this->Current().size() + total, SketchEntry{0, 0, 0, 0});
  bst_feature_t out_offset = 0;
  std::vector<bst_feature_t> new_columns_ptr{out_offset};
  for (size_t i = 0; i < num_columns_; ++i) {
    auto self_column = this->Column(i);
    auto that_column = that[i];

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
  timer.Stop(__func__);
}

void SketchContainer::AllReduce() {
  dh::safe_cuda(cudaSetDevice(device_));
  auto world = rabit::GetWorldSize();
  if (world == 1) {
    return;
  }
  if (!reducer_) {
    reducer_ = std::make_unique<dh::AllReducer>();
    reducer_->Init(device_, false);
  }
  auto d_columns_ptr = this->columns_ptr_.ConstDeviceSpan();
  dh::caching_device_vector<bst_feature_t> gathered_ptrs;
  reducer_->AllGather(d_columns_ptr.data(), d_columns_ptr.size(), &gathered_ptrs);
  std::vector<bst_feature_t> h_gathered_ptrs (gathered_ptrs.size());
  thrust::copy(gathered_ptrs.begin(), gathered_ptrs.end(), h_gathered_ptrs.begin());
  std::vector<size_t> recv_lengths;
  dh::caching_device_vector<char> recvbuf;
  reducer_->AllGather(this->Current().data().get(),
                      dh::ToSpan(this->Current()).size_bytes(), &recv_lengths,
                      &recvbuf);
  reducer_->Synchronize();

  auto s_recvbuf = dh::ToSpan(recvbuf);
  std::vector<Span<SketchEntry>> allworkers;
  size_t offset = 0;
  for (int32_t i = 0; i < world; ++i) {
    size_t length_as_bytes = recv_lengths[i];
    auto raw = s_recvbuf.subspan(offset, length_as_bytes);
    auto sketch = Span<SketchEntry>(reinterpret_cast<SketchEntry *>(raw.data()),
                                    length_as_bytes / sizeof(SketchEntry));
    allworkers.emplace_back(sketch);
    offset += length_as_bytes;
  }

  for (size_t i = 0; i < allworkers.size(); ++i) {
    auto worker = allworkers[i];
    auto worker_ptr = dh::ToSpan(gathered_ptrs).subspan(i * d_columns_ptr.size(), d_columns_ptr.size());
    std::vector<Span<SketchEntry>> columns;
    for (size_t j = 1; j < worker_ptr.size(); ++j) {
      columns.emplace_back(worker.subspan(worker_ptr[j], worker_ptr[j] - worker_ptr[j-1]));
    }
    this->Merge(columns);
  }
}

void SketchContainer::MakeCuts(HistogramCuts* p_cuts) {
  timer.Start(__func__);
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
  p_cuts->cut_ptrs_.SetDevice(0);
  auto& h_out_columns_ptr = p_cuts->cut_ptrs_.HostVector();
  h_out_columns_ptr.clear();
  h_out_columns_ptr.push_back(0);
  for (size_t i = 0; i < num_columns_; ++i) {
    h_out_columns_ptr.push_back(
        std::min(std::max(1ul, this->Column(i).size()), static_cast<size_t>(num_bins_)));
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

  // 1 thread for writing min value
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
  timer.Stop(__func__);
}
}  // namespace common
}  // namespace xgboost