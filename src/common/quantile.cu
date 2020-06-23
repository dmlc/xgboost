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

void PruneImpl(size_t to, common::Span<SketchEntry> entries,
               Span<SketchEntry> out,
               cudaStream_t stream = nullptr) {
  CHECK_GE(to, 2);
  if (entries.size() <= to) {
    dh::safe_cuda(cudaMemcpyAsync(out.data(), entries.data(),
                                  entries.size_bytes(),
                                  cudaMemcpyDeviceToDevice, stream));
    return;
  }
  CHECK_GE(out.size(), to);

  auto d_out = out;
  // 1 thread for each output.  See A.4 for detail.
  dh::LaunchN(0, to - 2, stream, [=] __device__(size_t tid) {
    tid += 1;
    float w = entries.back().rmin - entries.front().rmax;
    auto budget = static_cast<float>(d_out.size());
    assert(w != 0);
    assert(budget != 0);
    auto q = ((tid * w) / (to - 1) + entries.front().rmax);
    assert(tid < d_out.size());
    d_out[tid] = BinarySearchQuery(entries, q);
  });
  dh::LaunchN(0, 1, stream, [=]__device__(size_t) {
      assert(d_out.size() >= 2);
      d_out.front() = entries.front();
      d_out.back() = entries.back();
  });
}

void DeviceQuantile::Prune(size_t to) {
  monitor.Start(__func__);
  if (to > this->Current().size()) {
    return;
  }
  // PruneImpl(to, dh::ToSpan(this->Current()), &(this->Other()), stream_);
  this->Alternate();
  monitor.Stop(__func__);
}

void CopyTo(Span<SketchEntry> out,
            Span<SketchEntry const> src) {
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
    if (b_ind > d_y.size()) {
      printf("b_ind: %d, d_y.size(): %lu\n", b_ind, d_y.size());
    }
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

void DeviceQuantile::PushSorted(common::Span<SketchEntry> entries) {
  monitor.Start(__func__);
  dh::safe_cuda(cudaSetDevice(device_));
  SketchEntry *new_end =
      thrust::unique(thrust::device, entries.data(),
                     entries.data() + entries.size(), SketchUnique{});
  entries = entries.subspan(0, std::distance(entries.data(), new_end));

  MergeImpl(this->Data(), entries, &(this->Other()), stream_);
  this->Alternate();
  this->Prune(this->limit_size_);
  monitor.Stop(__func__);
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

void DeviceQuantile::MakeFromOthers(std::vector<DeviceQuantile> const& others) {
  dh::safe_cuda(cudaSetDevice(device_));
  std::vector<Span<SketchEntry const>> spans(others.size());
  for (size_t i = 0; i < others.size(); ++i) {
    spans[i] = Span<SketchEntry const>(others[i].Current().data().get(),
                                       others[i].Current().size());
  }
  this->SetMerge(spans);
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

void DeviceQuantile::Synchronize() {
  if (comm_) {
    comm_->Synchronize();
  }
}

void SketchContainer::Push(size_t entries_per_column,
                           const common::Span<SketchEntry>& entries,
                           const thrust::host_vector<size_t>& column_scan) {
  timer.Start(__func__);
  std::vector<Span<SketchEntry>> columns;
  for (size_t icol = 0; icol < num_columns_; ++icol) {
    size_t column_size = column_scan[icol + 1] - column_scan[icol];
    size_t num_available_cuts =
        std::min(size_t(entries_per_column), column_size);
    CHECK_GT(column_size, 0);  // FIXME
    // if (column_size == 0) continue;
    columns.emplace_back(entries.subspan(entries_per_column * icol, num_available_cuts));
  }
  CHECK_EQ(columns.size(), num_columns_);
  this->Merge(columns);
  this->Prune(limit_size_);

  timer.Stop(__func__);
}

size_t SketchContainer::Unique() {
  this->columns_ptr_.SetDevice(device_);
  Span<size_t> d_column_scan = this->columns_ptr_.DeviceSpan();
  CHECK_EQ(d_column_scan.size(), num_columns_ + 1);

  Span<SketchEntry> entries = dh::ToSpan(this->Current());

  d_column_scan = this->columns_ptr_.DeviceSpan();
  size_t n_uniques = SegmentedUnique(
      d_column_scan.data(), d_column_scan.data() + d_column_scan.size(),
      entries.data(), entries.data() + entries.size(), d_column_scan.data(),
      entries.data(),
      SketchUnique{});
  CHECK(!this->columns_ptr_.HostCanRead());
  // auto const& h_columns_ptr = this->columns_ptr_.ConstHostSpan();
  // CHECK_EQ(h_columns_ptr.size(), num_columns_ + 1);
  // CHECK_GT(h_columns_ptr.back(), 0);
  // CHECK_LT(h_columns_ptr.back(), std::numeric_limits<bst_feature_t>::max());

  this->Current().resize(n_uniques, SketchEntry{0, 0, 0, 0});
  // CHECK_EQ(h_columns_ptr.back(), n_uniques);
  return n_uniques;
}

void SketchContainer::Prune(size_t to) {
  timer.Start(__func__);
  auto n_uniques = this->Unique();
  // auto const& h_columns_ptr = this->columns_ptr_.ConstHostSpan();
  // CHECK_EQ(n_uniques, h_columns_ptr.back());

  size_t to_total = 0;
  HostDeviceVector<size_t> new_columns_ptr{to_total};
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
    size_t column_id =
        thrust::upper_bound(thrust::seq, d_columns_ptr_out.begin(),
                            d_columns_ptr_out.end(), idx) -
        1 - d_columns_ptr_out.begin();
    // size_t out_offset = d_columns_ptr_out[column_id];
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
      thrust::copy(thrust::device, c.data(), c.data() + c.size(),
                   this->Current().begin() + offset);
      offset += c.size();
      columns_ptr_.HostVector().emplace_back(offset);
    }

    CHECK_EQ(columns_ptr_.Size(), num_columns_ + 1);
    return;
  }

  // auto const& h_columns_ptr = columns_ptr_.ConstHostSpan();
  // CHECK_EQ(h_columns_ptr.size(), that.size() + 1);

  this->Other().resize(this->Current().size() + total, SketchEntry{0, 0, 0, 0});
  size_t out_offset = 0;
  std::vector<size_t> new_columns_ptr{out_offset};
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

void AddCutPoint(std::vector<SketchEntry> const &summary, int max_bin,
                 HistogramCuts *p_cuts_) {
  size_t required_cuts = std::min(summary.size(), static_cast<size_t>(max_bin));
  for (size_t i = 1; i < required_cuts; ++i) {
    bst_float cpt = summary[i].value;
    if (i == 1 || cpt > p_cuts_->cut_values_.ConstHostVector().back()) {
      p_cuts_->cut_values_.HostVector().push_back(cpt);
    }
  }
}

void SketchContainer::MakeCuts(HistogramCuts* p_cuts) {
  timer.Start(__func__);
  p_cuts->min_vals_.HostVector().resize(num_columns_);
  size_t global_max_rows = num_rows_;
  rabit::Allreduce<rabit::op::Sum>(&global_max_rows, 1);
  size_t intermediate_num_cuts =
      std::min(global_max_rows, static_cast<size_t>(num_bins_ * kFactor));
  this->Prune(intermediate_num_cuts);

  // Prune according to global number of rows.
  // for (auto& sketch : sketches_) {
  //   sketch.Prune(intermediate_num_cuts);
  //   sketch.AllReduce();
  // }
  this->Prune(num_bins_ + 1);
  for (size_t fid = 0; fid < num_columns_; ++fid) {
    // sketches_[fid].Synchronize();

    if (this->Column(fid).size() == 0) {  // Empty column
      p_cuts->min_vals_.HostVector().push_back(kRtEps);
      p_cuts->cut_values_.HostVector().push_back(kRtEps);
      auto cut_size = static_cast<uint32_t>(p_cuts->cut_values_.HostVector().size());
      p_cuts->cut_ptrs_.HostVector().push_back(cut_size);
      continue;
    }

    // sketches_[fid].Prune(num_bins_ + 1);
    std::vector<SketchEntry> entries(this->Column(fid).size());
    dh::safe_cuda(
        cudaMemcpyAsync(entries.data(), this->Column(fid).data(),
                        this->Column(fid).size_bytes(),
                        cudaMemcpyDeviceToHost));
    CHECK_GT(entries.size(), 0);
    const bst_float mval = entries[0].value;
    p_cuts->min_vals_.HostVector()[fid] = mval - (fabs(mval) + 1e-5);
    AddCutPoint(entries, num_bins_, p_cuts);
    // push a value that is greater than anything
    const bst_float cpt
        = (entries.size() > 0) ? entries.back().value : p_cuts->min_vals_.HostVector()[fid];
    // this must be bigger than last value in a scale
    const bst_float last = cpt + (fabs(cpt) + 1e-5);
    p_cuts->cut_values_.HostVector().push_back(last);

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(p_cuts->cut_values_.HostVector().size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(p_cuts->cut_values_.HostVector().size());
    CHECK_GT(cut_size, p_cuts->cut_ptrs_.HostVector().back());
    p_cuts->cut_ptrs_.HostVector().push_back(cut_size);
  }
  timer.Stop(__func__);
}
}  // namespace common
}  // namespace xgboost