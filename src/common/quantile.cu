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
  bool __device__ operator()(SketchEntry const& a, SketchEntry const& b) {
    return a.value == b.value;
  }
};


struct IsSorted {
  bool XGBOOST_DEVICE operator()(SketchEntry const& a, SketchEntry const& b) {
    return a.value <= b.value;
  }
};
}  // anonymous namespace

void PruneImpl(size_t to, common::Span<SketchEntry> entries,
               dh::caching_device_vector<SketchEntry> *p_out,
               cudaStream_t stream = nullptr) {
  dh::XGBCachingDeviceAllocator<SketchEntry> alloc;

  auto& out = *p_out;
  // Filter out duplicated values.  The unique call here really hurts performance when
  // number of columns is large.
  size_t unique_inputs = std::distance(
      entries.data(),
      thrust::unique(thrust::cuda::par(alloc).on(stream), entries.data(),
                     entries.data() + entries.size(), SketchUnique{}));
  CHECK_GE(to, 2);
  if (unique_inputs <= to) {
    p_out->resize(unique_inputs);
    dh::safe_cuda(cudaMemcpyAsync(p_out->data().get(), entries.data(),
                                  sizeof(SketchEntry) * unique_inputs,
                                  cudaMemcpyDeviceToDevice, stream));
    return;
  }
  entries = entries.subspan(0, unique_inputs);

  out.resize(to);
  auto d_out = dh::ToSpan(out);
  // 1 thread for each output.  See A.4 for detail.
  dh::LaunchN(0, to - 2, stream, [=] __device__(size_t tid) {
    tid += 1;
    float w = entries.back().rmin - entries.front().rmax;
    auto budget = static_cast<float>(d_out.size());
    assert(w != 0);
    assert(budget != 0);
    auto q = ((tid * w) / (to - 1) + entries.front().rmax);
    d_out[tid] = BinarySearchQuery(entries, q);
  });
  dh::LaunchN(0, 1, stream, [=]__device__(size_t) {
      d_out.front() = entries.front();
      d_out.back() = entries.back();
  });

  // if (!thrust::is_sorted(
  //     thrust::cuda::par.on(stream), out.begin(), out.end(),
  //     IsSorted{})) {

  //   auto it = thrust::is_sorted_until(thrust::cuda::par.on(stream), out.begin(), out.end(),
  //                                     IsSorted{});
  //   std::cout << std::setprecision(19) << " it: " << it - out.begin() << ", "
  //             << *(it - 1) << ", " << *it << ", "
  //             << IsSorted{}(*(it - 1), *it) <<  std::endl;

  //   for (size_t i = 1; i < out.size(); ++i) {
  //     if (!(IsSorted{}(out[i-1], out[i]))) {
  //       std::cout << std::setprecision(17) << out[i - 1] << " vs " << out[i] << ", ";
  //     }
  //   }
  //   std::cout << std::endl;
  //   LOG(FATAL) << "Not sorted";
  // }
}

void DeviceQuantile::Prune(size_t to) {
  monitor.Start(__func__);
  if (to > this->Current().size()) {
    return;
  }
  PruneImpl(to, dh::ToSpan(this->Current()), &(this->Other()), stream_);
  this->Alternate();
  monitor.Stop(__func__);
}

void CopyTo(dh::caching_device_vector<SketchEntry> *out,
            Span<SketchEntry const> src) {
  out->resize(src.size());
  dh::safe_cuda(cudaMemcpyAsync(out->data().get(), src.data(),
                                out->size() * sizeof(SketchEntry),
                                cudaMemcpyDefault));
}

// Merge d_x and d_y into out.  Because the final output depends on predicate (which
// summary does the output element come from) result by definition of merged rank.  So we
// run it in 2 phrases to obtain the merge path and then customize the standard merge
// algorithm.
void MergeImpl(Span<SketchEntry const> d_x, Span<SketchEntry const> d_y,
               dh::caching_device_vector<SketchEntry> *out, cudaStream_t stream = nullptr) {
  if (d_x.size() == 0) {
    CopyTo(out, d_y);
    return;
  }
  if (d_y.size() == 0) {
    CopyTo(out, d_x);
    return;
  }

  out->resize(d_x.size() + d_y.size());
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
  common::Span<Tuple> merge_path{reinterpret_cast<Tuple *>(out->data().get()), out->size()};
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
  auto d_out = Span<SketchEntry>{out->data().get(), d_x.size() + d_y.size()};

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
  dh::caching_device_vector<SketchEntry> buffer;
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
}  // namespace common
}  // namespace xgboost