/*!
 * Copyright 2020 by XGBoost Contributors
 */
#include "xgboost/span.h"
#include "quantile.h"
#include "quantile.cuh"
#include "hist_util.h"
#include "device_helpers.cuh"
#include "common.h"
#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

namespace xgboost {
namespace common {

using WQSketch = DenseCuts::WQSketch;
using SketchEntry = WQSketch::Entry;

__device__ SketchEntry BinarySearchQuery(Span<SketchEntry> entries, float rank) {
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

struct Comp {
  bool __device__ operator()(SketchEntry const& x, SketchEntry const& y) {
    return x.value <= y.value;
  }
};

template<typename DType, typename RType>
void WQSummary<DType, RType>::SetPruneDevice(const WQSummary &src, size_t maxsize) {
  if (src.size <= maxsize) {
    this->CopyFrom(src); return;
  }
  dh::caching_device_vector<SketchEntry> in(src.size);
  dh::safe_cuda(cudaMemcpyAsync(in.data().get(), src.data,
                                sizeof(SketchEntry) * src.size,
                                cudaMemcpyHostToDevice));
  // Filter out duplicated values.
  auto new_end = thrust::unique(thrust::device, in.begin(), in.end(),
                                [] __device__(SketchEntry a, SketchEntry b) {
                                  return a.value == b.value;
                                });
  size_t unique_inputs = std::distance(in.begin(), new_end);
  if (unique_inputs <= maxsize) {
    dh::safe_cuda(cudaMemcpyAsync(this->data, in.data().get(),
                                  sizeof(SketchEntry) * unique_inputs,
                                  cudaMemcpyDeviceToHost));
    this->size = unique_inputs;
  }
  auto entries = common::Span<SketchEntry>{in.data().get(),
                                           in.data().get() + unique_inputs};
  dh::caching_device_vector<SketchEntry> out(maxsize);
  auto d_out = dh::ToSpan(out);
  dh::LaunchN(0, maxsize - 2, [=] __device__(size_t tid) {
    tid += 1;
    float w = entries.back().rmin - entries.front().rmax;
    auto budget = static_cast<float>(d_out.size());
    assert(w != 0);
    assert(budget != 0);
    auto q = ((tid * w) / (maxsize - 1) + entries.front().rmax);
    d_out[tid] = BinarySearchQuery(entries, q);
  });
  dh::LaunchN(0, 1, [=]__device__(size_t tid) {
      d_out.front() = entries.front();
      d_out.back() = entries.back();
  });
  auto unique_end = thrust::unique(thrust::device, out.begin(), out.end(),
                 [] __device__(SketchEntry const &l, SketchEntry const &r) {
                   return l.value == r.value;
                 });
  size_t n_uniques = std::distance(out.begin(), unique_end);
  this->size = n_uniques;
  dh::safe_cuda(cudaMemcpyAsync(this->data, out.data().get(),
                                sizeof(SketchEntry) * n_uniques,
                                cudaMemcpyDeviceToHost));
}

void CopyTo(dh::caching_device_vector<SketchEntry> *out,
            Span<SketchEntry const> src) {
  out->resize(src.size());
  dh::safe_cuda(cudaMemcpyAsync(out->data().get(), src.data(),
                                out->size() * sizeof(SketchEntry),
                                cudaMemcpyHostToDevice));
}

void Merge(Span<SketchEntry const> d_x, Span<SketchEntry const> d_y,
           dh::caching_device_vector<SketchEntry> *out) {
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
      thrust::make_zip_iterator(thrust::make_tuple(place_holder, a_ind_iter));
  auto y_val_it =
      thrust::make_zip_iterator(thrust::make_tuple(place_holder, b_ind_iter));

  using Tuple = thrust::tuple<int32_t, int32_t>;
  auto get_ind = []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<1>(t); };
  auto get_a = []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<0>(t); };
  auto get_b = []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<1>(t); };

  dh::caching_device_vector<Tuple> merge_path(out->size());
  // Determine the merge path
  thrust::merge_by_key(thrust::device, a_key_it, a_key_it + d_x.size(), b_key_it,
                       b_key_it + d_y.size(), x_val_it, y_val_it,
                       thrust::make_discard_iterator(), merge_path.begin());
  // Compute the index for both a and b
  thrust::transform_exclusive_scan(
      thrust::device, merge_path.cbegin(), merge_path.cend(),
      merge_path.begin(),
      [=] __device__(Tuple const &t) {
        auto ind = get_ind(t);  // == 0 if element is from a
        // a_counter, b_counter
        return thrust::make_tuple(!ind, ind);
      },
      thrust::make_tuple(0, 0),
      [=] __device__(Tuple const &l, Tuple const &r) {
        return thrust::make_tuple(get_a(l) + get_a(r), get_b(l) + get_b(r));
      });

  auto d_merge_path = dh::ToSpan(merge_path);
  auto d_out = Span<SketchEntry>{out->data().get(), d_x.size() + d_y.size()};

  dh::LaunchN(0, d_out.size(), [=] __device__(size_t idx) {
    int32_t a_ind, b_ind;
    thrust::tie(a_ind, b_ind) = d_merge_path[idx];
    // Handle trailing elements.
    assert(a_ind <= d_x.size());
    if (a_ind == d_x.size()){
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

    // Merge procedure.
    assert(idx < d_out.size());
    if (x_elem.value == y_elem.value) {
      d_out[idx] =
          SketchEntry{x_elem.rmin + y_elem.rmin, x_elem.rmax + y_elem.rmax,
                      x_elem.wmin + y_elem.wmin, x_elem.value};
    } else if (x_elem.value < y_elem.value) {
      // elem from x is landed
      float yprev_min = b_ind == 0 ? 0.0f : d_y[b_ind - 1].RMinNext();
      d_out[idx] =
          SketchEntry{x_elem.rmin + yprev_min, x_elem.rmax + y_elem.RMaxPrev(),
                      x_elem.wmin, x_elem.value};
    } else {
      // elem from y is landed
      float xprev_min = a_ind == 0 ? 0.0f : d_x[a_ind - 1].RMinNext();
      d_out[idx] =
          SketchEntry{xprev_min + y_elem.rmin, x_elem.RMaxPrev() + y_elem.rmax,
                      y_elem.wmin, y_elem.value};
    }
  });
}

template<typename DType, typename RType>
void WQSummary<DType, RType>::DeviceSetCombined(WQSummary const& a, WQSummary const& b) {
  dh::caching_device_vector<SketchEntry> out(a.size + b.size);
  dh::safe_cuda(cudaMemcpyAsync(out.data().get(), this->data,
                                out.size() * sizeof(SketchEntry),
                                cudaMemcpyHostToDevice));
  auto cpu_data_ptr = this->data;
  this->data = out.data().get();

  dh::caching_device_vector<SketchEntry> a_data(a.size);
  WQSketch::Summary sa(a_data.data().get(), a.size);
  dh::safe_cuda(cudaMemcpyAsync(a_data.data().get(), a.data,
                                a.size * sizeof(SketchEntry),
                                cudaMemcpyHostToDevice));

  dh::caching_device_vector<SketchEntry> b_data(b.size);
  WQSketch::Summary sb(b_data.data().get(), b.size);
  dh::safe_cuda(cudaMemcpyAsync(b_data.data().get(), b.data,
                                b.size * sizeof(SketchEntry),
                                cudaMemcpyHostToDevice));

  auto d_x = Span<SketchEntry>{sa.data, sa.size};
  auto d_y = Span<SketchEntry>{sb.data, sb.size};
  Merge(d_x, d_y, &out);

  this->data = cpu_data_ptr;
  this->size = out.size();
  dh::safe_cuda(cudaMemcpyAsync(this->data, out.data().get(),
                                out.size() * sizeof(SketchEntry),

                                cudaMemcpyDeviceToHost));
}

template<typename DType, typename RType>
void WQSummary<DType, RType>::SetCombine(const WQSummary &sa, const WQSummary &sb) {
  this->DeviceSetCombined(sa, sb);
  return;

  if (sa.size == 0) {
    this->CopyFrom(sb); return;
  }
  if (sb.size == 0) {
    this->CopyFrom(sa); return;
  }
  CHECK(sa.size > 0 && sb.size > 0);
  const Entry *a = sa.data, *a_end = sa.data + sa.size;
  const Entry *b = sb.data, *b_end = sb.data + sb.size;
  // extended rmin value
  RType aprev_rmin = 0, bprev_rmin = 0;
  Entry *dst = this->data;
  while (a != a_end && b != b_end) {
    // duplicated value entry
    if (a->value == b->value) {
      *dst = Entry(a->rmin + b->rmin,
                   a->rmax + b->rmax,
                   a->wmin + b->wmin, a->value);
      aprev_rmin = a->RMinNext();
      bprev_rmin = b->RMinNext();
      ++dst; ++a; ++b;
    } else if (a->value < b->value) {
      *dst = Entry(a->rmin + bprev_rmin,
                   a->rmax + b->RMaxPrev(),
                   a->wmin, a->value);
      aprev_rmin = a->RMinNext();
      ++dst; ++a;
    } else {
      *dst = Entry(b->rmin + aprev_rmin,
                   b->rmax + a->RMaxPrev(),
                   b->wmin, b->value);
      bprev_rmin = b->RMinNext();
      ++dst; ++b;
    }
  }
  if (a != a_end) {
    RType brmax = (b_end - 1)->rmax;
    do {
      *dst = Entry(a->rmin + bprev_rmin, a->rmax + brmax, a->wmin, a->value);
      ++dst; ++a;
    } while (a != a_end);
  }
  if (b != b_end) {
    RType armax = (a_end - 1)->rmax;
    do {
      *dst = Entry(b->rmin + aprev_rmin, b->rmax + armax, b->wmin, b->value);
      ++dst; ++b;
    } while (b != b_end);
  }

  this->size = dst - data;

  const RType tol = 10;
  RType err_mingap, err_maxgap, err_wgap;
  this->FixError(&err_mingap, &err_maxgap, &err_wgap);
  if (err_mingap > tol || err_maxgap > tol || err_wgap > tol) {
    LOG(INFO) << "mingap=" << err_mingap
              << ", maxgap=" << err_maxgap
              << ", wgap=" << err_wgap;
  }
  CHECK(size <= sa.size + sb.size) << "bug in combine";
}

template class WQSummary<float, float>;

void ConstructCutMatrix(WQSketch::SummaryContainer const& summary, int max_bin, HistogramCuts* cuts) {
  size_t required_cuts = std::min(summary.size, static_cast<size_t>(max_bin));
  size_t ori_size = cuts->cut_values_.Size();
  cuts->cut_values_.Resize(ori_size + required_cuts);
  auto d_cut_values = cuts->cut_values_.DeviceSpan();
  auto data = Span<SketchEntry>{summary.data, summary.size};
  dh::LaunchN(0, required_cuts - 1, [=] __device__(size_t idx) {
    idx += 1;
    d_cut_values[idx + ori_size] = data[idx].value;
  });
}

void DeviceQuantile::MakeFromSorted(Span<SketchEntry> entries, int32_t device) {
  this->device_ = device;
  this->comm_.Init(device_);
  auto data = entries.data();
  SketchEntry *new_end =
      thrust::unique(thrust::device, data, data + entries.size(),
                     [] __device__(SketchEntry a, SketchEntry b) {
                       return a.value == b.value;
                     });
  static_assert(std::is_trivially_copy_constructible<SketchEntry>::value, "");
  static_assert(std::is_standard_layout<SketchEntry>::value, "");
  data_.resize(std::distance(data, new_end));
  dh::safe_cuda(cudaMemcpyAsync(
      data_.data().get(), entries.data(), sizeof(SketchEntry) * std::distance(data, new_end),
      cudaMemcpyDeviceToDevice));
}

void DeviceQuantile::SetMerge(std::vector<Span<SketchEntry const>> const& others) {
  auto x = others.front();
  dh::safe_cuda(cudaMemcpyAsync(this->data_.data().get(), x.data(),
                                this->data_.size() * sizeof(SketchEntry),
                                cudaMemcpyDeviceToDevice));
  dh::caching_device_vector<SketchEntry> buffer;
  for (size_t i = 1; i < others.size(); ++i) {
    auto x = dh::ToSpan(this->data_);
    auto const y = others[i];
    Merge(x, y, &buffer);
    this->data_.resize(buffer.size());
    dh::safe_cuda(cudaMemcpyAsync(this->data_.data().get(), buffer.data().get(),
                                  buffer.size() * sizeof(SketchEntry),
                                  cudaMemcpyDeviceToDevice));
  }
}

void DeviceQuantile::MakeFromOthers(std::vector<DeviceQuantile> const& others) {
  this->comm_.Init(device_);
  std::vector<Span<SketchEntry const>> spans(others.size());
  for (size_t i = 0; i < others.size(); ++i) {
    spans[i] = Span<SketchEntry const>(others[i].data_.data().get(),
                                       others[i].data_.size());
  }
  this->SetMerge(spans);
}

void DeviceQuantile::AllReduce() {
  dh::caching_device_vector<char> recvbuf;
  comm_.AllGather(data_.data().get(), data_.size() * sizeof(SketchEntry), &recvbuf);
  auto s_recvbuf = dh::ToSpan(recvbuf);
  size_t world = rabit::GetWorldSize();
  std::vector<Span<SketchEntry const>> allworkers;
  auto length_as_bytes = data_.size() * sizeof(SketchEntry);

  for (size_t i = 0; i < world; ++i) {
    auto raw = s_recvbuf.subspan(i * length_as_bytes, length_as_bytes);
    auto sketch = Span<SketchEntry>(reinterpret_cast<SketchEntry*>(raw.data()), data_.size());
    allworkers.emplace_back(sketch);
  }
  this->SetMerge(allworkers);
}
}  // namespace common
}  // namespace xgboost