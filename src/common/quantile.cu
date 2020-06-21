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
}  // anonymous namespace

void PruneImpl(size_t to, common::Span<SketchEntry> entries, dh::caching_device_vector<SketchEntry>* p_out) {
  auto& out = *p_out;
  // Filter out duplicated values.
  size_t unique_inputs = std::distance(
      entries.data(),
      thrust::unique(thrust::device, entries.data(),
                     entries.data() + entries.size(), SketchUnique{}));
  if (unique_inputs <= to) {
    p_out->resize(unique_inputs);
    dh::safe_cuda(cudaMemcpyAsync(p_out->data().get(), entries.data(),
                                  sizeof(SketchEntry) * unique_inputs,
                                  cudaMemcpyDeviceToHost));
  }
  entries = entries.subspan(0, unique_inputs);
  out.resize(to);
  auto d_out = dh::ToSpan(out);
  // 1 thread for each output.  See A.4 for detail.
  dh::LaunchN(0, to - 2, [=] __device__(size_t tid) {
    tid += 1;
    float w = entries.back().rmin - entries.front().rmax;
    auto budget = static_cast<float>(d_out.size());
    assert(w != 0);
    assert(budget != 0);
    auto q = ((tid * w) / (to - 1) + entries.front().rmax);
    d_out[tid] = BinarySearchQuery(entries, q);
  });
  dh::LaunchN(0, 1, [=]__device__(size_t tid) {
      d_out.front() = entries.front();
      d_out.back() = entries.back();
  });
  auto unique_end = thrust::unique(thrust::device, out.begin(), out.end(), SketchUnique{});
  size_t n_uniques = std::distance(out.begin(), unique_end);
  out.resize(n_uniques);
}

void DeviceQuantile::Prune(size_t to) {
  dh::caching_device_vector<SketchEntry> out;
  PruneImpl(to, this->Data(), &out);
  this->data_ = std::move(out);
}

template<typename DType, typename RType>
void WQSummary<DType, RType>::SetPruneDevice(const WQSummary &src, size_t maxsize) {
  if (src.size <= maxsize) {
    this->CopyFrom(src); return;
  }
  dh::caching_device_vector<SketchEntry> in(src.size);
  dh::safe_cuda(cudaMemcpyAsync(in.data().get(), src.data,
                                sizeof(SketchEntry) * src.size,
                                cudaMemcpyHostToDevice));
  dh::caching_device_vector<SketchEntry> out(maxsize);
  PruneImpl(maxsize, dh::ToSpan(in), &out);
  this->size = out.size();
  dh::safe_cuda(cudaMemcpyAsync(this->data, out.data().get(),
                                sizeof(SketchEntry) * out.size(),
                                cudaMemcpyDeviceToHost));
}

void CopyTo(dh::caching_device_vector<SketchEntry> *out,
            Span<SketchEntry const> src) {
  out->resize(src.size());
  dh::safe_cuda(cudaMemcpyAsync(out->data().get(), src.data(),
                                out->size() * sizeof(SketchEntry),
                                cudaMemcpyHostToDevice));
}

// Merge d_x and d_y into out.  Because the final output depends on predicate (which
// summary does the output element come from) result by definition of merged rank.  So we
// run it in 2 phrases to obtain the merge path and then customize the standard merge
// algorithm.
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
  auto get_a =   []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<0>(t); };
  auto get_b =   []XGBOOST_DEVICE(Tuple const& t) { return thrust::get<1>(t); };

  dh::caching_device_vector<Tuple> merge_path(out->size());
  // Determine the merge path
  thrust::merge_by_key(thrust::device, a_key_it, a_key_it + d_x.size(), b_key_it,
                       b_key_it + d_y.size(), x_val_it, y_val_it,
                       thrust::make_discard_iterator(), merge_path.begin());
  // Compute the index for both a and b (which of the element in a and b are used in each
  // comparison).  Take output [(a_0, b_0), (a_0, b_1), ...] as an example, the comparison
  // between (a_0, b_0) adds 1 step in the merge path.  Because b_0 is less than a_0 so
  // this step is torward the end of b.  After the comparison, index of b is incremented
  // by 1 from b_0 to b_1, and at the same time, b_0 is landed into output as the first
  // element in merge result.  Here we use the merge path to compute index for both a and
  // b along the merge path.  The output of this scan is a path showing each comparison.
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
         different summary.
    */
    assert(idx < d_out.size());
    if (x_elem.value == y_elem.value) {
      d_out[idx] =
          SketchEntry{x_elem.rmin + y_elem.rmin, x_elem.rmax + y_elem.rmax,
                      x_elem.wmin + y_elem.wmin, x_elem.value};
    } else if (x_elem.value < y_elem.value) {
      // elem from x is landed. yprev_min is the element in D_2 that's 1 rank less than
      // x_elem.
      float yprev_min = b_ind == 0 ? 0.0f : d_y[b_ind - 1].RMinNext();
      // rmin should equal to x_elem.rmin + x_elem.wmin.  But for implementation, the
      // weight is stored in a separated field and compute the extended definition on the
      // fly when needed.
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

template class WQSummary<float, float>;

void DeviceQuantile::PushSorted(common::Span<SketchEntry> entries) {
  dh::caching_device_vector<SketchEntry> out;
  SketchEntry *new_end =
      thrust::unique(thrust::device, entries.data(),
                     entries.data() + entries.size(), SketchUnique{});
  entries = entries.subspan(0, std::distance(entries.data(), new_end)) ;

  Merge(this->Data(), entries, &out);
  this->data_ = std::move(out);
  this->Prune(this->limit_size_);
}

void DeviceQuantile::MakeCuts(size_t max_rows, int max_bin, HistogramCuts* cuts) {
  constexpr int kFactor = 8;
  size_t global_max_rows = max_rows;
  rabit::Allreduce<rabit::op::Sum>(&global_max_rows, 1);
  size_t intermediate_num_cuts =
      std::min(global_max_rows, static_cast<size_t>(max_bin * kFactor));
  this->Prune(intermediate_num_cuts);
  this->AllReduce();

  size_t required_cuts = std::min(this->Data().size(), static_cast<size_t>(max_bin));
  cuts->cut_values_.SetDevice(this->device_);
  size_t ori_size = cuts->cut_values_.Size();
  cuts->cut_values_.Resize(ori_size + required_cuts);
  auto cut_values = cuts->cut_values_.HostVector();
  auto data = this->Data();

  cuts->min_vals_.SetDevice(this->device_);
  cuts->min_vals_.Resize(cuts->min_vals_.Size() + 1);
  auto d_min_vals = cuts->min_vals_.DeviceSpan();

  std::vector<SketchEntry> entries(this->Data().size());
  thrust::copy(this->data_.begin(), this->data_.end(), entries.begin());
  for (size_t i = 1; i < required_cuts; ++i) {
    float cpt = entries[i].value;
    if (i == 1 || cpt > cut_values.back()) {
      cut_values.push_back(cpt);
    }
  }
}

void DeviceQuantile::MakeFromSorted(Span<SketchEntry> entries, int32_t device) {
  this->device_ = device;
  auto data = entries.data();
  SketchEntry *new_end =
      thrust::unique(thrust::device, data, data + entries.size(), SketchUnique{});
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
  std::vector<Span<SketchEntry const>> spans(others.size());
  // We don't have k way merging, so iterate it through.
  for (size_t i = 0; i < others.size(); ++i) {
    spans[i] = Span<SketchEntry const>(others[i].data_.data().get(),
                                       others[i].data_.size());
  }
  this->SetMerge(spans);
}

void DeviceQuantile::AllReduce() {
  size_t world = rabit::GetWorldSize();
  if (world == 1) {
    return;
  }
  if (!comm_) {
    comm_ = std::make_unique<dh::AllReducer>();
    comm_->Init(device_);
  }

  dh::caching_device_vector<char> recvbuf;
  comm_->AllGather(data_.data().get(), data_.size() * sizeof(SketchEntry), &recvbuf);
  auto s_recvbuf = dh::ToSpan(recvbuf);
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