/**
 * Copyright 2020~2023, XGBoost contributors
 *
 * \brief GPU implementation of GK sketching.
 */
#ifndef XGBOOST_COMMON_QUANTILE_CUH_
#define XGBOOST_COMMON_QUANTILE_CUH_

#include <memory>

#include "categorical.h"
#include "device_helpers.cuh"
#include "quantile.h"
#include "timer.h"
#include "xgboost/data.h"
#include "xgboost/span.h"  // for IterSpan, Span

namespace xgboost::common {

class HistogramCuts;
using WQSketch = WQuantileSketch<bst_float, bst_float>;
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

template <typename EntryIter, typename ToSketchEntry>
void PruneImpl(common::Span<bst_row_t const> cuts_ptr, EntryIter sorted_data,
               Span<size_t const> columns_ptr_in,  // could be ptr for data or cuts
               Span<FeatureType const> feature_types, Span<SketchEntry> out_cuts,
               ToSketchEntry to_sketch_entry) {
  dh::LaunchN(out_cuts.size(), [=] __device__(size_t idx) {
    size_t column_id = dh::SegmentId(cuts_ptr, idx);
    auto out_column = out_cuts.subspan(
        cuts_ptr[column_id], cuts_ptr[column_id + 1] - cuts_ptr[column_id]);

    auto in_column_beg = columns_ptr_in[column_id];
    auto in_column =
        IterSpan{sorted_data + in_column_beg, columns_ptr_in[column_id + 1] - in_column_beg};
    // auto in_column = sorted_data.subspan(columns_ptr_in[column_id],
    //                                      columns_ptr_in[column_id + 1] - columns_ptr_in[column_id]);
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
  dh::safe_cuda(cudaMemcpyAsync(out.data(), src.data(), out.size_bytes(), cudaMemcpyDefault));
}

namespace detail {
struct SketchUnique {
  XGBOOST_DEVICE bool operator()(SketchEntry const& a, SketchEntry const& b) const {
    return a.value - b.value == 0;
  }
};
}  // namespace detail

/*!
 * \brief A container that holds the device sketches.  Sketching is performed per-column,
 *        but fused into single operation for performance.
 */
class SketchContainer {
 public:
  static constexpr float kFactor = WQSketch::kFactor;
  using OffsetT = bst_row_t;
  static_assert(sizeof(OffsetT) == sizeof(size_t), "Wrong type for sketch element offset.");

 private:
  Monitor timer_;
  HostDeviceVector<FeatureType> feature_types_;
  bst_row_t num_rows_;
  bst_feature_t num_columns_;
  int32_t num_bins_;
  int32_t device_;

  // Double buffer as neither prune nor merge can be performed inplace.
  dh::device_vector<SketchEntry> entries_a_;
  dh::device_vector<SketchEntry> entries_b_;
  bool current_buffer_ {true};
  // The container is just a CSC matrix.
  HostDeviceVector<OffsetT> columns_ptr_;
  HostDeviceVector<OffsetT> columns_ptr_b_;

  bool has_categorical_{false};

  dh::device_vector<SketchEntry>& Current() {
    if (current_buffer_) {
      return entries_a_;
    } else {
      return entries_b_;
    }
  }
  dh::device_vector<SketchEntry>& Other() {
    if (!current_buffer_) {
      return entries_a_;
    } else {
      return entries_b_;
    }
  }
  [[nodiscard]] dh::device_vector<SketchEntry> const& Current() const {
    return const_cast<SketchContainer*>(this)->Current();
  }
  [[nodiscard]] dh::device_vector<SketchEntry> const& Other() const {
    return const_cast<SketchContainer*>(this)->Other();
  }
  void Alternate() {
    current_buffer_ = !current_buffer_;
  }

  // Get the span of one column.
  Span<SketchEntry> Column(bst_feature_t i) {
    auto data = dh::ToSpan(this->Current());
    auto h_ptr = columns_ptr_.ConstHostSpan();
    auto c = data.subspan(h_ptr[i], h_ptr[i+1] - h_ptr[i]);
    return c;
  }

 public:
  /* \breif GPU quantile structure, with sketch data for each columns.
   *
   * \param max_bin     Maximum number of bins per column.
   * \param num_columns Total number of columns in dataset.
   * \param num_rows    Total number of rows in known dataset (typically the rows in current worker).
   * \param device      GPU ID.
   */
   SketchContainer(HostDeviceVector<FeatureType> const &feature_types,
                   int32_t max_bin, bst_feature_t num_columns,
                   bst_row_t num_rows, int32_t device)
       : num_rows_{num_rows},
         num_columns_{num_columns}, num_bins_{max_bin}, device_{device} {
     CHECK_GE(device, 0);
     // Initialize Sketches for this dmatrix
     this->columns_ptr_.SetDevice(device_);
     this->columns_ptr_.Resize(num_columns + 1);
     this->columns_ptr_b_.SetDevice(device_);
     this->columns_ptr_b_.Resize(num_columns + 1);

     this->feature_types_.Resize(feature_types.Size());
     this->feature_types_.Copy(feature_types);
     // Pull to device.
     this->feature_types_.SetDevice(device);
     this->feature_types_.ConstDeviceSpan();
     this->feature_types_.ConstHostSpan();

     auto d_feature_types = feature_types_.ConstDeviceSpan();
     has_categorical_ =
         !d_feature_types.empty() &&
         thrust::any_of(dh::tbegin(d_feature_types), dh::tend(d_feature_types),
                        common::IsCatOp{});

     timer_.Init(__func__);
   }
  /* \brief Return GPU ID for this container. */
  int32_t DeviceIdx() const { return device_; }
  /* \brief Whether the predictor matrix contains categorical features. */
  bool HasCategorical() const { return has_categorical_; }
  /* \brief Accumulate weights of duplicated entries in input. */
  size_t ScanInput(Span<SketchEntry> entries, Span<OffsetT> d_columns_ptr_in);
  /* Fix rounding error and re-establish invariance.  The error is mostly generated by the
   * addition inside `RMinNext` and subtraction in `RMaxPrev`. */
  void FixError();

  /**
   * \brief Push sorted entries.
   *
   * \param sorted_entries Iterator to sorted entries.
   * \param columns_ptr CSC pointer for entries.
   * \param cuts_ptr CSC pointer for cuts.
   * \param total_cuts Total number of cuts, equal to the back of cuts_ptr.
   * \param weights (optional) data weights.
   */
  template <typename EntryIter, typename WeightIter = Span<float const>::iterator>
  void Push(EntryIter sorted_entries, Span<size_t> columns_ptr, common::Span<OffsetT> cuts_ptr,
            size_t total_cuts, IterSpan<WeightIter> weights = {}) {
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
      auto to_sketch_entry = [] __device__(size_t sample_idx, auto const& column, size_t) {
        float rmin = sample_idx;
        float rmax = sample_idx + 1;
        return SketchEntry{rmin, rmax, 1, column[sample_idx].fvalue};
      };  // NOLINT
      PruneImpl(cuts_ptr, sorted_entries, columns_ptr, ft, out, to_sketch_entry);
     } else {
      auto to_sketch_entry = [weights, columns_ptr] __device__(
                                 size_t sample_idx, auto const& column, size_t column_id) {
        auto column_weights_scan = weights.subspan(columns_ptr[column_id], column.size());
        float rmin = sample_idx > 0 ? column_weights_scan[sample_idx - 1] : 0.0f;
        float rmax = column_weights_scan[sample_idx];
        float wmin = rmax - rmin;
        wmin = wmin < 0 ? kRtEps : wmin;  // GPU scan can generate floating error.
        return SketchEntry{rmin, rmax, wmin, column[sample_idx].fvalue};
      };  // NOLINT
      PruneImpl(cuts_ptr, sorted_entries, columns_ptr, ft, out, to_sketch_entry);
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
  /* \brief Prune the quantile structure.
   *
   * \param to The maximum size of pruned quantile.  If the size of quantile
   * structure is already less than `to`, then no operation is performed.
   */
  void Prune(size_t to);
  /* \brief Merge another set of sketch.
   * \param that columns of other.
   */
  void Merge(Span<OffsetT const> that_columns_ptr,
             Span<SketchEntry const> that);

  /* \brief Merge quantiles from other GPU workers. */
  void AllReduce();
  /* \brief Create the final histogram cut values. */
  void MakeCuts(HistogramCuts* cuts);

  Span<SketchEntry const> Data() const {
    return {this->Current().data().get(), this->Current().size()};
  }
  HostDeviceVector<FeatureType> const& FeatureTypes() const { return feature_types_; }

  Span<OffsetT const> ColumnsPtr() const { return this->columns_ptr_.ConstDeviceSpan(); }

  SketchContainer(SketchContainer&&) = default;
  SketchContainer& operator=(SketchContainer&&) = default;

  SketchContainer(const SketchContainer&) = delete;
  SketchContainer& operator=(const SketchContainer&) = delete;

  /* \brief Removes all the duplicated elements in quantile structure. */
  template <typename KeyComp = thrust::equal_to<size_t>>
  size_t Unique(KeyComp key_comp = thrust::equal_to<size_t>{}) {
    timer_.Start(__func__);
    dh::safe_cuda(cudaSetDevice(device_));
    this->columns_ptr_.SetDevice(device_);
    Span<OffsetT> d_column_scan = this->columns_ptr_.DeviceSpan();
    CHECK_EQ(d_column_scan.size(), num_columns_ + 1);
    Span<SketchEntry> entries = dh::ToSpan(this->Current());
    HostDeviceVector<OffsetT> scan_out(d_column_scan.size());
    scan_out.SetDevice(device_);
    auto d_scan_out = scan_out.DeviceSpan();
    dh::XGBCachingDeviceAllocator<char> alloc;

    d_column_scan = this->columns_ptr_.DeviceSpan();
    size_t n_uniques = dh::SegmentedUnique(
        thrust::cuda::par(alloc), d_column_scan.data(),
        d_column_scan.data() + d_column_scan.size(), entries.data(),
        entries.data() + entries.size(), scan_out.DevicePointer(),
        entries.data(), detail::SketchUnique{}, key_comp);
    this->columns_ptr_.Copy(scan_out);
    CHECK(!this->columns_ptr_.HostCanRead());

    this->Current().resize(n_uniques);
    timer_.Stop(__func__);
    return n_uniques;
  }
};
}  // namespace xgboost::common

#endif  // XGBOOST_COMMON_QUANTILE_CUH_
