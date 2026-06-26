/**
 * Copyright 2020-2025, XGBoost Contributors
 */
#ifndef XGBOOST_COMMON_QUANTILE_CUH_
#define XGBOOST_COMMON_QUANTILE_CUH_

#include <thrust/logical.h>  // for any_of

#include <algorithm>
#include <cstddef>     // for size_t
#include <functional>  // for equal_to

#include "categorical.h"
#include "common.h"          // for HumanMemUnit
#include "cuda_context.cuh"  // for CUDAContext
#include "cuda_rt_utils.h"   // for SetDevice
#include "device_helpers.cuh"
#include "error_msg.h"  // for InvalidMaxBin
#include "quantile.h"
#include "timer.h"
#include "xgboost/data.h"
#include "xgboost/span.h"

namespace xgboost::common {
class HistogramCuts;
using WQSketch = WQuantileSketch;
using SketchEntry = WQSketch::Entry;

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
  using OffsetT = bst_idx_t;
  static_assert(sizeof(OffsetT) == sizeof(size_t), "Wrong type for sketch element offset.");

 private:
  Monitor timer_;
  HostDeviceVector<FeatureType> feature_types_;
  bst_feature_t num_columns_;
  int32_t num_bins_;

  // The container is just a CSC matrix plus scratch storage for out-of-place transforms.
  dh::device_vector<SketchEntry> entries_;
  dh::device_vector<SketchEntry> entries_tmp_;
  dh::device_vector<SketchEntry> prune_buffer_;
  HostDeviceVector<OffsetT> columns_ptr_;
  HostDeviceVector<OffsetT> columns_ptr_tmp_;

  bool has_categorical_{false};
  std::size_t rows_seen_{0};

  void SetCurrentColumns(Span<OffsetT const> columns_ptr);
  void CommitScratch(std::size_t n_entries) {
    entries_.swap(entries_tmp_);
    columns_ptr_.Copy(columns_ptr_tmp_);
    entries_.resize(n_entries);
  }
  [[nodiscard]] std::size_t IntermediateCutsPerFeature() const {
    auto const eps = SketchEpsilon(num_bins_, std::max<std::size_t>(1, rows_seen_));
    return WQSketch::LimitSizeLevel(std::max<std::size_t>(1, rows_seen_), eps);
  }

  // Get the span of one column.
  Span<SketchEntry> Column(bst_feature_t i) {
    auto data = dh::ToSpan(this->entries_);
    auto h_ptr = columns_ptr_.ConstHostSpan();
    auto c = data.subspan(h_ptr[i], h_ptr[i+1] - h_ptr[i]);
    return c;
  }

 public:
  /* \breif GPU quantile structure, with sketch data for each columns.
   *
   * \param max_bin     Maximum number of bins per columns
   * \param num_columns Total number of columns in dataset.
   * \param device      GPU ID.
   */
  SketchContainer(HostDeviceVector<FeatureType> const& feature_types, bst_bin_t max_bin,
                  bst_feature_t num_columns, DeviceOrd device)
      : num_columns_{num_columns}, num_bins_{max_bin} {
    CHECK(device.IsCUDA());
    // Initialize Sketches for this dmatrix
    this->columns_ptr_.SetDevice(device);
    this->columns_ptr_.Resize(num_columns + 1, 0);
    this->columns_ptr_tmp_.SetDevice(device);
    this->columns_ptr_tmp_.Resize(num_columns + 1, 0);

    this->feature_types_.Resize(feature_types.Size());
    this->feature_types_.Copy(feature_types);
    // Pull to device.
    this->feature_types_.SetDevice(device);
    this->feature_types_.ConstDeviceSpan();
    this->feature_types_.ConstHostSpan();

    auto d_feature_types = feature_types_.ConstDeviceSpan();
    has_categorical_ =
        !d_feature_types.empty() &&
        thrust::any_of(dh::tbegin(d_feature_types), dh::tend(d_feature_types), common::IsCatOp{});
    CHECK_GE(max_bin, 2) << error::InvalidMaxBin();

    timer_.Init(__func__);
  }
  /**
   * @brief Calculate the memory cost of the container.
   */
  [[nodiscard]] std::size_t MemCapacityBytes() const {
    auto constexpr kE = sizeof(typename decltype(this->entries_)::value_type);
    auto n_bytes =
        (this->entries_.capacity() + this->entries_tmp_.capacity() + this->prune_buffer_.capacity()) *
        kE;
    n_bytes += (this->columns_ptr_.Size() + this->columns_ptr_tmp_.Size()) * sizeof(OffsetT);
    n_bytes += this->feature_types_.Size() * sizeof(FeatureType);

    return n_bytes;
  }
  [[nodiscard]] std::size_t MemCostBytes() const {
    auto constexpr kE = sizeof(typename decltype(this->entries_)::value_type);
    auto n_bytes =
        (this->entries_.size() + this->entries_tmp_.size() + this->prune_buffer_.size()) * kE;
    n_bytes += (this->columns_ptr_.Size() + this->columns_ptr_tmp_.Size()) * sizeof(OffsetT);
    n_bytes += this->feature_types_.Size() * sizeof(FeatureType);

    return n_bytes;
  }
  /* \brief Whether the predictor matrix contains categorical features. */
  bool HasCategorical() const { return has_categorical_; }
  /* \brief Accumulate weights of duplicated entries in input. */
  size_t ScanInput(Context const* ctx, Span<SketchEntry> entries, Span<OffsetT> d_columns_ptr_in);
  /* Fix rounding error and re-establish invariance.  The error is mostly generated by the
   * addition inside `RMinNext` and subtraction in `RMaxPrev`. */
  void FixError();

  /* \brief Push sorted entries.
   *
   * \param entries Sorted entries.
   * \param columns_ptr CSC pointer for entries.
   * \param weights (optional) data weights.
   */
  void Push(Context const* ctx, Span<Entry const> entries, Span<size_t> columns_ptr,
            bst_idx_t n_rows_in_batch, Span<float> weights = {});
  /**
   * @brief Prune the quantile structure.
   *
   * @param to The maximum size of pruned quantile.  If the size of quantile structure is
   *           already less than `to`, then no operation is performed.
   */
  void Prune(Context const* ctx, size_t to);
  /**
   * @brief Merge another set of sketch.
   *
   * @param that_columns_ptr Column pointer of the quantile summary being merged.
   * @param that Columns of the other quantile summary.
   */
  void Merge(Context const* ctx, Span<OffsetT const> that_columns_ptr,
             Span<SketchEntry const> that);
  /**
   * @brief Shrink the internal data structure to reduce memory usage. Can be used after
   *        prune.
   */
  void ShrinkToFit() {
    this->entries_.shrink_to_fit();
    this->entries_tmp_.clear();
    this->entries_tmp_.shrink_to_fit();
    LOG(DEBUG) << "Quantile memory cost:" << common::HumanMemUnit(this->MemCapacityBytes());
  }

  /* \brief Merge quantiles from other GPU workers. */
  void AllReduce(Context const* ctx, bool is_column_split);
  /* \brief Create the final histogram cut values. */
  [[nodiscard]] HistogramCuts MakeCuts(Context const* ctx, bool is_column_split);

  Span<SketchEntry const> Data() const { return {entries_.data().get(), entries_.size()}; }
  HostDeviceVector<FeatureType> const& FeatureTypes() const { return feature_types_; }
  Span<OffsetT const> ColumnsPtr() const { return columns_ptr_.ConstDeviceSpan(); }

  SketchContainer(SketchContainer&&) = default;
  SketchContainer& operator=(SketchContainer&&) = default;

  SketchContainer(const SketchContainer&) = delete;
  SketchContainer& operator=(const SketchContainer&) = delete;

};
}  // namespace xgboost::common

#endif  // XGBOOST_COMMON_QUANTILE_CUH_
