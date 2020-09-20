#ifndef XGBOOST_COMMON_QUANTILE_CUH_
#define XGBOOST_COMMON_QUANTILE_CUH_

#include <memory>

#include "xgboost/span.h"
#include "xgboost/data.h"
#include "device_helpers.cuh"
#include "quantile.h"
#include "timer.h"

namespace xgboost {
namespace common {

class HistogramCuts;
using WQSketch = WQuantileSketch<bst_float, bst_float>;
using SketchEntry = WQSketch::Entry;

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
  std::unique_ptr<dh::AllReducer> reducer_;
  HostDeviceVector<FeatureType> feature_types_;
  bst_row_t num_rows_;
  bst_feature_t num_columns_;
  int32_t num_bins_;
  int32_t device_;

  // Double buffer as neither prune nor merge can be performed inplace.
  dh::caching_device_vector<SketchEntry> entries_a_;
  dh::caching_device_vector<SketchEntry> entries_b_;
  bool current_buffer_ {true};
  // The container is just a CSC matrix.
  HostDeviceVector<OffsetT> columns_ptr_;

  dh::caching_device_vector<SketchEntry>& Current() {
    if (current_buffer_) {
      return entries_a_;
    } else {
      return entries_b_;
    }
  }
  dh::caching_device_vector<SketchEntry>& Other() {
    if (!current_buffer_) {
      return entries_a_;
    } else {
      return entries_b_;
    }
  }
  dh::caching_device_vector<SketchEntry> const& Current() const {
    return const_cast<SketchContainer*>(this)->Current();
  }
  dh::caching_device_vector<SketchEntry> const& Other() const {
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
   * \param max_bin     Maximum number of bins per columns
   * \param num_columns Total number of columns in dataset.
   * \param num_rows    Total number of rows in known dataset (typically the rows in current worker).
   * \param device      GPU ID.
   */
  SketchContainer(HostDeviceVector<FeatureType> const& feature_types,
                  int32_t max_bin,
                  bst_feature_t num_columns, bst_row_t num_rows,
                  int32_t device)
      : num_rows_{num_rows},
        num_columns_{num_columns}, num_bins_{max_bin}, device_{device} {
    // Initialize Sketches for this dmatrix
    this->columns_ptr_.SetDevice(device_);
    this->columns_ptr_.Resize(num_columns + 1);
    CHECK_GE(device, 0);
    this->feature_types_.Resize(feature_types.Size());
    this->feature_types_.Copy(feature_types);
    this->feature_types_.SetDevice(device);
    // pull to device early
    this->feature_types_.ConstDeviceSpan();
    this->feature_types_.ConstHostSpan();
    timer_.Init(__func__);
  }
  /* \brief Return GPU ID for this container. */
  int32_t DeviceIdx() const { return device_; }
  /* \brief Removes all the duplicated elements in quantile structure. */
  size_t Unique();
  /* Fix rounding error and re-establish invariance.  The error is mostly generated by the
   * addition inside `RMinNext` and subtraction in `RMaxPrev`. */
  void FixError();

  /* \brief Push sorted entries.
   *
   * \param entries Sorted entries.
   * \param columns_ptr CSC pointer for entries.
   * \param cuts_ptr CSC pointer for cuts.
   * \param total_cuts Total number of cuts, equal to the back of cuts_ptr.
   * \param weights (optional) data weights.
   */
  void Push(Span<Entry const> entries, Span<size_t> columns_ptr,
            common::Span<OffsetT const> cuts_ptr, size_t total_cuts,
            Span<float> weights = {});
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
};

namespace detail {
struct SketchUnique {
  XGBOOST_DEVICE bool operator()(SketchEntry const& a, SketchEntry const& b) const {
    return a.value - b.value == 0;
  }
};
}  // namespace detail
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_QUANTILE_CUH_
