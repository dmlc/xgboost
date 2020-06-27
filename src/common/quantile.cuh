#ifndef XGBOOST_COMMON_QUANTILE_CUH_
#define XGBOOST_COMMON_QUANTILE_CUH_

#include <memory>

#include "xgboost/span.h"
#include "device_helpers.cuh"
#include "quantile.h"
#include "timer.h"

namespace xgboost {
namespace common {

class HistogramCuts;
using WQSketch = WQuantileSketch<bst_float, bst_float>;
using SketchEntry = WQSketch::Entry;

/*!
 * \brief A container that holds the device sketches across all
 *  sparse page batches which are distributed to different devices.
 *  As sketches are aggregated by column, the mutex guards
 *  multiple devices pushing sketch summary for the same column
 *  across distinct rows.
 */
struct SketchContainer {
  static constexpr float kFactor = 8;
  Monitor timer;

 private:
  std::unique_ptr<dh::AllReducer> reducer_;
  size_t num_rows_;
  size_t num_columns_;
  int32_t num_bins_;
  size_t limit_size_;
  int32_t device_;

  // Double buffer as neither prune nor merge can be performed inplace.
  dh::caching_device_vector<SketchEntry> entries_a_;
  dh::caching_device_vector<SketchEntry> entries_b_;
  bool current_buffer_ {true};
  // The container is just a CSC matrix.
  HostDeviceVector<bst_feature_t> columns_ptr_;

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
  Span<SketchEntry> Column(size_t i) {
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
  SketchContainer(int max_bin, size_t num_columns, size_t num_rows, int32_t device) :
      num_rows_{num_rows}, num_columns_{num_columns}, num_bins_{max_bin}, device_{device} {
    // Initialize Sketches for this dmatrix
    auto eps = 1.0 / (WQSketch::kFactor * max_bin);
    size_t level;
    WQuantileSketch<float, float>::LimitSizeLevel(num_rows, eps, &limit_size_, &level);
    this->columns_ptr_.SetDevice(device_);
    this->columns_ptr_.Resize(num_columns + 1);
    limit_size_ *= level;  // ON GPU we don't have streaming algorithm.
    timer.Init(__func__);
  }
  /* \brief Removes all the duplicated elements in quantile structure. */
  size_t Unique();

  /* \brief Push a CSC structured cut matrix. */
  void Push(common::Span<size_t const> cuts_ptr,
            dh::caching_device_vector<SketchEntry>* entries);
  /* \brief Prune the quantile structure.
   *
   * \param to The maximum size of pruned quantile.  If the size of quantile structure is
   *           already less than `to`, then no operation is performed.
   */
  void Prune(size_t to);
  /* \brief Merge another set of sketch.
   * \param other columns of other.
   */
  void Merge(std::vector< Span<SketchEntry> >other);
  void Merge(Span<size_t const> that_columns_ptr,
             Span<SketchEntry const> other);

  /* \brief Merge quantiles from other GPU workers. */
  void AllReduce();
  /* \brief Create the final histogram cut values. */
  void MakeCuts(HistogramCuts* cuts);

  Span<SketchEntry const> Data() const {
    return {this->Current().data().get(), this->Current().size()};
  }

  // Prevent copying/assigning/moving this as its internals can't be
  // assigned/copied/moved
  SketchContainer(const SketchContainer&) = delete;
  SketchContainer(const SketchContainer&&) = delete;
  SketchContainer& operator=(const SketchContainer&) = delete;
  SketchContainer& operator=(const SketchContainer&&) = delete;
};
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_QUANTILE_CUH_