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
/*\brief Internal data structure for storing quantile for each data feature. */
class DeviceQuantile {
 public:
  using SketchEntry = WQSummary<float, float>::Entry;

 private:
  dh::device_vector<SketchEntry> data_;
  dh::device_vector<SketchEntry> data_b_;
  bool current_buffer_ {true};

  std::unique_ptr<dh::AllReducer> comm_;
  int32_t device_;
  size_t limit_size_;
  Monitor monitor;
  cudaStream_t stream_ { nullptr };

  dh::device_vector<SketchEntry>& Current() {
    if (current_buffer_) {
      return data_;
    } else {
      return data_b_;
    }
  }
  dh::device_vector<SketchEntry>& Other() {
    if (!current_buffer_) {
      return data_;
    } else {
      return data_b_;
    }
  }
  dh::device_vector<SketchEntry> const& Current() const {
    return const_cast<DeviceQuantile*>(this)->Current();
  }
  dh::device_vector<SketchEntry> const& Other() const {
    return const_cast<DeviceQuantile*>(this)->Other();
  }

  void Alternate() {
    current_buffer_ = !current_buffer_;
  }

  void SetMerge(std::vector<Span<SketchEntry const>> const& others);

 public:
  /* \brief Initialize the quantile structure.
   *
   * \param maxn   The maximum number of data points.
   * \param eps    Error rate.
   * \param device Device ordinal for data storage.
   */
  DeviceQuantile(size_t maxn, double eps, int32_t device, cudaStream_t stream) :
      device_{device}, stream_{stream} {
    size_t level;
    WQuantileSketch<float, float>::LimitSizeLevel(maxn, eps, &limit_size_, &level);
    limit_size_ *= level;  // ON GPU we don't have streaming algorithm.
    monitor.Init(__func__);
  }
  /* \brief Merge a set of quantiles */
  void MakeFromOthers(std::vector<DeviceQuantile> const& others);
  /* \brief Prune the quantile structure.
   *
   * \param to The maximum size of pruned quantile.  If the size of quantile structure is
   *           already less than `to`, then no operation is performed.
   */
  void Prune(size_t to);
  /* \brief Async merge quantiles from other GPU workers. */
  void AllReduce();
  /* \brief Sync the operations performed by AllReduce. */
  void Synchronize();
  /* \brief Push a sorted sequence into quantile. */
  void PushSorted(common::Span<SketchEntry> entries);

  common::Span<SketchEntry const> Data() const {
    cudaStreamSynchronize(stream_);
    return Span<SketchEntry const>(this->Current().data().get(), this->Current().size());
  }
};

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
  // std::vector<DeviceQuantile> sketches_;  // NOLINT
  // std::vector<cudaStream_t> streams_;
  // static constexpr int kOmpNumColsParallelizeLimit = 1000;
  static constexpr float kFactor = 8;
  Monitor timer;

 private:
  size_t num_rows_;
  size_t num_columns_;
  int32_t num_bins_;
  size_t limit_size_;
  int32_t device_;

  dh::device_vector<SketchEntry> entries_a_;
  dh::device_vector<SketchEntry> entries_b_;
  bool current_buffer_ {true};

  HostDeviceVector<bst_feature_t> columns_ptr_;

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
  dh::device_vector<SketchEntry> const& Current() const {
    return const_cast<SketchContainer*>(this)->Current();
  }
  dh::device_vector<SketchEntry> const& Other() const {
    return const_cast<SketchContainer*>(this)->Other();
  }
  void Alternate() {
    current_buffer_ = !current_buffer_;
  }

  Span<SketchEntry> Column(size_t i) {
    auto data = dh::ToSpan(this->Current());
    auto h_ptr = columns_ptr_.ConstHostSpan();
    auto c = data.subspan(h_ptr[i], h_ptr[i+1] - h_ptr[i]);
    return c;
  }

 public:
  SketchContainer(int max_bin, size_t num_columns, size_t num_rows, int32_t device = 0) :
      num_rows_{num_rows}, num_columns_{num_columns}, num_bins_{max_bin}, device_{device} {
    // Initialize Sketches for this dmatrix
    auto eps = 1.0 / (WQSketch::kFactor * max_bin);
    size_t level;
    WQuantileSketch<float, float>::LimitSizeLevel(num_rows, eps, &limit_size_, &level);
    limit_size_ *= level;  // ON GPU we don't have streaming algorithm.
    timer.Init(__func__);
  }
  size_t Unique();

  /**
   * \brief Pushes cuts to the sketches.
   *
   * \param entries_per_column  The entries per column.
   * \param entries             Vector of cuts from all columns, length
   * entries_per_column * num_columns. \param column_scan         Exclusive scan
   * of column sizes. Used to detect cases where there are fewer entries than we
   * have storage for.
   */
  void Push(size_t entries_per_column,
            const common::Span<SketchEntry>& entries,
            const thrust::host_vector<size_t>& column_scan);
  void Push(common::Span<size_t const> cuts_ptr,
            const common::Span<SketchEntry>& entries);

  void Prune(size_t to);

  void Merge(std::vector< Span<SketchEntry> >other);

  void MakeCuts(HistogramCuts* cuts);

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