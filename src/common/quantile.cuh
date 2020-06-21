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
class DeviceQuantile {
 public:
  using SketchEntry = WQSummary<float, float>::Entry;

 private:
  dh::caching_device_vector<SketchEntry> data_;
  std::unique_ptr<dh::AllReducer> comm_;
  int32_t device_;
  size_t limit_size_;
  Monitor monitor;

  void SetMerge(std::vector<Span<SketchEntry const>> const& others);

 public:
  /* \brief Initialize the quantile structure.
   *
   * \param maxn   The maximum number of data points.
   * \param eps    Error rate.
   * \param device Device ordinal for data storage.
   */
  DeviceQuantile(size_t maxn, double eps, int32_t device) : device_{device} {
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
    return Span<SketchEntry const>(this->data_.data().get(), this->data_.size());
  }
};
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_QUANTILE_CUH_