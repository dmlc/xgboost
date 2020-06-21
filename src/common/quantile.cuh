#ifndef XGBOOST_COMMON_QUANTILE_CUH_
#define XGBOOST_COMMON_QUANTILE_CUH_

#include <memory>

#include "xgboost/span.h"
#include "device_helpers.cuh"
#include "quantile.h"

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

  void SetMerge(std::vector<Span<SketchEntry const>> const& others);

 public:
  DeviceQuantile(size_t maxn, double eps, int32_t device) : device_{device} {
    size_t level;
    WQuantileSketch<float, float>::LimitSizeLevel(maxn, eps, &limit_size_, &level);
    std::cout << "maxn: " << maxn << ", "
              << "limit_size_: " << limit_size_ << ", "
              << "level: " << level << std::endl;
    limit_size_ *= level;
  }

  void MakeFromSorted(Span<SketchEntry> entries, int32_t device);
  void MakeFromOthers(std::vector<DeviceQuantile> const& others);
  void Prune(size_t to);
  void AllReduce();
  void PushSorted(common::Span<SketchEntry> entries);
  void MakeCuts(size_t max_rows, int max_bin, HistogramCuts* cuts);

  common::Span<SketchEntry const> Data() const {
    return Span<SketchEntry const>(this->data_.data().get(), this->data_.size());
  }
  common::Span<SketchEntry> Data() {
    return dh::ToSpan(this->data_);
  }
};
}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_QUANTILE_CUH_