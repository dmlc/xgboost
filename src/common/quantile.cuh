#ifndef XGBOOST_COMMON_QUANTILE_CUH_
#define XGBOOST_COMMON_QUANTILE_CUH_

#include <xgboost/span.h>
#include "device_helpers.cuh"
#include "quantile.h"

namespace xgboost {
namespace common {
class DeviceQuantile {
 public:
  using SketchEntry = WQSummary<float, float>::Entry;

 private:
  dh::caching_device_vector<SketchEntry> data_;
  dh::AllReducer comm_;
  int32_t device_;

  void SetMerge(std::vector<Span<SketchEntry const>> const& others);

 public:
  void MakeFromSorted(Span<SketchEntry> entries, int32_t device);
  void MakeFromOthers(std::vector<DeviceQuantile> const& others);
  void Prune(size_t to);
  void AllReduce();

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