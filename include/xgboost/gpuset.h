#ifndef XGBOOST_GPUSET_H_
#define XGBOOST_GPUSET_H_

#include <xgboost/range.h>
#include <xgboost/logging.h>
#include <limits>

namespace xgboost {

struct AllVisibleImpl {
  static int AllVisible();
};
/* \brief set of devices across which HostDeviceVector can be distributed.
 *
 * Currently implemented as a range, but can be changed later to something else,
 *   e.g. a bitset
 */
class GPUSet {
 public:
  using GpuIdType = int;
  static constexpr GpuIdType kAll = -1;

  explicit GPUSet(int start = 0, int ndevices = 0)
      : devices_(start, start + ndevices) {}

  static GPUSet Empty() { return GPUSet(); }

  static GPUSet Range(GpuIdType start, GpuIdType n_gpus) {
    return n_gpus <= 0 ? Empty() : GPUSet{start, n_gpus};
  }
  /*! \brief n_gpus and num_rows both are upper bounds. */
  static GPUSet All(GpuIdType gpu_id, GpuIdType n_gpus,
                    GpuIdType num_rows = std::numeric_limits<GpuIdType>::max());

  static GPUSet AllVisible() {
    GpuIdType n =  AllVisibleImpl::AllVisible();
    return Range(0, n);
  }

  size_t Size() const {
    GpuIdType size = *devices_.end() - *devices_.begin();
    GpuIdType res = size < 0 ? 0 : size;
    return static_cast<size_t>(res);
  }

  /*
   * By default, we have two configurations of identifying device, one
   * is the device id obtained from `cudaGetDevice'.  But we sometimes
   * store objects that allocated one for each device in a list, which
   * requires a zero-based index.
   *
   * Hence, `DeviceId' converts a zero-based index to actual device id,
   * `Index' converts a device id to a zero-based index.
   */
  GpuIdType DeviceId(size_t index) const {
    GpuIdType result = *devices_.begin() + static_cast<GpuIdType>(index);
    CHECK(Contains(result)) << "\nDevice " << result << " is not in GPUSet."
                            << "\nIndex: " << index
                            << "\nGPUSet: (" << *begin() << ", " << *end() << ")"
                            << std::endl;
    return result;
  }
  size_t Index(GpuIdType device) const {
    CHECK(Contains(device)) << "\nDevice " << device << " is not in GPUSet."
                            << "\nGPUSet: (" << *begin() << ", " << *end() << ")"
                            << std::endl;
    size_t result = static_cast<size_t>(device - *devices_.begin());
    return result;
  }

  bool IsEmpty() const { return Size() == 0; }

  bool Contains(GpuIdType device) const {
    return *devices_.begin() <= device && device < *devices_.end();
  }

  common::Range::Iterator begin() const { return devices_.begin(); }  // NOLINT
  common::Range::Iterator end() const { return devices_.end(); }      // NOLINT

  friend bool operator==(const GPUSet& lhs, const GPUSet& rhs) {
    return lhs.devices_ == rhs.devices_;
  }
  friend bool operator!=(const GPUSet& lhs, const GPUSet& rhs) {
    return !(lhs == rhs);
  }

 private:
  common::Range devices_;
};

}  // namespace xgboost

#endif  // XGBOOST_GPUSET_H_
