/**
 * Copyright 2017-2026, XGBoost contributors
 */

/**
 * @file host_device_vector.h
 * @brief A device-and-host vector abstraction layer.
 *
 * Why HostDeviceVector?<br/>
 * With CUDA, one has to explicitly manage memory through 'cudaMemcpy' calls.
 * This wrapper class hides this management from the users, thereby making it
 * easy to integrate GPU/CPU usage under a single interface.
 *
 * Initialization/Allocation:<br/>
 * One can choose to initialize the vector on CPU or GPU during constructor.
 * (use the 'devices' argument) Or, can choose to use the 'Resize' method to
 * allocate/resize memory explicitly, and use the 'SetDevice' method
 * to specify the device.
 *
 * Accessing underlying data:<br/>
 * Use 'HostVector' method to explicitly query for the underlying std::vector.
 * If you need the raw device pointer, use the 'DevicePointer' method. For perf
 * implications of these calls, see below.
 *
 * Accessing underling data and their perf implications:<br/>
 * There are 4 scenarios to be considered here:
 * HostVector and data on CPU --> no problems, std::vector returned immediately
 * HostVector but data on GPU --> this causes a cudaMemcpy to be issued internally.
 *                        subsequent calls to HostVector, will NOT incur this penalty.
 *                        (assuming 'DevicePointer' is not called in between)
 * DevicePointer but data on CPU  --> this causes a cudaMemcpy to be issued internally.
 *                        subsequent calls to DevicePointer, will NOT incur this penalty.
 *                        (assuming 'HostVector' is not called in between)
 * DevicePointer and data on GPU  --> no problems, the device ptr
 *                        will be returned immediately.
 *
 * What if xgboost is compiled without CUDA?<br/>
 * In that case, there's a special implementation which always falls-back to
 * working with std::vector. This logic can be found in host_device_vector.cc
 *
 * Why not consider CUDA unified memory?<br/>
 * We did consider. However, it poses complications if we need to support both
 * compiling with and without CUDA toolkit. It was easier to have
 * 'HostDeviceVector' with a special-case implementation in host_device_vector.cc
 *
 * @note: Size and Devices methods are thread-safe.
 */

#ifndef XGBOOST_HOST_DEVICE_VECTOR_H_
#define XGBOOST_HOST_DEVICE_VECTOR_H_

#include <xgboost/context.h>  // for DeviceOrd
#include <xgboost/span.h>     // for Span

#include <initializer_list>
#include <type_traits>
#include <vector>

namespace xgboost {

#ifdef __CUDACC__
// Sets a function to call instead of cudaSetDevice();
// only added for testing
void SetCudaSetDeviceHandler(void (*handler)(int));
#endif  // __CUDACC__

template <typename T>
struct HostDeviceVectorImpl;

/*!
 * \brief Controls data access from the GPU.
 *
 * Since a `HostDeviceVector` can have data on both the host and device, access control needs to be
 * maintained to keep the data consistent.
 *
 * There are 3 scenarios supported:
 *   - Data is being manipulated on device. GPU has write access, host doesn't have access.
 *   - Data is read-only on both the host and device.
 *   - Data is being manipulated on the host. Host has write access, device doesn't have access.
 */
enum GPUAccess {
  kNone,
  kRead,
  // write implies read
  kWrite
};

template <typename T>
class HostDeviceVector {
  static_assert(std::is_standard_layout_v<T>, "HostDeviceVector admits only POD types");

 public:
  using value_type = T;  // NOLINT

 public:
  explicit HostDeviceVector(size_t size = 0, T v = T(), DeviceOrd device = DeviceOrd::CPU(),
                            Context const* ctx = nullptr);
  HostDeviceVector(std::initializer_list<T> init, DeviceOrd device = DeviceOrd::CPU(),
                   Context const* ctx = nullptr);
  explicit HostDeviceVector(const std::vector<T>& init, DeviceOrd device = DeviceOrd::CPU(),
                            Context const* ctx = nullptr);
  ~HostDeviceVector();

  HostDeviceVector(const HostDeviceVector<T>&) = delete;
  HostDeviceVector(HostDeviceVector<T>&&);

  HostDeviceVector<T>& operator=(const HostDeviceVector<T>&) = delete;
  HostDeviceVector<T>& operator=(HostDeviceVector<T>&&);

  [[nodiscard]] bool Empty() const { return Size() == 0; }
  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] std::size_t SizeBytes() const { return this->Size() * sizeof(T); }
  [[nodiscard]] DeviceOrd Device() const;
  common::Span<T> DeviceSpan(Context const* ctx = nullptr);
  common::Span<const T> ConstDeviceSpan(Context const* ctx = nullptr) const;
  common::Span<const T> DeviceSpan(Context const* ctx = nullptr) const {
    return ConstDeviceSpan(ctx);
  }
  T* DevicePointer(Context const* ctx = nullptr);
  const T* ConstDevicePointer(Context const* ctx = nullptr) const;
  const T* DevicePointer(Context const* ctx = nullptr) const { return ConstDevicePointer(ctx); }

  T* HostPointer(Context const* ctx = nullptr) { return HostVector(ctx).data(); }
  common::Span<T> HostSpan(Context const* ctx = nullptr) {
    return common::Span<T>{HostVector(ctx)};
  }
  common::Span<T const> HostSpan(Context const* ctx = nullptr) const {
    return common::Span<T const>{HostVector(ctx)};
  }
  common::Span<T const> ConstHostSpan(Context const* ctx = nullptr) const { return HostSpan(ctx); }
  const T* ConstHostPointer(Context const* ctx = nullptr) const {
    return ConstHostVector(ctx).data();
  }
  const T* HostPointer(Context const* ctx = nullptr) const { return ConstHostPointer(ctx); }

  void Fill(T v, Context const* ctx = nullptr);
  void Copy(const HostDeviceVector<T>& other, Context const* ctx = nullptr);
  void Copy(const std::vector<T>& other, Context const* ctx = nullptr);
  void Copy(std::initializer_list<T> other, Context const* ctx = nullptr);

  void Extend(const HostDeviceVector<T>& other, Context const* ctx = nullptr);

  std::vector<T>& HostVector(Context const* ctx = nullptr);
  const std::vector<T>& ConstHostVector(Context const* ctx = nullptr) const;
  const std::vector<T>& HostVector(Context const* ctx = nullptr) const {
    return ConstHostVector(ctx);
  }

  [[nodiscard]] bool HostCanRead() const;
  [[nodiscard]] bool HostCanWrite() const;
  [[nodiscard]] bool DeviceCanRead() const;
  [[nodiscard]] bool DeviceCanWrite() const;
  [[nodiscard]] GPUAccess DeviceAccess() const;

  void SetDevice(DeviceOrd device, Context const* ctx = nullptr) const;

  void Resize(std::size_t new_size);
  void Resize(Context const* ctx, std::size_t new_size);

  /** @brief Resize and initialize the data if the new size is larger than the old size. */
  void Resize(Context const* ctx, std::size_t new_size, T v);

 private:
  HostDeviceVectorImpl<T>* impl_;
};

}  // namespace xgboost

#endif  // XGBOOST_HOST_DEVICE_VECTOR_H_
