/**
 * Copyright 2014-2023, XGBoost Contributors
 * \file context.h
 */
#ifndef XGBOOST_CONTEXT_H_
#define XGBOOST_CONTEXT_H_

#include <xgboost/base.h>       // for bst_d_ordinal_t
#include <xgboost/logging.h>    // for CHECK_GE
#include <xgboost/parameter.h>  // for XGBoostParameter

#include <cstdint>      // for int16_t, int32_t, int64_t
#include <memory>       // for shared_ptr
#include <string>       // for string, to_string
#include <type_traits>  // for invoke_result_t, is_same_v, underlying_type_t

namespace xgboost {

struct CUDAContext;

// symbolic names
struct DeviceSym {
  static auto constexpr CPU() { return "cpu"; }
  static auto constexpr CUDA() { return "cuda"; }
};

/**
 * @brief A type for device ordinal. The type is packed into 32-bit for efficient use in
 *        viewing types like `linalg::TensorView`.
 */
struct DeviceOrd {
  enum Type : std::int16_t { kCPU = 0, kCUDA = 1 } device{kCPU};
  // CUDA device ordinal.
  bst_d_ordinal_t ordinal{-1};

  [[nodiscard]] bool IsCUDA() const { return device == kCUDA; }
  [[nodiscard]] bool IsCPU() const { return device == kCPU; }

  DeviceOrd() = default;
  constexpr DeviceOrd(Type type, bst_d_ordinal_t ord) : device{type}, ordinal{ord} {}

  DeviceOrd(DeviceOrd const& that) = default;
  DeviceOrd& operator=(DeviceOrd const& that) = default;
  DeviceOrd(DeviceOrd&& that) = default;
  DeviceOrd& operator=(DeviceOrd&& that) = default;

  /**
   * @brief Constructor for CPU.
   */
  [[nodiscard]] constexpr static auto CPU() { return DeviceOrd{kCPU, -1}; }
  /**
   * @brief Constructor for CUDA device.
   *
   * @param ordinal CUDA device ordinal.
   */
  [[nodiscard]] static auto CUDA(bst_d_ordinal_t ordinal) { return DeviceOrd{kCUDA, ordinal}; }

  [[nodiscard]] bool operator==(DeviceOrd const& that) const {
    return device == that.device && ordinal == that.ordinal;
  }
  [[nodiscard]] bool operator!=(DeviceOrd const& that) const { return !(*this == that); }
  /**
   * @brief Get a string representation of the device and the ordinal.
   */
  [[nodiscard]] std::string Name() const {
    switch (device) {
      case DeviceOrd::kCPU:
        return DeviceSym::CPU();
      case DeviceOrd::kCUDA:
        return DeviceSym::CUDA() + (':' + std::to_string(ordinal));
      default: {
        LOG(FATAL) << "Unknown device.";
        return "";
      }
    }
  }
};

static_assert(sizeof(DeviceOrd) == sizeof(std::int32_t));

/**
 * @brief Runtime context for XGBoost. Contains information like threads and device.
 */
struct Context : public XGBoostParameter<Context> {
 private:
  std::string device{DeviceSym::CPU()};  // NOLINT
  // The device object for the current context. We are in the middle of replacing the
  // `gpu_id` with this device field.
  DeviceOrd device_{DeviceOrd::CPU()};

 public:
  // Constant representing the device ID of CPU.
  static bst_d_ordinal_t constexpr kCpuId = -1;
  static bst_d_ordinal_t constexpr InvalidOrdinal() { return -2; }
  static std::int64_t constexpr kDefaultSeed = 0;

 public:
  Context();

  template <typename Container>
  Args UpdateAllowUnknown(Container const& kwargs) {
    auto args = XGBoostParameter<Context>::UpdateAllowUnknown(kwargs);
    this->SetDeviceOrdinal(kwargs);
    return args;
  }

  std::int32_t gpu_id{kCpuId};
  // The number of threads to use if OpenMP is enabled. If equals 0, use the system default.
  std::int32_t nthread{0};  // NOLINT
  // stored random seed
  std::int64_t seed{kDefaultSeed};
  // whether seed the PRNG each iteration
  bool seed_per_iteration{false};
  // fail when gpu_id is invalid
  bool fail_on_invalid_gpu_id{false};
  bool validate_parameters{false};

  /**
   * @brief Configure the parameter `gpu_id'.
   *
   * @param require_gpu Whether GPU is explicitly required by the user through other
   *                    configurations.
   */
  void ConfigureGpuId(bool require_gpu);
  /**
   * @brief Returns the automatically chosen number of threads based on the `nthread`
   *        parameter and the system settting.
   */
  [[nodiscard]] std::int32_t Threads() const;
  /**
   * @brief Is XGBoost running on CPU?
   */
  [[nodiscard]] bool IsCPU() const { return Device().IsCPU(); }
  /**
   * @brief Is XGBoost running on a CUDA device?
   */
  [[nodiscard]] bool IsCUDA() const { return Device().IsCUDA(); }
  /**
   * @brief Get the current device and ordinal.
   */
  [[nodiscard]] DeviceOrd Device() const { return device_; }
  /**
   * @brief Get the CUDA device ordinal. -1 if XGBoost is running on CPU.
   */
  [[nodiscard]] bst_d_ordinal_t Ordinal() const { return Device().ordinal; }
  /**
   * @brief Name of the current device.
   */
  [[nodiscard]] std::string DeviceName() const { return Device().Name(); }
  /**
   * @brief Get a CUDA device context for allocator and stream.
   */
  [[nodiscard]] CUDAContext const* CUDACtx() const;

  /**
   * @brief Make a CUDA context based on the current context.
   *
   * @param ordinal The CUDA device ordinal.
   */
  [[nodiscard]] Context MakeCUDA(bst_d_ordinal_t ordinal = 0) const {
    Context ctx = *this;
    return ctx.SetDevice(DeviceOrd::CUDA(ordinal));
  }
  /**
   * @brief Make a CPU context based on the current context.
   */
  [[nodiscard]] Context MakeCPU() const {
    Context ctx = *this;
    return ctx.SetDevice(DeviceOrd::CPU());
  }
  /**
   * @brief Call function based on the current device.
   */
  template <typename CPUFn, typename CUDAFn>
  decltype(auto) DispatchDevice(CPUFn&& cpu_fn, CUDAFn&& cuda_fn) const {
    static_assert(std::is_same_v<std::invoke_result_t<CPUFn>, std::invoke_result_t<CUDAFn>>);
    switch (this->Device().device) {
      case DeviceOrd::kCPU:
        return cpu_fn();
      case DeviceOrd::kCUDA:
        return cuda_fn();
      default:
        // Do not use the device name as this is likely an internal error, the name
        // wouldn't be valid.
        LOG(FATAL) << "Unknown device type:"
                   << static_cast<std::underlying_type_t<DeviceOrd::Type>>(this->Device().device);
        break;
    }
    return std::invoke_result_t<CPUFn>();
  }

  // declare parameters
  DMLC_DECLARE_PARAMETER(Context) {
    DMLC_DECLARE_FIELD(seed)
        .set_default(kDefaultSeed)
        .describe("Random number seed during training.");
    DMLC_DECLARE_ALIAS(seed, random_state);
    DMLC_DECLARE_FIELD(seed_per_iteration)
        .set_default(false)
        .describe("Seed PRNG determnisticly via iterator number.");
    DMLC_DECLARE_FIELD(device).set_default(DeviceSym::CPU()).describe("Device ordinal.");
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe("Number of threads to use.");
    DMLC_DECLARE_ALIAS(nthread, n_jobs);
    DMLC_DECLARE_FIELD(fail_on_invalid_gpu_id)
        .set_default(false)
        .describe("Fail with error when gpu_id is invalid.");
    DMLC_DECLARE_FIELD(validate_parameters)
        .set_default(false)
        .describe("Enable checking whether parameters are used or not.");
  }

 private:
  void SetDeviceOrdinal(Args const& kwargs);
  Context& SetDevice(DeviceOrd d) {
    this->device_ = d;
    this->gpu_id = d.ordinal;  // this can be removed once we move away from `gpu_id`.
    this->device = d.Name();
    return *this;
  }

  // mutable for lazy cuda context initialization. This avoids initializing CUDA at load.
  // shared_ptr is used instead of unique_ptr as with unique_ptr it's difficult to define
  // p_impl while trying to hide CUDA code from the host compiler.
  mutable std::shared_ptr<CUDAContext> cuctx_;
  // cached value for CFS CPU limit. (used in containerized env)
  std::int32_t cfs_cpu_count_;  // NOLINT
};
}  // namespace xgboost

#endif  // XGBOOST_CONTEXT_H_
