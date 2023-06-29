/**
 * Copyright 2014-2023, XGBoost Contributors
 * \file context.h
 */
#ifndef XGBOOST_CONTEXT_H_
#define XGBOOST_CONTEXT_H_

#include <xgboost/base.h>  // for bst_d_ordinal_t
#include <xgboost/logging.h>
#include <xgboost/parameter.h>

#include <memory>  // std::shared_ptr
#include <string>

namespace xgboost {

struct CUDAContext;

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
   * @param bst_d_ordinal_t CUDA device ordinal.
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
        return "CPU";
      case DeviceOrd::kCUDA:
        return "CUDA:" + std::to_string(ordinal);
      default: {
        LOG(FATAL) << "Unknown device.";
        return "";
      }
    }
  }
};

static_assert(sizeof(DeviceOrd) == sizeof(std::int32_t));

struct Context : public XGBoostParameter<Context> {
 public:
  // Constant representing the device ID of CPU.
  static std::int32_t constexpr kCpuId = -1;
  static std::int64_t constexpr kDefaultSeed = 0;

 public:
  Context();

  // stored random seed
  std::int64_t seed{kDefaultSeed};
  // whether seed the PRNG each iteration
  bool seed_per_iteration{false};
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  std::int32_t nthread{0};
  // primary device, -1 means no gpu.
  std::int32_t gpu_id{kCpuId};
  // fail when gpu_id is invalid
  bool fail_on_invalid_gpu_id{false};
  bool validate_parameters{false};
  /**
   * @brief Configure the parameter `gpu_id'.
   *
   * @param require_gpu  Whether GPU is explicitly required from user.
   */
  void ConfigureGpuId(bool require_gpu);
  /*!
   * Return automatically chosen threads.
   */
  std::int32_t Threads() const;

  bool IsCPU() const { return gpu_id == kCpuId; }
  bool IsCUDA() const { return !IsCPU(); }

  DeviceOrd Device() const {
    return IsCPU() ? DeviceOrd::CPU() : DeviceOrd::CUDA(static_cast<bst_d_ordinal_t>(gpu_id));
  }
  /**
   * @brief Name of the current device.
   */
  std::string DeviceName() const { return Device().Name(); }

  CUDAContext const* CUDACtx() const;
  // Make a CUDA context based on the current context.
  Context MakeCUDA(std::int32_t device = 0) const {
    Context ctx = *this;
    ctx.gpu_id = device;
    return ctx;
  }
  Context MakeCPU() const {
    Context ctx = *this;
    ctx.gpu_id = kCpuId;
    return ctx;
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
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe("Number of threads to use.");
    DMLC_DECLARE_ALIAS(nthread, n_jobs);

    DMLC_DECLARE_FIELD(gpu_id).set_default(-1).set_lower_bound(-1).describe(
        "The primary GPU device ordinal.");
    DMLC_DECLARE_FIELD(fail_on_invalid_gpu_id)
        .set_default(false)
        .describe("Fail with error when gpu_id is invalid.");
    DMLC_DECLARE_FIELD(validate_parameters)
        .set_default(false)
        .describe("Enable checking whether parameters are used or not.");
  }

 private:
  // mutable for lazy initialization for cuda context to avoid initializing CUDA at load.
  // shared_ptr is used instead of unique_ptr as with unique_ptr it's difficult to define p_impl
  // while trying to hide CUDA code from host compiler.
  mutable std::shared_ptr<CUDAContext> cuctx_;
  // cached value for CFS CPU limit. (used in containerized env)
  std::int32_t cfs_cpu_count_;  // NOLINT
};
}  // namespace xgboost

#endif  // XGBOOST_CONTEXT_H_
