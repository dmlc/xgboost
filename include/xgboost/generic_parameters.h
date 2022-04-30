/*!
 * Copyright 2014-2019 by Contributors
 * \file generic_parameters.h
 */
#ifndef XGBOOST_GENERIC_PARAMETERS_H_
#define XGBOOST_GENERIC_PARAMETERS_H_

#include <xgboost/logging.h>
#include <xgboost/parameter.h>

#include <string>

namespace xgboost {

struct GenericParameter : public XGBoostParameter<GenericParameter> {
 private:
  // cached value for CFS CPU limit. (used in containerized env)
  int32_t cfs_cpu_count_;  // NOLINT

 public:
  // Constant representing the device ID of CPU.
  static int32_t constexpr kCpuId = -1;
  static int64_t constexpr kDefaultSeed = 0;

 public:
  GenericParameter();

  // stored random seed
  int64_t seed { kDefaultSeed };
  // whether seed the PRNG each iteration
  bool seed_per_iteration{false};
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  int nthread{0};
  // primary device, -1 means no gpu.
  int gpu_id{kCpuId};
  // fail when gpu_id is invalid
  bool fail_on_invalid_gpu_id {false};
  bool validate_parameters {false};

  /*!
   * \brief Configure the parameter `gpu_id'.
   *
   * \param require_gpu  Whether GPU is explicitly required from user.
   */
  void ConfigureGpuId(bool require_gpu);
  /*!
   * Return automatically chosen threads.
   */
  int32_t Threads() const;

  bool IsCPU() const { return gpu_id == kCpuId; }

  // declare parameters
  DMLC_DECLARE_PARAMETER(GenericParameter) {
    DMLC_DECLARE_FIELD(seed).set_default(kDefaultSeed).describe(
        "Random number seed during training.");
    DMLC_DECLARE_ALIAS(seed, random_state);
    DMLC_DECLARE_FIELD(seed_per_iteration)
        .set_default(false)
        .describe("Seed PRNG determnisticly via iterator number.");
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe(
        "Number of threads to use.");
    DMLC_DECLARE_ALIAS(nthread, n_jobs);

    DMLC_DECLARE_FIELD(gpu_id)
        .set_default(-1)
        .set_lower_bound(-1)
        .describe("The primary GPU device ordinal.");
    DMLC_DECLARE_FIELD(fail_on_invalid_gpu_id)
        .set_default(false)
        .describe("Fail with error when gpu_id is invalid.");
    DMLC_DECLARE_FIELD(validate_parameters)
        .set_default(false)
        .describe("Enable checking whether parameters are used or not.");
  }
};

using Context = GenericParameter;
}  // namespace xgboost

#endif  // XGBOOST_GENERIC_PARAMETERS_H_
