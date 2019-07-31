/*!
 * Copyright 2014-2019 by Contributors
 * \file learner.cc
 */
#ifndef XGBOOST_GENERIC_PARAMETERS_H_
#define XGBOOST_GENERIC_PARAMETERS_H_

#include <dmlc/parameter.h>
#include <xgboost/enum_class_param.h>

#include <string>

namespace xgboost {
struct GenericParameter : public dmlc::Parameter<GenericParameter> {
  // stored random seed
  int seed;
  // whether seed the PRNG each iteration
  bool seed_per_iteration;
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  int nthread;
  // primary device.
  int gpu_id;
  // number of devices to use, -1 implies using all available devices.
  int n_gpus;
  // If external memory is enabled for training/prediction
  bool external_memory;
  // Compute metrics and the base prediction on CPU
  bool transform_on_cpu;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GenericParameter) {
    DMLC_DECLARE_FIELD(seed).set_default(0).describe(
        "Random number seed during training.");
    DMLC_DECLARE_FIELD(seed_per_iteration)
        .set_default(false)
        .describe(
            "Seed PRNG determnisticly via iterator number, "
            "this option will be switched on automatically on distributed "
            "mode.");
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe(
        "Number of threads to use.");
    DMLC_DECLARE_FIELD(gpu_id)
        .set_default(0)
        .describe("The primary GPU device ordinal.");
    DMLC_DECLARE_FIELD(n_gpus)
        .set_default(0)
        .set_lower_bound(-1)
        .describe("Deprecated, please use distributed training with one "
                  "process per GPU. "
                  "Number of GPUs to use for multi-gpu algorithms.");
    DMLC_DECLARE_FIELD(external_memory)
        .set_default(false)
        .describe("If external memory is used for training and/or prediction");
    DMLC_DECLARE_FIELD(transform_on_cpu)
        .set_default(false)
        .describe("Compute metrics and base prediction on CPU. "
                  "This may be useful to save GPU memory at the expense of training time");
  }
};
}  // namespace xgboost

#endif  // XGBOOST_GENERIC_PARAMETERS_H_
