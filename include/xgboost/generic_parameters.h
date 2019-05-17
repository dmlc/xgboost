/*!
 * Copyright 2014-2019 by Contributors
 * \file learner.cc
 */
#ifndef XGBOOST_GENERIC_PARAMETERS_H_
#define XGBOOST_GENERIC_PARAMETERS_H_

#include <dmlc/parameter.h>
#include <xgboost/enum_class_param.h>
#include <xgboost/gpuset.h>

namespace xgboost {
enum class TreeMethod : int {
  kAuto = 0, kApprox = 1, kExact = 2, kHist = 3,
  kGPUExact = 4, kGPUHist = 5
};

enum class DataSplitMode : int {
  kAuto = 0, kCol = 1, kRow = 2
};
}  // namespace xgboost

DECLARE_FIELD_ENUM_CLASS(xgboost::TreeMethod);
DECLARE_FIELD_ENUM_CLASS(xgboost::DataSplitMode);

namespace xgboost {
struct LearnerTrainParam : public dmlc::Parameter<LearnerTrainParam> {
  // stored random seed
  int seed;
  // whether seed the PRNG each iteration
  bool seed_per_iteration;
  // data split mode, can be row, col, or none.
  DataSplitMode dsplit;
  // tree construction method
  TreeMethod tree_method;
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  int nthread;
  // flag to disable default metric
  int disable_default_eval_metric;
  // primary device.
  int gpu_id;
  // number of devices to use, -1 implies using all available devices.
  int n_gpus;

  // declare parameters
  DMLC_DECLARE_PARAMETER(LearnerTrainParam) {
    DMLC_DECLARE_FIELD(seed).set_default(0).describe(
        "Random number seed during training.");
    DMLC_DECLARE_FIELD(seed_per_iteration)
        .set_default(false)
        .describe(
            "Seed PRNG determnisticly via iterator number, "
            "this option will be switched on automatically on distributed "
            "mode.");
    DMLC_DECLARE_FIELD(dsplit)
        .set_default(DataSplitMode::kAuto)
        .add_enum("auto", DataSplitMode::kAuto)
        .add_enum("col", DataSplitMode::kCol)
        .add_enum("row", DataSplitMode::kRow)
        .describe("Data split mode for distributed training.");
    DMLC_DECLARE_FIELD(tree_method)
        .set_default(TreeMethod::kAuto)
        .add_enum("auto", TreeMethod::kAuto)
        .add_enum("approx", TreeMethod::kApprox)
        .add_enum("exact", TreeMethod::kExact)
        .add_enum("hist", TreeMethod::kHist)
        .add_enum("gpu_exact", TreeMethod::kGPUExact)
        .add_enum("gpu_hist", TreeMethod::kGPUHist)
        .describe("Choice of tree construction method.");
    DMLC_DECLARE_FIELD(nthread).set_default(0).describe(
        "Number of threads to use.");
    DMLC_DECLARE_FIELD(disable_default_eval_metric)
        .set_default(0)
        .describe("flag to disable default metric. Set to >0 to disable");
    DMLC_DECLARE_FIELD(gpu_id)
        .set_default(0)
        .describe("The primary GPU device ordinal.");
    DMLC_DECLARE_FIELD(n_gpus)
        .set_default(0)
        .set_lower_bound(-1)
        .describe("Number of GPUs to use for multi-gpu algorithms.");
  }

  GPUSet GetGPUSet() const {
    return GPUSet::All(gpu_id, n_gpus);
  }
};
}  // namespace xgboost

#endif  // XGBOOST_GENERIC_PARAMETERS_H_
