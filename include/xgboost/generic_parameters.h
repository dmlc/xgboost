/*!
 * Copyright 2014-2019 by Contributors
 * \file learner.cc
 */
#ifndef XGBOOST_GENERIC_PARAMETERS_H_
#define XGBOOST_GENERIC_PARAMETERS_H_

#include <xgboost/logging.h>
#include <xgboost/parameter.h>

#include <string>

namespace xgboost {
struct GenericParameter : public XGBoostParameter<GenericParameter> {
  // stored random seed
  int seed;
  // whether seed the PRNG each iteration
  bool seed_per_iteration;
  // number of threads to use if OpenMP is enabled
  // if equals 0, use system default
  int nthread;
  // primary device, -1 means no gpu.
  int gpu_id;

  void CheckDeprecated() {
    if (this->n_gpus != 0) {
      LOG(WARNING)
          << "\nn_gpus: "
          << this->__MANAGER__()->Find("n_gpus")->GetFieldInfo().description;
    }
  }

  // declare parameters
  DMLC_DECLARE_PARAMETER(GenericParameter) {
    DMLC_DECLARE_FIELD(seed).set_default(0).describe(
        "Random number seed during training.");
    DMLC_DECLARE_ALIAS(seed, random_state);
    DMLC_DECLARE_FIELD(seed_per_iteration)
        .set_default(false)
        .describe(
            "Seed PRNG determnisticly via iterator number, "
            "this option will be switched on automatically on distributed "
            "mode.");
    DMLC_DECLARE_FIELD(nthread).set_default(0)
        .set_lower_bound(-1)
        .describe("Number of threads to use.");
    DMLC_DECLARE_ALIAS(nthread, n_jobs);

    DMLC_DECLARE_FIELD(gpu_id)
        .set_default(-1)
        .set_lower_bound(-1)
        .describe("The primary GPU device ordinal.");
    DMLC_DECLARE_FIELD(n_gpus)
        .set_default(0)
        .set_range(0, 1)
        .describe(
"\n\tDeprecated. Single process multi-GPU training is no longer supported."
"\n\tPlease switch to distributed training with one process per GPU."
"\n\tThis can be done using Dask or Spark.  See documentation for details.");
  }

 private:
  // number of devices to use (deprecated).
  int n_gpus {0};
};
}  // namespace xgboost

#endif  // XGBOOST_GENERIC_PARAMETERS_H_
