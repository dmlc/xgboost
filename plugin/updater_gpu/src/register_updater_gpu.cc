/*!
 * Copyright 2017 XGBoost contributors
 */
#include <xgboost/tree_updater.h>
#include "updater_gpu.cuh"

namespace xgboost {
namespace tree {
DMLC_REGISTRY_FILE_TAG(updater_gpumaker);


XGBOOST_REGISTER_TREE_UPDATER(GPUMaker, "grow_gpu")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUMaker(); });

XGBOOST_REGISTER_TREE_UPDATER(GPUHistMaker, "grow_gpu_hist")
    .describe("Grow tree with GPU.")
    .set_body([]() { return new GPUHistMaker(); });
}  // namespace tree
}  // namespace xgboost
