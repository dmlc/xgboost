/*!
 * Copyright 2017 XGBoost contributors
 */
#include "updater_gpu.cuh"
#include <xgboost/tree_updater.h>
#include <vector>
#include <utility>
#include <string>
#include "../../../src/common/random.h"
#include "../../../src/common/sync.h"
#include "../../../src/tree/param.h"
#include "exact/gpu_builder.cuh"
#include "gpu_hist_builder.cuh"

namespace xgboost {
namespace tree {

GPUMaker::GPUMaker() : builder(new exact::GPUBuilder()) {}

void GPUMaker::Init(
    const std::vector<std::pair<std::string, std::string>>& args) {
  param.InitAllowUnknown(args);
  builder->Init(param);
}

void GPUMaker::Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
                      const std::vector<RegTree*>& trees) {
  GradStats::CheckInfo(dmat->info());
  // rescale learning rate according to size of trees
  float lr = param.learning_rate;
  param.learning_rate = lr / trees.size();
  builder->UpdateParam(param);

  try {
    // build tree
    for (size_t i = 0; i < trees.size(); ++i) {
      builder->Update(gpair, dmat, trees[i]);
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
  }
  param.learning_rate = lr;
}

GPUHistMaker::GPUHistMaker() : builder(new GPUHistBuilder()) {}

void GPUHistMaker::Init(
    const std::vector<std::pair<std::string, std::string>>& args) {
  param.InitAllowUnknown(args);
  builder->Init(param);
}

void GPUHistMaker::Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
                          const std::vector<RegTree*>& trees) {
  GradStats::CheckInfo(dmat->info());
  // rescale learning rate according to size of trees
  float lr = param.learning_rate;
  param.learning_rate = lr / trees.size();
  builder->UpdateParam(param);
  // build tree
  try {
    for (size_t i = 0; i < trees.size(); ++i) {
      builder->Update(gpair, dmat, trees[i]);
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "GPU plugin exception: " << e.what() << std::endl;
  }
  param.learning_rate = lr;
}

bool GPUHistMaker::UpdatePredictionCache(const DMatrix* data,
                                         std::vector<bst_float>* out_preds) {
  return builder->UpdatePredictionCache(data, out_preds);
}

}  // namespace tree
}  // namespace xgboost
