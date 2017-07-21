
/*!
 * Copyright 2017 XGBoost contributors
 */
#pragma once
#include <xgboost/tree_updater.h>
#include <memory>
#include "../../../src/tree/param.h"

namespace xgboost {
namespace tree {

// Forward declare builder classes
class GPUHistBuilder;
namespace exact {
template <typename node_id_t>
class GPUBuilder;
}

class GPUMaker : public TreeUpdater {
 protected:
  TrainParam param;
  std::unique_ptr<exact::GPUBuilder<int16_t>> builder;

 public:
  GPUMaker();
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override;
  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees);
};

class GPUHistMaker : public TreeUpdater {
 public:
  GPUHistMaker();
  void Init(
      const std::vector<std::pair<std::string, std::string>>& args) override;
  void Update(const std::vector<bst_gpair>& gpair, DMatrix* dmat,
              const std::vector<RegTree*>& trees) override;
  bool UpdatePredictionCache(const DMatrix* data,
                             std::vector<bst_float>* out_preds) override;

 protected:
  TrainParam param;
  std::unique_ptr<GPUHistBuilder> builder;
};
}  // namespace tree
}  // namespace xgboost
