/*!
 * Copyright 2016 Rory mitchell
 */
#pragma once
#include <xgboost/tree_updater.h>
#include <vector>
#include "../../src/tree/param.h"

namespace xgboost {

namespace tree {

struct gpu_gpair;
struct GPUData;

class GPUBuilder {
 public:
  GPUBuilder();
  void Init(const TrainParam &param);
  ~GPUBuilder();

  void UpdateParam(const TrainParam &param) { this->param = param; }

  void Update(const std::vector<bst_gpair> &gpair, DMatrix *p_fmat,
              RegTree *p_tree);

  void UpdateNodeId(int level);

 private:
  void InitData(const std::vector<bst_gpair> &gpair, DMatrix &fmat,  // NOLINT
                const RegTree &tree);

  void Sort(int level);
  void InitFirstNode();
  void ColsampleTree();

  TrainParam param;
  GPUData *gpu_data;
  std::vector<int> feature_set_tree;
  std::vector<int> feature_set_level;

  int multiscan_levels =
      5;  // Number of levels before switching to sorting algorithm
};
}  // namespace tree
}  // namespace xgboost
