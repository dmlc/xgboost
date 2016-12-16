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

  void Update(const std::vector<bst_gpair> &gpair, DMatrix *p_fmat,
              RegTree *p_tree);

  void UpdateNodeId(int level);
 private:
  void InitData(const std::vector<bst_gpair> &gpair, DMatrix &fmat, // NOLINT
                const RegTree &tree);

  float GetSubsamplingRate(MetaInfo info);
  void Sort(int level);
  void InitFirstNode();
  void CopyTree(RegTree &tree); // NOLINT

  TrainParam param;
  GPUData *gpu_data;

  int multiscan_levels =
     5;  // Number of levels before switching to sorting algorithm
};
}  // namespace tree
}  // namespace xgboost
