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

 private:
  void InitData(const std::vector<bst_gpair> &gpair, DMatrix &fmat, // NOLINT
                const RegTree &tree);

  void UpdateNodeId(int level);
  void Sort(int level);
  void InitFirstNode();
  void CopyTree(RegTree &tree); // NOLINT

  TrainParam param;
  GPUData *gpu_data;

  // Keep host copies of these arrays as the device versions change between
  // boosting iterations
  std::vector<float> fvalues;
  std::vector<bst_uint> instance_id;

  int multiscan_levels =
      5;  // Number of levels before switching to sorting algorithm
};
}  // namespace tree
}  // namespace xgboost
