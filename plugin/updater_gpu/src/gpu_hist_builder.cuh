/*!
 * Copyright 2016 Rory mitchell
 */
#pragma once
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <xgboost/tree_updater.h>
#include <cub/util_type.cuh>  // Need key value pair definition
#include <vector>
#include "../../src/common/hist_util.h"
#include "../../src/tree/param.h"
#include "device_helpers.cuh"
#include "types.cuh"

namespace xgboost {

namespace tree {

struct DeviceGMat {
  dh::dvec<int> gidx;
  dh::dvec<int> ridx;
  void Init(const common::GHistIndexMatrix &gmat);
};

struct HistBuilder {
  gpu_gpair *d_hist;
  int n_bins;
  __host__ __device__ HistBuilder(gpu_gpair *ptr, int n_bins);
  __device__ void Add(gpu_gpair gpair, int gidx, int nidx) const;
  __device__ gpu_gpair Get(int gidx, int nidx) const;
};

struct DeviceHist {
  int n_bins;
  dh::dvec<gpu_gpair> hist;

  void Init(int max_depth);

  void Reset();

  HistBuilder GetBuilder();

  gpu_gpair *GetLevelPtr(int depth);

  int LevelSize(int depth);
};

class GPUHistBuilder {
 public:
  GPUHistBuilder();
  ~GPUHistBuilder();
  void Init(const TrainParam &param);

  void UpdateParam(const TrainParam &param) {
    this->param = param;
    this->gpu_param = GPUTrainingParam(param.min_child_weight, param.reg_lambda,
                                       param.reg_alpha, param.max_delta_step);
  }

  void InitData(const std::vector<bst_gpair> &gpair, DMatrix &fmat,  // NOLINT
                const RegTree &tree);
  void Update(const std::vector<bst_gpair> &gpair, DMatrix *p_fmat,
              RegTree *p_tree);
  void BuildHist(int depth);
  void FindSplit(int depth);
  void InitFirstNode();
  void UpdatePosition();
  void UpdatePositionDense();
  void UpdatePositionSparse();
  void ColSampleTree();
  void ColSampleLevel();

  TrainParam param;
  GPUTrainingParam gpu_param;
  common::HistCutMatrix hmat_;
  common::GHistIndexMatrix gmat_;
  MetaInfo *info;
  bool initialised;
  bool is_dense;
  DeviceGMat device_matrix;

  dh::bulk_allocator ba;
  dh::CubMemory cub_mem;
  dh::dvec<int> gidx_feature_map;
  dh::dvec<int> hist_node_segments;
  dh::dvec<int> feature_segments;
  dh::dvec<float> gain;
  dh::dvec<NodeIdT> position;
  dh::dvec<NodeIdT> position_tmp;
  dh::dvec<float> gidx_fvalue_map;
  dh::dvec<float> fidx_min_map;
  DeviceHist hist;
  dh::dvec<cub::KeyValuePair<int, float>> argmax;
  dh::dvec<gpu_gpair> node_sums;
  dh::dvec<gpu_gpair> hist_scan;
  dh::dvec<gpu_gpair> device_gpair;
  dh::dvec<Node> nodes;
  dh::dvec<int> feature_flags;

  std::vector<int> feature_set_tree;
  std::vector<int> feature_set_level;
};
}  // namespace tree
}  // namespace xgboost
