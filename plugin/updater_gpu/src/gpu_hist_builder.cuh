/*!
 * Copyright 2017 XGBoost contributors
 */
#pragma once
#include <thrust/device_vector.h>
#include <xgboost/tree_updater.h>
#include <cub/util_type.cuh>  // Need key value pair definition
#include <vector>
#include "../../src/common/hist_util.h"
#include "../../src/tree/param.h"
#include "device_helpers.cuh"
#include "types.cuh"

#ifndef NCCL
#define NCCL 1
#endif

#if (NCCL)
#include "nccl.h"
#endif

namespace xgboost {

namespace tree {

struct DeviceGMat {
  dh::dvec<int> gidx;
  dh::dvec<int> ridx;
  void Init(int device_idx, const common::GHistIndexMatrix &gmat,
            bst_uint begin, bst_uint end);
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
  dh::dvec<gpu_gpair> data;

  void Init(int max_depth);

  void Reset(int device_idx);

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
  template <int BLOCK_THREADS>
  void FindSplitSpecialize(int depth);
  template <int BLOCK_THREADS>
  void LaunchFindSplit(int depth);
  void InitFirstNode(const std::vector<bst_gpair> &gpair);
  void UpdatePosition(int depth);
  void UpdatePositionDense(int depth);
  void UpdatePositionSparse(int depth);
  void ColSampleTree();
  void ColSampleLevel();
  bool UpdatePredictionCache(const DMatrix *data,
                             std::vector<bst_float> *p_out_preds);

  TrainParam param;
  GPUTrainingParam gpu_param;
  common::HistCutMatrix hmat_;
  common::GHistIndexMatrix gmat_;
  MetaInfo *info;
  bool initialised;
  bool is_dense;
  const DMatrix *p_last_fmat_;
  bool prediction_cache_initialised;

  // choose which memory type to use (DEVICE or DEVICE_MANAGED)
  dh::bulk_allocator<dh::memory_type::DEVICE> ba;
  //  dh::bulk_allocator<dh::memory_type::DEVICE_MANAGED> ba; // can't be used
  //  with NCCL
  dh::CubMemory cub_mem;

  std::vector<int> feature_set_tree;
  std::vector<int> feature_set_level;

  bst_uint num_rows;
  int n_devices;

  // below vectors are for each devices used
  std::vector<int> dList;
  std::vector<int> device_row_segments;
  std::vector<int> device_element_segments;

  std::vector<DeviceHist> hist_vec;
  std::vector<dh::dvec<Node>> nodes;
  std::vector<dh::dvec<Node>> nodes_temp;
  std::vector<dh::dvec<Node>> nodes_child_temp;
  std::vector<dh::dvec<bool>> left_child_smallest;
  std::vector<dh::dvec<bool>> left_child_smallest_temp;
  std::vector<dh::dvec<int>> feature_flags;
  std::vector<dh::dvec<float>> fidx_min_map;
  std::vector<dh::dvec<int>> feature_segments;
  std::vector<dh::dvec<bst_float>> prediction_cache;
  std::vector<dh::dvec<NodeIdT>> position;
  std::vector<dh::dvec<NodeIdT>> position_tmp;
  std::vector<DeviceGMat> device_matrix;
  std::vector<dh::dvec<gpu_gpair>> device_gpair;
  std::vector<dh::dvec<int>> gidx_feature_map;
  std::vector<dh::dvec<float>> gidx_fvalue_map;

  std::vector<cudaStream_t *> streams;
#if (NCCL)
  std::vector<ncclComm_t> comms;
  std::vector<std::vector<ncclComm_t>> find_split_comms;
#endif
};
}  // namespace tree
}  // namespace xgboost
