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
#include "../../src/common/compressed_iterator.h"
#include "device_helpers.cuh"
#include "types.cuh"
#include "nccl.h"

namespace xgboost {
namespace tree {

struct DeviceGMat {
  dh::dvec<common::compressed_byte_t> gidx_buffer;
  common::CompressedIterator<uint32_t> gidx;
  dh::dvec<size_t> row_ptr;
  void Init(int device_idx, const common::GHistIndexMatrix &gmat,
            bst_ulong element_begin, bst_ulong element_end, bst_ulong row_begin, bst_ulong row_end,int n_bins);
};

struct HistBuilder {
  bst_gpair_precise *d_hist;
  int n_bins;
  __host__ __device__ HistBuilder(bst_gpair_precise *ptr, int n_bins);
  __device__ void Add(bst_gpair_precise gpair, int gidx, int nidx) const;
  __device__ bst_gpair_precise Get(int gidx, int nidx) const;
};

struct DeviceHist {
  int n_bins;
  dh::dvec<bst_gpair_precise> data;

  void Init(int max_depth);

  void Reset(int device_idx);

  HistBuilder GetBuilder();

  bst_gpair_precise *GetLevelPtr(int depth);

  int LevelSize(int depth);
};

class GPUHistBuilder {
 public:
  GPUHistBuilder();
  ~GPUHistBuilder();
  void Init(const TrainParam &param);

  void UpdateParam(const TrainParam &param) {
    this->param = param;
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

  std::vector<int> feature_set_tree;
  std::vector<int> feature_set_level;

  bst_uint num_rows;
  int n_devices;

  // below vectors are for each devices used
  std::vector<int> dList;
  std::vector<int> device_row_segments;
  std::vector<size_t> device_element_segments;

  std::vector<dh::CubMemory> temp_memory;
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
  std::vector<dh::dvec<int>> position;
  std::vector<dh::dvec<int>> position_tmp;
  std::vector<DeviceGMat> device_matrix;
  std::vector<dh::dvec<bst_gpair>> device_gpair;
  std::vector<dh::dvec<int>> gidx_feature_map;
  std::vector<dh::dvec<float>> gidx_fvalue_map;

  std::vector<cudaStream_t *> streams;
  std::vector<ncclComm_t> comms;
  std::vector<std::vector<ncclComm_t>> find_split_comms;

  double cpu_init_time;
  double gpu_init_time;
  dh::Timer cpu_time;
  double gpu_time;
  
  
};
}  // namespace tree
}  // namespace xgboost
