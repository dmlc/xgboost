/*!
 * Copyright 2016 Rory mitchell
 */
#pragma once
#include <thrust/sequence.h>
#include <xgboost/logging.h>
#include <cub/cub.cuh>
#include <vector>
#include "../../src/tree/param.h"
#include "common.cuh"
#include "device_helpers.cuh"
#include "types.cuh"

namespace xgboost {
namespace tree {

struct GPUData {
  GPUData() : allocated(false), n_features(0), n_instances(0) {}

  bool allocated;
  int n_features;
  int n_instances;

  dh::bulk_allocator ba;
  GPUTrainingParam param;

  dh::dvec<float> fvalues;
  dh::dvec<float> fvalues_temp;
  dh::dvec<float> fvalues_cached;
  dh::dvec<int> foffsets;
  dh::dvec<bst_uint> instance_id;
  dh::dvec<bst_uint> instance_id_temp;
  dh::dvec<bst_uint> instance_id_cached;
  dh::dvec<int> feature_id;
  dh::dvec<NodeIdT> node_id;
  dh::dvec<NodeIdT> node_id_temp;
  dh::dvec<NodeIdT> node_id_instance;
  dh::dvec<gpu_gpair> gpair;
  dh::dvec<Node> nodes;
  dh::dvec<Split> split_candidates;
  dh::dvec<gpu_gpair> node_sums;
  dh::dvec<int> node_offsets;
  dh::dvec<int> sort_index_in;
  dh::dvec<int> sort_index_out;

  dh::dvec<char> cub_mem;

  dh::dvec<int> feature_flags;
  dh::dvec<int> feature_set;

  ItemIter items_iter;

  void Init(const std::vector<float> &in_fvalues,
            const std::vector<int> &in_foffsets,
            const std::vector<bst_uint> &in_instance_id,
            const std::vector<int> &in_feature_id,
            const std::vector<bst_gpair> &in_gpair, bst_uint n_instances_in,
            bst_uint n_features_in, int max_depth, const TrainParam &param_in) {
    n_features = n_features_in;
    n_instances = n_instances_in;

    uint32_t max_nodes = (1 << (max_depth + 1)) - 1;
    uint32_t max_nodes_level = 1 << max_depth;

    // Calculate memory for sort
    size_t cub_mem_size = 0;
    cub::DoubleBuffer<NodeIdT> db_key;
    cub::DoubleBuffer<int> db_value;

    cub::DeviceSegmentedRadixSort::SortPairs(
        cub_mem.data(), cub_mem_size, db_key, db_value, in_fvalues.size(),
        n_features, foffsets.data(), foffsets.data() + 1);

    // Allocate memory
    size_t free_memory = dh::available_memory();
    ba.allocate(&fvalues, in_fvalues.size(), &fvalues_temp, in_fvalues.size(),
                &fvalues_cached, in_fvalues.size(), &foffsets,
                in_foffsets.size(), &instance_id, in_instance_id.size(),
                &instance_id_temp, in_instance_id.size(), &instance_id_cached,
                in_instance_id.size(), &feature_id, in_feature_id.size(),
                &node_id, in_fvalues.size(), &node_id_temp, in_fvalues.size(),
                &node_id_instance, n_instances, &gpair, n_instances, &nodes,
                max_nodes, &split_candidates, max_nodes_level * n_features,
                &node_sums, max_nodes_level * n_features, &node_offsets,
                max_nodes_level * n_features, &sort_index_in, in_fvalues.size(),
                &sort_index_out, in_fvalues.size(), &cub_mem, cub_mem_size,
                &feature_flags, n_features, &feature_set, n_features);

    if (!param_in.silent) {
      const int mb_size = 1048576;
      LOG(CONSOLE) << "Allocated " << ba.size() / mb_size << "/"
                   << free_memory / mb_size << " MB on " << dh::device_name();
    }

    fvalues_cached = in_fvalues;
    foffsets = in_foffsets;
    instance_id_cached = in_instance_id;
    feature_id = in_feature_id;

    param = GPUTrainingParam(param_in.min_child_weight, param_in.reg_lambda,
                             param_in.reg_alpha, param_in.max_delta_step);

    allocated = true;

    this->Reset(in_gpair, param_in.subsample);

    items_iter = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_permutation_iterator(gpair.tbegin(), instance_id.tbegin()),
        fvalues.tbegin(), node_id.tbegin()));

    dh::safe_cuda(cudaGetLastError());
  }

  ~GPUData() {}

  // Reset memory for new boosting iteration
  void Reset(const std::vector<bst_gpair> &in_gpair, float subsample) {
    CHECK(allocated);
    gpair = in_gpair;
    subsample_gpair(&gpair, subsample);
    instance_id = instance_id_cached;
    fvalues = fvalues_cached;
    nodes.fill(Node());
    node_id_instance.fill(0);
    node_id.fill(0);
  }

  bool IsAllocated() { return allocated; }

  // Gather from node_id_instance into node_id according to instance_id
  void GatherNodeId() {
    // Update node_id for each item
    auto d_node_id = node_id.data();
    auto d_node_id_instance = node_id_instance.data();
    auto d_instance_id = instance_id.data();

    dh::launch_n(fvalues.size(), [=] __device__(bst_uint i) {
      d_node_id[i] = d_node_id_instance[d_instance_id[i]];
    });
  }
};
}  // namespace tree
}  // namespace xgboost
