/*!
 * Copyright 2016 Rory mitchell
*/
#include <cub/cub.cuh>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include "../../../src/common/random.h"
#include "device_helpers.cuh"
#include "find_split.cuh"
#include "gpu_builder.cuh"
#include "types_functions.cuh"
#include "gpu_data.cuh"

namespace xgboost {
namespace tree {

GPUBuilder::GPUBuilder() { gpu_data = new GPUData(); }

void GPUBuilder::Init(const TrainParam &param_in) {
  param = param_in;
  CHECK(param.max_depth < 16) << "Tree depth too large.";
}

GPUBuilder::~GPUBuilder() { delete gpu_data; }

void GPUBuilder::UpdateNodeId(int level) {
  auto *d_node_id_instance = gpu_data->node_id_instance.data();
  Node *d_nodes = gpu_data->nodes.data();

  dh::launch_n(gpu_data->node_id_instance.size(), [=] __device__(int i) {
    NodeIdT item_node_id = d_node_id_instance[i];

    if (item_node_id < 0) {
      return;
    }

    Node node = d_nodes[item_node_id];

    if (node.IsLeaf()) {
      d_node_id_instance[i] = -1;
    } else if (node.split.missing_left) {
      d_node_id_instance[i] = item_node_id * 2 + 1;
    } else {
      d_node_id_instance[i] = item_node_id * 2 + 2;
    }
  });

  dh::safe_cuda(cudaDeviceSynchronize());

  auto *d_fvalues = gpu_data->fvalues.data();
  auto *d_instance_id = gpu_data->instance_id.data();
  auto *d_node_id = gpu_data->node_id.data();
  auto *d_feature_id = gpu_data->feature_id.data();

  // Update node based on fvalue where exists
  dh::launch_n(gpu_data->fvalues.size(), [=] __device__(int i) {
    NodeIdT item_node_id = d_node_id[i];

    if (item_node_id < 0) {
      return;
    }

    Node node = d_nodes[item_node_id];

    if (node.IsLeaf()) {
      return;
    }

    int feature_id = d_feature_id[i];

    if (feature_id == node.split.findex) {
      float fvalue = d_fvalues[i];
      bst_uint instance_id = d_instance_id[i];

      if (fvalue < node.split.fvalue) {
        d_node_id_instance[instance_id] = item_node_id * 2 + 1;
      } else {
        d_node_id_instance[instance_id] = item_node_id * 2 + 2;
      }
    }
  });

  dh::safe_cuda(cudaDeviceSynchronize());

  gpu_data->GatherNodeId();
}

void GPUBuilder::Sort(int level) {
  thrust::sequence(gpu_data->sort_index_in.tbegin(),
                   gpu_data->sort_index_in.tend());

  cub::DoubleBuffer<NodeIdT> d_keys(gpu_data->node_id.data(),
                                    gpu_data->node_id_temp.data());
  cub::DoubleBuffer<int> d_values(gpu_data->sort_index_in.data(),
                                  gpu_data->sort_index_out.data());

  size_t temp_size = gpu_data->cub_mem.size();

  cub::DeviceSegmentedRadixSort::SortPairs(
      gpu_data->cub_mem.data(), temp_size, d_keys, d_values,
      gpu_data->fvalues.size(), gpu_data->n_features, gpu_data->foffsets.data(),
      gpu_data->foffsets.data() + 1);

  auto zip = thrust::make_zip_iterator(thrust::make_tuple(
      gpu_data->fvalues.tbegin(), gpu_data->instance_id.tbegin()));
  auto zip_temp = thrust::make_zip_iterator(thrust::make_tuple(
      gpu_data->fvalues_temp.tbegin(), gpu_data->instance_id_temp.tbegin()));
  thrust::gather(thrust::device_pointer_cast(d_values.Current()),
                 thrust::device_pointer_cast(d_values.Current()) +
                     gpu_data->sort_index_out.size(),
                 zip, zip_temp);
  thrust::copy(zip_temp, zip_temp + gpu_data->fvalues.size(), zip);

  if (d_keys.Current() == gpu_data->node_id_temp.data()) {
    thrust::copy(gpu_data->node_id_temp.tbegin(), gpu_data->node_id_temp.tend(),
                 gpu_data->node_id.tbegin());
  }
}

void GPUBuilder::ColsampleTree() {
  unsigned n = static_cast<unsigned>(
    param.colsample_bytree * gpu_data->n_features);
  CHECK_GT(n, 0);

  feature_set_tree.resize(gpu_data->n_features);
  std::iota(feature_set_tree.begin(), feature_set_tree.end(), 0);
  std::shuffle(feature_set_tree.begin(), feature_set_tree.end(),
    common::GlobalRandom());
}

void GPUBuilder::Update(const std::vector<bst_gpair> &gpair, DMatrix *p_fmat,
                        RegTree *p_tree) {
  try {
    dh::Timer update;
    dh::Timer t;
    this->InitData(gpair, *p_fmat, *p_tree);
    t.printElapsed("init data");
    this->InitFirstNode();
    this->ColsampleTree();

    for (int level = 0; level < param.max_depth; level++) {
      bool use_multiscan_algorithm = level < multiscan_levels;

      t.reset();
      if (level > 0) {
        dh::Timer update_node;
        this->UpdateNodeId(level);
        update_node.printElapsed("node");
      }

      if (level > 0 && !use_multiscan_algorithm) {
        dh::Timer s;
        this->Sort(level);
        s.printElapsed("sort");
      }

      dh::Timer split;
      find_split(gpu_data, param, level, use_multiscan_algorithm,
        feature_set_tree, &feature_set_level);

      split.printElapsed("split");

      t.printElapsed("level");
    }
    this->CopyTree(*p_tree);
    update.printElapsed("update");
  } catch (thrust::system_error &e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    exit(-1);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "Unknown exception." << std::endl;
    exit(-1);
  }
}

void GPUBuilder::InitData(const std::vector<bst_gpair> &gpair, DMatrix &fmat,
                          const RegTree &tree) {
  CHECK(fmat.SingleColBlock()) << "GPUMaker: must have single column block";

  if (gpu_data->IsAllocated()) {
    gpu_data->Reset(gpair, param.subsample);
    return;
  }

  dh::Timer t;

  MetaInfo info = fmat.info();

  std::vector<int> foffsets;
  foffsets.push_back(0);
  std::vector<int> feature_id;
  std::vector<float> fvalues;
  std::vector<bst_uint> instance_id;
  fvalues.reserve(info.num_col * info.num_row);
  instance_id.reserve(info.num_col * info.num_row);
  feature_id.reserve(info.num_col * info.num_row);

  dmlc::DataIter<ColBatch> *iter = fmat.ColIterator();

  while (iter->Next()) {
    const ColBatch &batch = iter->Value();

    for (int i = 0; i < batch.size; i++) {
      const ColBatch::Inst &col = batch[i];

      for (const ColBatch::Entry *it = col.data; it != col.data + col.length;
           it++) {
        bst_uint inst_id = it->index;
          fvalues.push_back(it->fvalue);
          instance_id.push_back(inst_id);
          feature_id.push_back(i);
      }
      foffsets.push_back(fvalues.size());
    }
  }

  t.printElapsed("dmatrix");
  t.reset();
  gpu_data->Init(fvalues, foffsets, instance_id, feature_id, gpair,
                 info.num_row, info.num_col, param.max_depth, param);

  t.printElapsed("gpu init");
}

void GPUBuilder::InitFirstNode() {
  // Build the root node on the CPU and copy to device
  gpu_gpair sum_gradients =
      thrust::reduce(gpu_data->gpair.tbegin(), gpu_data->gpair.tend(),
                     gpu_gpair(0, 0), cub::Sum());

  Node tmp = Node(
      sum_gradients,
      CalcGain(gpu_data->param, sum_gradients.grad(), sum_gradients.hess()),
      CalcWeight(gpu_data->param, sum_gradients.grad(), sum_gradients.hess()));

  thrust::copy_n(&tmp, 1, gpu_data->nodes.tbegin());
}

enum NodeType {
  NODE = 0,
  LEAF = 1,
  UNUSED = 2,
};

// Recursively label node types
void flag_nodes(const thrust::host_vector<Node> &nodes,
                std::vector<NodeType> *node_flags, int nid, NodeType type) {
  if (nid >= nodes.size() || type == UNUSED) {
    return;
  }

  const Node &n = nodes[nid];

  // Current node and all children are valid
  if (n.split.loss_chg > rt_eps) {
    (*node_flags)[nid] = NODE;
    flag_nodes(nodes, node_flags, nid * 2 + 1, NODE);
    flag_nodes(nodes, node_flags, nid * 2 + 2, NODE);
  } else {
    // Current node is leaf, therefore is valid but all children are invalid
    (*node_flags)[nid] = LEAF;
    flag_nodes(nodes, node_flags, nid * 2 + 1, UNUSED);
    flag_nodes(nodes, node_flags, nid * 2 + 2, UNUSED);
  }
}

// Copy gpu dense representation of tree to xgboost sparse representation
void GPUBuilder::CopyTree(RegTree &tree) {
  std::vector<Node> h_nodes = gpu_data->nodes.as_vector();
  std::vector<NodeType> node_flags(h_nodes.size(), UNUSED);
  flag_nodes(h_nodes, &node_flags, 0, NODE);

  int nid = 0;
  for (int gpu_nid = 0; gpu_nid < h_nodes.size(); gpu_nid++) {
    NodeType flag = node_flags[gpu_nid];
    const Node &n = h_nodes[gpu_nid];
    if (flag == NODE) {
      tree.AddChilds(nid);
      tree[nid].set_split(n.split.findex, n.split.fvalue, n.split.missing_left);
      tree.stat(nid).loss_chg = n.split.loss_chg;
      tree.stat(nid).base_weight = n.weight;
      tree.stat(nid).sum_hess = n.sum_gradients.hess();
      tree[tree[nid].cleft()].set_leaf(0);
      tree[tree[nid].cright()].set_leaf(0);
      nid++;
    } else if (flag == LEAF) {
      tree[nid].set_leaf(n.weight * param.learning_rate);
      tree.stat(nid).sum_hess = n.sum_gradients.hess();
      nid++;
    }
  }
}
}  // namespace tree
}  // namespace xgboost
