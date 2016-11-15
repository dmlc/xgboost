/*!
 * Copyright 2016 Rory mitchell
*/
#include "gpu_builder.cuh"
#include <stdio.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <cub/cub.cuh>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include "cuda_helpers.cuh"
#include "find_split.cuh"
#include "types_functions.cuh"

namespace xgboost {
namespace tree {
struct GPUData {
  GPUData() : allocated(false), n_features(0), n_instances(0) {}

  bool allocated;
  int n_features;
  int n_instances;

  GPUTrainingParam param;

  CubMemory cub_mem;

  thrust::device_vector<float> fvalues;
  thrust::device_vector<int> foffsets;
  thrust::device_vector<bst_uint> instance_id;
  thrust::device_vector<int> feature_id;
  thrust::device_vector<NodeIdT> node_id;
  thrust::device_vector<NodeIdT> node_id_temp;
  thrust::device_vector<NodeIdT> node_id_instance;
  thrust::device_vector<NodeIdT> node_id_instance_temp;
  thrust::device_vector<gpu_gpair> gpair;
  thrust::device_vector<Node> nodes;
  thrust::device_vector<Split> split_candidates;

  thrust::device_vector<Item> items;
  thrust::device_vector<Item> items_temp;

  thrust::device_vector<gpu_gpair> node_sums;
  thrust::device_vector<int> node_offsets;
  thrust::device_vector<int> sort_index_in;
  thrust::device_vector<int> sort_index_out;

  void Init(const std::vector<float> &in_fvalues,
            const std::vector<int> &in_foffsets,
            const std::vector<bst_uint> &in_instance_id,
            const std::vector<int> &in_feature_id,
            const std::vector<bst_gpair> &in_gpair, bst_uint n_instances_in,
            bst_uint n_features_in, int max_depth, const TrainParam &param_in) {
    Timer t;

    // Track allocated device memory
    size_t n_bytes = 0;

    n_features = n_features_in;
    n_instances = n_instances_in;

    fvalues = in_fvalues;
    n_bytes += size_bytes(fvalues);
    foffsets = in_foffsets;
    n_bytes += size_bytes(foffsets);
    instance_id = in_instance_id;
    n_bytes += size_bytes(instance_id);
    feature_id = in_feature_id;
    n_bytes += size_bytes(feature_id);

    param = GPUTrainingParam(param_in.min_child_weight, param_in.reg_lambda,
                             param_in.reg_alpha, param_in.max_delta_step);

    gpair = thrust::device_vector<gpu_gpair>(in_gpair.begin(), in_gpair.end());
    n_bytes += size_bytes(gpair);

    uint32_t max_nodes_level = 1 << max_depth;

    node_sums = thrust::device_vector<gpu_gpair>(max_nodes_level * n_features);
    n_bytes += size_bytes(node_sums);
    node_offsets = thrust::device_vector<int>(max_nodes_level * n_features);
    n_bytes += size_bytes(node_offsets);

    node_id_instance = thrust::device_vector<NodeIdT>(n_instances, 0);
    n_bytes += size_bytes(node_id_instance);

    node_id = thrust::device_vector<NodeIdT>(fvalues.size(), 0);
    n_bytes += size_bytes(node_id);
    node_id_temp = thrust::device_vector<NodeIdT>(fvalues.size());
    n_bytes += size_bytes(node_id_temp);

    uint32_t max_nodes = (1 << (max_depth + 1)) - 1;
    nodes = thrust::device_vector<Node>(max_nodes);
    n_bytes += size_bytes(nodes);

    split_candidates =
        thrust::device_vector<Split>(max_nodes_level * n_features);
    n_bytes += size_bytes(split_candidates);

    // Init items
    items = thrust::device_vector<Item>(fvalues.size());
    n_bytes += size_bytes(items);
    items_temp = thrust::device_vector<Item>(fvalues.size());
    n_bytes += size_bytes(items_temp);

    sort_index_in = thrust::device_vector<int>(fvalues.size());
    n_bytes += size_bytes(sort_index_in);
    sort_index_out = thrust::device_vector<int>(fvalues.size());
    n_bytes += size_bytes(sort_index_out);

    // std::cout << "Device memory allocated: " << n_bytes << "\n";

    this->CreateItems();
    allocated = true;
  }

  ~GPUData() {}

  // Create items array using gpair, instaoce_id, fvalue
  void CreateItems() {
    auto d_items = items.data();
    auto d_instance_id = instance_id.data();
    auto d_gpair = gpair.data();
    auto d_fvalue = fvalues.data();

    auto counting = thrust::make_counting_iterator<bst_uint>(0);
    thrust::for_each(counting, counting + fvalues.size(),
                     [=] __device__(bst_uint i) {
                       Item item;
                       item.instance_id = d_instance_id[i];
                       item.fvalue = d_fvalue[i];
                       item.gpair = d_gpair[item.instance_id];
                       d_items[i] = item;
                     });
  }

  // Reset memory for new boosting iteration
  void Reset(const std::vector<bst_gpair> &in_gpair,
             const std::vector<float> &in_fvalues,
             const std::vector<bst_uint> &in_instance_id) {
    CHECK(allocated);
    thrust::copy(in_gpair.begin(), in_gpair.end(), gpair.begin());
    thrust::fill(nodes.begin(), nodes.end(), Node());
    thrust::fill(node_id_instance.begin(), node_id_instance.end(), 0);
    thrust::fill(node_id.begin(), node_id.end(), 0);

    this->CreateItems();
  }

  bool IsAllocated() { return allocated; }

  // Gather from node_id_instance into node_id according to instance_id
  void GatherNodeId() {
    // Update node_id for each item
    auto d_items = items.data();
    auto d_node_id = node_id.data();
    auto d_node_id_instance = node_id_instance.data();

    auto counting = thrust::make_counting_iterator<bst_uint>(0);
    thrust::for_each(counting, counting + fvalues.size(),
                     [=] __device__(bst_uint i) {
                       Item item = d_items[i];
                       d_node_id[i] = d_node_id_instance[item.instance_id];
                     });
  }
};

GPUBuilder::GPUBuilder() { gpu_data = new GPUData(); }

void GPUBuilder::Init(const TrainParam &param_in) {
  param = param_in;
  CHECK(param.max_depth < 16) << "Max depth > 15 not supported.";
}

GPUBuilder::~GPUBuilder() { delete gpu_data; }

template <int ITEMS_PER_THREAD, typename OffsetT>
__global__ void update_nodeid_missing_kernel(NodeIdT *d_node_id_instance,
                                             Node *d_nodes, const OffsetT n) {
  for (auto i : grid_stride_range(OffsetT(0), n)) {
    NodeIdT item_node_id = d_node_id_instance[i];

    if (item_node_id < 0) {
      continue;
    }
    Node node = d_nodes[item_node_id];

    if (node.IsLeaf()) {
      d_node_id_instance[i] = -1;
    } else if (node.split.missing_left) {
      d_node_id_instance[i] = item_node_id * 2 + 1;
    } else {
      d_node_id_instance[i] = item_node_id * 2 + 2;
    }
  }
}

__device__ void load_as_words(const int n_nodes, Node *d_nodes, Node *s_nodes) {
  const int upper_range = n_nodes * (sizeof(Node) / sizeof(int));
  for (auto i : block_stride_range(0, upper_range)) {
    reinterpret_cast<int *>(s_nodes)[i] = reinterpret_cast<int *>(d_nodes)[i];
  }
}

template <int ITEMS_PER_THREAD>
__global__ void
update_nodeid_fvalue_kernel(NodeIdT *d_node_id, NodeIdT *d_node_id_instance,
                            Item *d_items, Node *d_nodes, const int n_nodes,
                            const int *d_foffsets, const int *d_feature_id,
                            const size_t n, const int n_features,
                            bool cache_nodes) {
  // Load nodes into shared memory
  extern __shared__ Node s_nodes[];

  if (cache_nodes) {
    load_as_words(n_nodes, d_nodes, s_nodes);
    __syncthreads();
  }

  for (auto i : grid_stride_range(size_t(0), n)) {
    Item item = d_items[i];
    NodeIdT item_node_id = d_node_id[i];

    if (item_node_id < 0) {
      continue;
    }

    Node node = cache_nodes ? s_nodes[item_node_id] : d_nodes[item_node_id];

    if (node.IsLeaf()) {
      continue;
    }

    int feature_id = d_feature_id[i];

    if (feature_id == node.split.findex) {
      if (item.fvalue < node.split.fvalue) {
        d_node_id_instance[item.instance_id] = item_node_id * 2 + 1;
      } else {
        d_node_id_instance[item.instance_id] = item_node_id * 2 + 2;
      }
    }
  }
}

void GPUBuilder::UpdateNodeId(int level) {
  // Update all nodes based on missing direction
  {
    const bst_uint n = gpu_data->node_id_instance.size();
    const bst_uint ITEMS_PER_THREAD = 8;
    const bst_uint BLOCK_THREADS = 256;
    const bst_uint GRID_SIZE =
        div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);

    update_nodeid_missing_kernel<
        ITEMS_PER_THREAD><<<GRID_SIZE, BLOCK_THREADS>>>(
        raw(gpu_data->node_id_instance), raw(gpu_data->nodes), n);

    safe_cuda(cudaDeviceSynchronize());
  }

  // Update node based on fvalue where exists
  {
    const bst_uint n = gpu_data->fvalues.size();
    const bst_uint ITEMS_PER_THREAD = 4;
    const bst_uint BLOCK_THREADS = 256;
    const bst_uint GRID_SIZE =
        div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);

    // Use smem cache version if possible
    const bool cache_nodes = level < 7;
    int n_nodes = (1 << (level + 1)) - 1;
    int smem_size = cache_nodes ? sizeof(Node) * n_nodes : 0;
    update_nodeid_fvalue_kernel<
        ITEMS_PER_THREAD><<<GRID_SIZE, BLOCK_THREADS, smem_size>>>(
        raw(gpu_data->node_id), raw(gpu_data->node_id_instance),
        raw(gpu_data->items), raw(gpu_data->nodes), n_nodes,
        raw(gpu_data->foffsets), raw(gpu_data->feature_id),
        gpu_data->fvalues.size(), gpu_data->n_features, cache_nodes);

    safe_cuda(cudaGetLastError());
    safe_cuda(cudaDeviceSynchronize());
  }

  gpu_data->GatherNodeId();
}

void GPUBuilder::Sort(int level) {
  thrust::sequence(gpu_data->sort_index_in.begin(),
                   gpu_data->sort_index_in.end());

  cub::DoubleBuffer<NodeIdT> d_keys(raw(gpu_data->node_id),
                                    raw(gpu_data->node_id_temp));
  cub::DoubleBuffer<int> d_values(raw(gpu_data->sort_index_in),
                                  raw(gpu_data->sort_index_out));

  if (!gpu_data->cub_mem.IsAllocated()) {
    cub::DeviceSegmentedRadixSort::SortPairs(
        gpu_data->cub_mem.d_temp_storage, gpu_data->cub_mem.temp_storage_bytes,
        d_keys, d_values, gpu_data->fvalues.size(), gpu_data->n_features,
        raw(gpu_data->foffsets), raw(gpu_data->foffsets) + 1);
    gpu_data->cub_mem.Allocate();
  }

  cub::DeviceSegmentedRadixSort::SortPairs(
      gpu_data->cub_mem.d_temp_storage, gpu_data->cub_mem.temp_storage_bytes,
      d_keys, d_values, gpu_data->fvalues.size(), gpu_data->n_features,
      raw(gpu_data->foffsets), raw(gpu_data->foffsets) + 1);

  thrust::gather(thrust::device_pointer_cast(d_values.Current()),
                 thrust::device_pointer_cast(d_values.Current()) +
                     gpu_data->sort_index_out.size(),
                 gpu_data->items.begin(), gpu_data->items_temp.begin());

  thrust::copy(gpu_data->items_temp.begin(), gpu_data->items_temp.end(),
               gpu_data->items.begin());

  if (d_keys.Current() == raw(gpu_data->node_id_temp)) {
    thrust::copy(gpu_data->node_id_temp.begin(), gpu_data->node_id_temp.end(),
                 gpu_data->node_id.begin());
  }
}

void GPUBuilder::Update(const std::vector<bst_gpair> &gpair, DMatrix *p_fmat,
                        RegTree *p_tree) {
  cudaProfilerStart();
  try {
    Timer update;
    Timer t;
    this->InitData(gpair, *p_fmat, *p_tree);
    t.printElapsed("init data");
    this->InitFirstNode();

    for (int level = 0; level < param.max_depth; level++) {
      bool use_multiscan_algorithm = level < multiscan_levels;

      t.reset();
      if (level > 0) {
        Timer update_node;
        this->UpdateNodeId(level);
        update_node.printElapsed("node");
      }

      if (level > 0 && !use_multiscan_algorithm) {
        Timer s;
        this->Sort(level);
        s.printElapsed("sort");
      }

      Timer split;
      find_split(raw(gpu_data->items), raw(gpu_data->split_candidates),
                 raw(gpu_data->node_id), raw(gpu_data->nodes),
                 (bst_uint)gpu_data->fvalues.size(), gpu_data->n_features,
                 raw(gpu_data->foffsets), raw(gpu_data->node_sums),
                 raw(gpu_data->node_offsets), gpu_data->param, level,
                 use_multiscan_algorithm);

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
  cudaProfilerStop();
}

void GPUBuilder::InitData(const std::vector<bst_gpair> &gpair, DMatrix &fmat,
                          const RegTree &tree) {
  CHECK_EQ(tree.param.num_nodes, tree.param.num_roots)
      << "ColMaker: can only grow new tree";

  CHECK(fmat.SingleColBlock()) << "GPUMaker: must have single column block";

  if (gpu_data->IsAllocated()) {
    gpu_data->Reset(gpair, fvalues, instance_id);
    return;
  }

  Timer t;

  MetaInfo info = fmat.info();
  dmlc::DataIter<ColBatch> *iter = fmat.ColIterator();

  std::vector<int> foffsets;
  foffsets.push_back(0);
  std::vector<int> feature_id;
  fvalues.reserve(info.num_col * info.num_row);
  instance_id.reserve(info.num_col * info.num_row);
  feature_id.reserve(info.num_col * info.num_row);

  while (iter->Next()) {
    const ColBatch &batch = iter->Value();

    for (int i = 0; i < batch.size; i++) {
      const ColBatch::Inst &col = batch[i];

      for (const ColBatch::Entry *it = col.data; it != col.data + col.length;
           it++) {
        fvalues.push_back(it->fvalue);
        instance_id.push_back(it->index);
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
      thrust::reduce(gpu_data->gpair.begin(), gpu_data->gpair.end(),
                     gpu_gpair(0, 0), cub::Sum());

  gpu_data->nodes[0] = Node(
      sum_gradients,
      CalcGain(gpu_data->param, sum_gradients.grad(), sum_gradients.hess()),
      CalcWeight(gpu_data->param, sum_gradients.grad(), sum_gradients.hess()));
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
  thrust::host_vector<Node> h_nodes = gpu_data->nodes;
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
