/*!
 * Copyright 2014 by Contributors
 * \file gpu_builder.cu
 * \brief GPU accelerated decision tree construction
 * \author Rory Mitchell
 */
#include "gpu_builder.cuh"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/count.h>
#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "cuda_helpers.cuh"
#include "cuda_profiler_api.h"
#include "gpu_gpair.cuh"
#include "find_split.cuh"

namespace xgboost {
namespace tree {
struct GPUData {
    GPUData()
        : allocated(false),
          n_features(0),
          n_instances(0) {
    }

    bool allocated;
    int n_features;
    int n_instances;

    GPUTrainingParam param;

    thrust::device_vector<float> fvalues;
    thrust::device_vector<bst_uint> foffsets;
    thrust::device_vector<bst_uint> instance_id;
    thrust::device_vector<int> feature_id;
    thrust::device_vector<int8_t> node_id;
    thrust::device_vector<int8_t> new_node_id;
    thrust::device_vector<gpu_gpair> gpair;
    thrust::device_vector<gpu_gpair> gpair_gather;
    thrust::device_vector<Node> nodes;
    thrust::device_vector<Split> split_candidates;

    typedef thrust::permutation_iterator<
        thrust::device_vector<int8_t>::iterator,
        thrust::device_vector<bst_uint>::iterator> perm;

    perm node_id_perm;
    perm new_node_id_perm;


    void Init(const std::vector<float>& in_fvalues,
        const std::vector<bst_uint>& in_foffsets,
        const std::vector<bst_uint>& in_instance_id,
        const std::vector<int>& in_feature_id,
        const std::vector<bst_gpair>& in_gpair,
        bst_uint n_instances_in,
        bst_uint n_features_in,
        int max_depth,
        const TrainParam& param_in) {
        n_features = n_features_in;
        n_instances = n_instances_in;

        fvalues = in_fvalues;
        foffsets = in_foffsets;
        instance_id = in_instance_id;
        feature_id = in_feature_id;

        param = GPUTrainingParam(param_in.min_child_weight,
            param_in.reg_lambda,
            param_in.reg_alpha,
            param_in.max_delta_step);

        gpair = thrust::device_vector<gpu_gpair>(in_gpair.begin(), in_gpair.end());

        gpair_gather = thrust::device_vector<gpu_gpair>(fvalues.size());
        thrust::gather(instance_id.begin(),
            instance_id.end(),
            gpair.begin(),
            gpair_gather.begin());

        uint32_t max_nodes_level = 1 << max_depth;

        node_id = thrust::device_vector<int8_t>(n_instances, 0);
        new_node_id = thrust::device_vector<int8_t>(n_instances);

        node_id_perm = thrust::make_permutation_iterator(
            node_id.begin(),
            instance_id.begin());
        new_node_id_perm = thrust::make_permutation_iterator(
            new_node_id.begin(),
            instance_id.begin());

        uint32_t max_nodes = (1 << (max_depth + 1)) - 1;
        nodes = thrust::device_vector<Node>(max_nodes);

        split_candidates = thrust::device_vector<Split>(max_nodes_level * n_features);
        allocated = true;
    }

    ~GPUData() {
    }

    // Reset memory for new boosting iteration
    void Reset(const std::vector<bst_gpair>& in_gpair) {
        CHECK(allocated);
        thrust::copy(in_gpair.begin(), in_gpair.end(), gpair.begin());
        thrust::gather(instance_id.begin(), instance_id.end(), gpair.begin(), gpair_gather.begin());
        thrust::fill(nodes.begin(), nodes.end(), Node());
        thrust::fill(node_id.begin(), node_id.end(), 0);
    }

    bool IsAllocated() {
        return allocated;
    }
};

GPUBuilder::GPUBuilder() {
    gpu_data = new GPUData();
}

void GPUBuilder::Init(const TrainParam& param_in) {
    param = param_in;
}

GPUBuilder::~GPUBuilder() {
    delete gpu_data;
}

template <typename OffsetT>
__device__ int find_feature_id(size_t i, const OffsetT* d_feature_offsets, int n_features) {
    for (int feature = 0; feature < n_features; feature++) {
        if (i < d_feature_offsets[feature]) {
            return feature - 1;
        }
    }

    return -1;
}

template <int ITEMS_PER_THREAD, typename NodeIdT, typename OffsetT>
__global__ void update_nodeid_missing_kernel(NodeIdT* d_node_id,
    NodeIdT* d_node_id_new,
    Node* d_nodes,
    const OffsetT n) {
    for (auto i : grid_stride_range(OffsetT(0), n)) {
        int8_t item_node_id = d_node_id[i];

        if (item_node_id < 0) {
            continue;
        }

        Node node = d_nodes[item_node_id];

        if (node.IsLeaf()) {
            d_node_id_new[i] = -1;
        } else if (node.split.missing_left) {
            d_node_id_new[i] = item_node_id * 2 + 1;
        } else {
            d_node_id_new[i] = item_node_id * 2 + 2;
        }
    }
}

__device__ void load_as_words(const int n_nodes, Node* d_nodes, Node* s_nodes) {
    const int upper_range = n_nodes * (sizeof(Node) / sizeof(int));
    for (auto i : block_stride_range(0, upper_range)) {
        reinterpret_cast<int*>(s_nodes)[i] = reinterpret_cast<int*>(d_nodes)[i];
    }
}

template <int ITEMS_PER_THREAD, typename NodeIdIterT, typename OffsetT>
__global__ void update_nodeid_fvalue_kernel(NodeIdIterT node_id,
    NodeIdIterT node_id_new,
    float* d_fvalue,
    Node* d_nodes,
    const int n_nodes,
    const OffsetT* d_feature_offsets,
    const int* d_feature_id,
    const OffsetT n,
    const int n_features) {

    // Load nodes into shared memory
    extern __shared__ Node s_nodes[];

    load_as_words(n_nodes, d_nodes, s_nodes);

    __syncthreads();

    for (auto i : grid_stride_range(OffsetT(0), n)) {
        int8_t item_node_id = node_id[i];

        if (item_node_id < 0) {
            continue;
        }

        Node node = s_nodes[item_node_id];

        if (node.IsLeaf()) {
            continue;
        }

        int feature_id = d_feature_id[i];

        if (feature_id == node.split.findex) {
            float fvalue = d_fvalue[i];

            if (fvalue < node.split.fvalue) {
                node_id_new[i] = item_node_id * 2 + 1;
            } else {
                node_id_new[i] = item_node_id * 2 + 2;
            }
        }
    }
}

void GPUBuilder::UpdateNodeId() {
    // Update all nodes based on missing direction
    {
        const bst_uint n = gpu_data->node_id.size();
        const bst_uint ITEMS_PER_THREAD = 8;
        const bst_uint BLOCK_THREADS = 256;
        const bst_uint GRID_SIZE = div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);

        update_nodeid_missing_kernel<ITEMS_PER_THREAD> << <GRID_SIZE,
            BLOCK_THREADS >> >(raw(gpu_data->node_id),
            raw(gpu_data->new_node_id),
            raw(gpu_data->nodes),
            n);

        safe_cuda(cudaDeviceSynchronize());
    }

    // Update node based on fvalue where exists
    {
        const bst_uint n = gpu_data->fvalues.size();
        const bst_uint ITEMS_PER_THREAD = 4;
        const bst_uint BLOCK_THREADS = 256;
        const bst_uint GRID_SIZE = div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);

        update_nodeid_fvalue_kernel<ITEMS_PER_THREAD>
            << <GRID_SIZE,
            BLOCK_THREADS, sizeof(Node) * gpu_data->nodes.size() >> >(gpu_data->node_id_perm,
            gpu_data->new_node_id_perm,
            raw(gpu_data->fvalues),
            raw(gpu_data->nodes),
            gpu_data->nodes.size(),
            raw(gpu_data->foffsets),
            raw(gpu_data->feature_id),
            (bst_uint)gpu_data->fvalues.size(),
            gpu_data->n_features);

        safe_cuda(cudaDeviceSynchronize());
    }

    thrust::copy(gpu_data->new_node_id.begin(),
        gpu_data->new_node_id.end(),
        gpu_data->node_id.begin());
}

void GPUBuilder::Update(const std::vector<bst_gpair>& gpair, DMatrix* p_fmat, RegTree* p_tree) {
    try {
        Timer update;
        Timer t;
        this->InitData(gpair, *p_fmat, *p_tree);
        t.printElapsed("init data");
        this->InitFirstNode();

        for (int level = 0; level < param.max_depth; level++) {
            t.reset();
            // Clear split candidates
            thrust::fill(gpu_data->split_candidates.begin(),
                gpu_data->split_candidates.end(),
                Split());

            find_split(raw(gpu_data->gpair_gather),
                raw(gpu_data->split_candidates),
                raw(gpu_data->fvalues),
                gpu_data->node_id_perm,
                raw(gpu_data->nodes),
                static_cast<bst_uint>(gpu_data->fvalues.size()),
                gpu_data->n_features,
                raw(gpu_data->foffsets),
                gpu_data->param, level);

            this->UpdateNodeId();
            t.printElapsed("level");
        }
        this->CopyTree(*p_tree);
        update.printElapsed("update");
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
        exit(-1);
    }
}

void GPUBuilder::InitData(const std::vector<bst_gpair>& gpair,
                          DMatrix& fmat,
                          const RegTree& tree) {
    CHECK_EQ(tree.param.num_nodes, tree.param.num_roots)
            << "ColMaker: can only grow new tree";

    CHECK(fmat.SingleColBlock()) << "GPUMaker: must have single column block";

    CHECK(param.max_depth <= 6) << "GPUMaker: Only supports trees of up to depth 6";

    if (gpu_data->IsAllocated()) {
        gpu_data->Reset(gpair);
        return;
    }

    Timer t;

    MetaInfo info = fmat.info();
    dmlc::DataIter<ColBatch>* iter = fmat.ColIterator();

    std::vector<float> fvalues;
    std::vector<bst_uint> foffsets;
    foffsets.push_back(0);
    std::vector<bst_uint> instance_id;
    std::vector<int> feature_id;
    fvalues.reserve(info.num_col * info.num_row);
    instance_id.reserve(info.num_col * info.num_row);
    feature_id.reserve(info.num_col * info.num_row);

    while (iter->Next()) {
        const ColBatch& batch = iter->Value();

        for (int i = 0; i < batch.size; i++) {
            const ColBatch::Inst& col = batch[i];

            for (const ColBatch::Entry *it = col.data; it != col.data+col.length; it++) {
                fvalues.push_back(it->fvalue);
                instance_id.push_back(it->index);
                feature_id.push_back(i);
            }
            foffsets.push_back(fvalues.size());
        }
    }

    t.printElapsed("read from dmatrix");
    t.reset();

    gpu_data->Init(fvalues,
        foffsets,
        instance_id,
        feature_id,
        gpair,
        info.num_row,
        info.num_col,
        param.max_depth,
        param);

    t.printElapsed("gpu_data init");
}

void GPUBuilder::InitFirstNode() {
    // Build the root node on the CPU and copy to device
    gpu_gpair sum_gradients = thrust::reduce(
            gpu_data->gpair.begin(),
            gpu_data->gpair.end(),
            gpu_gpair(0, 0), cub::Sum());
    gpu_data->nodes[0] = Node(sum_gradients, gpu_data->param);
}


enum NodeType {
    NODE = 0,
    LEAF = 1,
    UNUSED = 2,
};


// Recursively label node types
void flag_nodes(const thrust::host_vector<Node>& nodes, std::vector<NodeType>* node_flags,
                int nid, NodeType type) {
    if (nid >= nodes.size() || type == UNUSED) {
        return;
    }

    const Node& n = nodes[nid];

    // Current node and all children are valid
    if (n.split.loss_chg > rt_eps) {
        (*node_flags)[nid] = NODE;
        flag_nodes(nodes, node_flags, nid * 2 + 1, NODE);
        flag_nodes(nodes, node_flags, nid * 2 + 2, NODE);

        // Current node is leaf, therefore is valid but all children are invalid
    } else {
        (*node_flags)[nid] = LEAF;
        flag_nodes(nodes, node_flags, nid * 2 + 1, UNUSED);
        flag_nodes(nodes, node_flags, nid * 2 + 2, UNUSED);
    }
}

// Copy gpu dense representation of tree to xgboost sparse representation
void GPUBuilder::CopyTree(RegTree& tree) {
    thrust::host_vector<Node> h_nodes = gpu_data->nodes;
    std::vector<NodeType> node_flags(h_nodes.size(), UNUSED);
    flag_nodes(h_nodes, &node_flags, 0, NODE);

    int nid = 0;
    for (int gpu_nid = 0; gpu_nid < h_nodes.size(); gpu_nid++) {
        NodeType flag = node_flags[gpu_nid];
        const Node& n = h_nodes[gpu_nid];
        if (flag == NODE) {
            tree.AddChilds(nid);
            tree[nid].set_split(n.split.findex, n.split.fvalue, n.split.missing_left);
            tree.stat(nid).loss_chg = n.split.loss_chg;
            tree.stat(nid).base_weight = n.weight;
            tree.stat(nid).sum_hess = n.sum_gradients.hess();
            tree[tree[nid].cleft()].set_leaf(0);
            tree[tree[nid].cright()].set_leaf(0);
            nid++;
        }
        if (flag == LEAF) {
            tree[nid].set_leaf(n.weight * param.learning_rate);
            tree.stat(nid).sum_hess = n.sum_gradients.hess();
            nid++;
        }
    }
}
}  // namespace tree
}  // namespace xgboost
