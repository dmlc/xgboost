/*!
 * Copyright by Contributors 2017
 */
#include <dmlc/parameter.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <xgboost/data.h>
#include <xgboost/predictor.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>
#include <memory>
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(gpu_predictor);

/*! \brief prediction parameters */
struct GPUPredictionParam : public dmlc::Parameter<GPUPredictionParam> {
  int gpu_id;
  int n_gpus;
  bool silent;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GPUPredictionParam) {
    DMLC_DECLARE_FIELD(gpu_id).set_default(0).describe(
        "Device ordinal for GPU prediction.");
    DMLC_DECLARE_FIELD(n_gpus).set_default(1).describe(
        "Number of devices to use for prediction (NOT IMPLEMENTED).");
    DMLC_DECLARE_FIELD(silent).set_default(false).describe(
        "Do not print information during trainig.");
  }
};
DMLC_REGISTER_PARAMETER(GPUPredictionParam);

template <typename iter_t>
void increment_offset(iter_t begin_itr, iter_t end_itr, size_t amount) {
  thrust::transform(begin_itr, end_itr, begin_itr,
                    [=] __device__(size_t elem) { return elem + amount; });
}

/**
 * \struct  DeviceMatrix
 *
 * \brief A csr representation of the input matrix allocated on the device.
 */

struct DeviceMatrix {
  DMatrix* p_mat;  // Pointer to the original matrix on the host
  dh::bulk_allocator<dh::memory_type::DEVICE> ba;
  dh::dvec<size_t> row_ptr;
  dh::dvec<SparseBatch::Entry> data;
  thrust::device_vector<float> predictions;

  DeviceMatrix(DMatrix* dmat, int device_idx, bool silent) : p_mat(dmat) {
    dh::safe_cuda(cudaSetDevice(device_idx));
    auto info = dmat->info();
    ba.allocate(device_idx, silent, &row_ptr, info.num_row + 1, &data,
                info.num_nonzero);
    auto iter = dmat->RowIterator();
    iter->BeforeFirst();
    size_t data_offset = 0;
    while (iter->Next()) {
      auto batch = iter->Value();
      // Copy row ptr
      thrust::copy(batch.ind_ptr, batch.ind_ptr + batch.size + 1,
                   row_ptr.tbegin() + batch.base_rowid);
      if (batch.base_rowid > 0) {
        auto begin_itr = row_ptr.tbegin() + batch.base_rowid;
        auto end_itr = begin_itr + batch.size + 1;
        increment_offset(begin_itr, end_itr, batch.base_rowid);
      }
      // Copy data
      thrust::copy(batch.data_ptr, batch.data_ptr + batch.ind_ptr[batch.size],
                   data.tbegin() + data_offset);
      data_offset += batch.ind_ptr[batch.size];
    }
  }
};

/**
 * \struct  DevicePredictionNode
 *
 * \brief Packed 16 byte representation of a tree node for use in device
 * prediction
 */

struct DevicePredictionNode {
  XGBOOST_DEVICE DevicePredictionNode()
      : fidx(-1), left_child_idx(-1), right_child_idx(-1) {}

  union NodeValue {
    float leaf_weight;
    float fvalue;
  };

  int fidx;
  int left_child_idx;
  int right_child_idx;
  NodeValue val;

  DevicePredictionNode(const RegTree::Node& n) {  // NOLINT
    this->left_child_idx = n.cleft();
    this->right_child_idx = n.cright();
    this->fidx = n.split_index();
    if (n.default_left()) {
      fidx |= (1U << 31);
    }

    if (n.is_leaf()) {
      this->val.leaf_weight = n.leaf_value();
    } else {
      this->val.fvalue = n.split_cond();
    }
  }

  XGBOOST_DEVICE bool IsLeaf() const { return left_child_idx == -1; }

  XGBOOST_DEVICE int GetFidx() const { return fidx & ((1U << 31) - 1U); }

  XGBOOST_DEVICE bool MissingLeft() const { return (fidx >> 31) != 0; }

  XGBOOST_DEVICE int MissingIdx() const {
    if (MissingLeft()) {
      return this->left_child_idx;
    } else {
      return this->right_child_idx;
    }
  }

  XGBOOST_DEVICE float GetFvalue() const { return val.fvalue; }

  XGBOOST_DEVICE float GetWeight() const { return val.leaf_weight; }
};

struct ElementLoader {
  bool use_shared;
  size_t* d_row_ptr;
  SparseBatch::Entry* d_data;
  int num_features;
  float* smem;

  __device__ ElementLoader(bool use_shared, size_t* row_ptr,
                           SparseBatch::Entry* entry, int num_features,
                           float* smem, int num_rows)
      : use_shared(use_shared),
        d_row_ptr(row_ptr),
        d_data(entry),
        num_features(num_features),
        smem(smem) {
    // Copy instances
    if (use_shared) {
      bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
      int shared_elements = blockDim.x * num_features;
      dh::block_fill(smem, shared_elements, nanf(""));
      __syncthreads();
      if (global_idx < num_rows) {
        bst_uint elem_begin = d_row_ptr[global_idx];
        bst_uint elem_end = d_row_ptr[global_idx + 1];
        for (bst_uint elem_idx = elem_begin; elem_idx < elem_end; elem_idx++) {
          SparseBatch::Entry elem = d_data[elem_idx];
          smem[threadIdx.x * num_features + elem.index] = elem.fvalue;
        }
      }
      __syncthreads();
    }
  }
  __device__ float GetFvalue(int ridx, int fidx) {
    if (use_shared) {
      return smem[threadIdx.x * num_features + fidx];
    } else {
      // Binary search
      auto begin_ptr = d_data + d_row_ptr[ridx];
      auto end_ptr = d_data + d_row_ptr[ridx + 1];
      SparseBatch::Entry* previous_middle = nullptr;
      while (end_ptr != begin_ptr) {
        auto middle = begin_ptr + (end_ptr - begin_ptr) / 2;
        if (middle == previous_middle) {
          break;
        } else {
          previous_middle = middle;
        }

        if (middle->index == fidx) {
          return middle->fvalue;
        } else if (middle->index < fidx) {
          begin_ptr = middle;
        } else {
          end_ptr = middle;
        }
      }
      // Value is missing
      return nanf("");
    }
  }
};

__device__ float GetLeafWeight(bst_uint ridx, const DevicePredictionNode* tree,
                               ElementLoader* loader) {
  DevicePredictionNode n = tree[0];
  while (!n.IsLeaf()) {
    float fvalue = loader->GetFvalue(ridx, n.GetFidx());
    // Missing value
    if (isnan(fvalue)) {
      n = tree[n.MissingIdx()];
    } else {
      if (fvalue < n.GetFvalue()) {
        n = tree[n.left_child_idx];
      } else {
        n = tree[n.right_child_idx];
      }
    }
  }
  return n.GetWeight();
}

template <int BLOCK_THREADS>
__global__ void PredictKernel(const DevicePredictionNode* d_nodes,
                              float* d_out_predictions, int* d_tree_segments,
                              int* d_tree_group, size_t* d_row_ptr,
                              SparseBatch::Entry* d_data, int tree_begin,
                              int tree_end, int num_features, bst_uint num_rows,
                              bool use_shared, int num_group) {
  extern __shared__ float smem[];
  bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  ElementLoader loader(use_shared, d_row_ptr, d_data, num_features, smem,
                       num_rows);
  if (global_idx >= num_rows) return;
  if (num_group == 1) {
    float sum = 0;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      const DevicePredictionNode* d_tree =
          d_nodes + d_tree_segments[tree_idx - tree_begin];
      sum += GetLeafWeight(global_idx, d_tree, &loader);
    }
    d_out_predictions[global_idx] += sum;
  } else {
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      int tree_group = d_tree_group[tree_idx];
      const DevicePredictionNode* d_tree =
          d_nodes + d_tree_segments[tree_idx - tree_begin];
      bst_uint out_prediction_idx = global_idx * num_group + tree_group;
      d_out_predictions[out_prediction_idx] +=
          GetLeafWeight(global_idx, d_tree, &loader);
    }
  }
}

class GPUPredictor : public xgboost::Predictor {
 private:
  void DevicePredictInternal(DMatrix* dmat, std::vector<bst_float>* out_preds,
                             const gbm::GBTreeModel& model, int tree_begin,
                             int tree_end) {
    if (tree_end - tree_begin == 0) {
      return;
    }

    // Add dmatrix to device if not seen before
    if (this->device_matrix_cache_.find(dmat) ==
        this->device_matrix_cache_.end()) {
      this->device_matrix_cache_.emplace(
          dmat, std::unique_ptr<DeviceMatrix>(
                    new DeviceMatrix(dmat, param.gpu_id, param.silent)));
    }
    DeviceMatrix* device_matrix = device_matrix_cache_.find(dmat)->second.get();

    dh::safe_cuda(cudaSetDevice(param.gpu_id));
    CHECK_EQ(model.param.size_leaf_vector, 0);
    // Copy decision trees to device
    thrust::host_vector<int> h_tree_segments;
    h_tree_segments.reserve((tree_end - tree_end) + 1);
    int sum = 0;
    h_tree_segments.push_back(sum);
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      sum += model.trees[tree_idx]->GetNodes().size();
      h_tree_segments.push_back(sum);
    }

    thrust::host_vector<DevicePredictionNode> h_nodes(h_tree_segments.back());
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees[tree_idx]->GetNodes();
      std::copy(src_nodes.begin(), src_nodes.end(),
                h_nodes.begin() + h_tree_segments[tree_idx - tree_begin]);
    }

    nodes.resize(h_nodes.size());
    thrust::copy(h_nodes.begin(), h_nodes.end(), nodes.begin());
    tree_segments.resize(h_tree_segments.size());
    thrust::copy(h_tree_segments.begin(), h_tree_segments.end(),
                 tree_segments.begin());
    tree_group.resize(model.tree_info.size());
    thrust::copy(model.tree_info.begin(), model.tree_info.end(),
                 tree_group.begin());

    if (device_matrix->predictions.size() != out_preds->size()) {
      device_matrix->predictions.resize(out_preds->size());
      thrust::copy(out_preds->begin(), out_preds->end(),
                   device_matrix->predictions.begin());
    }

    const int BLOCK_THREADS = 128;
    const int GRID_SIZE =
        dh::div_round_up(device_matrix->row_ptr.size() - 1, BLOCK_THREADS);

    int shared_memory_bytes =
        sizeof(float) * device_matrix->p_mat->info().num_col * BLOCK_THREADS;
    bool use_shared = true;
    if (shared_memory_bytes > dh::max_shared_memory(param.gpu_id)) {
      shared_memory_bytes = 0;
      use_shared = false;
    }

    PredictKernel<BLOCK_THREADS>
        <<<GRID_SIZE, BLOCK_THREADS, shared_memory_bytes>>>(
            dh::raw(nodes), dh::raw(device_matrix->predictions),
            dh::raw(tree_segments), dh::raw(tree_group),
            device_matrix->row_ptr.data(), device_matrix->data.data(),
            tree_begin, tree_end, device_matrix->p_mat->info().num_col,
            device_matrix->p_mat->info().num_row, use_shared,
            model.param.num_output_group);

    dh::safe_cuda(cudaDeviceSynchronize());
    thrust::copy(device_matrix->predictions.begin(),
                 device_matrix->predictions.end(), out_preds->begin());
  }

 public:
  GPUPredictor() : cpu_predictor(Predictor::Create("cpu_predictor")) {}

  void PredictBatch(DMatrix* dmat, std::vector<bst_float>* out_preds,
                    const gbm::GBTreeModel& model, int tree_begin,
                    unsigned ntree_limit = 0) override {
    if (this->PredictFromCache(dmat, out_preds, model, ntree_limit)) {
      return;
    }
    this->InitOutPredictions(dmat->info(), out_preds, model);

    int tree_end = ntree_limit * model.param.num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      tree_end = static_cast<unsigned>(model.trees.size());
    }

    DevicePredictInternal(dmat, out_preds, model, tree_begin, tree_end);
  }

  void UpdatePredictionCache(
      const gbm::GBTreeModel& model,
      std::vector<std::unique_ptr<TreeUpdater>>* updaters,
      int num_new_trees) override {
    // dh::Timer t;
    int old_ntree = model.trees.size() - num_new_trees;
    // update cache entry
    for (auto& kv : cache_) {
      PredictionCacheEntry& e = kv.second;
      DMatrix* dmat = kv.first;

      if (e.predictions.size() == 0) {
        cpu_predictor->PredictBatch(dmat, &(e.predictions), model, 0,
                                    model.trees.size());
      } else if (model.param.num_output_group == 1 && updaters->size() > 0 &&
                 num_new_trees == 1 &&
                 updaters->back()->UpdatePredictionCache(e.data.get(),
                                                         &(e.predictions))) {
        {}  // do nothing
      } else {
        DevicePredictInternal(dmat, &(e.predictions), model, old_ntree,
                              model.trees.size());
      }
    }
  }

  void PredictInstance(const SparseBatch::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                       unsigned root_index) override {
    cpu_predictor->PredictInstance(inst, out_preds, model, root_index);
  }
  void PredictLeaf(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model,
                   unsigned ntree_limit) override {
    cpu_predictor->PredictLeaf(p_fmat, out_preds, model, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model,
                           unsigned ntree_limit,
                           bool approximate) override {
    cpu_predictor->PredictContribution(p_fmat, out_contribs, model,
                                       ntree_limit, approximate);
  }

  void Init(const std::vector<std::pair<std::string, std::string>>& cfg,
            const std::vector<std::shared_ptr<DMatrix>>& cache) override {
    Predictor::Init(cfg, cache);
    cpu_predictor->Init(cfg, cache);
    param.InitAllowUnknown(cfg);
  }

 private:
  GPUPredictionParam param;
  std::unique_ptr<Predictor> cpu_predictor;
  std::unordered_map<DMatrix*, std::unique_ptr<DeviceMatrix>>
      device_matrix_cache_;
  thrust::device_vector<DevicePredictionNode> nodes;
  thrust::device_vector<int> tree_segments;
  thrust::device_vector<int> tree_group;
};
XGBOOST_REGISTER_PREDICTOR(GPUPredictor, "gpu_predictor")
    .describe("Make predictions using GPU.")
    .set_body([]() { return new GPUPredictor(); });
}  // namespace predictor
}  // namespace xgboost
