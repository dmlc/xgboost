/*!
 * Copyright 2017-2018 by Contributors
 */
#include <dmlc/parameter.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <xgboost/data.h>
#include <xgboost/predictor.h>
#include <xgboost/tree_model.h>
#include <xgboost/tree_updater.h>
#include <memory>
#include "../common/common.h"
#include "../common/device_helpers.cuh"
#include "../common/host_device_vector.h"

namespace xgboost {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(gpu_predictor);

template <typename IterT>
void IncrementOffset(IterT begin_itr, IterT end_itr, size_t amount) {
  thrust::transform(begin_itr, end_itr, begin_itr,
                    [=] __device__(size_t elem) { return elem + amount; });
}

/**
 * \struct  DevicePredictionNode
 *
 * \brief Packed 16 byte representation of a tree node for use in device
 * prediction
 */
struct DevicePredictionNode {
  XGBOOST_DEVICE DevicePredictionNode()
      : fidx{-1}, left_child_idx{-1}, right_child_idx{-1} {}

  union NodeValue {
    float leaf_weight;
    float fvalue;
  };

  int fidx;
  int left_child_idx;
  int right_child_idx;
  NodeValue val;

  DevicePredictionNode(const RegTree::Node& n) {  // NOLINT
    static_assert(sizeof(DevicePredictionNode) == 16, "Size is not 16 bytes");
    this->left_child_idx = n.LeftChild();
    this->right_child_idx = n.RightChild();
    this->fidx = n.SplitIndex();
    if (n.DefaultLeft()) {
      fidx |= (1U << 31);
    }

    if (n.IsLeaf()) {
      this->val.leaf_weight = n.LeafValue();
    } else {
      this->val.fvalue = n.SplitCond();
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
  common::Span<const size_t> d_row_ptr;
  common::Span<const Entry> d_data;
  int num_features;
  float* smem;
  size_t entry_start;

  __device__ ElementLoader(bool use_shared, common::Span<const size_t> row_ptr,
                           common::Span<const Entry> entry, int num_features,
                           float* smem, int num_rows, size_t entry_start)
      : use_shared(use_shared),
        d_row_ptr(row_ptr),
        d_data(entry),
        num_features(num_features),
        smem(smem),
        entry_start(entry_start) {
    // Copy instances
    if (use_shared) {
      bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
      int shared_elements = blockDim.x * num_features;
      dh::BlockFill(smem, shared_elements, nanf(""));
      __syncthreads();
      if (global_idx < num_rows) {
        bst_uint elem_begin = d_row_ptr[global_idx];
        bst_uint elem_end = d_row_ptr[global_idx + 1];
        for (bst_uint elem_idx = elem_begin; elem_idx < elem_end; elem_idx++) {
          Entry elem = d_data[elem_idx - entry_start];
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
      auto begin_ptr = d_data.begin() + (d_row_ptr[ridx] - entry_start);
      auto end_ptr = d_data.begin() + (d_row_ptr[ridx + 1] - entry_start);
      common::Span<const Entry>::iterator previous_middle;
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
__global__ void PredictKernel(common::Span<const DevicePredictionNode> d_nodes,
                              common::Span<float> d_out_predictions,
                              common::Span<size_t> d_tree_segments,
                              common::Span<int> d_tree_group,
                              common::Span<const size_t> d_row_ptr,
                              common::Span<const Entry> d_data, size_t tree_begin,
                              size_t tree_end, size_t num_features,
                              size_t num_rows, size_t entry_start,
                              bool use_shared, int num_group) {
  extern __shared__ float smem[];
  bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  ElementLoader loader(use_shared, d_row_ptr, d_data, num_features, smem,
                       num_rows, entry_start);
  if (global_idx >= num_rows) return;
  if (num_group == 1) {
    float sum = 0;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      const DevicePredictionNode* d_tree =
          &d_nodes[d_tree_segments[tree_idx - tree_begin]];
      sum += GetLeafWeight(global_idx, d_tree, &loader);
    }
    d_out_predictions[global_idx] += sum;
  } else {
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      int tree_group = d_tree_group[tree_idx];
      const DevicePredictionNode* d_tree =
          &d_nodes[d_tree_segments[tree_idx - tree_begin]];
      bst_uint out_prediction_idx = global_idx * num_group + tree_group;
      d_out_predictions[out_prediction_idx] +=
          GetLeafWeight(global_idx, d_tree, &loader);
    }
  }
}

class GPUPredictor : public xgboost::Predictor {
 protected:
  struct DevicePredictionCacheEntry {
    std::shared_ptr<DMatrix> data;
    HostDeviceVector<bst_float> predictions;
  };

 private:
  struct DeviceShard {
    DeviceShard() : device_{-1} {}

    ~DeviceShard() {
      dh::safe_cuda(cudaSetDevice(device_));
    }

    void Init(int device) {
      this->device_ = device;
      max_shared_memory_bytes_ = dh::MaxSharedMemory(this->device_);
     }

    void InitModel(const gbm::GBTreeModel& model,
     const thrust::host_vector<size_t>& h_tree_segments,
     const thrust::host_vector<DevicePredictionNode>& h_nodes,
     size_t tree_begin, size_t tree_end) {
      dh::safe_cuda(cudaSetDevice(device_));
      nodes_.resize(h_nodes.size());
      dh::safe_cuda(cudaMemcpyAsync(nodes_.data().get(), h_nodes.data(),
                                    sizeof(DevicePredictionNode) * h_nodes.size(),
                                    cudaMemcpyHostToDevice));
      tree_segments_.resize(h_tree_segments.size());
      dh::safe_cuda(cudaMemcpyAsync(tree_segments_.data().get(), h_tree_segments.data(),
                                    sizeof(size_t) * h_tree_segments.size(),
                                    cudaMemcpyHostToDevice));
      tree_group_.resize(model.tree_info.size());
      dh::safe_cuda(cudaMemcpyAsync(tree_group_.data().get(), model.tree_info.data(),
                                    sizeof(int) * model.tree_info.size(),
                                    cudaMemcpyHostToDevice));
      this->tree_begin_ = tree_begin;
      this->tree_end_ = tree_end;
      this->num_group_ = model.param.num_output_group;
    }

    void PredictInternal
    (const SparsePage& batch, size_t num_features,
     HostDeviceVector<bst_float>* predictions) {
      if (predictions->DeviceSize() == 0) { return; }
      dh::safe_cuda(cudaSetDevice(device_));
      const int BLOCK_THREADS = 128;
      size_t num_rows = batch.offset.DeviceSize() - 1;
      const int GRID_SIZE = static_cast<int>(common::DivRoundUp(num_rows, BLOCK_THREADS));

      int shared_memory_bytes = static_cast<int>
        (sizeof(float) * num_features * BLOCK_THREADS);
      bool use_shared = true;
      if (shared_memory_bytes > max_shared_memory_bytes_) {
        shared_memory_bytes = 0;
        use_shared = false;
      }
      size_t entry_start = 0;

      PredictKernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS, shared_memory_bytes>>>
        (dh::ToSpan(nodes_), predictions->DeviceSpan(), dh::ToSpan(tree_segments_),
         dh::ToSpan(tree_group_), batch.offset.DeviceSpan(),
         batch.data.DeviceSpan(), this->tree_begin_, this->tree_end_, num_features,
         num_rows, entry_start, use_shared, this->num_group_);
    }

   private:
    int device_;
    dh::device_vector<DevicePredictionNode> nodes_;
    dh::device_vector<size_t> tree_segments_;
    dh::device_vector<int> tree_group_;
    size_t max_shared_memory_bytes_;
    size_t tree_begin_;
    size_t tree_end_;
    int num_group_;
  };

  void InitModel(const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end) {
    CHECK_EQ(model.param.size_leaf_vector, 0);
    // Copy decision trees to device
    thrust::host_vector<size_t> h_tree_segments;
    h_tree_segments.reserve((tree_end - tree_begin) + 1);
    size_t sum = 0;
    h_tree_segments.push_back(sum);
    for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      sum += model.trees.at(tree_idx)->GetNodes().size();
      h_tree_segments.push_back(sum);
    }

    thrust::host_vector<DevicePredictionNode> h_nodes(h_tree_segments.back());
    for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees.at(tree_idx)->GetNodes();
      std::copy(src_nodes.begin(), src_nodes.end(),
                h_nodes.begin() + h_tree_segments[tree_idx - tree_begin]);
    }
    dh::ExecuteIndexShards(&shards_, [&](int idx, DeviceShard &shard) {
      shard.InitModel(model, h_tree_segments, h_nodes, tree_begin, tree_end);
    });
  }

  void DevicePredictInternal(DMatrix* dmat,
                             HostDeviceVector<bst_float>* out_preds,
                             const gbm::GBTreeModel& model, size_t tree_begin,
                             size_t tree_end) {
    if (tree_end - tree_begin == 0) { return; }
    monitor_.StartCuda("DevicePredictInternal");

    InitModel(model, tree_begin, tree_end);

    size_t batch_offset = 0;
    auto* preds = out_preds;
    std::unique_ptr<HostDeviceVector<bst_float>> batch_preds{nullptr};
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      bool is_external_memory = batch.Size() < dmat->Info().num_row_;
      if (is_external_memory) {
        batch_preds.reset(new HostDeviceVector<bst_float>);
        batch_preds->Resize(batch.Size() * model.param.num_output_group);
        std::copy(out_preds->ConstHostVector().begin() + batch_offset,
                  out_preds->ConstHostVector().begin() + batch_offset + batch_preds->Size(),
                  batch_preds->HostVector().begin());
        batch_preds->SetDevice(device_);
        preds = batch_preds.get();
      }

      batch.offset.SetDevice(device_);
      batch.data.SetDevice(device_);
      dh::ExecuteIndexShards(&shards_, [&](int idx, DeviceShard& shard) {
        shard.PredictInternal(batch, model.param.num_feature, preds);
      });

      if (is_external_memory) {
        auto h_preds = preds->ConstHostVector();
        std::copy(h_preds.begin(), h_preds.end(), out_preds->HostVector().begin() + batch_offset);
      }
      batch_offset += batch.Size() * model.param.num_output_group;
    }

    monitor_.StopCuda("DevicePredictInternal");
  }

 public:
  GPUPredictor() : device_{-1} {};

  void PredictBatch(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                    const gbm::GBTreeModel& model, int tree_begin,
                    unsigned ntree_limit = 0) override {
    int device = learner_param_->gpu_id;
    CHECK_GE(device, 0);
    ConfigureShards(device);

    if (this->PredictFromCache(dmat, out_preds, model, ntree_limit)) {
      return;
    }
    this->InitOutPredictions(dmat->Info(), out_preds, model);

    int tree_end = ntree_limit * model.param.num_output_group;

    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      tree_end = static_cast<unsigned>(model.trees.size());
    }

    DevicePredictInternal(dmat, out_preds, model, tree_begin, tree_end);
  }

 protected:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const {
    size_t n_classes = model.param.num_output_group;
    size_t n = n_classes * info.num_row_;
    const HostDeviceVector<bst_float>& base_margin = info.base_margin_;
    out_preds->Resize(n);
    if (base_margin.Size() != 0) {
      CHECK_EQ(base_margin.Size(), n);
      out_preds->Copy(base_margin);
    } else {
      out_preds->Fill(model.base_margin);
    }
    out_preds->SetDevice(device_);
  }

  bool PredictFromCache(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                        const gbm::GBTreeModel& model, unsigned ntree_limit) {
    if (ntree_limit == 0 ||
        ntree_limit * model.param.num_output_group >= model.trees.size()) {
      auto it = cache_.find(dmat);
      if (it != cache_.end()) {
        const HostDeviceVector<bst_float>& y = it->second.predictions;
        if (y.Size() != 0) {
          monitor_.StartCuda("PredictFromCache");
          out_preds->SetDevice(y.DeviceIdx());
          out_preds->Resize(y.Size());
          out_preds->Copy(y);
          monitor_.StopCuda("PredictFromCache");
          return true;
        }
      }
    }
    return false;
  }

  void UpdatePredictionCache(
      const gbm::GBTreeModel& model,
      std::vector<std::unique_ptr<TreeUpdater>>* updaters,
      int num_new_trees) override {
    auto old_ntree = model.trees.size() - num_new_trees;
    // update cache entry
    for (auto& kv : cache_) {
      PredictionCacheEntry& e = kv.second;
      DMatrix* dmat = kv.first;
      HostDeviceVector<bst_float>& predictions = e.predictions;

      if (predictions.Size() == 0) {
        this->InitOutPredictions(dmat->Info(), &predictions, model);
      }

      if (model.param.num_output_group == 1 && updaters->size() > 0 &&
          num_new_trees == 1 &&
          updaters->back()->UpdatePredictionCache(e.data.get(), &predictions)) {
        // do nothing
      } else {
        DevicePredictInternal(dmat, &predictions, model, old_ntree, model.trees.size());
      }
    }
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                       unsigned root_index) override {
    LOG(FATAL) << "Internal error: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void PredictLeaf(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model,
                   unsigned ntree_limit) override {
    LOG(FATAL) << "Internal error: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model, unsigned ntree_limit,
                           bool approximate, int condition,
                           unsigned condition_feature) override {
    LOG(FATAL) << "Internal error: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model,
                                       unsigned ntree_limit,
                                       bool approximate) override {
    LOG(FATAL) << "Internal error: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void Configure(const std::vector<std::pair<std::string, std::string>>& cfg,
                 const std::vector<std::shared_ptr<DMatrix>>& cache) override {
    Predictor::Configure(cfg, cache);

    int device = learner_param_->gpu_id;
    if (device >= 0) {
      ConfigureShards(device);
    }
  }

 private:
  /*! \brief Re configure shards when GPUSet is changed. */
  void ConfigureShards(int device) {
    if (device_ == device) return;

    device_ = device;
    shards_.clear();
    shards_.resize(1);
    dh::ExecuteIndexShards(&shards_, [=](size_t i, DeviceShard& shard){
        shard.Init(device_);
      });
  }

  std::vector<DeviceShard> shards_;
  int device_;
  common::Monitor monitor_;
};

XGBOOST_REGISTER_PREDICTOR(GPUPredictor, "gpu_predictor")
    .describe("Make predictions using GPU.")
    .set_body([]() { return new GPUPredictor(); });

}  // namespace predictor
}  // namespace xgboost
