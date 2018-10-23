/*!
 * Copyright by Contributors 2017
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

/*! \brief prediction parameters */
struct GPUPredictionParam : public dmlc::Parameter<GPUPredictionParam> {
  int gpu_id;
  int n_gpus;
  bool silent;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GPUPredictionParam) {
    DMLC_DECLARE_FIELD(gpu_id).set_lower_bound(0).set_default(0).describe(
        "Device ordinal for GPU prediction.");
    DMLC_DECLARE_FIELD(n_gpus).set_lower_bound(-1).set_default(1).describe(
        "Number of devices to use for prediction.");
    DMLC_DECLARE_FIELD(silent).set_default(false).describe(
        "Do not print information during trainig.");
  }
};
DMLC_REGISTER_PARAMETER(GPUPredictionParam);

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
  void DeviceOffsets(const HostDeviceVector<size_t>& data, std::vector<size_t>* out_offsets) {
    auto& offsets = *out_offsets;
    offsets.resize(devices_.Size() + 1);
    offsets[0] = 0;
#pragma omp parallel for schedule(static, 1) if (devices_.Size() > 1)
    for (int shard = 0; shard < devices_.Size(); ++shard) {
      int device = devices_[shard];
      auto data_span = data.DeviceSpan(device);
      dh::safe_cuda(cudaSetDevice(device));
      // copy the last element from every shard
      dh::safe_cuda(cudaMemcpy(&offsets.at(shard + 1),
                               &data_span[data_span.size()-1],
                               sizeof(size_t), cudaMemcpyDeviceToHost));
    }
  }

  struct DeviceShard {
    DeviceShard() : device_(-1) {}
    void Init(int device) {
      this->device_ = device;
      max_shared_memory_bytes = dh::MaxSharedMemory(this->device_);
     }
    void PredictInternal
    (const SparsePage& batch, const MetaInfo& info,
     HostDeviceVector<bst_float>* predictions,
     const gbm::GBTreeModel& model,
     const thrust::host_vector<size_t>& h_tree_segments,
     const thrust::host_vector<DevicePredictionNode>& h_nodes,
     size_t tree_begin, size_t tree_end) {
      dh::safe_cuda(cudaSetDevice(device_));
      nodes.resize(h_nodes.size());
      dh::safe_cuda(cudaMemcpy(dh::Raw(nodes), h_nodes.data(),
                               sizeof(DevicePredictionNode) * h_nodes.size(),
                               cudaMemcpyHostToDevice));
      tree_segments.resize(h_tree_segments.size());

      dh::safe_cuda(cudaMemcpy(dh::Raw(tree_segments), h_tree_segments.data(),
                               sizeof(size_t) * h_tree_segments.size(),
                               cudaMemcpyHostToDevice));
      tree_group.resize(model.tree_info.size());

      dh::safe_cuda(cudaMemcpy(dh::Raw(tree_group), model.tree_info.data(),
                               sizeof(int) * model.tree_info.size(),
                               cudaMemcpyHostToDevice));

      const int BLOCK_THREADS = 128;
      size_t num_rows = batch.offset.DeviceSize(device_) - 1;

      const int GRID_SIZE = static_cast<int>(dh::DivRoundUp(num_rows, BLOCK_THREADS));

      int shared_memory_bytes = static_cast<int>
        (sizeof(float) * info.num_col_ * BLOCK_THREADS);
      bool use_shared = true;
      if (shared_memory_bytes > max_shared_memory_bytes) {
        shared_memory_bytes = 0;
        use_shared = false;
      }
      const auto& data_distr = batch.data.Distribution();
      int index = data_distr.Devices().Index(device_);
      size_t entry_start = data_distr.ShardStart(batch.data.Size(), index);

      PredictKernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS, shared_memory_bytes>>>
        (dh::ToSpan(nodes), predictions->DeviceSpan(device_), dh::ToSpan(tree_segments),
         dh::ToSpan(tree_group), batch.offset.DeviceSpan(device_),
         batch.data.DeviceSpan(device_), tree_begin, tree_end, info.num_col_,
         num_rows, entry_start, use_shared, model.param.num_output_group);

      dh::safe_cuda(cudaDeviceSynchronize());
    }

    int device_;
    thrust::device_vector<DevicePredictionNode> nodes;
    thrust::device_vector<size_t> tree_segments;
    thrust::device_vector<int> tree_group;
    size_t max_shared_memory_bytes;
  };

  void DevicePredictInternal(DMatrix* dmat,
                             HostDeviceVector<bst_float>* out_preds,
                             const gbm::GBTreeModel& model, size_t tree_begin,
                             size_t tree_end) {
    if (tree_end - tree_begin == 0) { return; }

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

    size_t i_batch = 0;

    for (const auto &batch : dmat->GetRowBatches()) {
      CHECK_EQ(i_batch, 0) << "External memory not supported";
      size_t n_rows = batch.offset.Size() - 1;
      // out_preds have been resharded and resized in InitOutPredictions()
      batch.offset.Reshard(GPUDistribution::Overlap(devices_, 1));
      std::vector<size_t> device_offsets;
      DeviceOffsets(batch.offset, &device_offsets);
      batch.data.Reshard(GPUDistribution::Explicit(devices_, device_offsets));
      dh::ExecuteShards(&shards, [&](DeviceShard& shard){
          shard.PredictInternal(batch, dmat->Info(), out_preds, model, h_tree_segments,
                                h_nodes, tree_begin, tree_end);
        });
      i_batch++;
    }
  }

 public:
  GPUPredictor() : cpu_predictor(Predictor::Create("cpu_predictor")) {}

  void PredictBatch(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                    const gbm::GBTreeModel& model, int tree_begin,
                    unsigned ntree_limit = 0) override {
    GPUSet devices = GPUSet::All(
        param.n_gpus, dmat->Info().num_row_).Normalised(param.gpu_id);
    ConfigureShards(devices);

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
    out_preds->Reshard(GPUDistribution::Granular(devices_, n_classes));
    out_preds->Resize(n);
    if (base_margin.Size() != 0) {
      CHECK_EQ(out_preds->Size(), n);
      out_preds->Copy(base_margin);
    } else {
      out_preds->Fill(model.base_margin);
    }
  }

  bool PredictFromCache(DMatrix* dmat, HostDeviceVector<bst_float>* out_preds,
                        const gbm::GBTreeModel& model, unsigned ntree_limit) {
    if (ntree_limit == 0 ||
        ntree_limit * model.param.num_output_group >= model.trees.size()) {
      auto it = cache_.find(dmat);
      if (it != cache_.end()) {
        const HostDeviceVector<bst_float>& y = it->second.predictions;
        if (y.Size() != 0) {
          out_preds->Reshard(y.Distribution());
          out_preds->Resize(y.Size());
          out_preds->Copy(y);
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
    cpu_predictor->PredictInstance(inst, out_preds, model, root_index);
  }
  void PredictLeaf(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model,
                   unsigned ntree_limit) override {
    cpu_predictor->PredictLeaf(p_fmat, out_preds, model, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model, unsigned ntree_limit,
                           bool approximate, int condition,
                           unsigned condition_feature) override {
    cpu_predictor->PredictContribution(p_fmat, out_contribs, model, ntree_limit,
                                       approximate, condition,
                                       condition_feature);
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model,
                                       unsigned ntree_limit,
                                       bool approximate) override {
    cpu_predictor->PredictInteractionContributions(p_fmat, out_contribs, model,
                                                   ntree_limit, approximate);
  }

  void Init(const std::vector<std::pair<std::string, std::string>>& cfg,
            const std::vector<std::shared_ptr<DMatrix>>& cache) override {
    Predictor::Init(cfg, cache);
    cpu_predictor->Init(cfg, cache);
    param.InitAllowUnknown(cfg);

    GPUSet devices = GPUSet::All(param.n_gpus).Normalised(param.gpu_id);
    ConfigureShards(devices);
  }

 private:
  /*! \brief Re configure shards when GPUSet is changed. */
  void ConfigureShards(GPUSet devices) {
    if (devices_ == devices) return;

    devices_ = devices;
    shards.clear();
    shards.resize(devices_.Size());
    dh::ExecuteIndexShards(&shards, [=](size_t i, DeviceShard& shard){
        shard.Init(devices_[i]);
      });
  }

  GPUPredictionParam param;
  std::unique_ptr<Predictor> cpu_predictor;
  std::vector<DeviceShard> shards;
  GPUSet devices_;
};

XGBOOST_REGISTER_PREDICTOR(GPUPredictor, "gpu_predictor")
    .describe("Make predictions using GPU.")
    .set_body([]() { return new GPUPredictor(); });

}  // namespace predictor
}  // namespace xgboost
