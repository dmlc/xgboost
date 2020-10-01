/*!
 * Copyright 2017-2020 by Contributors
 */
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <memory>

#include "xgboost/data.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"
#include "xgboost/host_device_vector.h"

#include "../gbm/gbtree_model.h"
#include "../data/ellpack_page.cuh"
#include "../data/device_adapter.cuh"
#include "../common/common.h"
#include "../common/device_helpers.cuh"

namespace xgboost {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(gpu_predictor);

struct SparsePageView {
  common::Span<const Entry> d_data;
  common::Span<const bst_row_t> d_row_ptr;

  XGBOOST_DEVICE SparsePageView(common::Span<const Entry> data,
                                common::Span<const bst_row_t> row_ptr) :
      d_data{data}, d_row_ptr{row_ptr} {}
};

struct SparsePageLoader {
  bool use_shared;
  common::Span<const bst_row_t> d_row_ptr;
  common::Span<const Entry> d_data;
  bst_feature_t num_features;
  float* smem;
  size_t entry_start;

  __device__ SparsePageLoader(SparsePageView data, bool use_shared, bst_feature_t num_features,
                              bst_row_t num_rows, size_t entry_start)
      : use_shared(use_shared),
        d_row_ptr(data.d_row_ptr),
        d_data(data.d_data),
        num_features(num_features),
        entry_start(entry_start) {
    extern __shared__ float _smem[];
    smem = _smem;
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
  __device__ float GetFvalue(int ridx, int fidx) const {
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

struct EllpackLoader {
  EllpackDeviceAccessor const& matrix;
  XGBOOST_DEVICE EllpackLoader(EllpackDeviceAccessor const& m, bool use_shared,
                               bst_feature_t num_features, bst_row_t num_rows,
                               size_t entry_start)
      : matrix{m} {}
  __device__ __forceinline__ float GetFvalue(int ridx, int fidx) const {
    auto gidx = matrix.GetBinIndex(ridx, fidx);
    if (gidx == -1) {
      return nan("");
    }
    // The gradient index needs to be shifted by one as min values are not included in the
    // cuts.
    if (gidx == matrix.feature_segments[fidx]) {
      return matrix.min_fvalue[fidx];
    }
    return matrix.gidx_fvalue_map[gidx - 1];
  }
};

template <typename Batch>
struct DeviceAdapterLoader {
  Batch batch;
  bst_feature_t columns;
  float* smem;
  bool use_shared;

  using BatchT = Batch;

  DEV_INLINE DeviceAdapterLoader(Batch const batch, bool use_shared,
                                 bst_feature_t num_features, bst_row_t num_rows,
                                 size_t entry_start) :
    batch{batch},
    columns{num_features},
    use_shared{use_shared} {
      extern __shared__ float _smem[];
      smem = _smem;
      if (use_shared) {
        uint32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
        size_t shared_elements = blockDim.x * num_features;
        dh::BlockFill(smem, shared_elements, nanf(""));
        __syncthreads();
        if (global_idx < num_rows) {
          auto beg = global_idx * columns;
          auto end = (global_idx + 1) * columns;
          for (size_t i = beg; i < end; ++i) {
            smem[threadIdx.x * num_features + (i - beg)] = batch.GetElement(i).value;
          }
        }
      }
      __syncthreads();
    }

  DEV_INLINE float GetFvalue(bst_row_t ridx, bst_feature_t fidx) const {
    if (use_shared) {
      return smem[threadIdx.x * columns + fidx];
    }
    return batch.GetElement(ridx * columns + fidx).value;
  }
};

template <typename Loader>
__device__ float GetLeafWeight(bst_uint ridx, const RegTree::Node* tree,
                               Loader* loader) {
  RegTree::Node n = tree[0];
  while (!n.IsLeaf()) {
    float fvalue = loader->GetFvalue(ridx, n.SplitIndex());
    // Missing value
    if (isnan(fvalue)) {
      n = tree[n.DefaultChild()];
    } else {
      if (fvalue < n.SplitCond()) {
        n = tree[n.LeftChild()];
      } else {
        n = tree[n.RightChild()];
      }
    }
  }
  return n.LeafValue();
}

template <typename Loader, typename Data>
__global__ void PredictKernel(Data data,
                              common::Span<const RegTree::Node> d_nodes,
                              common::Span<float> d_out_predictions,
                              common::Span<size_t> d_tree_segments,
                              common::Span<int> d_tree_group,
                              size_t tree_begin, size_t tree_end, size_t num_features,
                              size_t num_rows, size_t entry_start,
                              bool use_shared, int num_group) {
  bst_uint global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  Loader loader(data, use_shared, num_features, num_rows, entry_start);
  if (global_idx >= num_rows) return;
  if (num_group == 1) {
    float sum = 0;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      const RegTree::Node* d_tree =
          &d_nodes[d_tree_segments[tree_idx - tree_begin]];
      float leaf = GetLeafWeight(global_idx, d_tree, &loader);
      sum += leaf;
    }
    d_out_predictions[global_idx] += sum;
  } else {
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      int tree_group = d_tree_group[tree_idx];
      const RegTree::Node* d_tree =
          &d_nodes[d_tree_segments[tree_idx - tree_begin]];
      bst_uint out_prediction_idx = global_idx * num_group + tree_group;
      d_out_predictions[out_prediction_idx] +=
          GetLeafWeight(global_idx, d_tree, &loader);
    }
  }
}

class DeviceModel {
 public:
  dh::device_vector<RegTree::Node> nodes;
  dh::device_vector<size_t> tree_segments;
  dh::device_vector<int> tree_group;
  size_t tree_beg_;  // NOLINT
  size_t tree_end_;  // NOLINT
  int num_group;

  void CopyModel(const gbm::GBTreeModel& model,
                 const thrust::host_vector<size_t>& h_tree_segments,
                 const thrust::host_vector<RegTree::Node>& h_nodes,
                 size_t tree_begin, size_t tree_end) {
    nodes.resize(h_nodes.size());
    dh::safe_cuda(cudaMemcpyAsync(nodes.data().get(), h_nodes.data(),
                                  sizeof(RegTree::Node) * h_nodes.size(),
                                  cudaMemcpyHostToDevice));
    tree_segments.resize(h_tree_segments.size());
    dh::safe_cuda(cudaMemcpyAsync(tree_segments.data().get(), h_tree_segments.data(),
                                  sizeof(size_t) * h_tree_segments.size(),
                                  cudaMemcpyHostToDevice));
    tree_group.resize(model.tree_info.size());
    dh::safe_cuda(cudaMemcpyAsync(tree_group.data().get(), model.tree_info.data(),
                                  sizeof(int) * model.tree_info.size(),
                                  cudaMemcpyHostToDevice));
    this->tree_beg_ = tree_begin;
    this->tree_end_ = tree_end;
    this->num_group = model.learner_model_param->num_output_group;
  }

  void Init(const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end, int32_t gpu_id) {
    dh::safe_cuda(cudaSetDevice(gpu_id));
    CHECK_EQ(model.param.size_leaf_vector, 0);
    // Copy decision trees to device
    thrust::host_vector<size_t> h_tree_segments{};
    h_tree_segments.reserve((tree_end - tree_begin) + 1);
    size_t sum = 0;
    h_tree_segments.push_back(sum);
    for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      sum += model.trees.at(tree_idx)->GetNodes().size();
      h_tree_segments.push_back(sum);
    }

    thrust::host_vector<RegTree::Node> h_nodes(h_tree_segments.back());
    for (auto tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees.at(tree_idx)->GetNodes();
      std::copy(src_nodes.begin(), src_nodes.end(),
                h_nodes.begin() + h_tree_segments[tree_idx - tree_begin]);
    }
    CopyModel(model, h_tree_segments, h_nodes, tree_begin, tree_end);
  }
};

class GPUPredictor : public xgboost::Predictor {
 private:
  void PredictInternal(const SparsePage& batch, size_t num_features,
                       HostDeviceVector<bst_float>* predictions,
                       size_t batch_offset) {
    batch.offset.SetDevice(generic_param_->gpu_id);
    batch.data.SetDevice(generic_param_->gpu_id);
    const uint32_t BLOCK_THREADS = 128;
    size_t num_rows = batch.Size();
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(num_rows, BLOCK_THREADS));

    auto shared_memory_bytes =
        static_cast<size_t>(sizeof(float) * num_features * BLOCK_THREADS);
    bool use_shared = true;
    if (shared_memory_bytes > max_shared_memory_bytes_) {
      shared_memory_bytes = 0;
      use_shared = false;
    }
    size_t entry_start = 0;
    SparsePageView data{batch.data.DeviceSpan(), batch.offset.DeviceSpan()};
    dh::LaunchKernel {GRID_SIZE, BLOCK_THREADS, shared_memory_bytes} (
        PredictKernel<SparsePageLoader, SparsePageView>,
        data,
        dh::ToSpan(model_.nodes), predictions->DeviceSpan().subspan(batch_offset),
        dh::ToSpan(model_.tree_segments), dh::ToSpan(model_.tree_group),
        model_.tree_beg_, model_.tree_end_, num_features, num_rows,
        entry_start, use_shared, model_.num_group);
  }
  void PredictInternal(EllpackDeviceAccessor const& batch, HostDeviceVector<bst_float>* out_preds,
                       size_t batch_offset) {
    const uint32_t BLOCK_THREADS = 256;
    size_t num_rows = batch.n_rows;
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(num_rows, BLOCK_THREADS));

    bool use_shared = false;
    size_t entry_start = 0;
    dh::LaunchKernel {GRID_SIZE, BLOCK_THREADS} (
        PredictKernel<EllpackLoader, EllpackDeviceAccessor>,
        batch,
        dh::ToSpan(model_.nodes), out_preds->DeviceSpan().subspan(batch_offset),
        dh::ToSpan(model_.tree_segments), dh::ToSpan(model_.tree_group),
        model_.tree_beg_, model_.tree_end_, batch.NumFeatures(), num_rows,
        entry_start, use_shared, model_.num_group);
  }

  void DevicePredictInternal(DMatrix* dmat, HostDeviceVector<float>* out_preds,
                             const gbm::GBTreeModel& model, size_t tree_begin,
                             size_t tree_end) {
    dh::safe_cuda(cudaSetDevice(generic_param_->gpu_id));
    if (tree_end - tree_begin == 0) {
      return;
    }
    model_.Init(model, tree_begin, tree_end, generic_param_->gpu_id);
    out_preds->SetDevice(generic_param_->gpu_id);

    if (dmat->PageExists<SparsePage>()) {
      size_t batch_offset = 0;
      for (auto &batch : dmat->GetBatches<SparsePage>()) {
        this->PredictInternal(batch, model.learner_model_param->num_feature,
                              out_preds, batch_offset);
        batch_offset += batch.Size() * model.learner_model_param->num_output_group;
      }
    } else {
      size_t batch_offset = 0;
      for (auto const& page : dmat->GetBatches<EllpackPage>()) {
        this->PredictInternal(
            page.Impl()->GetDeviceAccessor(generic_param_->gpu_id), out_preds,
            batch_offset);
        batch_offset += page.Impl()->n_rows;
      }
    }
  }

 public:
  explicit GPUPredictor(GenericParameter const* generic_param) :
      Predictor::Predictor{generic_param} {}

  ~GPUPredictor() override {
    if (generic_param_->gpu_id >= 0) {
      dh::safe_cuda(cudaSetDevice(generic_param_->gpu_id));
    }
  }

  void PredictBatch(DMatrix* dmat, PredictionCacheEntry* predts,
                    const gbm::GBTreeModel& model, int tree_begin,
                    unsigned ntree_limit = 0) override {
    // This function is duplicated with CPU predictor PredictBatch, see comments in there.
    // FIXME(trivialfis): Remove the duplication.
    std::lock_guard<std::mutex> const guard(lock_);
    int device = generic_param_->gpu_id;
    CHECK_GE(device, 0) << "Set `gpu_id' to positive value for processing GPU data.";
    ConfigureDevice(device);

    CHECK_EQ(tree_begin, 0);
    auto* out_preds = &predts->predictions;
    CHECK_GE(predts->version, tree_begin);
    if (out_preds->Size() == 0 && dmat->Info().num_row_ != 0) {
      CHECK_EQ(predts->version, 0);
    }
    if (predts->version == 0) {
      this->InitOutPredictions(dmat->Info(), out_preds, model);
    }

    uint32_t const output_groups =  model.learner_model_param->num_output_group;
    CHECK_NE(output_groups, 0);

    uint32_t real_ntree_limit = ntree_limit * output_groups;
    if (real_ntree_limit == 0 || real_ntree_limit > model.trees.size()) {
      real_ntree_limit = static_cast<uint32_t>(model.trees.size());
    }

    uint32_t const end_version = (tree_begin + real_ntree_limit) / output_groups;

    if (predts->version > end_version) {
      CHECK_NE(ntree_limit, 0);
      this->InitOutPredictions(dmat->Info(), out_preds, model);
      predts->version = 0;
    }
    uint32_t const beg_version = predts->version;
    CHECK_LE(beg_version, end_version);

    if (beg_version < end_version) {
      this->DevicePredictInternal(dmat, out_preds, model,
                                  beg_version * output_groups,
                                  end_version * output_groups);
    }

    uint32_t delta = end_version - beg_version;
    CHECK_LE(delta, model.trees.size());
    predts->Update(delta);

    CHECK(out_preds->Size() == output_groups * dmat->Info().num_row_ ||
          out_preds->Size() == dmat->Info().num_row_);
  }

  template <typename Adapter, typename Loader>
  void DispatchedInplacePredict(dmlc::any const &x,
                                const gbm::GBTreeModel &model, float missing,
                                PredictionCacheEntry *out_preds,
                                uint32_t tree_begin, uint32_t tree_end) const {
    auto max_shared_memory_bytes = dh::MaxSharedMemory(this->generic_param_->gpu_id);
    uint32_t const output_groups =  model.learner_model_param->num_output_group;
    DeviceModel d_model;
    d_model.Init(model, tree_begin, tree_end, this->generic_param_->gpu_id);

    auto m = dmlc::get<std::shared_ptr<Adapter>>(x);
    CHECK_EQ(m->NumColumns(), model.learner_model_param->num_feature)
        << "Number of columns in data must equal to trained model.";
    CHECK_EQ(this->generic_param_->gpu_id, m->DeviceIdx())
        << "XGBoost is running on device: " << this->generic_param_->gpu_id << ", "
        << "but data is on: " << m->DeviceIdx();
    MetaInfo info;
    info.num_col_ = m->NumColumns();
    info.num_row_ = m->NumRows();
    this->InitOutPredictions(info, &(out_preds->predictions), model);

    const uint32_t BLOCK_THREADS = 128;
    auto GRID_SIZE = static_cast<uint32_t>(common::DivRoundUp(info.num_row_, BLOCK_THREADS));

    auto shared_memory_bytes =
        static_cast<size_t>(sizeof(float) * m->NumColumns() * BLOCK_THREADS);
    bool use_shared = true;
    if (shared_memory_bytes > max_shared_memory_bytes) {
      shared_memory_bytes = 0;
      use_shared = false;
    }
    size_t entry_start = 0;

    dh::LaunchKernel {GRID_SIZE, BLOCK_THREADS, shared_memory_bytes} (
        PredictKernel<Loader, typename Loader::BatchT>,
        m->Value(),
        dh::ToSpan(d_model.nodes), out_preds->predictions.DeviceSpan(),
        dh::ToSpan(d_model.tree_segments), dh::ToSpan(d_model.tree_group),
        tree_begin, tree_end, m->NumColumns(), info.num_row_,
        entry_start, use_shared, output_groups);
  }

  void InplacePredict(dmlc::any const &x, const gbm::GBTreeModel &model,
                      float missing, PredictionCacheEntry *out_preds,
                      uint32_t tree_begin, unsigned tree_end) const override {
    if (x.type() == typeid(std::shared_ptr<data::CupyAdapter>)) {
      this->DispatchedInplacePredict<
          data::CupyAdapter, DeviceAdapterLoader<data::CupyAdapterBatch>>(
          x, model, missing, out_preds, tree_begin, tree_end);
    } else if (x.type() == typeid(std::shared_ptr<data::CudfAdapter>)) {
      this->DispatchedInplacePredict<
          data::CudfAdapter, DeviceAdapterLoader<data::CudfAdapterBatch>>(
          x, model, missing, out_preds, tree_begin, tree_end);
    } else {
      LOG(FATAL) << "Only CuPy and CuDF are supported by GPU Predictor.";
    }
  }

 protected:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const {
    size_t n_classes = model.learner_model_param->num_output_group;
    size_t n = n_classes * info.num_row_;
    const HostDeviceVector<bst_float>& base_margin = info.base_margin_;
    out_preds->SetDevice(generic_param_->gpu_id);
    out_preds->Resize(n);
    if (base_margin.Size() != 0) {
      CHECK_EQ(base_margin.Size(), n);
      out_preds->Copy(base_margin);
    } else {
      out_preds->Fill(model.learner_model_param->base_score);
    }
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit) override {
    LOG(FATAL) << "[Internal error]: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void PredictLeaf(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model,
                   unsigned ntree_limit) override {
    LOG(FATAL) << "[Internal error]: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model, unsigned ntree_limit,
                           std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) override {
    LOG(FATAL) << "[Internal error]: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model,
                                       unsigned ntree_limit,
                                       std::vector<bst_float>* tree_weights,
                                       bool approximate) override {
    LOG(FATAL) << "[Internal error]: " << __func__
               << " is not implemented in GPU Predictor.";
  }

  void Configure(const std::vector<std::pair<std::string, std::string>>& cfg) override {
    Predictor::Configure(cfg);
  }

 private:
  /*! \brief Reconfigure the device when GPU is changed. */
  void ConfigureDevice(int device) {
    if (device >= 0) {
      max_shared_memory_bytes_ = dh::MaxSharedMemory(device);
    }
  }

  std::mutex lock_;
  DeviceModel model_;
  size_t max_shared_memory_bytes_;
};

XGBOOST_REGISTER_PREDICTOR(GPUPredictor, "gpu_predictor")
.describe("Make predictions using GPU.")
.set_body([](GenericParameter const* generic_param) {
            return new GPUPredictor(generic_param);
          });

}  // namespace predictor
}  // namespace xgboost
