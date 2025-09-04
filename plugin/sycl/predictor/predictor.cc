/*!
 * Copyright by Contributors 2017-2023
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#pragma GCC diagnostic pop

#include <cstddef>
#include <limits>
#include <mutex>

#include <sycl/sycl.hpp>

#include "../data.h"

#include "dmlc/registry.h"

#include "xgboost/tree_model.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_updater.h"
#include "../../../src/common/timer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../src/data/adapter.h"
#pragma GCC diagnostic pop
#include "../../src/common/math.h"
#include "../../src/gbm/gbtree_model.h"

#include "../device_manager.h"
#include "../device_properties.h"

namespace xgboost {
namespace sycl {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(predictor_sycl);

union NodeValue {
  float leaf_weight;
  float fvalue;
};

class Node {
  int fidx;
  int left_child_idx;
  int right_child_idx;
  NodeValue val;

 public:
  explicit Node(const RegTree::Node& n) {
    left_child_idx = n.LeftChild();
    right_child_idx = n.RightChild();
    fidx = n.SplitIndex();
    if (n.DefaultLeft()) {
      fidx |= (1U << 31);
    }

    if (n.IsLeaf()) {
      val.leaf_weight = n.LeafValue();
    } else {
      val.fvalue = n.SplitCond();
    }
  }

  int LeftChildIdx() const {return left_child_idx; }

  int RightChildIdx() const {return right_child_idx; }

  bool IsLeaf() const { return left_child_idx == -1; }

  int GetFidx() const { return fidx & ((1U << 31) - 1U); }

  bool MissingLeft() const { return (fidx >> 31) != 0; }

  int MissingIdx() const {
    if (MissingLeft()) {
      return left_child_idx;
    } else {
      return right_child_idx;
    }
  }

  float GetFvalue() const { return val.fvalue; }

  float GetWeight() const { return val.leaf_weight; }
};

class DeviceModel {
 public:
  USMVector<Node> nodes;
  HostDeviceVector<size_t> first_node_position;
  HostDeviceVector<int> tree_group;

  void SetDevice(DeviceOrd device) {
    first_node_position.SetDevice(device);
    tree_group.SetDevice(device);
  }

  void Init(::sycl::queue* qu, const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end) {
    int n_nodes = 0;
    first_node_position.Resize((tree_end - tree_begin) + 1);
    auto& first_node_position_host = first_node_position.HostVector();
    first_node_position_host[0] = n_nodes;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      if (model.trees[tree_idx]->HasCategoricalSplit()) {
        LOG(FATAL) << "Categorical features are not yet supported by sycl";
      }
      n_nodes += model.trees[tree_idx]->GetNodes().size();
      first_node_position_host[tree_idx - tree_begin + 1] = n_nodes;
    }

    nodes.Resize(qu, n_nodes);
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees[tree_idx]->GetNodes();
      size_t n_nodes_shift = first_node_position_host[tree_idx - tree_begin];
      for (size_t node_idx = 0; node_idx < src_nodes.size(); node_idx++) {
        nodes[node_idx + n_nodes_shift] = static_cast<Node>(src_nodes[node_idx]);
      }
    }

    int num_group = model.learner_model_param->num_output_group;
    if (num_group > 1) {
      tree_group.Resize(model.tree_info.size());
      auto& tree_group_host = tree_group.HostVector();
      for (size_t tree_idx = 0; tree_idx < model.tree_info.size(); tree_idx++)
        tree_group_host[tree_idx] = model.tree_info[tree_idx];
    }
  }
};

// Binary search
float BinarySearch(const Entry* begin_ptr, const Entry* end_ptr,
                   size_t col_idx, size_t num_features) {
  const size_t n_elems = end_ptr - begin_ptr;
  if (n_elems == num_features) {
    return (begin_ptr + col_idx)->fvalue;
  }

  // Since indexes are in range [0: num_features),
  // we can squeeze the search window from [0: n_elems) to [offset_left: offset_right)
  const size_t shift = (num_features - 1) - col_idx;
  const size_t offset_left = shift > n_elems - 1 ? 0 : std::max<size_t>(0, (n_elems - 1) - shift);
  const size_t offset_right = std::min<size_t>(col_idx + 1, n_elems);

  end_ptr = begin_ptr + offset_right;
  begin_ptr += offset_left;
  const Entry* previous_middle = nullptr;
  while (end_ptr != begin_ptr) {
    const Entry* middle = begin_ptr + (end_ptr - begin_ptr) / 2;
    if (middle == previous_middle) {
      break;
    } else {
      previous_middle = middle;
    }
    if (middle->index == col_idx) {
      return middle->fvalue;
    } else if (middle->index < col_idx) {
      begin_ptr = middle + 1;
    } else {
      end_ptr = middle;
    }
  }
  return std::numeric_limits<float>::quiet_NaN();
}

size_t NextNodeIdx(float fvalue, const Node& node) {
  if (std::isnan(fvalue)) {
    return node.MissingIdx();
  } else {
    if (fvalue < node.GetFvalue()) {
      return node.LeftChildIdx();
    } else {
      return node.RightChildIdx();
    }
  }
}

float GetLeafWeight(const Node* nodes, const Entry* first_entry,
                    const Entry* last_entry, size_t num_features) {
  size_t is_dense = (last_entry - first_entry == num_features);

  const Node* node = nodes;
  while (!node->IsLeaf()) {
    const float fvalue = is_dense ?
                         (first_entry + node->GetFidx())->fvalue :
                         BinarySearch(first_entry, last_entry, node->GetFidx(), num_features);
    node = nodes + NextNodeIdx(fvalue, *node);
  }
  return node->GetWeight();
}

float GetLeafWeight(const Node* nodes, const float* fval_buff) {
  const Node* node = nodes;
  while (!node->IsLeaf()) {
    const float fvalue = fval_buff[node->GetFidx()];
    node = nodes + NextNodeIdx(fvalue, *node);
  }
  return node->GetWeight();
}

class Predictor : public xgboost::Predictor {
 public:
  explicit Predictor(Context const* context) :
      xgboost::Predictor::Predictor{context},
      cpu_predictor(xgboost::Predictor::Create("cpu_predictor", context)),
      qu_(device_manager.GetQueue(context->Device())),
      device_prop_(qu_->get_device()) {}

  void PredictBatch(DMatrix *dmat, PredictionCacheEntry *predts,
                    const gbm::GBTreeModel &model, bst_tree_t tree_begin,
                    bst_tree_t tree_end = 0) const override {
    auto* out_preds = &predts->predictions;
    out_preds->SetDevice(ctx_->Device());
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }

    if (tree_begin < tree_end) {
      const bool any_missing = !(dmat->IsDense());
      if (any_missing) {
        DevicePredictInternal<true>(dmat, out_preds, model, tree_begin, tree_end);
      } else {
        DevicePredictInternal<false>(dmat, out_preds, model, tree_begin, tree_end);
      }
    }
  }

  bool InplacePredict(std::shared_ptr<DMatrix> p_m,
                      const gbm::GBTreeModel &model, float missing,
                      PredictionCacheEntry *out_preds, bst_tree_t tree_begin,
                      bst_tree_t tree_end) const override {
    LOG(WARNING) << "InplacePredict is not yet implemented for SYCL. CPU Predictor is used.";
    return cpu_predictor->InplacePredict(p_m, model, missing, out_preds, tree_begin, tree_end);
  }

  void PredictLeaf(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model, bst_tree_t ntree_limit) const override {
    LOG(WARNING) << "PredictLeaf is not yet implemented for SYCL. CPU Predictor is used.";
    cpu_predictor->PredictLeaf(p_fmat, out_preds, model, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                           const gbm::GBTreeModel& model, bst_tree_t ntree_limit,
                           const std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) const override {
    LOG(WARNING) << "PredictContribution is not yet implemented for SYCL. CPU Predictor is used.";
    cpu_predictor->PredictContribution(p_fmat, out_contribs, model, ntree_limit, tree_weights,
                                       approximate, condition, condition_feature);
  }

  void PredictInteractionContributions(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model, bst_tree_t ntree_limit,
                                       const std::vector<bst_float>* tree_weights,
                                       bool approximate) const override {
    LOG(WARNING) << "PredictInteractionContributions is not yet implemented for SYCL. "
                 << "CPU Predictor is used.";
    cpu_predictor->PredictInteractionContributions(p_fmat, out_contribs, model, ntree_limit,
                                                   tree_weights, approximate);
  }

 private:
  // 8KB fits EU registers
  static constexpr int kMaxFeatureBufferSize = 2048;

  // Relative cost of reading and writing for discrete and integrated devices.
  static constexpr float kCostCalibrationIntegrated = 64;
  static constexpr float kCostCalibrationDescrete = 4;

  template <bool any_missing, int kFeatureBufferSize = 8>
  void PredictKernelBufferDispatch(::sycl::event* event,
                                   const Entry* data,
                                   float* out_predictions,
                                   const size_t* row_ptr,
                                   size_t num_rows,
                                   size_t num_features,
                                   size_t num_group,
                                   size_t tree_begin,
                                   size_t tree_end,
                                   float sparsity) const {
    if constexpr (kFeatureBufferSize > kMaxFeatureBufferSize) {
      LOG(FATAL) << "Unreachable";
    } else {
      if (num_features > kFeatureBufferSize) {
        PredictKernelBufferDispatch<any_missing, 2 * kFeatureBufferSize>
                                   (event, data, out_predictions, row_ptr, num_rows,
                                    num_features, num_group, tree_begin, tree_end, sparsity);
      } else {
        PredictKernelBuffer<any_missing, kFeatureBufferSize>
                           (event, data, out_predictions, row_ptr, num_rows,
                            num_features, num_group, tree_begin, tree_end, sparsity);
      }
    }
  }

  size_t GetBlockSize(size_t n_nodes, size_t num_features, size_t num_rows, float sparsity) const {
    size_t max_compute_units = device_prop_.max_compute_units;
    size_t l2_size = device_prop_.l2_size;
    size_t sub_group_size = device_prop_.sub_group_size;
    size_t nodes_bytes = n_nodes * sizeof(Node);
    bool nodes_fit_l2 = l2_size > 2 * nodes_bytes;
    size_t block_size = nodes_fit_l2
                      // nodes and data fit L2
                      ? 0.8 * (l2_size - nodes_bytes) / (sparsity * num_features * sizeof(Entry))
                      // only data fit L2
                      : 0.8 * (l2_size) / (sparsity * num_features * sizeof(Entry));
    block_size = (block_size / sub_group_size) * sub_group_size;
    if (block_size < max_compute_units * sub_group_size) {
      block_size = max_compute_units * sub_group_size;
    }

    if (block_size > num_rows) block_size = num_rows;
    return block_size;
  }

  template <bool any_missing, int kFeatureBufferSize>
  void PredictKernelBuffer(::sycl::event* event,
                           const Entry* data,
                           float* out_predictions,
                           const size_t* row_ptr,
                           size_t num_rows,
                           size_t num_features,
                           size_t num_group,
                           size_t tree_begin,
                           size_t tree_end,
                           float sparsity) const {
    const Node* nodes = device_model.nodes.DataConst();
    const size_t* first_node_position = device_model.first_node_position.ConstDevicePointer();
    const int* tree_group = device_model.tree_group.ConstDevicePointer();

    size_t block_size = GetBlockSize(device_model.nodes.Size(),
                                     num_features, num_rows, sparsity);
    size_t n_blocks = num_rows / block_size + (num_rows % block_size > 0);

    for (size_t block = 0; block < n_blocks; ++block) {
      *event = qu_->submit([&](::sycl::handler& cgh) {
        cgh.depends_on(*event);
        cgh.parallel_for<>(::sycl::range<1>(block_size), [=](::sycl::id<1> pid) {
          int row_idx = block * block_size + pid[0];
          if (row_idx < num_rows) {
            const Entry* first_entry = data + row_ptr[row_idx];
            const Entry* last_entry = data + row_ptr[row_idx + 1];

            float fvalues[kFeatureBufferSize];
            if constexpr (any_missing) {
              for (size_t fid = 0; fid < num_features; ++fid) {
                fvalues[fid] = std::numeric_limits<float>::quiet_NaN();
              }
            }

            for (const Entry* entry = first_entry; entry < last_entry; entry += 1) {
              fvalues[entry->index] = entry->fvalue;
            }
            if (num_group == 1) {
              float& sum = out_predictions[row_idx];
              for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
                const Node* first_node = nodes + first_node_position[tree_idx - tree_begin];
                sum += GetLeafWeight(first_node, fvalues);
              }
            } else {
              for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
                const Node* first_node = nodes + first_node_position[tree_idx - tree_begin];
                int out_prediction_idx = row_idx * num_group + tree_group[tree_idx];
                out_predictions[out_prediction_idx] +=
                    GetLeafWeight(first_node, fvalues);
              }
            }
          }
        });
      });
    }
  }

  void PredictKernel(::sycl::event* event,
                     const Entry* data,
                     float* out_predictions,
                     const size_t* row_ptr,
                     size_t num_rows,
                     size_t num_features,
                     size_t num_group,
                     size_t tree_begin,
                     size_t tree_end,
                     float sparsity) const {
    const Node* nodes = device_model.nodes.DataConst();
    const size_t* first_node_position = device_model.first_node_position.ConstDevicePointer();
    const int* tree_group = device_model.tree_group.ConstDevicePointer();

    size_t block_size = GetBlockSize(device_model.nodes.Size(),
                                     num_features, num_rows, sparsity);
    size_t n_blocks = num_rows / block_size + (num_rows % block_size > 0);

    for (size_t block = 0; block < n_blocks; ++block) {
      *event = qu_->submit([&](::sycl::handler& cgh) {
        cgh.depends_on(*event);
        cgh.parallel_for<>(::sycl::range<1>(block_size), [=](::sycl::id<1> pid) {
          int row_idx = block * block_size + pid[0];
          if (row_idx < num_rows) {
            const Entry* first_entry = data + row_ptr[row_idx];
            const Entry* last_entry = data + row_ptr[row_idx + 1];

            if (num_group == 1) {
              float& sum = out_predictions[row_idx];
              for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
                const Node* first_node = nodes + first_node_position[tree_idx - tree_begin];
                sum += GetLeafWeight(first_node, first_entry, last_entry, num_features);
              }
            } else {
              for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
                const Node* first_node = nodes + first_node_position[tree_idx - tree_begin];
                int out_prediction_idx = row_idx * num_group + tree_group[tree_idx];
                out_predictions[out_prediction_idx] +=
                    GetLeafWeight(first_node, first_entry, last_entry, num_features);
              }
            }
          }
        });
      });
    }
  }

  template <bool any_missing>
  bool UseFvalueBuffer(size_t tree_begin,
                       size_t tree_end,
                       int num_features) const {
    size_t n_nodes = device_model.nodes.Size();
    size_t n_trees = tree_end - tree_begin;
    float av_depth = std::log2(static_cast<float>(n_nodes) / n_trees);
    // the last one is leaf
    float av_nodes_per_traversal = av_depth - 1;
    // number of reads in case of no-bufer
    float n_reads = av_nodes_per_traversal * n_trees;
    if (any_missing) {
      // we use binary search for sparse
      n_reads *= std::log2(static_cast<float>(num_features));
    }

    float cost_callibration = device_prop_.usm_host_allocations
                            ? kCostCalibrationIntegrated
                            : kCostCalibrationDescrete;

    // number of writes in local memory.
    float n_writes = num_features;
    bool use_fvalue_buffer = (num_features <= kMaxFeatureBufferSize) &&
                             (n_reads > cost_callibration * n_writes);
    return use_fvalue_buffer;
  }

  template <bool any_missing>
  void DevicePredictInternal(DMatrix *dmat,
                             HostDeviceVector<float>* out_preds,
                             const gbm::GBTreeModel& model,
                             size_t tree_begin,
                             size_t tree_end) const {
    if (tree_end - tree_begin == 0) return;
    if (out_preds->Size() == 0) return;

    device_model.Init(qu_, model, tree_begin, tree_end);

    int num_group = model.learner_model_param->num_output_group;
    int num_features = dmat->Info().num_col_;

    float* out_predictions = out_preds->DevicePointer();
    ::sycl::event event;
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      batch.data.SetDevice(ctx_->Device());
      batch.offset.SetDevice(ctx_->Device());
      const Entry* data = batch.data.ConstDevicePointer();
      const size_t* row_ptr = batch.offset.ConstDevicePointer();
      size_t batch_size = batch.Size();
      if (batch_size > 0) {
        const auto base_rowid = batch.base_rowid;

        float sparsity = static_cast<float>(batch.data.Size()) / (batch_size * num_features);
        if (UseFvalueBuffer<any_missing>(tree_begin, tree_end, num_features)) {
          PredictKernelBufferDispatch<any_missing>(&event, data,
                                                   out_predictions + base_rowid * num_group,
                                                   row_ptr, batch_size, num_features,
                                                   num_group, tree_begin, tree_end, sparsity);
        } else {
          PredictKernel(&event, data,
                        out_predictions + base_rowid * num_group,
                        row_ptr, batch_size, num_features,
                        num_group, tree_begin, tree_end, sparsity);
        }
      }
    }
    qu_->wait();
  }

  mutable DeviceModel device_model;
  DeviceManager device_manager;

  mutable ::sycl::queue* qu_ = nullptr;
  DeviceProperties device_prop_;

  std::unique_ptr<xgboost::Predictor> cpu_predictor;
};

XGBOOST_REGISTER_PREDICTOR(Predictor, "sycl_predictor")
.describe("Make predictions using SYCL.")
.set_body([](Context const* ctx) { return new Predictor(ctx); });

}  // namespace predictor
}  // namespace sycl
}  // namespace xgboost
