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

float GetLeafWeight(const Node* nodes, const float* fval_buff, const uint8_t* miss_buff) {
  const Node* node = nodes;
  while (!node->IsLeaf()) {
    if (miss_buff[node->GetFidx()] == 1) {
      node = nodes + node->MissingIdx();
    } else {
      const float fvalue = fval_buff[node->GetFidx()];
      if (fvalue < node->GetFvalue()) {
        node = nodes + node->LeftChildIdx();
      } else {
        node = nodes + node->RightChildIdx();
      }
    }
  }
  return node->GetWeight();
}

float GetLeafWeight(const Node* nodes, const float* fval_buff) {
  const Node* node = nodes;
  while (!node->IsLeaf()) {
    const float fvalue = fval_buff[node->GetFidx()];
    if (fvalue < node->GetFvalue()) {
      node = nodes + node->LeftChildIdx();
    } else {
      node = nodes + node->RightChildIdx();
    }
  }
  return node->GetWeight();
}

class Predictor : public xgboost::Predictor {
 public:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const override {
    device_model.SetDevice(ctx_->Device());
    CHECK_NE(model.learner_model_param->num_output_group, 0);
    size_t n = model.learner_model_param->num_output_group * info.num_row_;
    size_t base_margin_size = info.base_margin_.Data()->Size();
    out_preds->Resize(n);
    if (base_margin_size == n) {
      CHECK_EQ(out_preds->Size(), n);
      out_preds->Copy(*(info.base_margin_.Data()));
    } else {
      auto base_score = model.learner_model_param->BaseScore(ctx_)(0);
      if (base_margin_size > 0) {
        std::ostringstream oss;
        oss << "Ignoring the base margin, since it has incorrect length. "
            << "The base margin must be an array of length ";
        if (model.learner_model_param->num_output_group > 1) {
          oss << "[num_class] * [number of data points], i.e. "
              << model.learner_model_param->num_output_group << " * " << info.num_row_
              << " = " << n << ". ";
        } else {
          oss << "[number of data points], i.e. " << info.num_row_ << ". ";
        }
        oss << "Instead, all data points will use "
            << "base_score = " << base_score;
        LOG(WARNING) << oss.str();
      }
      out_preds->Fill(base_score);
    }
    needs_buffer_update = true;
  }

  explicit Predictor(Context const* context) :
      xgboost::Predictor::Predictor{context},
      cpu_predictor(xgboost::Predictor::Create("cpu_predictor", context)) {
        qu_ = device_manager.GetQueue(ctx_->Device());
      }

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
  template <bool any_missing>
  void PredictKernel(::sycl::event* event,
                     const Entry* data,
                     float* out_predictions,
                     const size_t* row_ptr,
                     size_t num_rows,
                     size_t num_features,
                     size_t num_group,
                     size_t tree_begin,
                     size_t tree_end) const {
    const Node* nodes = device_model.nodes.DataConst();
    const size_t* first_node_position = device_model.first_node_position.ConstDevicePointer();
    const int* tree_group = device_model.tree_group.ConstDevicePointer();

    float* fval_buff_ptr = fval_buff.Data();
    uint8_t* miss_buff_ptr = miss_buff.Data();
    bool needs_buffer_update = this->needs_buffer_update;

    *event = qu_->submit([&](::sycl::handler& cgh) {
      cgh.depends_on(*event);
      cgh.parallel_for<>(::sycl::range<1>(num_rows), [=](::sycl::id<1> pid) {
        int row_idx = pid[0];
        auto* fval_buff_row_ptr = fval_buff_ptr + num_features * row_idx;
        auto* miss_buff_row_ptr = miss_buff_ptr + num_features * row_idx;

        if (needs_buffer_update) {
          const Entry* first_entry = data + row_ptr[row_idx];
          const Entry* last_entry = data + row_ptr[row_idx + 1];
          for (const Entry* entry = first_entry; entry < last_entry; entry += 1) {
            fval_buff_row_ptr[entry->index] = entry->fvalue;
            if constexpr (any_missing) {
              miss_buff_row_ptr[entry->index] = 0;
            }
          }
        }

        if (num_group == 1) {
          float sum = 0.0;
          for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
            const Node* first_node = nodes + first_node_position[tree_idx - tree_begin];
            if constexpr (any_missing) {
              sum += GetLeafWeight(first_node, fval_buff_row_ptr, miss_buff_row_ptr);
            } else {
              sum += GetLeafWeight(first_node, fval_buff_row_ptr);
            }
          }
          out_predictions[row_idx] += sum;
        } else {
          for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
            const Node* first_node = nodes + first_node_position[tree_idx - tree_begin];
            int out_prediction_idx = row_idx * num_group + tree_group[tree_idx];
            if constexpr (any_missing) {
              out_predictions[out_prediction_idx] +=
                GetLeafWeight(first_node, fval_buff_row_ptr, miss_buff_row_ptr);
            } else {
              out_predictions[out_prediction_idx] +=
                GetLeafWeight(first_node, fval_buff_row_ptr);
            }
          }
        }
      });
    });
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

        if (needs_buffer_update) {
          fval_buff.ResizeNoCopy(qu_, num_features * batch_size);
          if constexpr (any_missing) {
            miss_buff.ResizeAndFill(qu_, num_features * batch_size, 1, &event);
          }
        }

        PredictKernel<any_missing>(&event, data, out_predictions + base_rowid,
                                   row_ptr, batch_size, num_features,
                                   num_group, tree_begin, tree_end);
        needs_buffer_update = (batch_size != out_preds->Size());
      }
    }
    qu_->wait();
  }

  mutable USMVector<float,   MemoryType::on_device> fval_buff;
  mutable USMVector<uint8_t, MemoryType::on_device> miss_buff;
  mutable DeviceModel device_model;
  mutable bool needs_buffer_update = true;

  mutable ::sycl::queue* qu_ = nullptr;

  DeviceManager device_manager;

  std::unique_ptr<xgboost::Predictor> cpu_predictor;
};

XGBOOST_REGISTER_PREDICTOR(Predictor, "sycl_predictor")
.describe("Make predictions using SYCL.")
.set_body([](Context const* ctx) { return new Predictor(ctx); });

}  // namespace predictor
}  // namespace sycl
}  // namespace xgboost
