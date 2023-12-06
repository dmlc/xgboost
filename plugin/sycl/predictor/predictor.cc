/*!
 * Copyright by Contributors 2017-2023
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <rabit/rabit.h>
#pragma GCC diagnostic pop

#include <cstddef>
#include <limits>
#include <mutex>

#include <CL/sycl.hpp>

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

/* Wrapper for descriptor of a tree node */
struct DeviceNode {
  DeviceNode()
      : fidx(-1), left_child_idx(-1), right_child_idx(-1) {}

  union NodeValue {
    float leaf_weight;
    float fvalue;
  };

  int fidx;
  int left_child_idx;
  int right_child_idx;
  NodeValue val;

  explicit DeviceNode(const RegTree::Node& n) {
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

  bool IsLeaf() const { return left_child_idx == -1; }

  int GetFidx() const { return fidx & ((1U << 31) - 1U); }

  bool MissingLeft() const { return (fidx >> 31) != 0; }

  int MissingIdx() const {
    if (MissingLeft()) {
      return this->left_child_idx;
    } else {
      return this->right_child_idx;
    }
  }

  float GetFvalue() const { return val.fvalue; }

  float GetWeight() const { return val.leaf_weight; }
};

/* SYCL implementation of a device model,
 * storing tree structure in USM buffers to provide access from device kernels
 */
class DeviceModel {
 public:
  ::sycl::queue qu_;
  USMVector<DeviceNode> nodes_;
  USMVector<size_t> tree_segments_;
  USMVector<int> tree_group_;
  size_t tree_beg_;
  size_t tree_end_;
  int num_group_;

  DeviceModel() {}

  ~DeviceModel() {}

  void Init(::sycl::queue qu, const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end) {
    qu_ = qu;

    tree_segments_.Resize(&qu_, (tree_end - tree_begin) + 1);
    int sum = 0;
    tree_segments_[0] = sum;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      if (model.trees[tree_idx]->HasCategoricalSplit()) {
        LOG(FATAL) << "Categorical features are not yet supported by sycl";
      }
      sum += model.trees[tree_idx]->GetNodes().size();
      tree_segments_[tree_idx - tree_begin + 1] = sum;
    }

    nodes_.Resize(&qu_, sum);
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees[tree_idx]->GetNodes();
      for (size_t node_idx = 0; node_idx < src_nodes.size(); node_idx++)
        nodes_[node_idx + tree_segments_[tree_idx - tree_begin]] =
          static_cast<DeviceNode>(src_nodes[node_idx]);
    }

    tree_group_.Resize(&qu_, model.tree_info.size());
    for (size_t tree_idx = 0; tree_idx < model.tree_info.size(); tree_idx++)
      tree_group_[tree_idx] = model.tree_info[tree_idx];

    tree_beg_ = tree_begin;
    tree_end_ = tree_end;
    num_group_ = model.learner_model_param->num_output_group;
  }
};

union NodeValue {
  float leaf_weight;
  float fvalue;
};

class Node {
  const int* fidx_;
  const int* left_child_idx_;
  const int* right_child_idx_;
  const NodeValue* val_;

 public:
  Node(const int* fidx, const int* left_child_idx, const int* right_child_idx, const NodeValue* val) : 
    fidx_(fidx), left_child_idx_(left_child_idx), right_child_idx_(right_child_idx), val_(val) {}

  int LeftChildIdx() const {return *left_child_idx_; }

  int RightChildIdx() const {return *right_child_idx_; }

  bool IsLeaf() const { return *left_child_idx_ == -1; }

  int GetFidx() const { return *fidx_ & ((1U << 31) - 1U); }

  bool MissingLeft() const { return (*fidx_ >> 31) != 0; }

  int MissingIdx() const {
    if (MissingLeft()) {
      return *left_child_idx_;
    } else {
      return *right_child_idx_;
    }
  }

  float GetFvalue() const { return (*val_).fvalue; }

  float GetWeight() const { return (*val_).leaf_weight; }
};

class DeviceModelNew {
  USMVector<int> fidx_;
  USMVector<int> left_child_idx_;
  USMVector<int> right_child_idx_;
  USMVector<NodeValue> val_;

  Node InitNode(const RegTree::Node& n, size_t node_idx) {
    left_child_idx_[node_idx] = n.LeftChild();
    right_child_idx_[node_idx] = n.RightChild();
    fidx_[node_idx] = n.SplitIndex();
    if (n.DefaultLeft()) {
      fidx_[node_idx] |= (1U << 31);
    }

    if (n.IsLeaf()) {
      val_[node_idx].leaf_weight = n.LeafValue();
    } else {
      val_[node_idx].fvalue = n.SplitCond();
    }
    return {fidx_.Data()            + node_idx, left_child_idx_.Data() + node_idx, 
            right_child_idx_.Data() + node_idx, val_.Data()            + node_idx};
  }

 public:
  USMVector<Node> nodes;
  USMVector<size_t> first_node_position;
  USMVector<int> tree_group;
  size_t tree_beg;
  size_t tree_end;
  int num_group;

  void Init(::sycl::queue* qu, const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end) {
    int n_nodes = 0;
    first_node_position.Resize(qu, (tree_end - tree_begin) + 1);
    first_node_position[0] = n_nodes;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      if (model.trees[tree_idx]->HasCategoricalSplit()) {
        LOG(FATAL) << "Categorical features are not yet supported by sycl";
      }
      n_nodes += model.trees[tree_idx]->GetNodes().size();
      first_node_position[tree_idx - tree_begin + 1] = n_nodes;
    }

    std::vector<::sycl::event> events(5);
    events[0] = fidx_.ResizeAsync(qu, n_nodes, events[0]);
    events[1] = left_child_idx_.ResizeAsync(qu, n_nodes, events[1]);
    events[2] = right_child_idx_.ResizeAsync(qu, n_nodes, events[2]);
    events[3] = val_.ResizeAsync(qu, n_nodes, events[3]);
    events[4] = nodes.ResizeAsync(qu, n_nodes, events[4]);

    ::sycl::event event;
    event.wait_and_throw(events);
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees[tree_idx]->GetNodes();
      size_t n_nodes_shift = first_node_position[tree_idx - tree_begin];
      for (size_t node_idx = 0; node_idx < src_nodes.size(); node_idx++) {
        nodes[node_idx + n_nodes_shift] = InitNode(src_nodes[node_idx], node_idx + n_nodes_shift);
      }
    }

    tree_group.Resize(qu, model.tree_info.size());
    for (size_t tree_idx = 0; tree_idx < model.tree_info.size(); tree_idx++)
      tree_group[tree_idx] = model.tree_info[tree_idx];

    tree_beg = tree_begin;
    tree_end = tree_end;
    num_group = model.learner_model_param->num_output_group;
  }
};

struct NodeResponse {
  float fvalue;
  uint8_t is_missing;
};

float GetLeafWeightNew(const Node* nodes, const NodeResponse* fval_buff) {
  const Node* node = nodes;
  while (!node->IsLeaf()) {
    const NodeResponse& response = fval_buff[node->GetFidx()];
    if (response.is_missing == 1) {
      node = nodes + node->MissingIdx();
    } else {
      if (response.fvalue < node->GetFvalue()) {
        node = nodes + node->LeftChildIdx();
      } else {
        node = nodes + node->RightChildIdx();
      }
    }
  }
  return node->GetWeight();
}

void DevicePredictInternalNew(::sycl::queue* qu,
                              const sycl::DeviceMatrix& dmat,
                              HostDeviceVector<float>* out_preds,
                              const gbm::GBTreeModel& model,
                              size_t tree_begin,
                              size_t tree_end) {
  if (tree_end - tree_begin == 0) return;
  if (out_preds->HostVector().size() == 0) return;

  DeviceModelNew device_model;
  device_model.Init(qu, model, tree_begin, tree_end);

  const Node* nodes = device_model.nodes.DataConst();
  const size_t* first_node_position = device_model.first_node_position.DataConst();
  const int* tree_group = device_model.tree_group.DataConst();
  const size_t* row_ptr = dmat.row_ptr.DataConst();
  const Entry* data = dmat.data.DataConst();
  int num_features = dmat.p_mat->Info().num_col_;
  int num_rows = dmat.row_ptr.Size() - 1;
  int num_group = model.learner_model_param->num_output_group;

  USMVector<NodeResponse, MemoryType::on_device> fval_buff(qu, num_features * num_rows);
  auto* fval_buff_ptr = fval_buff.Data();

  constexpr NodeResponse missing_response = {0.0, 1};
  std::vector<::sycl::event> events(1);
  events[0] = qu->fill(fval_buff_ptr, missing_response, num_features * num_rows);

  events[0] = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.parallel_for<>(::sycl::range<1>(num_rows), [=](::sycl::id<1> pid) {
      int global_idx = pid[0];
      auto* fval_buff_row_ptr = fval_buff_ptr + num_features * global_idx;

      const Entry* begin_ptr = data + row_ptr[global_idx];
      const Entry* end_ptr = data + row_ptr[global_idx + 1];
      for (const Entry* entry = begin_ptr; entry < end_ptr; entry += 1) {
        fval_buff_row_ptr[entry->index] = {entry->fvalue, 0};
      }
    });
  });

  auto& out_preds_vec = out_preds->HostVector();
  ::sycl::buffer<float, 1> out_preds_buf(out_preds_vec.data(), out_preds_vec.size());
  events[0] = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(events[0]);
    auto out_predictions = out_preds_buf.template get_access<::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<>(::sycl::range<1>(num_rows), [=](::sycl::id<1> pid) {
      int global_idx = pid[0];
      auto* fval_buff_row_ptr = fval_buff_ptr + num_features * global_idx;
      if (num_group == 1) {
        float sum = 0.0;
        for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
          const Node* first_node = nodes + first_node_position[tree_idx - tree_begin];
          sum += GetLeafWeightNew(first_node, fval_buff_row_ptr);
        }
        out_predictions[global_idx] += sum;
      } else {
        for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
          const Node* first_node = nodes + first_node_position[tree_idx - tree_begin];
          int out_prediction_idx = global_idx * num_group + tree_group[tree_idx];
          out_predictions[out_prediction_idx] += GetLeafWeightNew(first_node, fval_buff_row_ptr);
        }
      }
    });
  });
  qu->wait();
}


float GetFvalue(int ridx, int fidx, Entry* data, size_t* row_ptr, bool* is_missing) {
  // Binary search
  auto begin_ptr = data + row_ptr[ridx];
  auto end_ptr = data + row_ptr[ridx + 1];
  Entry* previous_middle = nullptr;
  while (end_ptr != begin_ptr) {
    auto middle = begin_ptr + (end_ptr - begin_ptr) / 2;
    if (middle == previous_middle) {
      break;
    } else {
      previous_middle = middle;
    }

    if (middle->index == fidx) {
      *is_missing = false;
      return middle->fvalue;
    } else if (middle->index < fidx) {
      begin_ptr = middle;
    } else {
      end_ptr = middle;
    }
  }
  *is_missing = true;
  return 0.0;
}

float GetLeafWeight(int ridx, const DeviceNode* tree, Entry* data, size_t* row_ptr) {
  DeviceNode n = tree[0];
  int node_id = 0;
  bool is_missing;
  while (!n.IsLeaf()) {
    float fvalue = GetFvalue(ridx, n.GetFidx(), data, row_ptr, &is_missing);
    // Missing value
    if (is_missing) {
      n = tree[n.MissingIdx()];
    } else {
      if (fvalue < n.GetFvalue()) {
        node_id = n.left_child_idx;
        n = tree[n.left_child_idx];
      } else {
        node_id = n.right_child_idx;
        n = tree[n.right_child_idx];
      }
    }
  }
  return n.GetWeight();
}

void DevicePredictInternal(::sycl::queue qu,
                           sycl::DeviceMatrix* dmat,
                           HostDeviceVector<float>* out_preds,
                           const gbm::GBTreeModel& model,
                           size_t tree_begin,
                           size_t tree_end) {
  if (tree_end - tree_begin == 0) return;
  if (out_preds->HostVector().size() == 0) return;

  DeviceModel device_model;
  device_model.Init(qu, model, tree_begin, tree_end);

  auto& out_preds_vec = out_preds->HostVector();

  DeviceNode* nodes = device_model.nodes_.Data();
  ::sycl::buffer<float, 1> out_preds_buf(out_preds_vec.data(), out_preds_vec.size());
  size_t* tree_segments = device_model.tree_segments_.Data();
  int* tree_group = device_model.tree_group_.Data();
  size_t* row_ptr = dmat->row_ptr.Data();
  Entry* data = dmat->data.Data();
  int num_features = dmat->p_mat->Info().num_col_;
  int num_rows = dmat->row_ptr.Size() - 1;
  int num_group = model.learner_model_param->num_output_group;

  qu.submit([&](::sycl::handler& cgh) {
    auto out_predictions = out_preds_buf.template get_access<::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<>(::sycl::range<1>(num_rows), [=](::sycl::id<1> pid) {
      int global_idx = pid[0];
      if (global_idx >= num_rows) return;
      if (num_group == 1) {
        float sum = 0.0;
        for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
          const DeviceNode* tree = nodes + tree_segments[tree_idx - tree_begin];
          sum += GetLeafWeight(global_idx, tree, data, row_ptr);
        }
        out_predictions[global_idx] += sum;
      } else {
        for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
          const DeviceNode* tree = nodes + tree_segments[tree_idx - tree_begin];
          int out_prediction_idx = global_idx * num_group + tree_group[tree_idx];
          out_predictions[out_prediction_idx] += GetLeafWeight(global_idx, tree, data, row_ptr);
        }
      }
    });
  }).wait();
}

class Predictor : public xgboost::Predictor {
  mutable xgboost::common::Monitor monitor;
 protected:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const override {
    monitor.Start(__func__);
    CHECK_NE(model.learner_model_param->num_output_group, 0);
    size_t n = model.learner_model_param->num_output_group * info.num_row_;
    const auto& base_margin = info.base_margin_.Data()->HostVector();
    out_preds->Resize(n);
    std::vector<bst_float>& out_preds_h = out_preds->HostVector();
    if (base_margin.size() == n) {
      CHECK_EQ(out_preds->Size(), n);
      std::copy(base_margin.begin(), base_margin.end(), out_preds_h.begin());
    } else {
      auto base_score = model.learner_model_param->BaseScore(ctx_)(0);
      if (!base_margin.empty()) {
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
      std::fill(out_preds_h.begin(), out_preds_h.end(), base_score);
    }
    monitor.Stop(__func__);
  }

 public:
  explicit Predictor(Context const* context) :
      xgboost::Predictor::Predictor{context},
      cpu_predictor(xgboost::Predictor::Create("cpu_predictor", context)) {monitor.Init("SyclPredictor"); }

  void PredictBatch(DMatrix *dmat, PredictionCacheEntry *predts,
                    const gbm::GBTreeModel &model, uint32_t tree_begin,
                    uint32_t tree_end = 0) const override {
    monitor.Start(__func__);
    ::sycl::queue qu = device_manager.GetQueue(ctx_->Device());
    // TODO(razdoburdin): remove temporary workaround after cache fix
    sycl::DeviceMatrix device_matrix(qu, dmat, &monitor);

    auto* out_preds = &predts->predictions;
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }

    if (tree_begin < tree_end) {
      DevicePredictInternalNew(&qu, device_matrix, out_preds, model, tree_begin, tree_end);
    }
    monitor.Stop(__func__);
  }

  bool InplacePredict(std::shared_ptr<DMatrix> p_m,
                      const gbm::GBTreeModel &model, float missing,
                      PredictionCacheEntry *out_preds, uint32_t tree_begin,
                      unsigned tree_end) const override {
    LOG(WARNING) << "InplacePredict is not yet implemented for SYCL. CPU Predictor is used.";
    return cpu_predictor->InplacePredict(p_m, model, missing, out_preds, tree_begin, tree_end);
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                       bool is_column_split) const override {
    LOG(WARNING) << "PredictInstance is not yet implemented for SYCL. CPU Predictor is used.";
    cpu_predictor->PredictInstance(inst, out_preds, model, ntree_limit, is_column_split);
  }

  void PredictLeaf(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model, unsigned ntree_limit) const override {
    LOG(WARNING) << "PredictLeaf is not yet implemented for SYCL. CPU Predictor is used.";
    cpu_predictor->PredictLeaf(p_fmat, out_preds, model, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                           const gbm::GBTreeModel& model, uint32_t ntree_limit,
                           const std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) const override {
    LOG(WARNING) << "PredictContribution is not yet implemented for SYCL. CPU Predictor is used.";
    cpu_predictor->PredictContribution(p_fmat, out_contribs, model, ntree_limit, tree_weights,
                                       approximate, condition, condition_feature);
  }

  void PredictInteractionContributions(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                                       const std::vector<bst_float>* tree_weights,
                                       bool approximate) const override {
    LOG(WARNING) << "PredictInteractionContributions is not yet implemented for SYCL. "
                 << "CPU Predictor is used.";
    cpu_predictor->PredictInteractionContributions(p_fmat, out_contribs, model, ntree_limit,
                                                   tree_weights, approximate);
  }

 private:
  DeviceManager device_manager;

  std::unique_ptr<xgboost::Predictor> cpu_predictor;
};

XGBOOST_REGISTER_PREDICTOR(Predictor, "sycl_predictor")
.describe("Make predictions using SYCL.")
.set_body([](Context const* ctx) { return new Predictor(ctx); });

}  // namespace predictor
}  // namespace sycl
}  // namespace xgboost
