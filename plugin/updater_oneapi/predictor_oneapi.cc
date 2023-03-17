/*!
 * Copyright by Contributors 2017-2020
 */
#include <any>  // for any
#include <cstddef>
#include <limits>
#include <mutex>

#include "../../src/common/math.h"
#include "../../src/data/adapter.h"
#include "../../src/gbm/gbtree_model.h"
#include "CL/sycl.hpp"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/logging.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(predictor_oneapi);

/*! \brief Element from a sparse vector */
struct EntryOneAPI {
  /*! \brief feature index */
  bst_feature_t index;
  /*! \brief feature value */
  bst_float fvalue;
  /*! \brief default constructor */
  EntryOneAPI() = default;
  /*!
   * \brief constructor with index and value
   * \param index The feature or row index.
   * \param fvalue The feature value.
   */
  EntryOneAPI(bst_feature_t index, bst_float fvalue) : index(index), fvalue(fvalue) {}

  EntryOneAPI(const Entry& entry) : index(entry.index), fvalue(entry.fvalue) {}

  /*! \brief reversely compare feature values */
  inline static bool CmpValue(const EntryOneAPI& a, const EntryOneAPI& b) {
    return a.fvalue < b.fvalue;
  }
  inline bool operator==(const EntryOneAPI& other) const {
    return (this->index == other.index && this->fvalue == other.fvalue);
  }
};

struct DeviceMatrixOneAPI {
  DMatrix* p_mat;  // Pointer to the original matrix on the host
  cl::sycl::queue qu_;
  size_t* row_ptr;
  size_t row_ptr_size;
  EntryOneAPI* data;

  DeviceMatrixOneAPI(DMatrix* dmat, cl::sycl::queue qu) : p_mat(dmat), qu_(qu) {
    size_t num_row = 0;
    size_t num_nonzero = 0;
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      const auto& data_vec = batch.data.HostVector();
      const auto& offset_vec = batch.offset.HostVector();
      num_nonzero += data_vec.size();
      num_row += batch.Size();
    }

    row_ptr = cl::sycl::malloc_shared<size_t>(num_row + 1, qu_);
    data = cl::sycl::malloc_shared<EntryOneAPI>(num_nonzero, qu_);

    size_t data_offset = 0;
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      const auto& data_vec = batch.data.HostVector();
      const auto& offset_vec = batch.offset.HostVector();
      size_t batch_size = batch.Size();
      if (batch_size > 0) {
        std::copy(offset_vec.data(), offset_vec.data() + batch_size,
                  row_ptr + batch.base_rowid);
        if (batch.base_rowid > 0) {
          for(size_t i = 0; i < batch_size; i++)
            row_ptr[i + batch.base_rowid] += batch.base_rowid;
        }
        std::copy(data_vec.data(), data_vec.data() + offset_vec[batch_size],
                  data + data_offset);
        data_offset += offset_vec[batch_size];
      }
    }
    row_ptr[num_row] = data_offset;
    row_ptr_size = num_row + 1;
  }

  ~DeviceMatrixOneAPI() {
    if (row_ptr) {
      cl::sycl::free(row_ptr, qu_);
    }
    if (data) {
      cl::sycl::free(data, qu_);
    }
  }
};

struct DeviceNodeOneAPI {
  DeviceNodeOneAPI()
      : fidx(-1), left_child_idx(-1), right_child_idx(-1) {}

  union NodeValue {
    float leaf_weight;
    float fvalue;
  };

  int fidx;
  int left_child_idx;
  int right_child_idx;
  NodeValue val;

  DeviceNodeOneAPI(const RegTree::Node& n) {  // NOLINT
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

class DeviceModelOneAPI {
 public:
  cl::sycl::queue qu_;
  DeviceNodeOneAPI* nodes;
  size_t* tree_segments;
  int* tree_group;
  size_t tree_beg_;
  size_t tree_end_;
  int num_group;

  DeviceModelOneAPI() : nodes(nullptr), tree_segments(nullptr), tree_group(nullptr) {}

  ~DeviceModelOneAPI() {
    Reset();
  }

  void Reset() {
    if (nodes)
      cl::sycl::free(nodes, qu_);
    if (tree_segments)
      cl::sycl::free(tree_segments, qu_);
    if (tree_group)
      cl::sycl::free(tree_group, qu_);
  }

  void Init(const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end, cl::sycl::queue qu) {
    qu_ = qu;
    CHECK_EQ(model.param.size_leaf_vector, 0);
    Reset();

    tree_segments = cl::sycl::malloc_shared<size_t>((tree_end - tree_begin) + 1, qu_);
    int sum = 0;
    tree_segments[0] = sum;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      sum += model.trees[tree_idx]->GetNodes().size();
      tree_segments[tree_idx - tree_begin + 1] = sum;
    }

    nodes = cl::sycl::malloc_shared<DeviceNodeOneAPI>(sum, qu_);
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees[tree_idx]->GetNodes();
      for (size_t node_idx = 0; node_idx < src_nodes.size(); node_idx++)
        nodes[node_idx + tree_segments[tree_idx - tree_begin]] = src_nodes[node_idx];
    }

    tree_group = cl::sycl::malloc_shared<int>(model.tree_info.size(), qu_);
    for (size_t tree_idx = 0; tree_idx < model.tree_info.size(); tree_idx++)
      tree_group[tree_idx] = model.tree_info[tree_idx];

    tree_beg_ = tree_begin;
    tree_end_ = tree_end;
    num_group = model.learner_model_param->num_output_group;
  }
};

float GetFvalue(int ridx, int fidx, EntryOneAPI* data, size_t* row_ptr, bool& is_missing) {
  // Binary search
  auto begin_ptr = data + row_ptr[ridx];
  auto end_ptr = data + row_ptr[ridx + 1];
  EntryOneAPI* previous_middle = nullptr;
  while (end_ptr != begin_ptr) {
    auto middle = begin_ptr + (end_ptr - begin_ptr) / 2;
    if (middle == previous_middle) {
      break;
    } else {
      previous_middle = middle;
    }

    if (middle->index == fidx) {
      is_missing = false;
      return middle->fvalue;
    } else if (middle->index < fidx) {
      begin_ptr = middle;
    } else {
      end_ptr = middle;
    }
  }
  is_missing = true;
  return 0.0;
}

float GetLeafWeight(int ridx, const DeviceNodeOneAPI* tree, EntryOneAPI* data, size_t* row_ptr) {
  DeviceNodeOneAPI n = tree[0];
  int node_id = 0;
  bool is_missing;
  while (!n.IsLeaf()) {
    float fvalue = GetFvalue(ridx, n.GetFidx(), data, row_ptr, is_missing);
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

class PredictorOneAPI : public Predictor {
 protected:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const {
    CHECK_NE(model.learner_model_param->num_output_group, 0);
    size_t n = model.learner_model_param->num_output_group * info.num_row_;
    const auto& base_margin = info.base_margin_.HostVector();
    out_preds->Resize(n);
    std::vector<bst_float>& out_preds_h = out_preds->HostVector();
    if (base_margin.size() == n) {
      CHECK_EQ(out_preds->Size(), n);
      std::copy(base_margin.begin(), base_margin.end(), out_preds_h.begin());
    } else {
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
            << "base_score = " << model.learner_model_param->base_score;
        LOG(WARNING) << oss.str();
      }
      std::fill(out_preds_h.begin(), out_preds_h.end(),
                model.learner_model_param->base_score);
    }
  }

  void DevicePredictInternal(DeviceMatrixOneAPI* dmat, HostDeviceVector<float>* out_preds,
                             const gbm::GBTreeModel& model, size_t tree_begin,
                             size_t tree_end) {
    if (tree_end - tree_begin == 0) {
      return;
    }
    model_.Init(model, tree_begin, tree_end, qu_);

    auto& out_preds_vec = out_preds->HostVector();

    DeviceNodeOneAPI* nodes = model_.nodes;
    cl::sycl::buffer<float, 1> out_preds_buf(out_preds_vec.data(), out_preds_vec.size());
    size_t* tree_segments = model_.tree_segments;
    int* tree_group = model_.tree_group;
    size_t* row_ptr = dmat->row_ptr;
    EntryOneAPI* data = dmat->data;
    int num_features = dmat->p_mat->Info().num_col_;
    int num_rows = dmat->row_ptr_size - 1;
    int num_group = model.learner_model_param->num_output_group;

    qu_.submit([&](cl::sycl::handler& cgh) {
      auto out_predictions = out_preds_buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<class PredictInternal>(cl::sycl::range<1>(num_rows), [=](cl::sycl::id<1> pid) {
        int global_idx = pid[0];
        if (global_idx >= num_rows) return;
        if (num_group == 1) {
          float sum = 0.0;
          for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
            const DeviceNodeOneAPI* tree = nodes + tree_segments[tree_idx - tree_begin];
            sum += GetLeafWeight(global_idx, tree, data, row_ptr);
          }
          out_predictions[global_idx] += sum;
        } else {
          for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
            const DeviceNodeOneAPI* tree = nodes + tree_segments[tree_idx - tree_begin];
            int out_prediction_idx = global_idx * num_group + tree_group[tree_idx];
            out_predictions[out_prediction_idx] += GetLeafWeight(global_idx, tree, data, row_ptr);
          }
        }
      });
    }).wait();
  }

 public:
  explicit PredictorOneAPI(Context const* generic_param) :
      Predictor::Predictor{generic_param}, cpu_predictor(Predictor::Create("cpu_predictor", generic_param)) {
    cl::sycl::default_selector selector;
    qu_ = cl::sycl::queue(selector);
  }

  // ntree_limit is a very problematic parameter, as it's ambiguous in the context of
  // multi-output and forest.  Same problem exists for tree_begin
  void PredictBatch(DMatrix* dmat, PredictionCacheEntry* predts,
                    const gbm::GBTreeModel& model, int tree_begin,
                    uint32_t const ntree_limit = 0) override {
    if (this->device_matrix_cache_.find(dmat) ==
        this->device_matrix_cache_.end()) {
      this->device_matrix_cache_.emplace(
          dmat, std::unique_ptr<DeviceMatrixOneAPI>(
                    new DeviceMatrixOneAPI(dmat, qu_)));
    }
    DeviceMatrixOneAPI* device_matrix = device_matrix_cache_.find(dmat)->second.get();

    // tree_begin is not used, right now we just enforce it to be 0.
    CHECK_EQ(tree_begin, 0);
    auto* out_preds = &predts->predictions;
    CHECK_GE(predts->version, tree_begin);
    if (out_preds->Size() == 0 && dmat->Info().num_row_ != 0) {
      CHECK_EQ(predts->version, 0);
    }
    if (predts->version == 0) {
      // out_preds->Size() can be non-zero as it's initialized here before any tree is
      // built at the 0^th iterator.
      this->InitOutPredictions(dmat->Info(), out_preds, model);
    }

    uint32_t const output_groups = model.learner_model_param->num_output_group;
    CHECK_NE(output_groups, 0);
    // Right now we just assume ntree_limit provided by users means number of tree layers
    // in the context of multi-output model
    uint32_t real_ntree_limit = ntree_limit * output_groups;
    if (real_ntree_limit == 0 || real_ntree_limit > model.trees.size()) {
      real_ntree_limit = static_cast<uint32_t>(model.trees.size());
    }

    uint32_t const end_version = (tree_begin + real_ntree_limit) / output_groups;
    // When users have provided ntree_limit, end_version can be lesser, cache is violated
    if (predts->version > end_version) {
      CHECK_NE(ntree_limit, 0);
      this->InitOutPredictions(dmat->Info(), out_preds, model);
      predts->version = 0;
    }
    uint32_t const beg_version = predts->version;
    CHECK_LE(beg_version, end_version);

    if (beg_version < end_version) {
      DevicePredictInternal(device_matrix, out_preds, model,
                            beg_version * output_groups,
                            end_version * output_groups);
    }

    // delta means {size of forest} * {number of newly accumulated layers}
    uint32_t delta = end_version - beg_version;
    CHECK_LE(delta, model.trees.size());
    predts->Update(delta);

    CHECK(out_preds->Size() == output_groups * dmat->Info().num_row_ ||
          out_preds->Size() == dmat->Info().num_row_);
  }

  void InplacePredict(std::any const& x, const gbm::GBTreeModel& model, float missing,
                      PredictionCacheEntry* out_preds, uint32_t tree_begin,
                      unsigned tree_end) const override {
    cpu_predictor->InplacePredict(x, model, missing, out_preds, tree_begin, tree_end);
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit) override {
    cpu_predictor->PredictInstance(inst, out_preds, model, ntree_limit);
  }

  void PredictLeaf(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model, unsigned ntree_limit) override {
    cpu_predictor->PredictLeaf(p_fmat, out_preds, model, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat, std::vector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model, uint32_t ntree_limit,
                           std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) override {
    cpu_predictor->PredictContribution(p_fmat, out_contribs, model, ntree_limit, tree_weights, approximate, condition, condition_feature);
  }

  void PredictInteractionContributions(DMatrix* p_fmat, std::vector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                                       std::vector<bst_float>* tree_weights,
                                       bool approximate) override {
    cpu_predictor->PredictInteractionContributions(p_fmat, out_contribs, model, ntree_limit, tree_weights, approximate);
  }

 private:
  cl::sycl::queue qu_;
  DeviceModelOneAPI model_;

  std::mutex lock_;
  std::unique_ptr<Predictor> cpu_predictor;

  std::unordered_map<DMatrix*, std::unique_ptr<DeviceMatrixOneAPI>>
      device_matrix_cache_;
};

XGBOOST_REGISTER_PREDICTOR(PredictorOneAPI, "oneapi_predictor")
.describe("Make predictions using DPC++.")
.set_body([](Context const* generic_param) {
            return new PredictorOneAPI(generic_param);
          });
}  // namespace predictor
}  // namespace xgboost
