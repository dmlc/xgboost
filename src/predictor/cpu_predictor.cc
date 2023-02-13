/**
 * Copyright 2017-2023 by XGBoost Contributors
 */
#include <algorithm>  // for max, fill, min
#include <any>        // for any, any_cast
#include <cassert>    // for assert
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t, int32_t, uint64_t
#include <memory>     // for unique_ptr, shared_ptr
#include <ostream>    // for char_traits, operator<<, basic_ostream
#include <typeinfo>   // for type_info
#include <vector>     // for vector

#include "../collective/communicator-inl.h"   // for Allreduce, IsDistributed
#include "../collective/communicator.h"       // for Operation
#include "../common/bitfield.h"               // for RBitField8
#include "../common/categorical.h"            // for IsCat, Decision
#include "../common/common.h"                 // for DivRoundUp
#include "../common/math.h"                   // for CheckNAN
#include "../common/threading_utils.h"        // for ParallelFor
#include "../data/adapter.h"                  // for ArrayAdapter, CSRAdapter, CSRArrayAdapter
#include "../data/gradient_index.h"           // for GHistIndexMatrix
#include "../data/proxy_dmatrix.h"            // for DMatrixProxy
#include "../gbm/gbtree_model.h"              // for GBTreeModel, GBTreeModelParam
#include "cpu_treeshap.h"                     // for CalculateContributions
#include "dmlc/registry.h"                    // for DMLC_REGISTRY_FILE_TAG
#include "predict_fn.h"                       // for GetNextNode, GetNextNodeMulti
#include "xgboost/base.h"                     // for bst_float, bst_node_t, bst_omp_uint, bst_fe...
#include "xgboost/context.h"                  // for Context
#include "xgboost/data.h"                     // for Entry, DMatrix, MetaInfo, SparsePage, Batch...
#include "xgboost/host_device_vector.h"       // for HostDeviceVector
#include "xgboost/learner.h"                  // for LearnerModelParam
#include "xgboost/linalg.h"                   // for TensorView, All, VectorView, Tensor
#include "xgboost/logging.h"                  // for LogCheck_EQ, CHECK_EQ, CHECK, LogCheck_NE
#include "xgboost/multi_target_tree_model.h"  // for MultiTargetTree
#include "xgboost/predictor.h"                // for PredictionCacheEntry, Predictor, PredictorReg
#include "xgboost/span.h"                     // for Span
#include "xgboost/tree_model.h"               // for RegTree, MTNotImplemented, RTreeNodeStat

namespace xgboost::predictor {

DMLC_REGISTRY_FILE_TAG(cpu_predictor);

namespace scalar {
template <bool has_missing, bool has_categorical>
bst_node_t GetLeafIndex(RegTree const &tree, const RegTree::FVec &feat,
                        RegTree::CategoricalSplitMatrix const &cats) {
  bst_node_t nidx{0};
  while (!tree[nidx].IsLeaf()) {
    bst_feature_t split_index = tree[nidx].SplitIndex();
    auto fvalue = feat.GetFvalue(split_index);
    nidx = GetNextNode<has_missing, has_categorical>(
        tree[nidx], nidx, fvalue, has_missing && feat.IsMissing(split_index), cats);
  }
  return nidx;
}

bst_float PredValue(const SparsePage::Inst &inst,
                    const std::vector<std::unique_ptr<RegTree>> &trees,
                    const std::vector<int> &tree_info, std::int32_t bst_group,
                    RegTree::FVec *p_feats, std::uint32_t tree_begin, std::uint32_t tree_end) {
  bst_float psum = 0.0f;
  p_feats->Fill(inst);
  for (size_t i = tree_begin; i < tree_end; ++i) {
    if (tree_info[i] == bst_group) {
      auto const &tree = *trees[i];
      bool has_categorical = tree.HasCategoricalSplit();
      auto cats = tree.GetCategoriesMatrix();
      bst_node_t nidx = -1;
      if (has_categorical) {
        nidx = GetLeafIndex<true, true>(tree, *p_feats, cats);
      } else {
        nidx = GetLeafIndex<true, false>(tree, *p_feats, cats);
      }
      psum += (*trees[i])[nidx].LeafValue();
    }
  }
  p_feats->Drop();
  return psum;
}

template <bool has_categorical>
bst_float PredValueByOneTree(const RegTree::FVec &p_feats, RegTree const &tree,
                             RegTree::CategoricalSplitMatrix const &cats) {
  const bst_node_t leaf = p_feats.HasMissing()
                              ? GetLeafIndex<true, has_categorical>(tree, p_feats, cats)
                              : GetLeafIndex<false, has_categorical>(tree, p_feats, cats);
  return tree[leaf].LeafValue();
}
}  // namespace scalar

namespace multi {
template <bool has_missing, bool has_categorical>
bst_node_t GetLeafIndex(MultiTargetTree const &tree, const RegTree::FVec &feat,
                        RegTree::CategoricalSplitMatrix const &cats) {
  bst_node_t nidx{0};
  while (!tree.IsLeaf(nidx)) {
    unsigned split_index = tree.SplitIndex(nidx);
    auto fvalue = feat.GetFvalue(split_index);
    nidx = GetNextNodeMulti<has_missing, has_categorical>(
        tree, nidx, fvalue, has_missing && feat.IsMissing(split_index), cats);
  }
  return nidx;
}

template <bool has_categorical>
void PredValueByOneTree(RegTree::FVec const &p_feats, MultiTargetTree const &tree,
                        RegTree::CategoricalSplitMatrix const &cats,
                        linalg::VectorView<float> out_predt) {
  bst_node_t const leaf = p_feats.HasMissing()
                              ? GetLeafIndex<true, has_categorical>(tree, p_feats, cats)
                              : GetLeafIndex<false, has_categorical>(tree, p_feats, cats);
  auto leaf_value = tree.LeafValue(leaf);
  assert(out_predt.Shape(0) == leaf_value.Shape(0) && "shape mismatch.");
  for (size_t i = 0; i < leaf_value.Size(); ++i) {
    out_predt(i) += leaf_value(i);
  }
}
}  // namespace multi

namespace {
void PredictByAllTrees(gbm::GBTreeModel const &model, std::uint32_t const tree_begin,
                       std::uint32_t const tree_end, std::size_t const predict_offset,
                       std::vector<RegTree::FVec> const &thread_temp, std::size_t const offset,
                       std::size_t const block_size, linalg::MatrixView<float> out_predt) {
  for (std::uint32_t tree_id = tree_begin; tree_id < tree_end; ++tree_id) {
    auto const &tree = *model.trees.at(tree_id);
    auto const &cats = tree.GetCategoriesMatrix();
    bool has_categorical = tree.HasCategoricalSplit();

    if (tree.IsMultiTarget()) {
      if (has_categorical) {
        for (std::size_t i = 0; i < block_size; ++i) {
          auto t_predts = out_predt.Slice(predict_offset + i, linalg::All());
          multi::PredValueByOneTree<true>(thread_temp[offset + i], *tree.GetMultiTargetTree(), cats,
                                          t_predts);
        }
      } else {
        for (std::size_t i = 0; i < block_size; ++i) {
          auto t_predts = out_predt.Slice(predict_offset + i, linalg::All());
          multi::PredValueByOneTree<false>(thread_temp[offset + i], *tree.GetMultiTargetTree(),
                                           cats, t_predts);
        }
      }
    } else {
      auto const gid = model.tree_info[tree_id];
      if (has_categorical) {
        for (std::size_t i = 0; i < block_size; ++i) {
          out_predt(predict_offset + i, gid) +=
              scalar::PredValueByOneTree<true>(thread_temp[offset + i], tree, cats);
        }
      } else {
        for (std::size_t i = 0; i < block_size; ++i) {
          out_predt(predict_offset + i, gid) +=
              scalar::PredValueByOneTree<true>(thread_temp[offset + i], tree, cats);
        }
      }
    }
  }
}

template <typename DataView>
void FVecFill(const size_t block_size, const size_t batch_offset, const int num_feature,
              DataView *batch, const size_t fvec_offset, std::vector<RegTree::FVec> *p_feats) {
  for (size_t i = 0; i < block_size; ++i) {
    RegTree::FVec &feats = (*p_feats)[fvec_offset + i];
    if (feats.Size() == 0) {
      feats.Init(num_feature);
    }
    const SparsePage::Inst inst = (*batch)[batch_offset + i];
    feats.Fill(inst);
  }
}

void FVecDrop(std::size_t const block_size, std::size_t const fvec_offset,
              std::vector<RegTree::FVec> *p_feats) {
  for (size_t i = 0; i < block_size; ++i) {
    RegTree::FVec &feats = (*p_feats)[fvec_offset + i];
    feats.Drop();
  }
}

static std::size_t constexpr kUnroll = 8;

struct SparsePageView {
  bst_row_t base_rowid;
  HostSparsePageView view;

  explicit SparsePageView(SparsePage const *p) : base_rowid{p->base_rowid} { view = p->GetView(); }
  SparsePage::Inst operator[](size_t i) { return view[i]; }
  size_t Size() const { return view.Size(); }
};

struct GHistIndexMatrixView {
 private:
  GHistIndexMatrix const &page_;
  std::uint64_t const n_features_;
  common::Span<FeatureType const> ft_;
  common::Span<Entry> workspace_;
  std::vector<size_t> current_unroll_;

  std::vector<std::uint32_t> const& ptrs_;
  std::vector<float> const& mins_;
  std::vector<float> const& values_;

 public:
  size_t base_rowid;

 public:
  GHistIndexMatrixView(GHistIndexMatrix const &_page, uint64_t n_feat,
                       common::Span<FeatureType const> ft, common::Span<Entry> workplace,
                       int32_t n_threads)
      : page_{_page},
        n_features_{n_feat},
        ft_{ft},
        workspace_{workplace},
        current_unroll_(n_threads > 0 ? n_threads : 1, 0),
        ptrs_{_page.cut.Ptrs()},
        mins_{_page.cut.MinValues()},
        values_{_page.cut.Values()},
        base_rowid{_page.base_rowid} {}

  SparsePage::Inst operator[](size_t r) {
    auto t = omp_get_thread_num();
    auto const beg = (n_features_ * kUnroll * t) + (current_unroll_[t] * n_features_);
    size_t non_missing{static_cast<std::size_t>(beg)};

    for (bst_feature_t c = 0; c < n_features_; ++c) {
      float f = page_.GetFvalue(ptrs_, values_, mins_, r, c, common::IsCat(ft_, c));
      if (!common::CheckNAN(f)) {
        workspace_[non_missing] = Entry{c, f};
        ++non_missing;
      }
    }

    auto ret = workspace_.subspan(beg, non_missing - beg);
    current_unroll_[t]++;
    if (current_unroll_[t] == kUnroll) {
      current_unroll_[t] = 0;
    }
    return ret;
  }
  size_t Size() const { return page_.Size(); }
};

template <typename Adapter>
class AdapterView {
  Adapter* adapter_;
  float missing_;
  common::Span<Entry> workspace_;
  std::vector<size_t> current_unroll_;

 public:
  explicit AdapterView(Adapter *adapter, float missing, common::Span<Entry> workplace,
                       int32_t n_threads)
      : adapter_{adapter},
        missing_{missing},
        workspace_{workplace},
        current_unroll_(n_threads > 0 ? n_threads : 1, 0) {}
  SparsePage::Inst operator[](size_t i) {
    bst_feature_t columns = adapter_->NumColumns();
    auto const &batch = adapter_->Value();
    auto row = batch.GetLine(i);
    auto t = omp_get_thread_num();
    auto const beg = (columns * kUnroll * t) + (current_unroll_[t] * columns);
    size_t non_missing {beg};
    for (size_t c = 0; c < row.Size(); ++c) {
      auto e = row.GetElement(c);
      if (missing_ != e.value && !common::CheckNAN(e.value)) {
        workspace_[non_missing] =
            Entry{static_cast<bst_feature_t>(e.column_idx), e.value};
        ++non_missing;
      }
    }
    auto ret = workspace_.subspan(beg, non_missing - beg);
    current_unroll_[t]++;
    if (current_unroll_[t] == kUnroll) {
      current_unroll_[t] = 0;
    }
    return ret;
  }

  size_t Size() const { return adapter_->NumRows(); }

  bst_row_t const static base_rowid = 0;  // NOLINT
};

template <typename DataView, size_t block_of_rows_size>
void PredictBatchByBlockOfRowsKernel(DataView batch, gbm::GBTreeModel const &model,
                                     std::uint32_t tree_begin, std::uint32_t tree_end,
                                     std::vector<RegTree::FVec> *p_thread_temp, int32_t n_threads,
                                     linalg::TensorView<float, 2> out_predt) {
  auto &thread_temp = *p_thread_temp;

  // parallel over local batch
  const auto nsize = static_cast<bst_omp_uint>(batch.Size());
  const int num_feature = model.learner_model_param->num_feature;
  omp_ulong n_blocks = common::DivRoundUp(nsize, block_of_rows_size);

  common::ParallelFor(n_blocks, n_threads, [&](bst_omp_uint block_id) {
    const size_t batch_offset = block_id * block_of_rows_size;
    const size_t block_size = std::min(nsize - batch_offset, block_of_rows_size);
    const size_t fvec_offset = omp_get_thread_num() * block_of_rows_size;

    FVecFill(block_size, batch_offset, num_feature, &batch, fvec_offset, p_thread_temp);
    // process block of rows through all trees to keep cache locality
    PredictByAllTrees(model, tree_begin, tree_end, batch_offset + batch.base_rowid, thread_temp,
                      fvec_offset, block_size, out_predt);
    FVecDrop(block_size, fvec_offset, p_thread_temp);
  });
}

float FillNodeMeanValues(RegTree const *tree, bst_node_t nidx, std::vector<float> *mean_values) {
  bst_float result;
  auto &node = (*tree)[nidx];
  auto &node_mean_values = *mean_values;
  if (node.IsLeaf()) {
    result = node.LeafValue();
  } else {
    result = FillNodeMeanValues(tree, node.LeftChild(), mean_values) *
             tree->Stat(node.LeftChild()).sum_hess;
    result += FillNodeMeanValues(tree, node.RightChild(), mean_values) *
              tree->Stat(node.RightChild()).sum_hess;
    result /= tree->Stat(nidx).sum_hess;
  }
  node_mean_values[nidx] = result;
  return result;
}

void FillNodeMeanValues(RegTree const* tree, std::vector<float>* mean_values) {
  size_t num_nodes = tree->NumNodes();
  if (mean_values->size() == num_nodes) {
    return;
  }
  mean_values->resize(num_nodes);
  FillNodeMeanValues(tree, 0, mean_values);
}

// init thread buffers
static void InitThreadTemp(int nthread, std::vector<RegTree::FVec> *out) {
  int prev_thread_temp_size = out->size();
  if (prev_thread_temp_size < nthread) {
    out->resize(nthread, RegTree::FVec());
  }
}
}  // anonymous namespace

/**
 * @brief A helper class for prediction when the DMatrix is split by column.
 *
 * When data is split by column, a local DMatrix only contains a subset of features. All the workers
 * in a distributed/federated environment need to cooperate to produce a prediction. This is done in
 * two passes with the help of bit vectors.
 *
 * First pass:
 * for each tree:
 *   for each row:
 *     for each node:
 *       if the feature is available and passes the filter, mark the corresponding decision bit
 *       if the feature is missing, mark the missing bit
 *
 * Once the two bit vectors are populated, run allreduce on both, using bitwise OR for the decision
 * bits, and bitwise AND for the missing bits.
 *
 * Second pass:
 * for each tree:
 *   for each row:
 *     find the leaf node using the decision and missing bits, return the leaf value
 *
 * The size of the decision/missing bit vector is:
 *   number of rows in a batch * sum(number of nodes in each tree)
 */
class ColumnSplitHelper {
 public:
  ColumnSplitHelper(std::int32_t n_threads, gbm::GBTreeModel const &model, uint32_t tree_begin,
                    uint32_t tree_end)
      : n_threads_{n_threads}, model_{model}, tree_begin_{tree_begin}, tree_end_{tree_end} {
    auto const n_trees = tree_end_ - tree_begin_;
    tree_sizes_.resize(n_trees);
    tree_offsets_.resize(n_trees);
    for (decltype(tree_begin) i = 0; i < n_trees; i++) {
      auto const &tree = *model_.trees[tree_begin_ + i];
      tree_sizes_[i] = tree.GetNodes().size();
    }
    // std::exclusive_scan (only available in c++17) equivalent to get tree offsets.
    tree_offsets_[0] = 0;
    for (decltype(tree_begin) i = 1; i < n_trees; i++) {
      tree_offsets_[i] = tree_offsets_[i - 1] + tree_sizes_[i - 1];
    }
    bits_per_row_ = tree_offsets_.back() + tree_sizes_.back();

    InitThreadTemp(n_threads_ * kBlockOfRowsSize, &feat_vecs_);
  }

  // Disable copy (and move) semantics.
  ColumnSplitHelper(ColumnSplitHelper const &) = delete;
  ColumnSplitHelper &operator=(ColumnSplitHelper const &) = delete;
  ColumnSplitHelper(ColumnSplitHelper &&) noexcept = delete;
  ColumnSplitHelper &operator=(ColumnSplitHelper &&) noexcept = delete;

  void PredictDMatrix(DMatrix *p_fmat, std::vector<bst_float> *out_preds) {
    CHECK(xgboost::collective::IsDistributed())
        << "column-split prediction is only supported for distributed training";

    for (auto const &batch : p_fmat->GetBatches<SparsePage>()) {
      CHECK_EQ(out_preds->size(),
               p_fmat->Info().num_row_ * model_.learner_model_param->num_output_group);
      PredictBatchKernel<SparsePageView, kBlockOfRowsSize>(SparsePageView{&batch}, out_preds);
    }
  }

 private:
  using BitVector = RBitField8;

  void InitBitVectors(std::size_t n_rows) {
    n_rows_ = n_rows;
    auto const size = BitVector::ComputeStorageSize(bits_per_row_ * n_rows_);
    decision_storage_.resize(size);
    decision_bits_ = BitVector(common::Span<BitVector::value_type>(decision_storage_));
    missing_storage_.resize(size);
    missing_bits_ = BitVector(common::Span<BitVector::value_type>(missing_storage_));
  }

  void ClearBitVectors() {
    std::fill(decision_storage_.begin(), decision_storage_.end(), 0);
    std::fill(missing_storage_.begin(), missing_storage_.end(), 0);
  }

  std::size_t BitIndex(std::size_t tree_id, std::size_t row_id, std::size_t node_id) const {
    size_t tree_index = tree_id - tree_begin_;
    return tree_offsets_[tree_index] * n_rows_ + row_id * tree_sizes_[tree_index] + node_id;
  }

  void AllreduceBitVectors() {
    collective::Allreduce<collective::Operation::kBitwiseOR>(decision_storage_.data(),
                                                             decision_storage_.size());
    collective::Allreduce<collective::Operation::kBitwiseAND>(missing_storage_.data(),
                                                              missing_storage_.size());
  }

  void MaskOneTree(RegTree::FVec const &feat, std::size_t tree_id, std::size_t row_id) {
    auto const &tree = *model_.trees[tree_id];
    auto const &cats = tree.GetCategoriesMatrix();
    auto const has_categorical = tree.HasCategoricalSplit();
    bst_node_t n_nodes = tree.GetNodes().size();

    for (bst_node_t nid = 0; nid < n_nodes; nid++) {
      auto const &node = tree[nid];
      if (node.IsDeleted() || node.IsLeaf()) {
        continue;
      }

      auto const bit_index = BitIndex(tree_id, row_id, nid);
      unsigned split_index = node.SplitIndex();
      if (feat.IsMissing(split_index)) {
        missing_bits_.Set(bit_index);
        continue;
      }

      auto const fvalue = feat.GetFvalue(split_index);
      if (has_categorical && common::IsCat(cats.split_type, nid)) {
        auto const node_categories =
            cats.categories.subspan(cats.node_ptr[nid].beg, cats.node_ptr[nid].size);
        if (!common::Decision(node_categories, fvalue)) {
          decision_bits_.Set(bit_index);
        }
        continue;
      }

      if (fvalue >= node.SplitCond()) {
        decision_bits_.Set(bit_index);
      }
    }
  }

  void MaskAllTrees(std::size_t batch_offset, std::size_t fvec_offset, std::size_t block_size) {
    for (auto tree_id = tree_begin_; tree_id < tree_end_; ++tree_id) {
      for (size_t i = 0; i < block_size; ++i) {
        MaskOneTree(feat_vecs_[fvec_offset + i], tree_id, batch_offset + i);
      }
    }
  }

  bst_node_t GetNextNode(RegTree::Node const &node, std::size_t bit_index) {
    if (missing_bits_.Check(bit_index)) {
      return node.DefaultChild();
    } else {
      return node.LeftChild() + decision_bits_.Check(bit_index);
    }
  }

  bst_node_t GetLeafIndex(RegTree const &tree, std::size_t tree_id, std::size_t row_id) {
    bst_node_t nid = 0;
    while (!tree[nid].IsLeaf()) {
      auto const bit_index = BitIndex(tree_id, row_id, nid);
      nid = GetNextNode(tree[nid], bit_index);
    }
    return nid;
  }

  bst_float PredictOneTree(std::size_t tree_id, std::size_t row_id) {
    auto const &tree = *model_.trees[tree_id];
    auto const leaf = GetLeafIndex(tree, tree_id, row_id);
    return tree[leaf].LeafValue();
  }

  void PredictAllTrees(std::vector<bst_float> *out_preds, std::size_t batch_offset,
                       std::size_t predict_offset, std::size_t num_group, std::size_t block_size) {
    auto &preds = *out_preds;
    for (size_t tree_id = tree_begin_; tree_id < tree_end_; ++tree_id) {
      auto const gid = model_.tree_info[tree_id];
      for (size_t i = 0; i < block_size; ++i) {
        preds[(predict_offset + i) * num_group + gid] += PredictOneTree(tree_id, batch_offset + i);
      }
    }
  }

  template <typename DataView, size_t block_of_rows_size>
  void PredictBatchKernel(DataView batch, std::vector<bst_float> *out_preds) {
    auto const num_group = model_.learner_model_param->num_output_group;

    // parallel over local batch
    auto const nsize = batch.Size();
    auto const num_feature = model_.learner_model_param->num_feature;
    auto const n_blocks = common::DivRoundUp(nsize, block_of_rows_size);
    InitBitVectors(nsize);

    // auto block_id has the same type as `n_blocks`.
    common::ParallelFor(n_blocks, n_threads_, [&](auto block_id) {
      auto const batch_offset = block_id * block_of_rows_size;
      auto const block_size = std::min(static_cast<std::size_t>(nsize - batch_offset),
                                       static_cast<std::size_t>(block_of_rows_size));
      auto const fvec_offset = omp_get_thread_num() * block_of_rows_size;

      FVecFill(block_size, batch_offset, num_feature, &batch, fvec_offset, &feat_vecs_);
      MaskAllTrees(batch_offset, fvec_offset, block_size);
      FVecDrop(block_size, fvec_offset, &feat_vecs_);
    });

    AllreduceBitVectors();

    // auto block_id has the same type as `n_blocks`.
    common::ParallelFor(n_blocks, n_threads_, [&](auto block_id) {
      auto const batch_offset = block_id * block_of_rows_size;
      auto const block_size = std::min(static_cast<std::size_t>(nsize - batch_offset),
                                       static_cast<std::size_t>(block_of_rows_size));
      PredictAllTrees(out_preds, batch_offset, batch_offset + batch.base_rowid, num_group,
                      block_size);
    });

    ClearBitVectors();
  }

  static std::size_t constexpr kBlockOfRowsSize = 64;

  std::int32_t const n_threads_;
  gbm::GBTreeModel const &model_;
  uint32_t const tree_begin_;
  uint32_t const tree_end_;

  std::vector<std::size_t> tree_sizes_{};
  std::vector<std::size_t> tree_offsets_{};
  std::size_t bits_per_row_{};
  std::vector<RegTree::FVec> feat_vecs_{};

  std::size_t n_rows_;
  /**
   * @brief Stores decision bit for each split node.
   *
   * Conceptually it's a 3-dimensional bit matrix:
   *   - 1st dimension is the tree index, from `tree_begin_` to `tree_end_`.
   *   - 2nd dimension is the row index, for each row in the batch.
   *   - 3rd dimension is the node id, for each node in the tree.
   *
   * Since we have to ship the whole thing over the wire to do an allreduce, the matrix is flattened
   * into a 1-dimensional array.
   *
   * First, it's divided by the tree index:
   *
   * [ tree 0 ] [ tree 1 ] ...
   *
   * Then each tree is divided by row:
   *
   * [             tree 0              ] [           tree 1     ] ...
   * [ row 0 ] [ row 1 ] ... [ row n-1 ] [ row 0 ] ...
   *
   * Finally, each row is divided by the node id:
   *
   * [                             tree 0                                         ]
   * [              row 0                 ] [        row 1           ] ...
   * [ node 0 ] [ node 1 ] ... [ node n-1 ] [ node 0 ] ...
   *
   * The first two dimensions are fixed length, while the last dimension is variable length since
   * each tree may have a different number of nodes. We precompute the tree offsets, which are the
   * cumulative sums of tree sizes. The index of tree t, row r, node n is:
   *   index(t, r, n) = tree_offsets[t] * n_rows + r * tree_sizes[t] + n
   */
  std::vector<BitVector::value_type> decision_storage_{};
  BitVector decision_bits_{};
  /**
   * @brief Stores whether the feature is missing for each split node.
   *
   * See above for the storage layout.
   */
  std::vector<BitVector::value_type> missing_storage_{};
  BitVector missing_bits_{};
};

class CPUPredictor : public Predictor {
 protected:
  void PredictDMatrix(DMatrix *p_fmat, std::vector<bst_float> *out_preds,
                      gbm::GBTreeModel const &model, int32_t tree_begin, int32_t tree_end) const {
    if (p_fmat->Info().IsColumnSplit()) {
      ColumnSplitHelper helper(this->ctx_->Threads(), model, tree_begin, tree_end);
      helper.PredictDMatrix(p_fmat, out_preds);
      return;
    }

    auto const n_threads = this->ctx_->Threads();
    constexpr double kDensityThresh = .5;
    size_t total =
        std::max(p_fmat->Info().num_row_ * p_fmat->Info().num_col_, static_cast<uint64_t>(1));
    double density = static_cast<double>(p_fmat->Info().num_nonzero_) / static_cast<double>(total);
    bool blocked = density > kDensityThresh;

    std::vector<RegTree::FVec> feat_vecs;
    InitThreadTemp(n_threads * (blocked ? kBlockOfRowsSize : 1), &feat_vecs);

    std::size_t n_samples = p_fmat->Info().num_row_;
    std::size_t n_groups = model.learner_model_param->OutputLength();
    CHECK_EQ(out_preds->size(), n_samples * n_groups);
    linalg::TensorView<float, 2> out_predt{*out_preds, {n_samples, n_groups}, ctx_->gpu_id};

    if (!p_fmat->PageExists<SparsePage>()) {
      std::vector<Entry> workspace(p_fmat->Info().num_col_ * kUnroll * n_threads);
      auto ft = p_fmat->Info().feature_types.ConstHostVector();
      for (auto const &batch : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, {})) {
        if (blocked) {
          PredictBatchByBlockOfRowsKernel<GHistIndexMatrixView, kBlockOfRowsSize>(
              GHistIndexMatrixView{batch, p_fmat->Info().num_col_, ft, workspace, n_threads}, model,
              tree_begin, tree_end, &feat_vecs, n_threads, out_predt);
        } else {
          PredictBatchByBlockOfRowsKernel<GHistIndexMatrixView, 1>(
              GHistIndexMatrixView{batch, p_fmat->Info().num_col_, ft, workspace, n_threads}, model,
              tree_begin, tree_end, &feat_vecs, n_threads, out_predt);
        }
      }
    } else {
      for (auto const &batch : p_fmat->GetBatches<SparsePage>()) {
        if (blocked) {
          PredictBatchByBlockOfRowsKernel<SparsePageView, kBlockOfRowsSize>(
              SparsePageView{&batch}, model, tree_begin, tree_end, &feat_vecs, n_threads,
              out_predt);

        } else {
          PredictBatchByBlockOfRowsKernel<SparsePageView, 1>(SparsePageView{&batch}, model,
                                                             tree_begin, tree_end, &feat_vecs,
                                                             n_threads, out_predt);
        }
      }
    }
  }

 public:
  explicit CPUPredictor(Context const *ctx) : Predictor::Predictor{ctx} {}

  void PredictBatch(DMatrix *dmat, PredictionCacheEntry *predts, const gbm::GBTreeModel &model,
                    uint32_t tree_begin, uint32_t tree_end = 0) const override {
    auto *out_preds = &predts->predictions;
    // This is actually already handled in gbm, but large amount of tests rely on the
    // behaviour.
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }
    this->PredictDMatrix(dmat, &out_preds->HostVector(), model, tree_begin, tree_end);
  }

  template <typename Adapter, size_t kBlockSize>
  void DispatchedInplacePredict(std::any const &x, std::shared_ptr<DMatrix> p_m,
                                const gbm::GBTreeModel &model, float missing,
                                PredictionCacheEntry *out_preds, uint32_t tree_begin,
                                uint32_t tree_end) const {
    auto const n_threads = this->ctx_->Threads();
    auto m = std::any_cast<std::shared_ptr<Adapter>>(x);
    CHECK_EQ(m->NumColumns(), model.learner_model_param->num_feature)
        << "Number of columns in data must equal to trained model.";
    if (p_m) {
      p_m->Info().num_row_ = m->NumRows();
      this->InitOutPredictions(p_m->Info(), &(out_preds->predictions), model);
    } else {
      MetaInfo info;
      info.num_row_ = m->NumRows();
      this->InitOutPredictions(info, &(out_preds->predictions), model);
    }

    std::vector<Entry> workspace(m->NumColumns() * kUnroll * n_threads);
    auto &predictions = out_preds->predictions.HostVector();
    std::vector<RegTree::FVec> thread_temp;
    InitThreadTemp(n_threads * kBlockSize, &thread_temp);
    std::size_t n_groups = model.learner_model_param->OutputLength();
    linalg::TensorView<float, 2> out_predt{predictions, {m->NumRows(), n_groups}, Context::kCpuId};
    PredictBatchByBlockOfRowsKernel<AdapterView<Adapter>, kBlockSize>(
        AdapterView<Adapter>(m.get(), missing, common::Span<Entry>{workspace}, n_threads), model,
        tree_begin, tree_end, &thread_temp, n_threads, out_predt);
  }

  bool InplacePredict(std::shared_ptr<DMatrix> p_m, const gbm::GBTreeModel &model, float missing,
                      PredictionCacheEntry *out_preds, uint32_t tree_begin,
                      unsigned tree_end) const override {
    auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
    CHECK(proxy)<< "Inplace predict accepts only DMatrixProxy as input.";
    auto x = proxy->Adapter();
    if (x.type() == typeid(std::shared_ptr<data::DenseAdapter>)) {
      this->DispatchedInplacePredict<data::DenseAdapter, kBlockOfRowsSize>(
          x, p_m, model, missing, out_preds, tree_begin, tree_end);
    } else if (x.type() == typeid(std::shared_ptr<data::CSRAdapter>)) {
      this->DispatchedInplacePredict<data::CSRAdapter, 1>(x, p_m, model, missing, out_preds,
                                                          tree_begin, tree_end);
    } else if (x.type() == typeid(std::shared_ptr<data::ArrayAdapter>)) {
      this->DispatchedInplacePredict<data::ArrayAdapter, kBlockOfRowsSize>(
          x, p_m, model, missing, out_preds, tree_begin, tree_end);
    } else if (x.type() == typeid(std::shared_ptr<data::CSRArrayAdapter>)) {
      this->DispatchedInplacePredict<data::CSRArrayAdapter, 1>(x, p_m, model, missing, out_preds,
                                                               tree_begin, tree_end);
    } else {
      return false;
    }
    return true;
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit) const override {
    CHECK(!model.learner_model_param->IsVectorLeaf()) << "predict instance" << MTNotImplemented();
    std::vector<RegTree::FVec> feat_vecs;
    feat_vecs.resize(1, RegTree::FVec());
    feat_vecs[0].Init(model.learner_model_param->num_feature);
    ntree_limit *= model.learner_model_param->num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    out_preds->resize(model.learner_model_param->num_output_group);
    auto base_score = model.learner_model_param->BaseScore(ctx_)(0);
    // loop over output groups
    for (uint32_t gid = 0; gid < model.learner_model_param->num_output_group; ++gid) {
      (*out_preds)[gid] = scalar::PredValue(inst, model.trees, model.tree_info, gid, &feat_vecs[0],
                                            0, ntree_limit) +
                          base_score;
    }
  }

  void PredictLeaf(DMatrix *p_fmat, HostDeviceVector<bst_float> *out_preds,
                   const gbm::GBTreeModel &model, unsigned ntree_limit) const override {
    auto const n_threads = this->ctx_->Threads();
    std::vector<RegTree::FVec> feat_vecs;
    const int num_feature = model.learner_model_param->num_feature;
    InitThreadTemp(n_threads, &feat_vecs);
    const MetaInfo &info = p_fmat->Info();
    // number of valid trees
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    std::vector<bst_float> &preds = out_preds->HostVector();
    preds.resize(info.num_row_ * ntree_limit);
    // start collecting the prediction
    for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
      // parallel over local batch
      auto page = batch.GetView();
      common::ParallelFor(page.Size(), n_threads, [&](auto i) {
        const int tid = omp_get_thread_num();
        auto ridx = static_cast<size_t>(batch.base_rowid + i);
        RegTree::FVec &feats = feat_vecs[tid];
        if (feats.Size() == 0) {
          feats.Init(num_feature);
        }
        feats.Fill(page[i]);
        for (std::uint32_t j = 0; j < ntree_limit; ++j) {
          auto const &tree = *model.trees[j];
          auto const &cats = tree.GetCategoriesMatrix();
          bst_node_t nidx;
          if (tree.IsMultiTarget()) {
            nidx = multi::GetLeafIndex<true, true>(*tree.GetMultiTargetTree(), feats, cats);
          } else {
            nidx = scalar::GetLeafIndex<true, true>(tree, feats, cats);
          }
          preds[ridx * ntree_limit + j] = static_cast<bst_float>(nidx);
        }
        feats.Drop();
      });
    }
  }

  void PredictContribution(DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                           const gbm::GBTreeModel &model, uint32_t ntree_limit,
                           std::vector<bst_float> const *tree_weights, bool approximate,
                           int condition, unsigned condition_feature) const override {
    CHECK(!model.learner_model_param->IsVectorLeaf())
        << "Predict contribution" << MTNotImplemented();
    auto const n_threads = this->ctx_->Threads();
    const int num_feature = model.learner_model_param->num_feature;
    std::vector<RegTree::FVec> feat_vecs;
    InitThreadTemp(n_threads, &feat_vecs);
    const MetaInfo& info = p_fmat->Info();
    // number of valid trees
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    const int ngroup = model.learner_model_param->num_output_group;
    CHECK_NE(ngroup, 0);
    size_t const ncolumns = num_feature + 1;
    CHECK_NE(ncolumns, 0);
    // allocate space for (number of features + bias) times the number of rows
    std::vector<bst_float>& contribs = out_contribs->HostVector();
    contribs.resize(info.num_row_ * ncolumns * model.learner_model_param->num_output_group);
    // make sure contributions is zeroed, we could be reusing a previously
    // allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    // initialize tree node mean values
    std::vector<std::vector<float>> mean_values(ntree_limit);
    common::ParallelFor(ntree_limit, n_threads, [&](bst_omp_uint i) {
      FillNodeMeanValues(model.trees[i].get(), &(mean_values[i]));
    });
    auto base_margin = info.base_margin_.View(Context::kCpuId);
    auto base_score = model.learner_model_param->BaseScore(Context::kCpuId)(0);
    // start collecting the contributions
    for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
      auto page = batch.GetView();
      // parallel over local batch
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
      common::ParallelFor(nsize, n_threads, [&](bst_omp_uint i) {
        auto row_idx = static_cast<size_t>(batch.base_rowid + i);
        RegTree::FVec &feats = feat_vecs[omp_get_thread_num()];
        if (feats.Size() == 0) {
          feats.Init(num_feature);
        }
        std::vector<bst_float> this_tree_contribs(ncolumns);
        // loop over all classes
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float* p_contribs = &contribs[(row_idx * ngroup + gid) * ncolumns];
          feats.Fill(page[i]);
          // calculate contributions
          for (unsigned j = 0; j < ntree_limit; ++j) {
            auto *tree_mean_values = &mean_values.at(j);
            std::fill(this_tree_contribs.begin(), this_tree_contribs.end(), 0);
            if (model.tree_info[j] != gid) {
              continue;
            }
            if (!approximate) {
              CalculateContributions(*model.trees[j], feats, tree_mean_values,
                                     &this_tree_contribs[0], condition, condition_feature);
            } else {
              model.trees[j]->CalculateContributionsApprox(
                  feats, tree_mean_values, &this_tree_contribs[0]);
            }
            for (size_t ci = 0; ci < ncolumns; ++ci) {
              p_contribs[ci] +=
                  this_tree_contribs[ci] *
                  (tree_weights == nullptr ? 1 : (*tree_weights)[j]);
            }
          }
          feats.Drop();
          // add base margin to BIAS
          if (base_margin.Size() != 0) {
            CHECK_EQ(base_margin.Shape(1), ngroup);
            p_contribs[ncolumns - 1] += base_margin(row_idx, gid);
          } else {
            p_contribs[ncolumns - 1] += base_score;
          }
        }
      });
    }
  }

  void PredictInteractionContributions(DMatrix *p_fmat, HostDeviceVector<bst_float> *out_contribs,
                                       const gbm::GBTreeModel &model, unsigned ntree_limit,
                                       std::vector<bst_float> const *tree_weights,
                                       bool approximate) const override {
    CHECK(!model.learner_model_param->IsVectorLeaf())
        << "Predict interaction contribution" << MTNotImplemented();
    const MetaInfo& info = p_fmat->Info();
    const int ngroup = model.learner_model_param->num_output_group;
    size_t const ncolumns = model.learner_model_param->num_feature;
    const unsigned row_chunk = ngroup * (ncolumns + 1) * (ncolumns + 1);
    const unsigned mrow_chunk = (ncolumns + 1) * (ncolumns + 1);
    const unsigned crow_chunk = ngroup * (ncolumns + 1);

    // allocate space for (number of features^2) times the number of rows and tmp off/on contribs
    std::vector<bst_float>& contribs = out_contribs->HostVector();
    contribs.resize(info.num_row_ * ngroup * (ncolumns + 1) * (ncolumns + 1));
    HostDeviceVector<bst_float> contribs_off_hdv(info.num_row_ * ngroup * (ncolumns + 1));
    auto &contribs_off = contribs_off_hdv.HostVector();
    HostDeviceVector<bst_float> contribs_on_hdv(info.num_row_ * ngroup * (ncolumns + 1));
    auto &contribs_on = contribs_on_hdv.HostVector();
    HostDeviceVector<bst_float> contribs_diag_hdv(info.num_row_ * ngroup * (ncolumns + 1));
    auto &contribs_diag = contribs_diag_hdv.HostVector();

    // Compute the difference in effects when conditioning on each of the features on and off
    // see: Axiomatic characterizations of probabilistic and
    //      cardinal-probabilistic interaction indices
    PredictContribution(p_fmat, &contribs_diag_hdv, model, ntree_limit,
                        tree_weights, approximate, 0, 0);
    for (size_t i = 0; i < ncolumns + 1; ++i) {
      PredictContribution(p_fmat, &contribs_off_hdv, model, ntree_limit,
                          tree_weights, approximate, -1, i);
      PredictContribution(p_fmat, &contribs_on_hdv, model, ntree_limit,
                          tree_weights, approximate, 1, i);

      for (size_t j = 0; j < info.num_row_; ++j) {
        for (int l = 0; l < ngroup; ++l) {
          const unsigned o_offset = j * row_chunk + l * mrow_chunk + i * (ncolumns + 1);
          const unsigned c_offset = j * crow_chunk + l * (ncolumns + 1);
          contribs[o_offset + i] = 0;
          for (size_t k = 0; k < ncolumns + 1; ++k) {
            // fill in the diagonal with additive effects, and off-diagonal with the interactions
            if (k == i) {
              contribs[o_offset + i] += contribs_diag[c_offset + k];
            } else {
              contribs[o_offset + k] = (contribs_on[c_offset + k] - contribs_off[c_offset + k])/2.0;
              contribs[o_offset + i] -= contribs[o_offset + k];
            }
          }
        }
      }
    }
  }

 private:
  static size_t constexpr kBlockOfRowsSize = 64;
};

XGBOOST_REGISTER_PREDICTOR(CPUPredictor, "cpu_predictor")
    .describe("Make predictions using CPU.")
    .set_body([](Context const *ctx) { return new CPUPredictor(ctx); });
}  // namespace xgboost::predictor
