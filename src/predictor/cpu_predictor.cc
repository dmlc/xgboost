/**
 * Copyright 2017-2025, XGBoost Contributors
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
#include "../collective/allreduce.h"
#include "../common/bitfield.h"               // for RBitField8
#include "../common/common.h"                 // for DivRoundUp
#include "../common/error_msg.h"              // for InplacePredictProxy
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

template <bool has_categorical>
[[nodiscard]] float PredValueByOneTree(const RegTree::FVec &p_feats, RegTree const &tree,
                                       RegTree::CategoricalSplitMatrix const &cats) noexcept(true) {
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
    bst_feature_t split_index = tree.SplitIndex(nidx);
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
              scalar::PredValueByOneTree<false>(thread_temp[offset + i], tree, cats);
        }
      }
    }
  }
}

template <typename DataView>
void FVecFill(std::size_t const block_size, std::size_t const batch_offset,
              bst_feature_t n_features, DataView *p_batch, std::size_t const fvec_offset,
              std::vector<RegTree::FVec> *p_feats) {
  auto &feats_vec = *p_feats;
  auto &batch = *p_batch;
  for (std::size_t i = 0; i < block_size; ++i) {
    RegTree::FVec &feats = feats_vec[fvec_offset + i];
    if (feats.Size() == 0) {
      feats.Init(n_features);
    }
    batch.Fill(batch_offset + i, &feats);
  }
}

void FVecDrop(std::size_t const block_size, std::size_t const fvec_offset,
              std::vector<RegTree::FVec> *p_feats) {
  for (size_t i = 0; i < block_size; ++i) {
    RegTree::FVec &feats = (*p_feats)[fvec_offset + i];
    feats.Drop();
  }
}

// Convert a single sample in batch view to FVec
template <typename BatchView>
struct DataToFeatVec {
  void Fill(bst_idx_t ridx, RegTree::FVec *p_feats) const {
    auto &feats = *p_feats;
    auto n_valid = static_cast<BatchView const *>(this)->DoFill(ridx, feats.Data().data());
    feats.HasMissing(n_valid != feats.Size());
  }
};

struct SparsePageView : public DataToFeatVec<SparsePageView> {
  bst_idx_t base_rowid;
  HostSparsePageView view;

  explicit SparsePageView(SparsePage const *p) : base_rowid{p->base_rowid} { view = p->GetView(); }
  [[nodiscard]] std::size_t Size() const { return view.Size(); }

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float *out) const {
    auto p_data = view[ridx].data();

    for (std::size_t i = 0, n = view[ridx].size(); i < n; ++i) {
      auto const &entry = p_data[i];
      out[entry.index] = entry.fvalue;
    }

    return view[ridx].size();
  }
};

struct GHistIndexMatrixView : public DataToFeatVec<GHistIndexMatrixView> {
 private:
  GHistIndexMatrix const &page_;
  common::Span<FeatureType const> ft_;

  std::vector<std::uint32_t> const &ptrs_;
  std::vector<float> const &mins_;
  std::vector<float> const &values_;

 public:
  bst_idx_t const base_rowid;

 public:
  GHistIndexMatrixView(GHistIndexMatrix const &_page, common::Span<FeatureType const> ft)
      : page_{_page},
        ft_{ft},
        ptrs_{_page.cut.Ptrs()},
        mins_{_page.cut.MinValues()},
        values_{_page.cut.Values()},
        base_rowid{_page.base_rowid} {}

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float* out) const {
    auto gridx = ridx + this->base_rowid;
    auto n_features = page_.Features();

    bst_idx_t n_non_missings = 0;
    if (page_.IsDense()) {
      common::DispatchBinType(page_.index.GetBinTypeSize(), [&](auto t) {
        using T = decltype(t);
        auto ptr = page_.index.data<T>();
        auto rbeg = page_.row_ptr[ridx];
        for (bst_feature_t fidx = 0; fidx < n_features; ++fidx) {
          bst_bin_t bin_idx;
          float fvalue;
          if (common::IsCat(ft_, fidx)) {
            bin_idx = page_.GetGindex(gridx, fidx);
            fvalue = this->values_[bin_idx];
          } else {
            bin_idx = ptr[rbeg + fidx] + page_.index.Offset()[fidx];
            fvalue =
                common::HistogramCuts::NumericBinValue(this->ptrs_, values_, mins_, fidx, bin_idx);
          }
          out[fidx] = fvalue;
        }
      });
      n_non_missings += n_features;
    } else {
      for (bst_feature_t fidx = 0; fidx < n_features; ++fidx) {
        float f = page_.GetFvalue(ptrs_, values_, mins_, gridx, fidx, common::IsCat(ft_, fidx));
        if (!common::CheckNAN(f)) {
          out[fidx] = f;
          n_non_missings++;
        }
      }
    }
    return n_non_missings;
  }

  [[nodiscard]] auto Size() const { return page_.Size(); }
};

template <typename Adapter>
class AdapterView : public DataToFeatVec<AdapterView<Adapter>> {
  Adapter const *adapter_;
  float missing_;

 public:
  explicit AdapterView(Adapter const *adapter, float missing)
      : adapter_{adapter}, missing_{missing} {}

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float *out) const {
    auto const &batch = adapter_->Value();
    auto row = batch.GetLine(ridx);
    bst_idx_t n_non_missings = 0;
    for (size_t c = 0; c < row.Size(); ++c) {
      auto e = row.GetElement(c);
      if (missing_ != e.value && !common::CheckNAN(e.value)) {
        out[e.column_idx] = e.value;
        n_non_missings++;
      }
    }
    return n_non_missings;
  }

  [[nodiscard]] size_t Size() const { return adapter_->NumRows(); }

  bst_idx_t const static base_rowid = 0;  // NOLINT
};

template <typename DataView, std::size_t kBlockOfRowsSize>
void PredictBatchByBlockOfRowsKernel(DataView batch, gbm::GBTreeModel const &model,
                                     bst_tree_t tree_begin, bst_tree_t tree_end,
                                     std::vector<RegTree::FVec> *p_thread_temp,
                                     std::int32_t n_threads,
                                     linalg::TensorView<float, 2> out_predt) {
  auto &thread_temp = *p_thread_temp;

  // Parallel over local batches
  auto const n_samples = batch.Size();
  auto const n_features = model.learner_model_param->num_feature;
  auto const n_blocks = common::DivRoundUp(n_samples, kBlockOfRowsSize);

  common::ParallelFor(n_blocks, n_threads, [&](auto block_id) {
    auto const batch_offset = block_id * kBlockOfRowsSize;
    auto const block_size =
        std::min(static_cast<std::size_t>(n_samples - batch_offset), kBlockOfRowsSize);
    auto const fvec_offset = omp_get_thread_num() * kBlockOfRowsSize;

    FVecFill(block_size, batch_offset, n_features, &batch, fvec_offset, p_thread_temp);
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
  auto n_nodes = tree->NumNodes();
  if (static_cast<decltype(n_nodes)>(mean_values->size()) == n_nodes) {
    return;
  }
  mean_values->resize(n_nodes);
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

  void PredictDMatrix(Context const *ctx, DMatrix *p_fmat, std::vector<bst_float> *out_preds) {
    CHECK(xgboost::collective::IsDistributed())
        << "column-split prediction is only supported for distributed training";

    for (auto const &batch : p_fmat->GetBatches<SparsePage>()) {
      CHECK_EQ(out_preds->size(),
               p_fmat->Info().num_row_ * model_.learner_model_param->num_output_group);
      PredictBatchKernel<SparsePageView, kBlockOfRowsSize>(ctx, SparsePageView{&batch}, out_preds);
    }
  }

  void PredictLeaf(Context const* ctx, DMatrix *p_fmat, std::vector<bst_float> *out_preds) {
    CHECK(xgboost::collective::IsDistributed())
        << "column-split prediction is only supported for distributed training";

    for (auto const &batch : p_fmat->GetBatches<SparsePage>()) {
      CHECK_EQ(out_preds->size(), p_fmat->Info().num_row_ * (tree_end_ - tree_begin_));
      PredictBatchKernel<SparsePageView, kBlockOfRowsSize, true>(ctx, SparsePageView{&batch},
                                                                 out_preds);
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

  [[nodiscard]] std::size_t BitIndex(std::size_t tree_id, std::size_t row_id,
                                     std::size_t node_id) const {
    size_t tree_index = tree_id - tree_begin_;
    return tree_offsets_[tree_index] * n_rows_ + row_id * tree_sizes_[tree_index] + node_id;
  }

  void AllreduceBitVectors(Context const *ctx) {
    auto rc = collective::Success() << [&] {
      return collective::Allreduce(
          ctx, linalg::MakeVec(decision_storage_.data(), decision_storage_.size()),
          collective::Op::kBitwiseOR);
    } << [&] {
      return collective::Allreduce(
          ctx, linalg::MakeVec(missing_storage_.data(), missing_storage_.size()),
          collective::Op::kBitwiseAND);
    };
    collective::SafeColl(rc);
  }

  void MaskOneTree(RegTree::FVec const &feat, std::size_t tree_id, std::size_t row_id) {
    auto const &tree = *model_.trees[tree_id];
    auto const &cats = tree.GetCategoriesMatrix();
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
      auto const decision = tree.HasCategoricalSplit()
                                ? GetDecision<true>(node, nid, fvalue, cats)
                                : GetDecision<false>(node, nid, fvalue, cats);
      if (decision) {
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
      return node.LeftChild() + !decision_bits_.Check(bit_index);
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

  template <bool predict_leaf = false>
  bst_float PredictOneTree(std::size_t tree_id, std::size_t row_id) {
    auto const &tree = *model_.trees[tree_id];
    auto const leaf = GetLeafIndex(tree, tree_id, row_id);
    if constexpr (predict_leaf) {
      return static_cast<bst_float>(leaf);
    } else {
      return tree[leaf].LeafValue();
    }
  }

  template <bool predict_leaf = false>
  void PredictAllTrees(std::vector<bst_float> *out_preds, std::size_t batch_offset,
                       std::size_t predict_offset, std::size_t num_group, std::size_t block_size) {
    auto &preds = *out_preds;
    for (size_t tree_id = tree_begin_; tree_id < tree_end_; ++tree_id) {
      auto const gid = model_.tree_info[tree_id];
      for (size_t i = 0; i < block_size; ++i) {
        auto const result = PredictOneTree<predict_leaf>(tree_id, batch_offset + i);
        if constexpr (predict_leaf) {
          preds[(predict_offset + i) * (tree_end_ - tree_begin_) + tree_id] = result;
        } else {
          preds[(predict_offset + i) * num_group + gid] += result;
        }
      }
    }
  }

  template <typename DataView, size_t block_of_rows_size, bool predict_leaf = false>
  void PredictBatchKernel(Context const* ctx, DataView batch, std::vector<bst_float> *out_preds) {
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

    AllreduceBitVectors(ctx);

    // auto block_id has the same type as `n_blocks`.
    common::ParallelFor(n_blocks, n_threads_, [&](auto block_id) {
      auto const batch_offset = block_id * block_of_rows_size;
      auto const block_size = std::min(static_cast<std::size_t>(nsize - batch_offset),
                                       static_cast<std::size_t>(block_of_rows_size));
      PredictAllTrees<predict_leaf>(out_preds, batch_offset, batch_offset + batch.base_rowid,
                                    num_group, block_size);
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
  void PredictDMatrix(DMatrix *p_fmat, std::vector<float> *out_preds, gbm::GBTreeModel const &model,
                      bst_tree_t tree_begin, bst_tree_t tree_end) const {
    if (p_fmat->Info().IsColumnSplit()) {
      CHECK(!model.learner_model_param->IsVectorLeaf())
          << "Predict DMatrix with column split" << MTNotImplemented();

      ColumnSplitHelper helper(this->ctx_->Threads(), model, tree_begin, tree_end);
      helper.PredictDMatrix(ctx_, p_fmat, out_preds);
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
    auto out_predt = linalg::MakeTensorView(ctx_, *out_preds, n_samples, n_groups);

    if (!p_fmat->PageExists<SparsePage>()) {
      auto ft = p_fmat->Info().feature_types.ConstHostVector();
      for (auto const &batch : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, {})) {
        if (blocked) {
          PredictBatchByBlockOfRowsKernel<GHistIndexMatrixView, kBlockOfRowsSize>(
              GHistIndexMatrixView{batch, ft}, model, tree_begin, tree_end, &feat_vecs, n_threads,
              out_predt);
        } else {
          PredictBatchByBlockOfRowsKernel<GHistIndexMatrixView, 1>(
              GHistIndexMatrixView{batch, ft}, model, tree_begin, tree_end, &feat_vecs, n_threads,
              out_predt);
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

  template <typename DataView>
  void PredictContributionKernel(
      DataView batch, const MetaInfo &info, const gbm::GBTreeModel &model,
      const std::vector<bst_float> *tree_weights, std::vector<std::vector<float>> *mean_values,
      std::vector<RegTree::FVec> *feat_vecs, std::vector<bst_float> *contribs,
      bst_tree_t ntree_limit, bool approximate, int condition, unsigned condition_feature) const {
    const int num_feature = model.learner_model_param->num_feature;
    const int ngroup = model.learner_model_param->num_output_group;
    CHECK_NE(ngroup, 0);
    size_t const ncolumns = num_feature + 1;
    CHECK_NE(ncolumns, 0);
    auto device = ctx_->Device().IsSycl() ? DeviceOrd::CPU() : ctx_->Device();
    auto base_margin = info.base_margin_.View(device);
    auto base_score = model.learner_model_param->BaseScore(device)(0);

    // parallel over local batch
    common::ParallelFor(batch.Size(), this->ctx_->Threads(), [&](auto i) {
      auto row_idx = batch.base_rowid + i;
      RegTree::FVec &feats = (*feat_vecs)[omp_get_thread_num()];
      if (feats.Size() == 0) {
        feats.Init(num_feature);
      }
      std::vector<bst_float> this_tree_contribs(ncolumns);
      // loop over all classes
      for (int gid = 0; gid < ngroup; ++gid) {
        bst_float *p_contribs = &(*contribs)[(row_idx * ngroup + gid) * ncolumns];
        batch.Fill(i, &feats);
        // calculate contributions
        for (bst_tree_t j = 0; j < ntree_limit; ++j) {
          auto *tree_mean_values = &mean_values->at(j);
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

 public:
  explicit CPUPredictor(Context const *ctx) : Predictor::Predictor{ctx} {}

  void PredictBatch(DMatrix *dmat, PredictionCacheEntry *predts, gbm::GBTreeModel const &model,
                    bst_tree_t tree_begin, bst_tree_t tree_end = 0) const override {
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
                                PredictionCacheEntry *out_preds, bst_tree_t tree_begin,
                                bst_tree_t tree_end) const {
    auto const n_threads = this->ctx_->Threads();
    auto m = std::any_cast<std::shared_ptr<Adapter>>(x);
    CHECK_EQ(m->NumColumns(), model.learner_model_param->num_feature)
        << "Number of columns in data must equal to trained model.";
    CHECK(p_m);
    CHECK_EQ(p_m->Info().num_row_, m->NumRows());
    CHECK_EQ(p_m->Info().num_col_, m->NumColumns());
    this->InitOutPredictions(p_m->Info(), &(out_preds->predictions), model);

    auto &predictions = out_preds->predictions.HostVector();
    std::vector<RegTree::FVec> thread_temp;
    InitThreadTemp(n_threads * kBlockSize, &thread_temp);
    std::size_t n_groups = model.learner_model_param->OutputLength();
    auto out_predt = linalg::MakeTensorView(ctx_, predictions, m->NumRows(), n_groups);
    PredictBatchByBlockOfRowsKernel<AdapterView<Adapter>, kBlockSize>(
        AdapterView<Adapter>(m.get(), missing), model, tree_begin, tree_end, &thread_temp,
        n_threads, out_predt);
  }

  bool InplacePredict(std::shared_ptr<DMatrix> p_m, const gbm::GBTreeModel &model, float missing,
                      PredictionCacheEntry *out_preds, bst_tree_t tree_begin,
                      bst_tree_t tree_end) const override {
    auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
    CHECK(proxy)<< error::InplacePredictProxy();
    CHECK(!p_m->Info().IsColumnSplit())
        << "Inplace predict support for column-wise data split is not yet implemented.";
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
    } else if (x.type() == typeid(std::shared_ptr<data::ColumnarAdapter>)) {
      this->DispatchedInplacePredict<data::ColumnarAdapter, kBlockOfRowsSize>(
          x, p_m, model, missing, out_preds, tree_begin, tree_end);
    } else {
      return false;
    }
    return true;
  }

  void PredictLeaf(DMatrix *p_fmat, HostDeviceVector<float> *out_preds,
                   gbm::GBTreeModel const &model, bst_tree_t ntree_limit) const override {
    auto const n_threads = this->ctx_->Threads();
    // number of valid trees
    ntree_limit = GetTreeLimit(model.trees, ntree_limit);
    const MetaInfo &info = p_fmat->Info();
    std::vector<bst_float> &preds = out_preds->HostVector();
    preds.resize(info.num_row_ * ntree_limit);

    if (p_fmat->Info().IsColumnSplit()) {
      CHECK(!model.learner_model_param->IsVectorLeaf())
          << "Predict leaf with column split" << MTNotImplemented();

      ColumnSplitHelper helper(n_threads, model, 0, ntree_limit);
      helper.PredictLeaf(ctx_, p_fmat, &preds);
      return;
    }

    std::vector<RegTree::FVec> feat_vecs;
    const int num_feature = model.learner_model_param->num_feature;
    InitThreadTemp(n_threads, &feat_vecs);
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
        for (bst_tree_t j = 0; j < ntree_limit; ++j) {
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
                           const gbm::GBTreeModel &model, bst_tree_t ntree_limit,
                           std::vector<bst_float> const *tree_weights, bool approximate,
                           int condition, unsigned condition_feature) const override {
    CHECK(!model.learner_model_param->IsVectorLeaf())
        << "Predict contribution" << MTNotImplemented();
    CHECK(!p_fmat->Info().IsColumnSplit())
        << "Predict contribution support for column-wise data split is not yet implemented.";
    auto const n_threads = this->ctx_->Threads();
    std::vector<RegTree::FVec> feat_vecs;
    InitThreadTemp(n_threads, &feat_vecs);
    const MetaInfo& info = p_fmat->Info();
    // number of valid trees
    ntree_limit = GetTreeLimit(model.trees, ntree_limit);
    size_t const ncolumns = model.learner_model_param->num_feature + 1;
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
    // start collecting the contributions
    if (!p_fmat->PageExists<SparsePage>()) {
      auto ft = p_fmat->Info().feature_types.ConstHostVector();
      for (const auto &batch : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, {})) {
        PredictContributionKernel(GHistIndexMatrixView{batch, ft}, info, model, tree_weights,
                                  &mean_values, &feat_vecs, &contribs, ntree_limit, approximate,
                                  condition, condition_feature);
      }
    } else {
      for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
        PredictContributionKernel(
            SparsePageView{&batch}, info, model, tree_weights, &mean_values, &feat_vecs,
            &contribs, ntree_limit, approximate, condition, condition_feature);
      }
    }
  }

  void PredictInteractionContributions(DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                                       gbm::GBTreeModel const &model, bst_tree_t ntree_limit,
                                       std::vector<float> const *tree_weights,
                                       bool approximate) const override {
    CHECK(!model.learner_model_param->IsVectorLeaf())
        << "Predict interaction contribution" << MTNotImplemented();
    CHECK(!p_fmat->Info().IsColumnSplit()) << "Predict interaction contribution support for "
                                              "column-wise data split is not yet implemented.";
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
