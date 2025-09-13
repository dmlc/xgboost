/**
 * Copyright 2017-2025, XGBoost Contributors
 */
#include <algorithm>  // for max, fill, min
#include <cassert>    // for assert
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t, int32_t, uint64_t
#include <memory>     // for unique_ptr, shared_ptr
#include <ostream>    // for char_traits, operator<<, basic_ostream
#include <vector>     // for vector

#include "../collective/allreduce.h"          // for Allreduce
#include "../collective/communicator-inl.h"   // for IsDistributed
#include "../common/bitfield.h"               // for RBitField8
#include "../common/column_matrix.h"          // for ColumnMatrix
#include "../common/error_msg.h"              // for InplacePredictProxy
#include "../common/math.h"                   // for CheckNAN
#include "../common/threading_utils.h"        // for ParallelFor
#include "../data/adapter.h"                  // for ArrayAdapter, CSRAdapter, CSRArrayAdapter
#include "../data/cat_container.h"            // for CatContainer
#include "../data/gradient_index.h"           // for GHistIndexMatrix
#include "../data/proxy_dmatrix.h"            // for DMatrixProxy
#include "../gbm/gbtree_model.h"              // for GBTreeModel, GBTreeModelParam
#include "dmlc/registry.h"                    // for DMLC_REGISTRY_FILE_TAG
#include "predict_fn.h"                       // for GetNextNode, GetNextNodeMulti
#include "array_tree_layout.h"                // for ProcessArrayTree
#include "treeshap.h"                         // for CalculateContributions
#include "utils.h"                            // for CheckProxyDMatrix
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
                        RegTree::CategoricalSplitMatrix const &cats, bst_node_t nidx) {
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
                                       RegTree::CategoricalSplitMatrix const &cats,
                                       bst_node_t nidx) noexcept(true) {
  const bst_node_t leaf = p_feats.HasMissing()
                              ? GetLeafIndex<true, has_categorical>(tree, p_feats, cats, nidx)
                              : GetLeafIndex<false, has_categorical>(tree, p_feats, cats, nidx);
  return tree[leaf].LeafValue();
}

template <bool has_categorical, bool any_missing, bool use_array_tree_layout>
void PredValueByOneTree(const RegTree& tree,
                        std::size_t const predict_offset,
                        common::Span<RegTree::FVec> fvec_tloc,
                        std::size_t const block_size,
                        linalg::MatrixView<float> out_predt,
                        bst_node_t* p_nidx, int depth, int gid) {
  auto const &cats = tree.GetCategoriesMatrix();
  if constexpr (use_array_tree_layout) {
    ProcessArrayTree<has_categorical, any_missing>(tree, cats, fvec_tloc, block_size, p_nidx,
                                                   depth);
  }
  for (std::size_t i = 0; i < block_size; ++i) {
    bst_node_t nidx = 0;
    /*
     * If array_tree_layout was used, we start processing from the nidx calculated using
     * the array tree.
     */
    if constexpr (use_array_tree_layout) {
      nidx = p_nidx[i];
      p_nidx[i] = 0;
    }
    out_predt(predict_offset + i, gid) +=
        PredValueByOneTree<has_categorical>(fvec_tloc[i], tree, cats, nidx);
  }
}
}  // namespace scalar

namespace multi {
template <bool has_missing, bool has_categorical>
bst_node_t GetLeafIndex(MultiTargetTree const &tree, const RegTree::FVec &feat,
                        RegTree::CategoricalSplitMatrix const &cats,
                        bst_node_t nidx) {
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
                        linalg::VectorView<float> out_predt, bst_node_t nidx) {
  bst_node_t const leaf = p_feats.HasMissing()
                              ? GetLeafIndex<true, has_categorical>(tree, p_feats, cats, nidx)
                              : GetLeafIndex<false, has_categorical>(tree, p_feats, cats, nidx);
  auto leaf_value = tree.LeafValue(leaf);
  assert(out_predt.Shape(0) == leaf_value.Shape(0) && "shape mismatch.");
  for (size_t i = 0; i < leaf_value.Size(); ++i) {
    out_predt(i) += leaf_value(i);
  }
}

template <bool has_categorical, bool any_missing, bool use_array_tree_layout>
void PredValueByOneTree(const RegTree &tree, std::size_t const predict_offset,
                        common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
                        linalg::MatrixView<float> out_predt, bst_node_t *p_nidx, bst_node_t depth) {
  const auto &mt_tree = *(tree.GetMultiTargetTree());
  auto const &cats = tree.GetCategoriesMatrix();
  if constexpr (use_array_tree_layout) {
    ProcessArrayTree<has_categorical, any_missing>(tree, cats, fvec_tloc, block_size, p_nidx,
                                                   depth);
  }
  for (std::size_t i = 0; i < block_size; ++i) {
    bst_node_t nidx = 0;
    if constexpr (use_array_tree_layout) {
      nidx = p_nidx[i];
      p_nidx[i] = 0;
    }
    auto t_predts = out_predt.Slice(predict_offset + i, linalg::All());
    PredValueByOneTree<has_categorical>(fvec_tloc[i], mt_tree, cats, t_predts, nidx);
  }
}
}  // namespace multi

namespace {
template <bool use_array_tree_layout, bool any_missing>
void PredictBlockByAllTrees(gbm::GBTreeModel const &model, bst_tree_t const tree_begin,
                            bst_tree_t const tree_end, std::size_t const predict_offset,
                            common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
                            linalg::MatrixView<float> out_predt,
                            const std::vector<int>& tree_depth) {
  std::vector<bst_node_t> nidx;
  if constexpr (use_array_tree_layout) {
    nidx.resize(block_size, 0);
  }
  for (bst_tree_t tree_id = tree_begin; tree_id < tree_end; ++tree_id) {
    auto const &tree = *model.trees.at(tree_id);
    bool has_categorical = tree.HasCategoricalSplit();

    int depth = use_array_tree_layout ? tree_depth[tree_id - tree_begin] : 0;
    if (tree.IsMultiTarget()) {
      if (has_categorical) {
        multi::PredValueByOneTree<true, any_missing, use_array_tree_layout>
          (tree, predict_offset, fvec_tloc, block_size, out_predt, nidx.data(), depth);
      } else {
        multi::PredValueByOneTree<false, any_missing, use_array_tree_layout>
          (tree, predict_offset, fvec_tloc, block_size, out_predt, nidx.data(), depth);
      }
    } else {
      auto const gid = model.tree_info[tree_id];
      if (has_categorical) {
        scalar::PredValueByOneTree<true, any_missing, use_array_tree_layout>
          (tree, predict_offset, fvec_tloc, block_size, out_predt, nidx.data(), depth, gid);
      } else {
        scalar::PredValueByOneTree<false, any_missing, use_array_tree_layout>
          (tree, predict_offset, fvec_tloc, block_size, out_predt, nidx.data(), depth, gid);
      }
    }
  }
}

// Dispatch between template implementations
void DispatchArrayLayout(gbm::GBTreeModel const &model, bst_tree_t const tree_begin,
                         bst_tree_t const tree_end, std::size_t const predict_offset,
                         common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
                         linalg::MatrixView<float> out_predt, const std::vector<int> &tree_depth,
                         bool any_missing) {
  /*
   * We transform trees to array layout for each block of data to avoid memory overheads.
   * It makes the array layout inefficient for block_size == 1
   */
  const bool use_array_tree_layout = block_size > 1;
  if (use_array_tree_layout) {
    // Recheck if the current block has missing values.
    if (any_missing) {
      any_missing = false;
      for (std::size_t i = 0; i < block_size; ++i) {
        any_missing |= fvec_tloc[i].HasMissing();
        if (any_missing) {
          break;
        }
      }
    }
    if (any_missing) {
      PredictBlockByAllTrees<true, true>(model, tree_begin, tree_end, predict_offset, fvec_tloc,
                                         block_size, out_predt, tree_depth);
    } else {
      PredictBlockByAllTrees<true, false>(model, tree_begin, tree_end, predict_offset, fvec_tloc,
                                          block_size, out_predt, tree_depth);
    }
  } else {
    PredictBlockByAllTrees<false, true>(model, tree_begin, tree_end, predict_offset, fvec_tloc,
                                        block_size, out_predt, tree_depth);
  }
}

bool ShouldUseBlock(DMatrix *p_fmat) {
  // Threshold to use block-based prediction.
  constexpr double kDensityThresh = .125;
  bst_idx_t n_samples = p_fmat->Info().num_row_;
  bst_idx_t total = std::max(n_samples * p_fmat->Info().num_col_, static_cast<bst_idx_t>(1));
  double density = static_cast<double>(p_fmat->Info().num_nonzero_) / static_cast<double>(total);
  bool blocked = density > kDensityThresh;
  return blocked;
}

using cpu_impl::MakeCatAccessor;

// Convert a single sample in batch view to FVec
template <typename BatchView>
struct DataToFeatVec {
  void Fill(bst_idx_t ridx, RegTree::FVec *p_feats) const {
    auto &feats = *p_feats;
    auto n_valid = static_cast<BatchView const *>(this)->DoFill(ridx, feats.Data().data());
    feats.HasMissing(n_valid != feats.Size());
  }

  // Fill the data into the feature vector.
  void FVecFill(common::Range1d const &block, bst_feature_t n_features,
                common::Span<RegTree::FVec> s_feats_vec) const {
    auto feats_vec = s_feats_vec.data();
    for (std::size_t i = 0; i < block.Size(); ++i) {
      RegTree::FVec &feats = feats_vec[i];
      if (feats.Size() == 0) {
        feats.Init(n_features);
      }
      this->Fill(block.begin() + i, &feats);
    }
  }
  // Clear the feature vector.
  static void FVecDrop(common::Span<RegTree::FVec> s_feats) {
    auto p_feats = s_feats.data();
    for (size_t i = 0, n = s_feats.size(); i < n; ++i) {
      p_feats[i].Drop();
    }
  }
};

template <typename EncAccessor>
class SparsePageView : public DataToFeatVec<SparsePageView<EncAccessor>> {
  EncAccessor const &acc_;
  HostSparsePageView const view_;

 public:
  bst_idx_t const base_rowid;

  SparsePageView(HostSparsePageView const p, bst_idx_t base_rowid, EncAccessor const &acc)
      : acc_{acc}, view_{p}, base_rowid{base_rowid} {}
  [[nodiscard]] std::size_t Size() const { return view_.Size(); }

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float *out) const {
    auto p_data = view_[ridx].data();

    for (std::size_t i = 0, n = view_[ridx].size(); i < n; ++i) {
      auto const &entry = p_data[i];
      out[entry.index] = acc_(entry);
    }

    return view_[ridx].size();
  }
};

template <typename EncAccessor>
class GHistIndexMatrixView : public DataToFeatVec<GHistIndexMatrixView<EncAccessor>> {
 private:
  GHistIndexMatrix const &page_;
  EncAccessor const &acc_;
  common::Span<FeatureType const> ft_;

  std::vector<std::uint32_t> const &ptrs_;
  std::vector<float> const &mins_;
  std::vector<float> const &values_;
  common::ColumnMatrix const &columns_;

 public:
  bst_idx_t const base_rowid;

 public:
  GHistIndexMatrixView(GHistIndexMatrix const &_page, EncAccessor const &acc,
                       common::Span<FeatureType const> ft)
      : page_{_page},
        acc_{acc},
        ft_{ft},
        ptrs_{_page.cut.Ptrs()},
        mins_{_page.cut.MinValues()},
        values_{_page.cut.Values()},
        columns_{page_.Transpose()},
        base_rowid{_page.base_rowid} {}

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float *out) const {
    auto gridx = ridx + this->base_rowid;
    auto n_features = page_.Features();

    bst_idx_t n_non_missings = 0;
    if (page_.IsDense()) {
      common::DispatchBinType(page_.index.GetBinTypeSize(), [&](auto t) {
        using T = decltype(t);
        auto ptr = this->page_.index.template data<T>();
        auto rbeg = this->page_.row_ptr[ridx];
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
          out[fidx] = acc_(fvalue, fidx);
        }
      });
      n_non_missings += n_features;
    } else {
      for (bst_feature_t fidx = 0; fidx < n_features; ++fidx) {
        float fvalue = std::numeric_limits<float>::quiet_NaN();
        bool is_cat = common::IsCat(ft_, fidx);
        if (columns_.GetColumnType(fidx) == common::kSparseColumn) {
          // Special handling for extremely sparse data. Just binary search.
          auto bin_idx = page_.GetGindex(gridx, fidx);
          if (bin_idx != -1) {
            if (is_cat) {
              fvalue = values_[bin_idx];
            } else {
              fvalue = common::HistogramCuts::NumericBinValue(this->ptrs_, values_, mins_, fidx,
                                                              bin_idx);
            }
          }
        } else {
          fvalue = page_.GetFvalue(ptrs_, values_, mins_, gridx, fidx, is_cat);
        }
        if (!common::CheckNAN(fvalue)) {
          out[fidx] = acc_(fvalue, fidx);
          n_non_missings++;
        }
      }
    }
    return n_non_missings;
  }

  [[nodiscard]] bst_idx_t Size() const { return page_.Size(); }
};

template <typename Adapter, typename EncAccessor>
class AdapterView : public DataToFeatVec<AdapterView<Adapter, EncAccessor>> {
  Adapter const *adapter_;
  float missing_;
  EncAccessor const &acc_;

 public:
  explicit AdapterView(Adapter const *adapter, float missing, EncAccessor const &acc)
      : adapter_{adapter}, missing_{missing}, acc_{acc} {}

  [[nodiscard]] bst_idx_t DoFill(bst_idx_t ridx, float *out) const {
    auto const &batch = adapter_->Value();
    auto row = batch.GetLine(ridx);
    bst_idx_t n_non_missings = 0;
    for (size_t c = 0; c < row.Size(); ++c) {
      auto e = row.GetElement(c);
      if (missing_ != e.value && !common::CheckNAN(e.value)) {
        auto fvalue = this->acc_(e);
        out[e.column_idx] = fvalue;
        n_non_missings++;
      }
    }
    return n_non_missings;
  }

  [[nodiscard]] bst_idx_t Size() const { return adapter_->NumRows(); }

  bst_idx_t const static base_rowid = 0;  // NOLINT
};

// Ordinal re-coder.
struct EncAccessorPolicy {
 private:
  std::vector<int32_t> mapping_;

 public:
  EncAccessorPolicy() = default;

  EncAccessorPolicy &operator=(EncAccessorPolicy const &that) = delete;
  EncAccessorPolicy(EncAccessorPolicy const &that) = delete;

  EncAccessorPolicy &operator=(EncAccessorPolicy &&that) = default;
  EncAccessorPolicy(EncAccessorPolicy &&that) = default;

  [[nodiscard]] auto MakeAccessor(Context const *ctx, enc::HostColumnsView new_enc,
                                  gbm::GBTreeModel const &model) {
    auto [acc, mapping] = MakeCatAccessor(ctx, new_enc, model.Cats());
    this->mapping_ = std::move(mapping);
    return acc;
  }
};

struct NullEncAccessorPolicy {
  template <typename... Args>
  [[nodiscard]] auto MakeAccessor(Args &&...) const {
    return NoOpAccessor{};
  }
};

// Block-based parallel.
struct BlockPolicy {
  constexpr static std::size_t kBlockOfRowsSize = 64;
};

struct NullBlockPolicy {
  constexpr static std::size_t kBlockOfRowsSize = 1;
};

/**
 * @brief Policy class, requires a block policy and an accessor policy.
 */
template <typename... Args>
struct LaunchConfig : public Args... {
  Context const *ctx;
  DMatrix *p_fmat;
  gbm::GBTreeModel const &model;

  LaunchConfig(Context const *ctx, DMatrix *p_fmat, gbm::GBTreeModel const &model)
      : ctx{ctx}, p_fmat{p_fmat}, model{model} {}

  LaunchConfig(LaunchConfig const &that) = delete;
  LaunchConfig &operator=(LaunchConfig const &that) = delete;
  LaunchConfig(LaunchConfig &&that) = default;
  LaunchConfig &operator=(LaunchConfig &&that) = default;

  // Helper for running prediction with DMatrix inputs.
  template <typename Fn>
  void ForEachBatch(Fn &&fn) {
    auto acc = this->MakeAccessor(ctx, p_fmat->Cats()->HostView(), model);

    if (!p_fmat->PageExists<SparsePage>()) {
      auto ft = p_fmat->Info().feature_types.ConstHostVector();
      for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx, {})) {
        fn(GHistIndexMatrixView{page, acc, ft});
      }
    } else {
      for (auto const &page : p_fmat->GetBatches<SparsePage>()) {
        // bool any_missing = !page.IsDense();
        fn(SparsePageView{page.GetView(), page.base_rowid, acc});
      }
    }
  }
};

/**
 * @brief Dispatch for the prediction function.
 *
 * @tparam Fn         A function that accepts a @ref LaunchConfig object.
 * @tparam NeedRecode Given a DMatrix input, returns whether we need to recode the categorical
 *                    features.
 */
template <typename Fn, typename NeedRecode>
void LaunchPredict(Context const *ctx, DMatrix *p_fmat, gbm::GBTreeModel const &model, Fn &&fn,
                   NeedRecode &&need_recode) {
  bool blocked = ShouldUseBlock(p_fmat);

  if (blocked) {
    if (model.Cats()->HasCategorical() && need_recode(p_fmat)) {
      using Policy = LaunchConfig<BlockPolicy, EncAccessorPolicy>;
      fn(Policy{ctx, p_fmat, model});
    } else {
      using Policy = LaunchConfig<BlockPolicy, NullEncAccessorPolicy>;
      fn(Policy{ctx, p_fmat, model});
    }
  } else {
    if (model.Cats()->HasCategorical() && need_recode(p_fmat)) {
      using Policy = LaunchConfig<NullBlockPolicy, EncAccessorPolicy>;
      fn(Policy{ctx, p_fmat, model});
    } else {
      using Policy = LaunchConfig<NullBlockPolicy, NullEncAccessorPolicy>;
      fn(Policy{ctx, p_fmat, model});
    }
  }
}

template <typename Fn>
void LaunchPredict(Context const *ctx, DMatrix *p_fmat, gbm::GBTreeModel const &model, Fn &&fn) {
  LaunchPredict(ctx, p_fmat, model, fn,
                [](DMatrix const *p_fmat) { return p_fmat->Cats()->NeedRecode(); });
}

/**
 * @brief Thread-local buffer for the feature matrix.
 */
template <std::size_t kBlockOfRowsSize>
class ThreadTmp {
 private:
  std::vector<RegTree::FVec> feat_vecs_;

 public:
  /**
   * @param blocked Whether block-based parallelism is used.
   */
  explicit ThreadTmp(std::int32_t n_threads) {
    std::size_t n = n_threads * kBlockOfRowsSize;
    std::size_t prev_thread_temp_size = feat_vecs_.size();
    if (prev_thread_temp_size < n) {
      feat_vecs_.resize(n, RegTree::FVec{});
    }
  }
  /**
   * @brief Get a thread local buffer.
   *
   * @param n The size of the thread local block.
   */
  common::Span<RegTree::FVec> ThreadBuffer(std::size_t n) {
    std::int32_t thread_idx = omp_get_thread_num();
    auto const fvec_offset = thread_idx * kBlockOfRowsSize;
    auto fvec_tloc = common::Span{feat_vecs_}.subspan(fvec_offset, n);
    return fvec_tloc;
  }
};

template <std::size_t kBlockOfRowsSize, typename DataView>
void PredictBatchByBlockKernel(DataView const &batch, gbm::GBTreeModel const &model,
                               bst_tree_t tree_begin, bst_tree_t tree_end,
                               ThreadTmp<kBlockOfRowsSize> *p_fvec, std::int32_t n_threads,
                               bool any_missing,
                               linalg::TensorView<float, 2> out_predt) {
  auto &fvec = *p_fvec;
  // Parallel over local batches
  auto const n_samples = batch.Size();
  auto const n_features = model.learner_model_param->num_feature;

  /* Precalculate depth for each tree.
   * These values are required only for the ArrayLayout optimization,
   * so we don't need them if kBlockOfRowsSize == 1
   */
  std::vector<int> tree_depth;
  if constexpr (kBlockOfRowsSize > 1) {
    tree_depth.resize(tree_end - tree_begin);
    common::ParallelFor(tree_end - tree_begin, n_threads, [&](auto i) {
      bst_tree_t tree_id = tree_begin + i;
      tree_depth[i] = model.trees.at(tree_id)->MaxDepth();
    });
  }

  common::ParallelFor1d<kBlockOfRowsSize>(n_samples, n_threads, [&](auto &&block) {
    auto fvec_tloc = fvec.ThreadBuffer(block.Size());

    batch.FVecFill(block, n_features, fvec_tloc);
    DispatchArrayLayout(model, tree_begin, tree_end, block.begin() + batch.base_rowid, fvec_tloc,
                        block.Size(), out_predt, tree_depth, any_missing);
    batch.FVecDrop(fvec_tloc);
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

void FillNodeMeanValues(RegTree const *tree, std::vector<float> *mean_values) {
  auto n_nodes = tree->NumNodes();
  if (static_cast<decltype(n_nodes)>(mean_values->size()) == n_nodes) {
    return;
  }
  mean_values->resize(n_nodes);
  FillNodeMeanValues(tree, 0, mean_values);
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
  ColumnSplitHelper(std::int32_t n_threads, gbm::GBTreeModel const &model, bst_tree_t tree_begin,
                    bst_tree_t tree_end)
      : n_threads_{n_threads},
        model_{model},
        tree_begin_{tree_begin},
        tree_end_{tree_end},
        feat_vecs_{n_threads} {
    CHECK(!model.learner_model_param->IsVectorLeaf())
        << "Predict DMatrix with column split" << MTNotImplemented();
    CHECK(!model.Cats()->HasCategorical())
        << "Categorical feature is not yet supported with column-split.";
    CHECK(xgboost::collective::IsDistributed())
        << "column-split prediction is only supported for distributed training";

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
    // Add the size of the last tree since this is exclusive_scan
    bits_per_row_ = tree_offsets_.back() + tree_sizes_.back();
  }

  // Disable copy (and move) semantics.
  ColumnSplitHelper(ColumnSplitHelper const &) = delete;
  ColumnSplitHelper &operator=(ColumnSplitHelper const &) = delete;
  ColumnSplitHelper(ColumnSplitHelper &&) noexcept = delete;
  ColumnSplitHelper &operator=(ColumnSplitHelper &&) noexcept = delete;

  void PredictDMatrix(Context const *ctx, DMatrix *p_fmat, std::vector<bst_float> *out_preds) {
    if (!p_fmat->PageExists<SparsePage>()) {
      LOG(FATAL) << "Predict with `QuantileDMatrix` is not supported with column-split.";
    }
    for (auto const &batch : p_fmat->GetBatches<SparsePage>()) {
      CHECK_EQ(out_preds->size(),
               p_fmat->Info().num_row_ * model_.learner_model_param->num_output_group);
      PredictBatchKernel<kBlockOfRowsSize>(
          ctx, SparsePageView{batch.GetView(), batch.base_rowid, NoOpAccessor{}}, out_preds);
    }
  }

  void PredictLeaf(Context const *ctx, DMatrix *p_fmat, std::vector<bst_float> *out_preds) {
    for (auto const &batch : p_fmat->GetBatches<SparsePage>()) {
      CHECK_EQ(out_preds->size(), p_fmat->Info().num_row_ * (tree_end_ - tree_begin_));
      PredictBatchKernel<kBlockOfRowsSize, true>(
          ctx, SparsePageView{batch.GetView(), batch.base_rowid, NoOpAccessor{}}, out_preds);
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

  void MaskAllTrees(std::size_t batch_offset, common::Span<RegTree::FVec> feat_vecs,
                    std::size_t block_size) {
    for (auto tree_id = tree_begin_; tree_id < tree_end_; ++tree_id) {
      for (size_t i = 0; i < block_size; ++i) {
        MaskOneTree(feat_vecs[i], tree_id, batch_offset + i);
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
    for (auto tree_id = tree_begin_; tree_id < tree_end_; ++tree_id) {
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

  template <size_t block_of_rows_size, bool predict_leaf = false, typename DataView>
  void PredictBatchKernel(Context const *ctx, DataView batch, std::vector<bst_float> *out_preds) {
    auto const num_group = model_.learner_model_param->num_output_group;

    // parallel over local batch
    auto const n_samples = batch.Size();
    auto const n_features = model_.learner_model_param->num_feature;

    InitBitVectors(n_samples);

    common::ParallelFor1d<kBlockOfRowsSize>(n_samples, n_threads_, [&](auto &&block) {
      auto fvec_tloc = feat_vecs_.ThreadBuffer(block.Size());

      batch.FVecFill(block, n_features, fvec_tloc);
      MaskAllTrees(block.begin(), fvec_tloc, block.Size());
      batch.FVecDrop(fvec_tloc);
    });

    AllreduceBitVectors(ctx);

    common::ParallelFor1d<kBlockOfRowsSize>(n_samples, n_threads_, [&](auto &&block) {
      PredictAllTrees<predict_leaf>(out_preds, block.begin(), block.begin() + batch.base_rowid,
                                    num_group, block.Size());
    });

    ClearBitVectors();
  }

  static std::size_t constexpr kBlockOfRowsSize = BlockPolicy::kBlockOfRowsSize;

  std::int32_t const n_threads_;
  gbm::GBTreeModel const &model_;
  bst_tree_t const tree_begin_;
  bst_tree_t const tree_end_;

  std::vector<std::size_t> tree_sizes_{};
  std::vector<std::size_t> tree_offsets_{};
  std::size_t bits_per_row_{};
  ThreadTmp<kBlockOfRowsSize> feat_vecs_;

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
      ColumnSplitHelper helper(this->ctx_->Threads(), model, tree_begin, tree_end);
      helper.PredictDMatrix(ctx_, p_fmat, out_preds);
      return;
    }

    auto const n_threads = this->ctx_->Threads();

    // Create a writable view on the output prediction vector.
    bst_idx_t n_groups = model.learner_model_param->OutputLength();
    bst_idx_t n_samples = p_fmat->Info().num_row_;
    CHECK_EQ(out_preds->size(), n_samples * n_groups);
    auto out_predt = linalg::MakeTensorView(ctx_, *out_preds, n_samples, n_groups);
    bool any_missing = !(p_fmat->IsDense());

    LaunchPredict(this->ctx_, p_fmat, model, [&](auto &&policy) {
      using Policy = common::GetValueT<decltype(policy)>;
      ThreadTmp<Policy::kBlockOfRowsSize> feat_vecs{n_threads};
      policy.ForEachBatch([&](auto &&batch) {
        PredictBatchByBlockKernel<Policy::kBlockOfRowsSize>(batch, model, tree_begin, tree_end,
                                                            &feat_vecs, n_threads, any_missing,
                                                            out_predt);
      });
    });
  }

  template <typename DataView>
  void PredictContributionKernel(DataView batch, const MetaInfo &info,
                                 const gbm::GBTreeModel &model,
                                 const std::vector<bst_float> *tree_weights,
                                 std::vector<std::vector<float>> *mean_values,
                                 ThreadTmp<1> *feat_vecs, std::vector<bst_float> *contribs,
                                 bst_tree_t ntree_limit, bool approximate, int condition,
                                 unsigned condition_feature) const {
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
      RegTree::FVec &feats = feat_vecs->ThreadBuffer(1).front();
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
            CalculateContributions(*model.trees[j], feats, tree_mean_values, &this_tree_contribs[0],
                                   condition, condition_feature);
          } else {
            CalculateContributionsApprox(*model.trees[j], feats, tree_mean_values,
                                         &this_tree_contribs[0]);
          }
          for (size_t ci = 0; ci < ncolumns; ++ci) {
            p_contribs[ci] +=
                this_tree_contribs[ci] * (tree_weights == nullptr ? 1 : (*tree_weights)[j]);
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

  [[nodiscard]] bool InplacePredict(std::shared_ptr<DMatrix> p_m, gbm::GBTreeModel const &model,
                                    float missing, PredictionCacheEntry *out_preds,
                                    bst_tree_t tree_begin, bst_tree_t tree_end) const override {
    auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
    CHECK(proxy) << error::InplacePredictProxy();

    this->InitOutPredictions(p_m->Info(), &(out_preds->predictions), model);
    auto &predictions = out_preds->predictions.HostVector();
    bool any_missing = true;

    auto const n_threads = this->ctx_->Threads();
    // Always use block as we don't know the nnz.
    ThreadTmp<BlockPolicy::kBlockOfRowsSize> feat_vecs{n_threads};
    bst_idx_t n_groups = model.learner_model_param->OutputLength();

    auto kernel = [&](auto &&view) {
      auto out_predt = linalg::MakeTensorView(ctx_, predictions, view.Size(), n_groups);
      PredictBatchByBlockKernel<BlockPolicy::kBlockOfRowsSize>(view, model, tree_begin, tree_end,
                                                               &feat_vecs, n_threads, any_missing,
                                                               out_predt);
    };
    auto dispatch = [&](auto x) {
      using AdapterT = typename decltype(x)::element_type;
      CheckProxyDMatrix(x, proxy, model.learner_model_param);
      LaunchPredict(
          this->ctx_, proxy, model,
          [&](auto &&policy) {
            if constexpr (std::is_same_v<AdapterT, data::ColumnarAdapter>) {
              auto view =
                  AdapterView{x.get(), missing, policy.MakeAccessor(ctx_, x->Cats(), model)};
              kernel(view);
            } else {
              auto view = AdapterView{x.get(), missing, NoOpAccessor{}};
              kernel(view);
            }
          },
          [&](auto) {
            if constexpr (std::is_same_v<AdapterT, data::ColumnarAdapter>) {
              return !x->Cats().Empty();
            } else {
              return false;
            }
          });
    };

    bool type_error = false;
    data::cpu_impl::DispatchAny<false>(proxy, dispatch, &type_error);
    return !type_error;
  }

  void PredictLeaf(DMatrix *p_fmat, HostDeviceVector<float> *out_preds,
                   gbm::GBTreeModel const &model, bst_tree_t ntree_limit) const override {
    auto const n_threads = this->ctx_->Threads();
    // number of valid trees
    ntree_limit = GetTreeLimit(model.trees, ntree_limit);
    const MetaInfo &info = p_fmat->Info();
    std::vector<float> &preds = out_preds->HostVector();
    preds.resize(info.num_row_ * ntree_limit);

    if (p_fmat->Info().IsColumnSplit()) {
      ColumnSplitHelper helper(n_threads, model, 0, ntree_limit);
      helper.PredictLeaf(ctx_, p_fmat, &preds);
      return;
    }

    auto n_features = model.learner_model_param->num_feature;
    ThreadTmp<1> feat_vecs{n_threads};

    LaunchPredict(this->ctx_, p_fmat, model, [&](auto &&policy) {
      policy.ForEachBatch([&](auto &&batch) {
        common::ParallelFor1d<1>(batch.Size(), n_threads, [&](auto &&block) {
          auto ridx = static_cast<bst_idx_t>(batch.base_rowid + block.begin());
          auto fvec_tloc = feat_vecs.ThreadBuffer(block.Size());
          batch.FVecFill(block, n_features, fvec_tloc);

          for (bst_tree_t j = 0; j < ntree_limit; ++j) {
            auto const &tree = *model.trees[j];
            auto const &cats = tree.GetCategoriesMatrix();
            bst_node_t nidx = 0;
            if (tree.IsMultiTarget()) {
              nidx = multi::GetLeafIndex<true, true>(*tree.GetMultiTargetTree(), fvec_tloc.front(),
                                                     cats, nidx);
            } else {
              nidx = scalar::GetLeafIndex<true, true>(tree, fvec_tloc.front(), cats, nidx);
            }
            preds[ridx * ntree_limit + j] = static_cast<float>(nidx);
          }
          batch.FVecDrop(fvec_tloc);
        });
      });
    });
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
    ThreadTmp<1> feat_vecs{n_threads};
    const MetaInfo &info = p_fmat->Info();
    // number of valid trees
    ntree_limit = GetTreeLimit(model.trees, ntree_limit);
    size_t const ncolumns = model.learner_model_param->num_feature + 1;
    // allocate space for (number of features + bias) times the number of rows
    std::vector<bst_float> &contribs = out_contribs->HostVector();
    contribs.resize(info.num_row_ * ncolumns * model.learner_model_param->num_output_group);
    // make sure contributions is zeroed, we could be reusing a previously
    // allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    // initialize tree node mean values
    std::vector<std::vector<float>> mean_values(ntree_limit);
    common::ParallelFor(ntree_limit, n_threads, [&](bst_omp_uint i) {
      FillNodeMeanValues(model.trees[i].get(), &(mean_values[i]));
    });

    LaunchPredict(this->ctx_, p_fmat, model, [&](auto &&policy) {
      policy.ForEachBatch([&](auto &&batch) {
        PredictContributionKernel(batch, info, model, tree_weights, &mean_values, &feat_vecs,
                                  &contribs, ntree_limit, approximate, condition,
                                  condition_feature);
      });
    });
  }

  void PredictInteractionContributions(DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                                       gbm::GBTreeModel const &model, bst_tree_t ntree_limit,
                                       std::vector<float> const *tree_weights,
                                       bool approximate) const override {
    CHECK(!model.learner_model_param->IsVectorLeaf())
        << "Predict interaction contribution" << MTNotImplemented();
    CHECK(!p_fmat->Info().IsColumnSplit()) << "Predict interaction contribution support for "
                                              "column-wise data split is not yet implemented.";
    const MetaInfo &info = p_fmat->Info();
    auto const ngroup = model.learner_model_param->num_output_group;
    auto const ncolumns = model.learner_model_param->num_feature;
    const unsigned row_chunk = ngroup * (ncolumns + 1) * (ncolumns + 1);
    const unsigned mrow_chunk = (ncolumns + 1) * (ncolumns + 1);
    const unsigned crow_chunk = ngroup * (ncolumns + 1);

    // allocate space for (number of features^2) times the number of rows and tmp off/on contribs
    std::vector<bst_float> &contribs = out_contribs->HostVector();
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
    PredictContribution(p_fmat, &contribs_diag_hdv, model, ntree_limit, tree_weights, approximate,
                        0, 0);
    for (size_t i = 0; i < ncolumns + 1; ++i) {
      PredictContribution(p_fmat, &contribs_off_hdv, model, ntree_limit, tree_weights, approximate,
                          -1, i);
      PredictContribution(p_fmat, &contribs_on_hdv, model, ntree_limit, tree_weights, approximate,
                          1, i);

      for (size_t j = 0; j < info.num_row_; ++j) {
        for (std::remove_const_t<decltype(ngroup)> l = 0; l < ngroup; ++l) {
          const unsigned o_offset = j * row_chunk + l * mrow_chunk + i * (ncolumns + 1);
          const unsigned c_offset = j * crow_chunk + l * (ncolumns + 1);
          contribs[o_offset + i] = 0;
          for (size_t k = 0; k < ncolumns + 1; ++k) {
            // fill in the diagonal with additive effects, and off-diagonal with the interactions
            if (k == i) {
              contribs[o_offset + i] += contribs_diag[c_offset + k];
            } else {
              contribs[o_offset + k] =
                  (contribs_on[c_offset + k] - contribs_off[c_offset + k]) / 2.0;
              contribs[o_offset + i] -= contribs[o_offset + k];
            }
          }
        }
      }
    }
  }
};

XGBOOST_REGISTER_PREDICTOR(CPUPredictor, "cpu_predictor")
    .describe("Make predictions using CPU.")
    .set_body([](Context const *ctx) { return new CPUPredictor(ctx); });
}  // namespace xgboost::predictor
