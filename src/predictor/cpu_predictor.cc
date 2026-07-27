/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include <algorithm>  // for max, fill, min
#include <cassert>    // for assert
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t, int32_t, uint64_t
#include <memory>     // for unique_ptr, shared_ptr
#include <vector>     // for vector

#include "../collective/allreduce.h"         // for Allreduce
#include "../collective/communicator-inl.h"  // for IsDistributed
#include "../common/bitfield.h"              // for RBitField8
#include "../common/column_matrix.h"         // for ColumnMatrix
#include "../common/error_msg.h"             // for InplacePredictProxy
#include "../common/math.h"                  // for CheckNAN
#include "../common/optional_weight.h"       // for OptionalWeights
#include "../common/threading_utils.h"       // for ParallelFor
#include "../data/adapter.h"                 // for ArrayAdapter, CSRAdapter, CSRArrayAdapter
#include "../data/cat_container.h"           // for CatContainer
#include "../data/gradient_index.h"          // for GHistIndexMatrix
#include "../data/proxy_dmatrix.h"           // for DMatrixProxy
#include "../gbm/gbtree_model.h"             // for GBTreeModel, GBTreeModelParam
#include "array_tree_layout.h"               // for ProcessArrayTree
#include "data_accessor.h"                   // for GHistIndexMatrixView, SparsePageView
#include "dmlc/registry.h"                   // for DMLC_REGISTRY_FILE_TAG
#include "gbtree_view.h"                     // for GBTreeModelView
#include "interpretability/shap.h"  // for ShapValues, ApproxFeatureImportance, ShapInteractionValues
#include "predict_fn.h"             // for GetNextNode, GetNextNodeMulti
#include "utils.h"                  // for CheckProxyDMatrix
#include "xgboost/base.h"           // for bst_float, bst_node_t, bst_omp_uint, bst_fe...
#include "xgboost/context.h"        // for Context
#include "xgboost/data.h"           // for Entry, DMatrix, MetaInfo, SparsePage, Batch...
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

namespace {
using TreeViewVar = std::variant<tree::ScalarTreeView, tree::MultiTargetTreeView>;
struct CopyViews {
  void operator()(std::vector<TreeViewVar> *p_dst, std::vector<TreeViewVar> &&src) const {
    std::swap(src, *p_dst);
  }
};

template <typename T>
using Vec = std::vector<T, std::allocator<T>>;
// The input device should be DeviceOrd::CPU() instead of Context::Device(). The GBTree
// has an optimization to use CPU predictor when the DMatrix SparsePage is on CPU, even if
// the context is a CUDA context.
using HostModel = GBTreeModelView<Vec, TreeViewVar, CopyViews>;

template <bool has_missing, bool has_categorical, typename TreeView>
bst_node_t GetLeafIndex(TreeView const &tree, const RegTree::FVec &feat,
                        RegTree::CategoricalSplitMatrix const &cats, bst_node_t nidx) {
  while (!tree.IsLeaf(nidx)) {
    bst_feature_t split_index = tree.SplitIndex(nidx);
    auto fvalue = feat.GetFvalue(split_index);
    nidx = GetNextNode<has_missing, has_categorical>(
        tree, nidx, fvalue, has_missing && feat.IsMissing(split_index), cats);
  }
  return nidx;
}
}  // namespace

namespace scalar {
template <bool has_categorical>
[[nodiscard]] float PredValueByOneTree(const RegTree::FVec &p_feats,
                                       tree::ScalarTreeView const &tree,
                                       RegTree::CategoricalSplitMatrix const &cats,
                                       bst_node_t nidx) noexcept(true) {
  const bst_node_t leaf = p_feats.HasMissing()
                              ? GetLeafIndex<true, has_categorical>(tree, p_feats, cats, nidx)
                              : GetLeafIndex<false, has_categorical>(tree, p_feats, cats, nidx);
  return tree.LeafValue(leaf);
}

template <bool has_categorical, bool any_missing, bool use_array_tree_layout>
void PredValueByOneTree(tree::ScalarTreeView const &tree, std::size_t const predict_offset,
                        common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
                        linalg::MatrixView<float> out_predt, bst_node_t *p_nidx, int depth, int gid,
                        float tree_weight) {
  auto const &cats = tree.GetCategoriesMatrix();
  if constexpr (use_array_tree_layout) {
    ProcessArrayTree<has_categorical, any_missing>(tree, fvec_tloc, block_size, p_nidx, depth);
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
        PredValueByOneTree<has_categorical>(fvec_tloc[i], tree, cats, nidx) * tree_weight;
  }
}
}  // namespace scalar

namespace multi {
template <bool has_categorical>
void PredValueByOneTree(RegTree::FVec const &p_feats, tree::MultiTargetTreeView const &tree,
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
void PredValueByOneTree(tree::MultiTargetTreeView const &tree, std::size_t const predict_offset,
                        common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
                        linalg::MatrixView<float> out_predt, bst_node_t *p_nidx, bst_node_t depth,
                        float tree_weight) {
  auto const &cats = tree.GetCategoriesMatrix();
  if constexpr (use_array_tree_layout) {
    ProcessArrayTree<has_categorical, any_missing>(tree, fvec_tloc, block_size, p_nidx, depth);
  }
  for (std::size_t i = 0; i < block_size; ++i) {
    bst_node_t nidx = RegTree::kRoot;
    if constexpr (use_array_tree_layout) {
      nidx = p_nidx[i];
      p_nidx[i] = RegTree::kRoot;
    }
    auto leaf = fvec_tloc[i].HasMissing()
                    ? GetLeafIndex<true, has_categorical>(tree, fvec_tloc[i], cats, nidx)
                    : GetLeafIndex<false, has_categorical>(tree, fvec_tloc[i], cats, nidx);
    auto leaf_value = tree.LeafValue(leaf);
    auto t_predts = out_predt.Slice(predict_offset + i, linalg::All());
    assert(t_predts.Shape(0) == leaf_value.Shape(0) && "shape mismatch.");
    for (size_t j = 0; j < leaf_value.Size(); ++j) {
      t_predts(j) += leaf_value(j) * tree_weight;
    }
  }
}
}  // namespace multi

namespace {
template <bool use_array_tree_layout, bool any_missing>
void PredictBlockByAllTrees(HostModel const &model, std::size_t const predict_offset,
                            common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
                            linalg::MatrixView<float> out_predt, const std::vector<int> &tree_depth,
                            common::OptionalWeights tree_weights) {
  std::vector<bst_node_t> nidx;
  if constexpr (use_array_tree_layout) {
    nidx.resize(block_size, 0);
  }
  auto trees = model.Trees();
  for (bst_tree_t tree_id = 0, n_trees = model.Trees().size(); tree_id < n_trees; ++tree_id) {
    bst_node_t depth = use_array_tree_layout ? tree_depth[tree_id] : 0;
    auto weight = tree_weights[tree_id];
    std::visit(
        enc::Overloaded{[&](tree::ScalarTreeView const &tree) {
                          bool has_categorical = tree.HasCategoricalSplit();
                          auto const gid = model.tree_groups[tree_id];
                          if (has_categorical) {
                            scalar::PredValueByOneTree<true, any_missing, use_array_tree_layout>(
                                tree, predict_offset, fvec_tloc, block_size, out_predt, nidx.data(),
                                depth, gid, weight);
                          } else {
                            scalar::PredValueByOneTree<false, any_missing, use_array_tree_layout>(
                                tree, predict_offset, fvec_tloc, block_size, out_predt, nidx.data(),
                                depth, gid, weight);
                          }
                        },
                        [&](tree::MultiTargetTreeView const &tree) {
                          bool has_categorical = tree.HasCategoricalSplit();
                          if (has_categorical) {
                            multi::PredValueByOneTree<true, any_missing, use_array_tree_layout>(
                                tree, predict_offset, fvec_tloc, block_size, out_predt, nidx.data(),
                                depth, weight);
                          } else {
                            multi::PredValueByOneTree<false, any_missing, use_array_tree_layout>(
                                tree, predict_offset, fvec_tloc, block_size, out_predt, nidx.data(),
                                depth, weight);
                          }
                        }},
        trees[tree_id]);
  }
}

// Dispatch between template implementations
void DispatchArrayLayout(HostModel const &model, std::size_t const predict_offset,
                         common::Span<RegTree::FVec> fvec_tloc, std::size_t const block_size,
                         linalg::MatrixView<float> out_predt, const std::vector<int> &tree_depth,
                         bool any_missing, common::OptionalWeights tree_weights) {
  auto n_trees = model.tree_end - model.tree_begin;
  CHECK_EQ(n_trees, model.Trees().size());
  /*
   * We transform trees to array layout for each block of data to avoid memory overheads.
   * It makes the array layout inefficient for block_size == 1
   */
  const bool use_array_tree_layout = block_size > 1;
  if (use_array_tree_layout) {
    CHECK_EQ(n_trees, tree_depth.size());
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
      PredictBlockByAllTrees<true, true>(model, predict_offset, fvec_tloc, block_size, out_predt,
                                         tree_depth, tree_weights);
    } else {
      PredictBlockByAllTrees<true, false>(model, predict_offset, fvec_tloc, block_size, out_predt,
                                          tree_depth, tree_weights);
    }
  } else {
    PredictBlockByAllTrees<false, true>(model, predict_offset, fvec_tloc, block_size, out_predt,
                                        tree_depth, tree_weights);
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
    std::swap(mapping, this->mapping_);
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
void PredictBatchByBlockKernel(DataView const &batch, HostModel const &model,
                               ThreadTmp<kBlockOfRowsSize> *p_fvec, std::int32_t n_threads,
                               bool any_missing, linalg::TensorView<float, 2> out_predt,
                               common::OptionalWeights tree_weights) {
  auto &fvec = *p_fvec;
  // Parallel over local batches
  auto const n_samples = batch.Size();
  auto const n_features = model.n_features;

  /* Precalculate depth for each tree.
   * These values are required only for the ArrayLayout optimization,
   * so we don't need them if kBlockOfRowsSize == 1. They are equally unused
   * when every block has size 1 (n_samples <= 1): DispatchArrayLayout only
   * reads tree_depth when block_size > 1. Computing them walks every node of
   * every tree, which would otherwise dominate single-row inplace prediction.
   */
  std::vector<int> tree_depth;
  if constexpr (kBlockOfRowsSize > 1) {
    if (n_samples > 1) {
      tree_depth.resize(model.tree_end - model.tree_begin);
      CHECK_EQ(tree_depth.size(), model.Trees().size());
      common::ParallelFor(model.tree_end - model.tree_begin, n_threads, [&](auto i) {
        std::visit([&](auto &&tree) { tree_depth[i] = tree.MaxDepth(); }, model.Trees()[i]);
      });
    }
  }
  common::ParallelFor1d<kBlockOfRowsSize>(n_samples, n_threads, [&](auto &&block) {
    auto fvec_tloc = fvec.ThreadBuffer(block.Size());

    batch.FVecFill(block, n_features, fvec_tloc);
    DispatchArrayLayout(model, block.begin() + batch.base_rowid, fvec_tloc, block.Size(), out_predt,
                        tree_depth, any_missing, tree_weights);
    batch.FVecDrop(fvec_tloc);
  });
}

}  // anonymous namespace

class CPUPredictor : public Predictor {
 protected:
  void PredictDMatrix(DMatrix *p_fmat, std::vector<float> *out_preds, gbm::GBTreeModel const &model,
                      bst_tree_t tree_begin, bst_tree_t tree_end,
                      common::OptionalWeights tree_weights) const {
    auto const n_threads = this->ctx_->Threads();

    // Create a writable view on the output prediction vector.
    bst_idx_t n_groups = model.learner_model_param->OutputLength();
    bst_idx_t n_samples = p_fmat->Info().num_row_;
    CHECK_EQ(out_preds->size(), n_samples * n_groups);
    auto out_predt = linalg::MakeTensorView(ctx_, *out_preds, n_samples, n_groups);
    bool any_missing = !(p_fmat->IsDense());
    auto const h_model =
        HostModel{DeviceOrd::CPU(), model, false, tree_begin, tree_end, CopyViews{}};

    LaunchPredict(this->ctx_, p_fmat, model, [&](auto &&policy) {
      using Policy = common::GetValueT<decltype(policy)>;
      ThreadTmp<Policy::kBlockOfRowsSize> feat_vecs{n_threads};
      policy.ForEachBatch([&](auto &&batch) {
        PredictBatchByBlockKernel<Policy::kBlockOfRowsSize>(batch, h_model, &feat_vecs, n_threads,
                                                            any_missing, out_predt, tree_weights);
      });
    });
  }

 public:
  explicit CPUPredictor(Context const *ctx) : Predictor::Predictor{ctx} {}

  void PredictBatch(DMatrix *dmat, PredictionCacheEntry *predts, gbm::GBTreeModel const &model,
                    bst_tree_t tree_begin, bst_tree_t tree_end = 0,
                    std::vector<float> const *tree_weights_override = nullptr) const override {
    auto *out_preds = &predts->predictions;
    // This is actually already handled in gbm, but large amount of tests rely on the
    // behaviour.
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }
    auto const *tree_weights =
        tree_weights_override == nullptr ? model.TreeWeights() : tree_weights_override;
    auto weights = tree_weights == nullptr ? common::OptionalWeights{1.0f}
                                           : common::OptionalWeights{common::Span<float const>{
                                                 tree_weights->data() + tree_begin,
                                                 static_cast<std::size_t>(tree_end - tree_begin)}};
    this->PredictDMatrix(dmat, &out_preds->HostVector(), model, tree_begin, tree_end, weights);
  }

  [[nodiscard]] bool InplacePredict(std::shared_ptr<DMatrix> p_m, gbm::GBTreeModel const &model,
                                    float missing, PredictionCacheEntry *out_preds,
                                    bst_tree_t tree_begin, bst_tree_t tree_end) const override {
    auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
    CHECK(proxy) << error::InplacePredictProxy();
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }

    this->InitOutPredictions(p_m->Info(), &(out_preds->predictions), model);
    auto &predictions = out_preds->predictions.HostVector();
    bool any_missing = true;

    auto const n_threads = this->ctx_->Threads();
    // Always use block as we don't know the nnz.
    ThreadTmp<BlockPolicy::kBlockOfRowsSize> feat_vecs{n_threads};
    bst_idx_t n_groups = model.learner_model_param->OutputLength();
    auto const h_model =
        HostModel{DeviceOrd::CPU(), model, false, tree_begin, tree_end, CopyViews{}};
    auto const *tree_weights = model.TreeWeights();
    auto weights = tree_weights == nullptr ? common::OptionalWeights{1.0f}
                                           : common::OptionalWeights{common::Span<float const>{
                                                 tree_weights->data() + tree_begin,
                                                 static_cast<std::size_t>(tree_end - tree_begin)}};

    auto kernel = [&](auto &&view) {
      auto out_predt = linalg::MakeTensorView(ctx_, predictions, view.Size(), n_groups);
      PredictBatchByBlockKernel<BlockPolicy::kBlockOfRowsSize>(view, h_model, &feat_vecs, n_threads,
                                                               any_missing, out_predt, weights);
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

    auto n_features = model.learner_model_param->num_feature;
    ThreadTmp<1> feat_vecs{n_threads};

    auto const h_model = HostModel{DeviceOrd::CPU(), model, false, 0, ntree_limit, CopyViews{}};
    LaunchPredict(this->ctx_, p_fmat, model, [&](auto &&policy) {
      policy.ForEachBatch([&](auto &&batch) {
        common::ParallelFor1d<1>(batch.Size(), n_threads, [&](auto &&block) {
          auto ridx = static_cast<bst_idx_t>(batch.base_rowid + block.begin());
          auto fvec_tloc = feat_vecs.ThreadBuffer(block.Size());
          batch.FVecFill(block, n_features, fvec_tloc);

          for (bst_tree_t j = 0; j < ntree_limit; ++j) {
            bst_node_t nidx = std::visit(
                [&](auto &&tree) {
                  return GetLeafIndex<true, true>(tree, fvec_tloc.front(),
                                                  tree.GetCategoriesMatrix(), RegTree::kRoot);
                },
                h_model.Trees()[j]);
            preds[ridx * ntree_limit + j] = static_cast<float>(nidx);
          }
          batch.FVecDrop(fvec_tloc);
        });
      });
    });
  }

  void PredictContribution(DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                           const gbm::GBTreeModel &model, bst_tree_t ntree_limit, bool approximate,
                           int condition, unsigned condition_feature) const override {
    auto const *tree_weights = model.TreeWeights();
    if (approximate) {
      interpretability::ApproxFeatureImportance(this->ctx_, p_fmat, out_contribs, model,
                                                ntree_limit, tree_weights);
    } else {
      interpretability::ShapValues(this->ctx_, p_fmat, out_contribs, model, ntree_limit,
                                   tree_weights, condition, condition_feature);
    }
  }

  void PredictInteractionContributions(DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                                       gbm::GBTreeModel const &model, bst_tree_t ntree_limit,
                                       bool approximate) const override {
    auto const *tree_weights = model.TreeWeights();
    interpretability::ShapInteractionValues(this->ctx_, p_fmat, out_contribs, model, ntree_limit,
                                            tree_weights, approximate);
  }
};

XGBOOST_REGISTER_PREDICTOR(CPUPredictor, "cpu_predictor")
    .describe("Make predictions using CPU.")
    .set_body([](Context const *ctx) { return new CPUPredictor(ctx); });
}  // namespace xgboost::predictor
