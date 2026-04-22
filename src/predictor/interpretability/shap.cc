/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include "shap.h"

#include <algorithm>    // for copy, fill
#include <array>        // for array
#include <cmath>        // for abs
#include <cstdint>      // for uint32_t
#include <limits>       // for numeric_limits
#include <type_traits>  // for remove_const_t
#include <vector>       // for vector

#include "../../common/threading_utils.h"  // for ParallelFor
#include "../../gbm/gbtree_model.h"        // for GBTreeModel
#include "../../tree/tree_view.h"          // for ScalarTreeView
#include "../data_accessor.h"              // for GHistIndexMatrixView
#include "../predict_fn.h"                 // for GetTreeLimit
#include "dmlc/omp.h"                      // for omp_get_thread_num
#include "quadrature.h"
#include "xgboost/base.h"        // for bst_omp_uint
#include "xgboost/logging.h"     // for CHECK
#include "xgboost/span.h"        // for Span
#include "xgboost/tree_model.h"  // for MTNotImplemented

namespace xgboost::interpretability {
namespace {
void ValidateTreeWeights(std::vector<float> const *tree_weights, bst_tree_t tree_end) {
  if (tree_weights == nullptr) {
    return;
  }
  CHECK_GE(tree_weights->size(), static_cast<std::size_t>(tree_end));
}

float FillNodeMeanValues(tree::ScalarTreeView const &tree, bst_node_t nidx,
                         std::vector<float> *mean_values) {
  float result;
  auto &node_mean_values = *mean_values;
  if (tree.IsLeaf(nidx)) {
    result = tree.LeafValue(nidx);
  } else {
    result = FillNodeMeanValues(tree, tree.LeftChild(nidx), mean_values) *
             tree.Stat(tree.LeftChild(nidx)).sum_hess;
    result += FillNodeMeanValues(tree, tree.RightChild(nidx), mean_values) *
              tree.Stat(tree.RightChild(nidx)).sum_hess;
    result /= tree.Stat(nidx).sum_hess;
  }
  node_mean_values[nidx] = result;
  return result;
}

void FillNodeMeanValues(tree::ScalarTreeView const &tree, std::vector<float> *mean_values) {
  auto n_nodes = tree.Size();
  if (static_cast<decltype(n_nodes)>(mean_values->size()) == n_nodes) {
    return;
  }
  mean_values->resize(n_nodes);
  FillNodeMeanValues(tree, 0, mean_values);
}

double FillRootMeanValue(tree::ScalarTreeView const &tree, bst_node_t nidx) {
  if (tree.IsLeaf(nidx)) {
    return tree.LeafValue(nidx);
  }
  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  double result = FillRootMeanValue(tree, left) * tree.SumHess(left);
  result += FillRootMeanValue(tree, right) * tree.SumHess(right);
  result /= tree.SumHess(nidx);
  return result;
}

void ValidateQuadratureTreeShapCovers(tree::ScalarTreeView const &tree, bst_node_t nidx) {
  if (tree.IsLeaf(nidx)) {
    return;
  }

  CHECK_GT(tree.SumHess(nidx), 0.0f)
      << "CPU QuadratureTreeSHAP is undefined for trees with non-positive cover at split nodes.";

  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  CHECK_GT(tree.SumHess(left), 0.0f)
      << "CPU QuadratureTreeSHAP is undefined for trees with non-positive cover at child nodes.";
  CHECK_GT(tree.SumHess(right), 0.0f)
      << "CPU QuadratureTreeSHAP is undefined for trees with non-positive cover at child nodes.";

  ValidateQuadratureTreeShapCovers(tree, left);
  ValidateQuadratureTreeShapCovers(tree, right);
}

void CalculateApproxContributions(tree::ScalarTreeView const &tree, RegTree::FVec const &feats,
                                  std::vector<float> *mean_values,
                                  std::vector<bst_float> *out_contribs) {
  CHECK_EQ(out_contribs->size(), feats.Size() + 1);
  CHECK_GT(mean_values->size(), 0U);
  bst_feature_t split_index = 0;
  float node_value = (*mean_values)[0];
  out_contribs->back() += node_value;
  if (tree.IsLeaf(RegTree::kRoot)) {
    return;
  }

  bst_node_t nidx = RegTree::kRoot;
  auto const &cats = tree.GetCategoriesMatrix();
  while (!tree.IsLeaf(nidx)) {
    split_index = tree.SplitIndex(nidx);
    nidx = predictor::GetNextNode<true, true>(tree, nidx, feats.GetFvalue(split_index),
                                              feats.IsMissing(split_index), cats);
    auto new_value = (*mean_values)[nidx];
    (*out_contribs)[split_index] += new_value - node_value;
    node_value = new_value;
  }
  (*out_contribs)[split_index] += tree.LeafValue(nidx) - node_value;
}

// Keep the CPU quadrature recurrence on the same fixed 8-point rule as the GPU path so the hot
// loops stay small and the compiler can fully unroll the basis update and extraction work.
constexpr std::size_t kQuadratureTreeShapPoints = 8;
constexpr double kQuadratureTreeShapBuildQeps = 1e-15;
constexpr float kQuadratureTreeShapUnseen = -999.0f;

struct QuadratureRule {
  std::array<float, kQuadratureTreeShapPoints> nodes{};
  std::array<float, kQuadratureTreeShapPoints> weights{};
};
using QuadratureBuffer = std::array<float, kQuadratureTreeShapPoints>;

QuadratureRule const &GetQuadratureRule() {
  static QuadratureRule const kRule = [] {
    auto const rule_d =
        detail::MakeEndpointQuadrature<kQuadratureTreeShapPoints>(kQuadratureTreeShapBuildQeps);
    QuadratureRule out;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      out.nodes[i] = static_cast<float>(rule_d.nodes[i]);
      out.weights[i] = static_cast<float>(rule_d.weights[i]);
    }
    return out;
  }();
  return kRule;
}

void AddInPlace(QuadratureBuffer *lhs, QuadratureBuffer const &rhs) {
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    (*lhs)[i] += rhs[i];
  }
}

float ExtractQuadratureDelta(QuadratureRule const &rule, QuadratureBuffer const &h_vals,
                             float p_enter, float p_exit) {
  float acc = 0.0f;
  if (p_enter != 1.0f) {
    auto const alpha_enter = p_enter - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      acc += alpha_enter * h_vals[i] / (1.0f + alpha_enter * rule.nodes[i]);
    }
  }
  if (p_exit != 1.0f) {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      acc -= alpha_exit * h_vals[i] / (1.0f + alpha_exit * rule.nodes[i]);
    }
  }
  return acc;
}

float ExtractQuadratureInteractionDelta(QuadratureRule const &rule, QuadratureBuffer const &h_vals,
                                        float p_enter, float p_exit, float q_partner) {
  if (q_partner == 1.0f) {
    return 0.0f;
  }

  auto const alpha_partner = q_partner - 1.0f;
  auto const alpha_enter = p_enter - 1.0f;

  float acc = 0.0f;
  if (p_exit == 1.0f) {
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      auto const edge_delta = alpha_enter / (1.0f + alpha_enter * rule.nodes[i]);
      acc += alpha_partner * h_vals[i] * edge_delta / (1.0f + alpha_partner * rule.nodes[i]);
    }
  } else {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      auto const edge_delta = alpha_enter / (1.0f + alpha_enter * rule.nodes[i]) -
                              alpha_exit / (1.0f + alpha_exit * rule.nodes[i]);
      acc += alpha_partner * h_vals[i] * edge_delta / (1.0f + alpha_partner * rule.nodes[i]);
    }
  }
  return acc;
}

void WriteWeightedLeafReturn(tree::ScalarTreeView const &tree, QuadratureRule const &rule,
                             bst_node_t nidx, QuadratureBuffer const &c_vals, float w_prod,
                             QuadratureBuffer *out_h) {
  auto const leaf_scale = w_prod * tree.LeafValue(nidx);
  for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
    (*out_h)[i] = c_vals[i] * leaf_scale * rule.weights[i];
  }
}

// Dense row-local output view for additive contributions.
template <typename T>
struct ContributionVectorView {
  T *data;
  std::size_t size;

  T &operator[](std::size_t idx) const { return data[idx]; }
};

// Dense row-local output view for interaction matrices. Future formulations can target this sink
// directly instead of open-coding flattened indexing arithmetic.
template <typename T>
struct DenseInteractionMatrixView {
  T *data;
  std::size_t ncolumns;

  T &operator()(std::size_t i, std::size_t j) const { return data[i * ncolumns + j]; }
};

// One active split on the current root-to-node path. Traversal owns the push/pop discipline, while
// formulations can inspect the live path without duplicating duplicate-feature bookkeeping.
struct QuadraturePathElement {
  bst_feature_t split_index;
  float p_child;
};

// Read-only formulation view of the current root-to-node path. Traversal keeps ownership of the
// stack so different contribution formulations can inspect the same live path state.
struct QuadraturePathView {
  common::Span<QuadraturePathElement const> elements;

  [[nodiscard]] auto Depth() const { return elements.size(); }
  [[nodiscard]] bool Empty() const { return elements.empty(); }
  [[nodiscard]] auto Entries() const { return elements; }

  [[nodiscard]] auto CurrentSplit() const -> QuadraturePathElement const & {
    CHECK(!elements.empty());
    return elements.back();
  }

  // Iterate the active path once per feature, newest-to-oldest. Later duplicate splits are the
  // live ones for path-local partner lookups, so older duplicates are hidden from formulations.
  template <typename Fn>
  void ForEachUniqueFeature(Fn &&fn) const {
    for (std::size_t i = elements.size(); i != 0; --i) {
      auto const idx = i - 1;
      auto const split_index = elements[idx].split_index;
      bool shadowed = false;
      for (std::size_t newer = elements.size(); newer > i; --newer) {
        if (elements[newer - 1].split_index == split_index) {
          shadowed = true;
          break;
        }
      }
      if (!shadowed) {
        fn(idx, elements[idx]);
      }
    }
  }
};

struct EmptyQuadraturePathState {
  void Reset() const {}
  void Push(bst_feature_t, float) const {}
  void Pop(bst_feature_t) const {}
  [[nodiscard]] auto View() const { return QuadraturePathView{{}}; }
};

struct LiveQuadraturePathState {
  std::vector<QuadraturePathElement> *path;

  void Reset() const { path->clear(); }

  void Push(bst_feature_t split_index, float p_child) const {
    path->push_back(QuadraturePathElement{split_index, p_child});
  }

  void Pop(bst_feature_t) const { path->pop_back(); }

  [[nodiscard]] auto View() const {
    return QuadraturePathView{common::Span<QuadraturePathElement const>{*path}};
  }
};

// Current additive SHAP formulation. It consumes the weighted subtree return and writes one
// feature contribution per return edge.
struct AdditiveContributionFormulation {
  EmptyQuadraturePathState path_state;
  ContributionVectorView<float> phi;

  explicit AdditiveContributionFormulation(ContributionVectorView<float> phi) : phi{phi} {}

  void ResetPath() const { path_state.Reset(); }
  void PushPathSplit(bst_feature_t split_index, float p_child) const {
    path_state.Push(split_index, p_child);
  }
  void PopPathSplit(bst_feature_t split_index) const { path_state.Pop(split_index); }

  void HandleLeaf(tree::ScalarTreeView const &tree, QuadratureRule const &rule, bst_node_t nidx,
                  QuadratureBuffer const &c_vals, float w_prod, QuadratureBuffer *out_h) const {
    WriteWeightedLeafReturn(tree, rule, nidx, c_vals, w_prod, out_h);
  }

  void HandleReturn(QuadratureRule const &rule, bst_feature_t split_index,
                    QuadratureBuffer const &h_vals, float p_enter, float p_exit) const {
    phi[split_index] += ExtractQuadratureDelta(rule, h_vals, p_enter, p_exit);
  }
};

// First path-local interaction formulation built on top of the quadrature traversal. It keeps the
// traversal and weighted subtree return shared with additive SHAP, and only changes how return
// edges are written into the dense interaction sink.
struct InteractionContributionFormulation {
  LiveQuadraturePathState path_state;
  ContributionVectorView<float> phi_diag;
  DenseInteractionMatrixView<float> phi_interactions;
  float scale;

  InteractionContributionFormulation(LiveQuadraturePathState path_state,
                                     ContributionVectorView<float> phi_diag,
                                     DenseInteractionMatrixView<float> phi_interactions,
                                     float scale)
      : path_state{path_state},
        phi_diag{phi_diag},
        phi_interactions{phi_interactions},
        scale{scale} {}

  void ResetPath() const { path_state.Reset(); }
  void PushPathSplit(bst_feature_t split_index, float p_child) const {
    path_state.Push(split_index, p_child);
  }
  void PopPathSplit(bst_feature_t split_index) const { path_state.Pop(split_index); }

  // Traversal still needs a weighted subtree return, so the interaction path shares the additive
  // leaf behavior and changes only the return-edge algebra.
  void HandleLeaf(tree::ScalarTreeView const &tree, QuadratureRule const &rule, bst_node_t nidx,
                  QuadratureBuffer const &c_vals, float w_prod, QuadratureBuffer *out_h) const {
    WriteWeightedLeafReturn(tree, rule, nidx, c_vals, w_prod, out_h);
  }

  // Walk the live unique path excluding the current split. A pairwise formulation can distribute
  // the current edge effect across these partner features without reimplementing duplicate logic.
  template <typename Fn>
  void ForEachPartner(QuadraturePathView path, Fn &&fn) const {
    CHECK(!path.Empty());
    auto const current_split = path.CurrentSplit().split_index;
    bool skipped_current = false;
    path.ForEachUniqueFeature([&](std::size_t, QuadraturePathElement const &element) {
      if (!skipped_current && element.split_index == current_split) {
        skipped_current = true;
        return;
      }
      fn(element);
    });
  }

  void AccumulatePair(bst_feature_t split_index, QuadraturePathElement const &partner,
                      float pair_delta) const {
    auto const i = static_cast<std::size_t>(split_index);
    auto const j = static_cast<std::size_t>(partner.split_index);
    phi_interactions(i, j) += scale * pair_delta;
  }

  void HandleReturn(QuadratureRule const &rule, bst_feature_t split_index,
                    QuadratureBuffer const &h_vals, float p_enter, float p_exit) const {
    auto path = path_state.View();
    phi_diag[split_index] += scale * ExtractQuadratureDelta(rule, h_vals, p_enter, p_exit);

    this->ForEachPartner(path, [&](QuadraturePathElement const &partner) {
      auto pair_delta =
          ExtractQuadratureInteractionDelta(rule, h_vals, p_enter, p_exit, partner.p_child);
      this->AccumulatePair(split_index, partner, pair_delta);
    });
  }
};

// Tree-walk engine for quadrature formulations. It owns feature evaluation, child descent, and
// the live path-probability state, then hands leaf/return events to the selected formulation.
template <typename ContributionFormulation>
struct QuadratureTreeShapRunner {
  tree::ScalarTreeView const &tree;
  RegTree::FVec const &feat;
  QuadratureRule const &rule;
  std::vector<float> *path_prob;
  ContributionFormulation formulation;

  [[nodiscard]] bool EvaluateGoesLeft(bst_node_t nidx) const {
    auto split_index = tree.SplitIndex(nidx);
    auto const &cats = tree.GetCategoriesMatrix();
    auto next = predictor::GetNextNode<true, true>(tree, nidx, feat.GetFvalue(split_index),
                                                   feat.IsMissing(split_index), cats);
    return next == tree.LeftChild(nidx);
  }

  [[nodiscard]] float ChildWeight(bst_node_t parent, bst_node_t child) const {
    auto parent_cover = tree.Stat(parent).sum_hess;
    CHECK_GT(parent_cover, 0.0f);
    return tree.Stat(child).sum_hess / parent_cover;
  }

  void VisitChild(bst_node_t split_node, bst_node_t child_node, float child_weight, bool satisfies,
                  QuadratureBuffer const &c_vals, float w_prod, QuadratureBuffer *out_h) {
    auto split_index = tree.SplitIndex(split_node);
    auto p_old = (*path_prob)[split_index];

    float p_e = 0.0f;
    if (p_old == kQuadratureTreeShapUnseen) {
      p_e = satisfies ? 1.0f / child_weight : 0.0f;
    } else {
      p_e = satisfies ? p_old / child_weight : 0.0f;
    }

    auto c_child = c_vals;
    auto alpha_e = p_e - 1.0f;
    for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
      c_child[i] *= 1.0f + alpha_e * rule.nodes[i];
    }

    if (p_old != kQuadratureTreeShapUnseen) {
      auto alpha_old = p_old - 1.0f;
      if (alpha_old != 0.0f) {
        for (std::size_t i = 0; i < kQuadratureTreeShapPoints; ++i) {
          c_child[i] /= 1.0f + alpha_old * rule.nodes[i];
        }
      }
    }

    (*path_prob)[split_index] = p_e;
    formulation.PushPathSplit(split_index, p_e);
    this->RunNode(child_node, c_child, w_prod * child_weight, out_h);
    formulation.HandleReturn(rule, split_index, *out_h, p_e,
                             p_old == kQuadratureTreeShapUnseen ? 1.0f : p_old);
    formulation.PopPathSplit(split_index);
    (*path_prob)[split_index] = p_old;
  }

  void RunNode(bst_node_t nidx, QuadratureBuffer const &c_vals, float w_prod,
               QuadratureBuffer *out_h) {
    if (tree.IsLeaf(nidx)) {
      formulation.HandleLeaf(tree, rule, nidx, c_vals, w_prod, out_h);
      return;
    }

    auto left = tree.LeftChild(nidx);
    auto right = tree.RightChild(nidx);
    auto left_weight = this->ChildWeight(nidx, left);
    auto right_weight = this->ChildWeight(nidx, right);
    auto goes_left = this->EvaluateGoesLeft(nidx);

    QuadratureBuffer right_h{};

    this->VisitChild(nidx, left, left_weight, goes_left, c_vals, w_prod, out_h);
    this->VisitChild(nidx, right, right_weight, !goes_left, c_vals, w_prod, &right_h);
    AddInPlace(out_h, right_h);
  }

  void Run() {
    formulation.ResetPath();
    if (tree.IsLeaf(RegTree::kRoot)) {
      return;
    }

    QuadratureBuffer c_init{};
    c_init.fill(1.0f);
    QuadratureBuffer h_vals{};
    this->RunNode(RegTree::kRoot, c_init, 1.0f, &h_vals);
  }
};

struct QuadratureTreeShapModelData {
  std::vector<tree::ScalarTreeView> trees;
  std::vector<std::vector<bst_tree_t>> trees_by_group;
  std::vector<float> weights;
  std::vector<float> group_root_mean_sums;
};

QuadratureTreeShapModelData MakeQuadratureTreeShapModelData(
    gbm::GBTreeModel const &model, bst_tree_t tree_end, std::vector<float> const *tree_weights) {
  auto const n_trees = static_cast<std::size_t>(tree_end);
  auto const h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  auto const n_groups = model.learner_model_param->num_output_group;

  QuadratureTreeShapModelData out;
  out.trees.reserve(n_trees);
  out.trees_by_group.resize(n_groups);
  out.weights.resize(n_trees, 1.0f);
  out.group_root_mean_sums.resize(n_groups, 0.0f);

  for (std::size_t i = 0; i < n_trees; ++i) {
    out.trees.emplace_back(model.trees[i].get());
  }
  for (bst_tree_t i = 0; i < tree_end; ++i) {
    auto gid = h_tree_groups[i];
    auto weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[i];
    out.trees_by_group[gid].push_back(i);
    out.weights[i] = weight;
    ValidateQuadratureTreeShapCovers(out.trees[i], RegTree::kRoot);
    out.group_root_mean_sums[gid] +=
        static_cast<float>(FillRootMeanValue(out.trees[i], RegTree::kRoot) * weight);
  }
  return out;
}

template <typename EncAccessor, typename Fn>
void DispatchByBatchView(Context const *ctx, DMatrix *p_fmat, EncAccessor acc, Fn &&fn) {
  using AccT = std::decay_t<EncAccessor>;
  if (p_fmat->PageExists<SparsePage>()) {
    for (auto const &page : p_fmat->GetBatches<SparsePage>()) {
      predictor::SparsePageView<AccT> view{page.GetView(), page.base_rowid, acc};
      fn(view);
    }
  } else {
    auto ft = p_fmat->Info().feature_types.ConstHostVector();
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx, {})) {
      predictor::GHistIndexMatrixView<AccT> view{page, acc, ft};
      fn(view);
    }
  }
}

template <typename Fn>
void LaunchShap(Context const *ctx, DMatrix *p_fmat, gbm::GBTreeModel const &model, Fn &&fn) {
  if (model.Cats()->HasCategorical() && p_fmat->Cats()->NeedRecode()) {
    auto new_enc = p_fmat->Cats()->HostView();
    auto [acc, mapping] = ::xgboost::cpu_impl::MakeCatAccessor(ctx, new_enc, model.Cats());
    DispatchByBatchView(ctx, p_fmat, acc, fn);
  } else {
    DispatchByBatchView(ctx, p_fmat, NoOpAccessor{}, fn);
  }
}
}  // namespace

namespace cpu_impl {
void QuadratureTreeShapValues(Context const *ctx, DMatrix *p_fmat,
                              HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                              bst_tree_t tree_end, std::vector<float> const *tree_weights,
                              std::size_t quadrature_points);
void QuadratureTreeShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                                         HostDeviceVector<float> *out_contribs,
                                         gbm::GBTreeModel const &model, bst_tree_t tree_end,
                                         std::vector<float> const *tree_weights,
                                         std::size_t quadrature_points);

void ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                gbm::GBTreeModel const &model, bst_tree_t tree_end,
                std::vector<float> const *tree_weights, int condition, unsigned condition_feature) {
  CHECK_EQ(condition, 0) << "CPU exact SHAP uses fixed 8-point QuadratureTreeSHAP and does not "
                            "support conditioned exact attributions.";
  CHECK_EQ(condition_feature, 0U)
      << "CPU exact SHAP uses fixed 8-point QuadratureTreeSHAP and does "
         "not support conditioned exact attributions.";
  QuadratureTreeShapValues(ctx, p_fmat, out_contribs, model, tree_end, tree_weights,
                           kQuadratureTreeShapPoints);
}

void QuadratureTreeShapValues(Context const *ctx, DMatrix *p_fmat,
                              HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                              bst_tree_t tree_end, std::vector<float> const *tree_weights,
                              std::size_t quadrature_points) {
  CHECK(!model.learner_model_param->IsVectorLeaf()) << "Predict contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  CHECK_EQ(quadrature_points, kQuadratureTreeShapPoints)
      << "CPU QuadratureTreeSHAP currently uses a fixed quadrature size of "
      << kQuadratureTreeShapPoints << ".";
  MetaInfo const &info = p_fmat->Info();
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  CHECK_GE(tree_end, 0);
  ValidateTreeWeights(tree_weights, tree_end);
  auto const n_threads = ctx->Threads();
  auto const n_groups = model.learner_model_param->num_output_group;
  auto const n_features = model.learner_model_param->num_feature;
  size_t const ncolumns = model.learner_model_param->num_feature + 1;
  std::vector<bst_float> &contribs = out_contribs->HostVector();
  contribs.resize(info.num_row_ * ncolumns * model.learner_model_param->num_output_group);
  std::fill(contribs.begin(), contribs.end(), 0.0f);
  CHECK_NE(n_groups, 0);
  auto const &rule = GetQuadratureRule();
  auto const base_score = model.learner_model_param->BaseScore(DeviceOrd::CPU());
  auto model_data = MakeQuadratureTreeShapModelData(model, tree_end, tree_weights);
  std::vector<RegTree::FVec> feats_tloc(n_threads);
  std::vector<std::vector<float>> contribs_tloc(n_threads, std::vector<float>(ncolumns));
  std::vector<std::vector<float>> path_prob_tloc(
      n_threads, std::vector<float>(n_features, kQuadratureTreeShapUnseen));

  auto device = ctx->Device().IsSycl() ? DeviceOrd::CPU() : ctx->Device();
  auto base_margin = info.base_margin_.View(device);

  auto process_view = [&](auto &&view) {
    common::ParallelFor(view.Size(), n_threads, [&](auto i) {
      auto tid = omp_get_thread_num();
      auto &feats = feats_tloc[tid];
      if (feats.Size() == 0) {
        feats.Init(model.learner_model_param->num_feature);
      }
      auto &this_tree_contribs = contribs_tloc[tid];
      auto &path_prob = path_prob_tloc[tid];
      auto row_idx = view.base_rowid + i;
      auto n_valid = view.DoFill(i, feats.Data().data());
      feats.HasMissing(n_valid != feats.Size());
      for (bst_target_t gid = 0; gid < n_groups; ++gid) {
        float *p_contribs = &contribs[(row_idx * n_groups + gid) * ncolumns];
        for (auto j : model_data.trees_by_group[gid]) {
          std::fill(this_tree_contribs.begin(), this_tree_contribs.end(), 0.0f);
          auto formulation = AdditiveContributionFormulation{{this_tree_contribs.data(), ncolumns}};
          auto runner = QuadratureTreeShapRunner<AdditiveContributionFormulation>{
              model_data.trees[j], feats, rule, &path_prob, formulation};
          runner.Run();
          auto const weight = model_data.weights[j];
          for (size_t ci = 0; ci + 1 < ncolumns; ++ci) {
            p_contribs[ci] += this_tree_contribs[ci] * weight;
          }
        }
        p_contribs[ncolumns - 1] += model_data.group_root_mean_sums[gid];
        if (base_margin.Size() != 0) {
          CHECK_EQ(base_margin.Shape(1), n_groups);
          p_contribs[ncolumns - 1] += base_margin(row_idx, gid);
        } else {
          p_contribs[ncolumns - 1] += base_score(gid);
        }
      }
      feats.Drop();
    });
  };

  LaunchShap(ctx, p_fmat, model, process_view);
}

void QuadratureTreeShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                                         HostDeviceVector<float> *out_contribs,
                                         gbm::GBTreeModel const &model, bst_tree_t tree_end,
                                         std::vector<float> const *tree_weights,
                                         std::size_t quadrature_points) {
  CHECK(!model.learner_model_param->IsVectorLeaf())
      << "Predict interaction contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit()) << "Predict interaction contribution support for "
                                            "column-wise data split is not yet implemented.";
  CHECK_EQ(quadrature_points, kQuadratureTreeShapPoints)
      << "CPU QuadratureTreeSHAP currently uses a fixed quadrature size of "
      << kQuadratureTreeShapPoints << ".";

  MetaInfo const &info = p_fmat->Info();
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  CHECK_GE(tree_end, 0);
  ValidateTreeWeights(tree_weights, tree_end);

  auto const n_threads = ctx->Threads();
  auto const n_groups = model.learner_model_param->num_output_group;
  auto const n_features = model.learner_model_param->num_feature;
  auto const ncolumns = n_features + 1;
  auto const row_chunk = n_groups * ncolumns * ncolumns;
  auto const matrix_chunk = ncolumns * ncolumns;

  std::vector<bst_float> &contribs = out_contribs->HostVector();
  contribs.resize(info.num_row_ * row_chunk);
  std::fill(contribs.begin(), contribs.end(), 0.0f);

  auto const &rule = GetQuadratureRule();
  auto const base_score = model.learner_model_param->BaseScore(DeviceOrd::CPU());
  auto model_data = MakeQuadratureTreeShapModelData(model, tree_end, tree_weights);
  std::vector<RegTree::FVec> feats_tloc(n_threads);
  std::vector<std::vector<QuadraturePathElement>> path_tloc(n_threads);
  std::vector<std::vector<float>> path_prob_tloc(
      n_threads, std::vector<float>(n_features, kQuadratureTreeShapUnseen));
  std::vector<std::vector<float>> diag_tloc(n_threads, std::vector<float>(ncolumns));

  auto device = ctx->Device().IsSycl() ? DeviceOrd::CPU() : ctx->Device();
  auto base_margin = info.base_margin_.View(device);

  auto process_view = [&](auto &&view) {
    common::ParallelFor(view.Size(), n_threads, [&](auto i) {
      auto tid = omp_get_thread_num();
      auto &feats = feats_tloc[tid];
      if (feats.Size() == 0) {
        feats.Init(model.learner_model_param->num_feature);
      }
      auto &path = path_tloc[tid];
      auto &path_prob = path_prob_tloc[tid];
      auto &diag = diag_tloc[tid];
      auto row_idx = view.base_rowid + i;
      auto n_valid = view.DoFill(i, feats.Data().data());
      feats.HasMissing(n_valid != feats.Size());

      for (bst_target_t gid = 0; gid < n_groups; ++gid) {
        auto const offset = (row_idx * n_groups + gid) * matrix_chunk;
        auto matrix = DenseInteractionMatrixView<bst_float>{contribs.data() + offset, ncolumns};
        std::fill(diag.begin(), diag.end(), 0.0f);

        for (auto j : model_data.trees_by_group[gid]) {
          auto formulation = InteractionContributionFormulation{{&path},
                                                                {diag.data(), ncolumns},
                                                                {matrix.data, matrix.ncolumns},
                                                                model_data.weights[j]};
          auto runner = QuadratureTreeShapRunner<InteractionContributionFormulation>{
              model_data.trees[j], feats, rule, &path_prob, formulation};
          runner.Run();
        }

        diag[ncolumns - 1] += model_data.group_root_mean_sums[gid];
        if (base_margin.Size() != 0) {
          CHECK_EQ(base_margin.Shape(1), n_groups);
          diag[ncolumns - 1] += base_margin(row_idx, gid);
        } else {
          diag[ncolumns - 1] += base_score(gid);
        }

        // The path-local return updates populate row-wise off-diagonal effects. Average the two
        // directional estimates so the final matrix is explicitly symmetric.
        for (size_t r = 0; r < ncolumns; ++r) {
          for (size_t c = r + 1; c < ncolumns; ++c) {
            auto const sym = 0.5f * (matrix(r, c) + matrix(c, r));
            matrix(r, c) = sym;
            matrix(c, r) = sym;
          }
        }

        // Match the incumbent interaction semantics: each diagonal entry is the additive SHAP
        // value minus the off-diagonal interactions in that row.
        for (size_t r = 0; r < ncolumns; ++r) {
          float value = diag[r];
          for (size_t c = 0; c < ncolumns; ++c) {
            if (c != r) {
              value -= matrix(r, c);
            }
          }
          matrix(r, r) = value;
        }
      }

      feats.Drop();
    });
  };

  LaunchShap(ctx, p_fmat, model, process_view);
}

void ApproxFeatureImportance(Context const *ctx, DMatrix *p_fmat,
                             HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                             bst_tree_t tree_end, std::vector<float> const *tree_weights) {
  CHECK(!model.learner_model_param->IsVectorLeaf()) << "Predict contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  MetaInfo const &info = p_fmat->Info();
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  CHECK_GE(tree_end, 0);
  ValidateTreeWeights(tree_weights, tree_end);
  auto const n_trees = static_cast<std::size_t>(tree_end);
  auto const n_threads = ctx->Threads();
  size_t const ncolumns = model.learner_model_param->num_feature + 1;
  std::vector<bst_float> &contribs = out_contribs->HostVector();
  contribs.resize(info.num_row_ * ncolumns * model.learner_model_param->num_output_group);
  std::fill(contribs.begin(), contribs.end(), 0);
  std::vector<std::vector<float>> mean_values(n_trees);
  common::ParallelFor(n_trees, n_threads, [&](auto i) {
    FillNodeMeanValues(model.trees[i]->HostScView(), &(mean_values[i]));
  });

  auto const n_groups = model.learner_model_param->num_output_group;
  CHECK_NE(n_groups, 0);
  auto const base_score = model.learner_model_param->BaseScore(DeviceOrd::CPU());
  auto const h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  std::vector<RegTree::FVec> feats_tloc(n_threads);
  std::vector<std::vector<bst_float>> contribs_tloc(n_threads, std::vector<bst_float>(ncolumns));

  auto device = ctx->Device().IsSycl() ? DeviceOrd::CPU() : ctx->Device();
  auto base_margin = info.base_margin_.View(device);

  auto process_view = [&](auto &&view) {
    common::ParallelFor(view.Size(), n_threads, [&](auto i) {
      auto tid = omp_get_thread_num();
      auto &feats = feats_tloc[tid];
      if (feats.Size() == 0) {
        feats.Init(model.learner_model_param->num_feature);
      }
      auto &this_tree_contribs = contribs_tloc[tid];
      auto row_idx = view.base_rowid + i;
      auto n_valid = view.DoFill(i, feats.Data().data());
      feats.HasMissing(n_valid != feats.Size());
      for (bst_target_t gid = 0; gid < n_groups; ++gid) {
        float *p_contribs = &contribs[(row_idx * n_groups + gid) * ncolumns];
        for (bst_tree_t j = 0; j < tree_end; ++j) {
          if (h_tree_groups[j] != gid) {
            continue;
          }
          std::fill(this_tree_contribs.begin(), this_tree_contribs.end(), 0);
          auto const sc_tree = model.trees[j]->HostScView();
          CalculateApproxContributions(sc_tree, feats, &mean_values[j], &this_tree_contribs);
          for (size_t ci = 0; ci < ncolumns; ++ci) {
            p_contribs[ci] +=
                this_tree_contribs[ci] * (tree_weights == nullptr ? 1 : (*tree_weights)[j]);
          }
        }
        if (base_margin.Size() != 0) {
          CHECK_EQ(base_margin.Shape(1), n_groups);
          p_contribs[ncolumns - 1] += base_margin(row_idx, gid);
        } else {
          p_contribs[ncolumns - 1] += base_score(gid);
        }
      }
      feats.Drop();
    });
  };

  LaunchShap(ctx, p_fmat, model, process_view);
}

void ShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                           HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                           bst_tree_t tree_end, std::vector<float> const *tree_weights,
                           bool approximate) {
  if (!approximate) {
    QuadratureTreeShapInteractionValues(ctx, p_fmat, out_contribs, model, tree_end, tree_weights,
                                        kQuadratureTreeShapPoints);
    return;
  }
  CHECK(!model.learner_model_param->IsVectorLeaf())
      << "Predict interaction contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit()) << "Predict interaction contribution support for "
                                            "column-wise data split is not yet implemented.";
  MetaInfo const &info = p_fmat->Info();
  auto const ngroup = model.learner_model_param->num_output_group;
  auto const ncolumns = model.learner_model_param->num_feature;
  const std::size_t row_chunk = ngroup * (ncolumns + 1) * (ncolumns + 1);
  const std::size_t mrow_chunk = (ncolumns + 1) * (ncolumns + 1);
  const std::size_t crow_chunk = ngroup * (ncolumns + 1);

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
  ApproxFeatureImportance(ctx, p_fmat, &contribs_diag_hdv, model, tree_end, tree_weights);
  for (size_t i = 0; i < ncolumns + 1; ++i) {
    ApproxFeatureImportance(ctx, p_fmat, &contribs_off_hdv, model, tree_end, tree_weights);
    ApproxFeatureImportance(ctx, p_fmat, &contribs_on_hdv, model, tree_end, tree_weights);

    for (size_t j = 0; j < info.num_row_; ++j) {
      for (std::remove_const_t<decltype(ngroup)> l = 0; l < ngroup; ++l) {
        const std::size_t o_offset = j * row_chunk + l * mrow_chunk;
        const std::size_t c_offset = j * crow_chunk + l * (ncolumns + 1);
        auto matrix =
            DenseInteractionMatrixView<bst_float>{contribs.data() + o_offset, ncolumns + 1};
        auto diag =
            ContributionVectorView<bst_float>{contribs_diag.data() + c_offset, ncolumns + 1};
        auto off = ContributionVectorView<bst_float>{contribs_off.data() + c_offset, ncolumns + 1};
        auto on = ContributionVectorView<bst_float>{contribs_on.data() + c_offset, ncolumns + 1};
        matrix(i, i) = 0;
        for (size_t k = 0; k < ncolumns + 1; ++k) {
          // fill in the diagonal with additive effects, and off-diagonal with the interactions
          if (k == i) {
            matrix(i, i) += diag[k];
          } else {
            matrix(i, k) = (on[k] - off[k]) / 2.0f;
            matrix(i, i) -= matrix(i, k);
          }
        }
      }
    }
  }
}
}  // namespace cpu_impl
}  // namespace xgboost::interpretability
