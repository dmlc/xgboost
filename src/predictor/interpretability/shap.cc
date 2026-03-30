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

struct PathElement {
  int feature_index;
  float zero_fraction;
  float one_fraction;
  float pweight;
  PathElement() = default;
  PathElement(int i, float z, float o, float w)
      : feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}
};

void ExtendPath(PathElement *unique_path, std::uint32_t unique_depth, float zero_fraction,
                float one_fraction, int feature_index) {
  unique_path[unique_depth].feature_index = feature_index;
  unique_path[unique_depth].zero_fraction = zero_fraction;
  unique_path[unique_depth].one_fraction = one_fraction;
  unique_path[unique_depth].pweight = (unique_depth == 0 ? 1.0f : 0.0f);
  for (int i = static_cast<int>(unique_depth) - 1; i >= 0; --i) {
    unique_path[i + 1].pweight +=
        one_fraction * unique_path[i].pweight * (i + 1) / static_cast<float>(unique_depth + 1);
    unique_path[i].pweight = zero_fraction * unique_path[i].pweight * (unique_depth - i) /
                             static_cast<float>(unique_depth + 1);
  }
}

void UnwindPath(PathElement *unique_path, std::uint32_t unique_depth, std::uint32_t path_index) {
  auto const one_fraction = unique_path[path_index].one_fraction;
  auto const zero_fraction = unique_path[path_index].zero_fraction;
  float next_one_portion = unique_path[unique_depth].pweight;

  for (int i = static_cast<int>(unique_depth) - 1; i >= 0; --i) {
    if (one_fraction != 0.0f) {
      auto const tmp = unique_path[i].pweight;
      unique_path[i].pweight =
          next_one_portion * (unique_depth + 1) / static_cast<float>((i + 1) * one_fraction);
      next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (unique_depth - i) /
                                   static_cast<float>(unique_depth + 1);
    } else {
      unique_path[i].pweight = unique_path[i].pweight * (unique_depth + 1) /
                               static_cast<float>(zero_fraction * (unique_depth - i));
    }
  }

  for (auto i = path_index; i < unique_depth; ++i) {
    unique_path[i].feature_index = unique_path[i + 1].feature_index;
    unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction;
    unique_path[i].one_fraction = unique_path[i + 1].one_fraction;
  }
}

float UnwoundPathSum(PathElement const *unique_path, std::uint32_t unique_depth,
                     std::uint32_t path_index) {
  auto const one_fraction = unique_path[path_index].one_fraction;
  auto const zero_fraction = unique_path[path_index].zero_fraction;
  float next_one_portion = unique_path[unique_depth].pweight;
  float total = 0.0f;
  for (int i = static_cast<int>(unique_depth) - 1; i >= 0; --i) {
    if (one_fraction != 0.0f) {
      auto const tmp =
          next_one_portion * (unique_depth + 1) / static_cast<float>((i + 1) * one_fraction);
      total += tmp;
      next_one_portion =
          unique_path[i].pweight -
          tmp * zero_fraction * ((unique_depth - i) / static_cast<float>(unique_depth + 1));
    } else if (zero_fraction != 0.0f) {
      total += (unique_path[i].pweight / zero_fraction) /
               ((unique_depth - i) / static_cast<float>(unique_depth + 1));
    } else {
      CHECK_EQ(unique_path[i].pweight, 0.0f) << "Unique path " << i << " must have zero weight";
    }
  }
  return total;
}

void TreeShap(tree::ScalarTreeView const &tree, RegTree::FVec const &feat, float *phi,
              bst_node_t nidx, std::uint32_t unique_depth, PathElement *parent_unique_path,
              float parent_zero_fraction, float parent_one_fraction, int parent_feature_index,
              int condition, std::uint32_t condition_feature, float condition_fraction) {
  if (condition_fraction == 0.0f) {
    return;
  }

  PathElement *unique_path = parent_unique_path + unique_depth + 1;
  std::copy(parent_unique_path, parent_unique_path + unique_depth + 1, unique_path);
  if (condition == 0 || condition_feature != static_cast<std::uint32_t>(parent_feature_index)) {
    ExtendPath(unique_path, unique_depth, parent_zero_fraction, parent_one_fraction,
               parent_feature_index);
  }

  auto const split_index = tree.SplitIndex(nidx);
  if (tree.IsLeaf(nidx)) {
    for (std::uint32_t i = 1; i <= unique_depth; ++i) {
      auto const w = UnwoundPathSum(unique_path, unique_depth, i);
      auto const &el = unique_path[i];
      phi[el.feature_index] +=
          w * (el.one_fraction - el.zero_fraction) * tree.LeafValue(nidx) * condition_fraction;
    }
    return;
  }

  auto const &cats = tree.GetCategoriesMatrix();
  auto hot_index = predictor::GetNextNode<true, true>(tree, nidx, feat.GetFvalue(split_index),
                                                      feat.IsMissing(split_index), cats);
  auto const cold_index =
      (hot_index == tree.LeftChild(nidx) ? tree.RightChild(nidx) : tree.LeftChild(nidx));
  auto const w = tree.Stat(nidx).sum_hess;
  auto const hot_zero_fraction = tree.Stat(hot_index).sum_hess / w;
  auto const cold_zero_fraction = tree.Stat(cold_index).sum_hess / w;
  float incoming_zero_fraction = 1.0f;
  float incoming_one_fraction = 1.0f;

  std::uint32_t path_index = 0;
  for (; path_index <= unique_depth; ++path_index) {
    if (static_cast<std::uint32_t>(unique_path[path_index].feature_index) == split_index) {
      break;
    }
  }
  if (path_index != unique_depth + 1) {
    incoming_zero_fraction = unique_path[path_index].zero_fraction;
    incoming_one_fraction = unique_path[path_index].one_fraction;
    UnwindPath(unique_path, unique_depth, path_index);
    unique_depth -= 1;
  }

  float hot_condition_fraction = condition_fraction;
  float cold_condition_fraction = condition_fraction;
  if (condition > 0 && split_index == condition_feature) {
    cold_condition_fraction = 0.0f;
    unique_depth -= 1;
  } else if (condition < 0 && split_index == condition_feature) {
    hot_condition_fraction *= hot_zero_fraction;
    cold_condition_fraction *= cold_zero_fraction;
    unique_depth -= 1;
  }

  TreeShap(tree, feat, phi, hot_index, unique_depth + 1, unique_path,
           hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction, split_index,
           condition, condition_feature, hot_condition_fraction);
  TreeShap(tree, feat, phi, cold_index, unique_depth + 1, unique_path,
           cold_zero_fraction * incoming_zero_fraction, 0.0f, split_index, condition,
           condition_feature, cold_condition_fraction);
}

void CalculateContributions(tree::ScalarTreeView const &tree, RegTree::FVec const &feat,
                            std::vector<float> *mean_values, float *out_contribs, int condition,
                            std::uint32_t condition_feature) {
  if (condition == 0) {
    out_contribs[feat.Size()] += (*mean_values)[RegTree::kRoot];
  }

  auto const maxd = tree.MaxDepth() + 2;
  std::vector<PathElement> unique_path_data((maxd * (maxd + 1)) / 2);
  TreeShap(tree, feat, out_contribs, RegTree::kRoot, 0, unique_path_data.data(), 1.0f, 1.0f, -1,
           condition, condition_feature, 1.0f);
}

// Keep the CPU quadrature recurrence on the same fixed 8-point rule as the GPU path so the hot
// loops stay small and the compiler can fully unroll the basis update and extraction work.
constexpr std::size_t kQuadratureShapPoints = 8;
constexpr double kQuadratureShapBuildQeps = 1e-15;
constexpr float kQuadratureShapUnseen = -999.0f;

struct QuadratureRule {
  std::array<float, kQuadratureShapPoints> nodes{};
  std::array<float, kQuadratureShapPoints> weights{};
};
using QuadratureBuffer = std::array<float, kQuadratureShapPoints>;

QuadratureRule const &GetQuadratureRule() {
  static QuadratureRule const rule = [] {
    auto const rule_d =
        detail::MakeEndpointQuadrature<kQuadratureShapPoints>(kQuadratureShapBuildQeps);
    QuadratureRule out;
    for (std::size_t i = 0; i < kQuadratureShapPoints; ++i) {
      out.nodes[i] = static_cast<float>(rule_d.nodes[i]);
      out.weights[i] = static_cast<float>(rule_d.weights[i]);
    }
    return out;
  }();
  return rule;
}

void ScaleInPlace(QuadratureBuffer *h_vals, float scale) {
  for (auto &v : *h_vals) {
    v *= scale;
  }
}

void AddInPlace(QuadratureBuffer *lhs, QuadratureBuffer const &rhs) {
  for (std::size_t i = 0; i < kQuadratureShapPoints; ++i) {
    (*lhs)[i] += rhs[i];
  }
}

float ExtractQuadratureDelta(QuadratureRule const &rule, QuadratureBuffer const &h_vals,
                             float p_enter, float p_exit) {
  float acc = 0.0f;
  if (p_enter != 1.0f) {
    auto const alpha_enter = p_enter - 1.0f;
    for (std::size_t i = 0; i < kQuadratureShapPoints; ++i) {
      acc += alpha_enter * h_vals[i] / (1.0f + alpha_enter * rule.nodes[i]);
    }
  }
  if (p_exit != 1.0f) {
    auto const alpha_exit = p_exit - 1.0f;
    for (std::size_t i = 0; i < kQuadratureShapPoints; ++i) {
      acc -= alpha_exit * h_vals[i] / (1.0f + alpha_exit * rule.nodes[i]);
    }
  }
  return acc;
}

struct QuadratureShapTreeRunner {
  tree::ScalarTreeView const &tree;
  RegTree::FVec const &feat;
  QuadratureRule const &rule;
  std::vector<float> *path_prob;
  float *phi;

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

  void HandleLeaf(bst_node_t nidx, QuadratureBuffer const &c_vals, float w_prod,
                  QuadratureBuffer *out_h) const {
    auto const leaf_scale = w_prod * tree.LeafValue(nidx);
    for (std::size_t i = 0; i < kQuadratureShapPoints; ++i) {
      (*out_h)[i] = c_vals[i] * leaf_scale * rule.weights[i];
    }
  }

  void ExtractContribution(bst_feature_t split_index, QuadratureBuffer const &h_vals, float p_enter,
                           float p_exit) {
    phi[split_index] += ExtractQuadratureDelta(rule, h_vals, p_enter, p_exit);
  }

  void VisitChild(bst_node_t split_node, bst_node_t child_node, float child_weight, bool satisfies,
                  QuadratureBuffer const &c_vals, float w_prod, QuadratureBuffer *out_h) {
    auto split_index = tree.SplitIndex(split_node);
    auto p_old = (*path_prob)[split_index];

    float p_e = 0.0f;
    float p_up = 0.0f;
    if (p_old == kQuadratureShapUnseen) {
      p_e = satisfies ? 1.0f / child_weight : 0.0f;
      p_up = 1.0f;
    } else if (p_old == 0.0f) {
      p_e = 0.0f;
      p_up = 0.0f;
    } else {
      p_e = satisfies ? p_old / child_weight : 0.0f;
      p_up = p_old;
    }

    auto c_child = c_vals;
    auto alpha_e = p_e - 1.0f;
    for (std::size_t i = 0; i < kQuadratureShapPoints; ++i) {
      c_child[i] *= 1.0f + alpha_e * rule.nodes[i];
    }

    if (p_old != kQuadratureShapUnseen) {
      auto alpha_old = p_old - 1.0f;
      if (alpha_old != 0.0f) {
        for (std::size_t i = 0; i < kQuadratureShapPoints; ++i) {
          c_child[i] /= 1.0f + alpha_old * rule.nodes[i];
        }
      }
    }

    (*path_prob)[split_index] = p_e;
    this->RunNode(child_node, c_child, w_prod * child_weight, out_h);
    (*path_prob)[split_index] = p_old;
    this->ExtractContribution(split_index, *out_h, p_e, p_up);
  }

  void RunNode(bst_node_t nidx, QuadratureBuffer const &c_vals, float w_prod,
               QuadratureBuffer *out_h) {
    if (tree.IsLeaf(nidx)) {
      this->HandleLeaf(nidx, c_vals, w_prod, out_h);
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
    if (tree.IsLeaf(RegTree::kRoot)) {
      return;
    }

    QuadratureBuffer c_init{};
    c_init.fill(1.0f);
    QuadratureBuffer h_vals{};
    this->RunNode(RegTree::kRoot, c_init, 1.0f, &h_vals);
  }
};

struct QuadratureShapModelData {
  std::vector<tree::ScalarTreeView> trees;
  std::vector<std::vector<bst_tree_t>> trees_by_group;
  std::vector<float> weights;
  std::vector<float> group_root_mean_sums;
};

QuadratureShapModelData MakeQuadratureShapModelData(gbm::GBTreeModel const &model,
                                                    bst_tree_t tree_end,
                                                    std::vector<float> const *tree_weights) {
  auto const n_trees = static_cast<std::size_t>(tree_end);
  auto const h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  auto const n_groups = model.learner_model_param->num_output_group;

  QuadratureShapModelData out;
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
    out.group_root_mean_sums[gid] +=
        static_cast<float>(detail::FillRootMeanValue(out.trees[i], RegTree::kRoot) * weight);
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
void ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                gbm::GBTreeModel const &model, bst_tree_t tree_end,
                std::vector<float> const *tree_weights, int condition, unsigned condition_feature) {
  CHECK(!model.learner_model_param->IsVectorLeaf()) << "Predict contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  MetaInfo const &info = p_fmat->Info();
  // number of valid trees
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  CHECK_GE(tree_end, 0);
  ValidateTreeWeights(tree_weights, tree_end);
  auto const n_trees = static_cast<std::size_t>(tree_end);
  auto const n_threads = ctx->Threads();
  size_t const ncolumns = model.learner_model_param->num_feature + 1;
  // allocate space for (number of features + bias) times the number of rows
  std::vector<bst_float> &contribs = out_contribs->HostVector();
  contribs.resize(info.num_row_ * ncolumns * model.learner_model_param->num_output_group);
  // make sure contributions is zeroed, we could be reusing a previously allocated one
  std::fill(contribs.begin(), contribs.end(), 0);
  // initialize tree node mean values
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
          CalculateContributions(sc_tree, feats, &mean_values[j], this_tree_contribs.data(),
                                 condition, condition_feature);
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

void QuadratureShapValues(Context const *ctx, DMatrix *p_fmat,
                          HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                          bst_tree_t tree_end, std::vector<float> const *tree_weights,
                          std::size_t quadrature_points) {
  CHECK(!model.learner_model_param->IsVectorLeaf()) << "Predict contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  CHECK_EQ(quadrature_points, kQuadratureShapPoints)
      << "CPU QuadratureSHAP currently uses a fixed quadrature size of " << kQuadratureShapPoints
      << ".";
  MetaInfo const &info = p_fmat->Info();
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  CHECK_GE(tree_end, 0);
  ValidateTreeWeights(tree_weights, tree_end);
  auto const n_trees = static_cast<std::size_t>(tree_end);
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
  auto model_data = MakeQuadratureShapModelData(model, tree_end, tree_weights);
  std::vector<RegTree::FVec> feats_tloc(n_threads);
  std::vector<std::vector<float>> contribs_tloc(n_threads, std::vector<float>(ncolumns));
  std::vector<std::vector<float>> path_prob_tloc(
      n_threads, std::vector<float>(n_features, kQuadratureShapUnseen));

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
          auto runner = QuadratureShapTreeRunner{model_data.trees[j], feats, rule, &path_prob,
                                                 this_tree_contribs.data()};
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
  CHECK(!model.learner_model_param->IsVectorLeaf())
      << "Predict interaction contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit()) << "Predict interaction contribution support for "
                                            "column-wise data split is not yet implemented.";
  MetaInfo const &info = p_fmat->Info();
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
  if (approximate) {
    ApproxFeatureImportance(ctx, p_fmat, &contribs_diag_hdv, model, tree_end, tree_weights);
  } else {
    ShapValues(ctx, p_fmat, &contribs_diag_hdv, model, tree_end, tree_weights, 0, 0);
  }
  for (size_t i = 0; i < ncolumns + 1; ++i) {
    if (approximate) {
      ApproxFeatureImportance(ctx, p_fmat, &contribs_off_hdv, model, tree_end, tree_weights);
      ApproxFeatureImportance(ctx, p_fmat, &contribs_on_hdv, model, tree_end, tree_weights);
    } else {
      ShapValues(ctx, p_fmat, &contribs_off_hdv, model, tree_end, tree_weights, -1, i);
      ShapValues(ctx, p_fmat, &contribs_on_hdv, model, tree_end, tree_weights, 1, i);
    }

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
            contribs[o_offset + k] = (contribs_on[c_offset + k] - contribs_off[c_offset + k]) / 2.0;
            contribs[o_offset + i] -= contribs[o_offset + k];
          }
        }
      }
    }
  }
}
}  // namespace cpu_impl
}  // namespace xgboost::interpretability
