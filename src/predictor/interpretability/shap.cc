/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include "shap.h"

#include <algorithm>    // for fill
#include <array>        // for array
#include <cmath>        // for abs
#include <limits>       // for numeric_limits
#include <type_traits>  // for remove_const_t
#include <vector>       // for vector

#include "../../common/threading_utils.h"  // for ParallelFor
#include "../../gbm/gbtree_model.h"        // for GBTreeModel
#include "../../tree/tree_view.h"          // for ScalarTreeView
#include "../data_accessor.h"              // for GHistIndexMatrixView
#include "../predict_fn.h"                 // for GetTreeLimit
#include "../treeshap.h"                   // for CalculateContributions
#include "dmlc/omp.h"                      // for omp_get_thread_num
#include "xgboost/base.h"                  // for bst_omp_uint
#include "xgboost/logging.h"               // for CHECK
#include "xgboost/tree_model.h"            // for MTNotImplemented

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
  CalculateContributionsApprox(tree, feats, mean_values, out_contribs->data());
}

constexpr std::size_t kV6QuadraturePoints = 30;
constexpr double kV6Qeps = 1e-15;
constexpr double kV6Unseen = -999.0;
using V6Quad = std::array<double, kV6QuadraturePoints>;

V6Quad const &V6Nodes() {
  static constexpr V6Quad kNodes = {
      1.55325796267524740557e-03, 8.16593836012641238753e-03, 1.99890675158462260974e-02,
      3.68999762853628454629e-02, 5.87197321039736319648e-02, 8.52171188086158215569e-02,
      1.16111283947586907406e-01, 1.51074752603342077339e-01, 1.89736908505378554235e-01,
      2.31687925928990068325e-01, 2.76483115230955422970e-01, 3.23647637234560914266e-01,
      3.72681536916055100583e-01, 4.23065043195708256896e-01, 4.74264078722341164696e-01,
      5.25735921277658890816e-01, 5.76934956804291743104e-01, 6.27318463083944899417e-01,
      6.76352362765439085734e-01, 7.23516884769044521519e-01, 7.68312074071009876164e-01,
      8.10263091494621390254e-01, 8.48925247396657978172e-01, 8.83888716052413148105e-01,
      9.14782881191384178443e-01, 9.41280267896026368035e-01, 9.63100023714637210048e-01,
      9.80010932484153718391e-01, 9.91834061639873532101e-01, 9.98446742037324752594e-01};
  return kNodes;
}

V6Quad const &V6Weights() {
  static constexpr V6Quad kWeights = {
      3.98409624808451681005e-03, 9.23323415554584518705e-03, 1.43923539416612698838e-02,
      1.93995962848134140266e-02, 2.42013364152968944720e-02, 2.87465781088096158924e-02,
      3.29871149410902175791e-02, 3.68779873688524495456e-02, 4.03779476147099816719e-02,
      4.34498936005414254646e-02, 4.60612611188929710337e-02, 4.81843685873219601534e-02,
      4.97967102933974670176e-02, 5.08811948742026384784e-02, 5.14263264467792954870e-02,
      5.14263264467792954870e-02, 5.08811948742026384784e-02, 4.97967102933974670176e-02,
      4.81843685873219601534e-02, 4.60612611188929710337e-02, 4.34498936005414254646e-02,
      4.03779476147099816719e-02, 3.68779873688524495456e-02, 3.29871149410902175791e-02,
      2.87465781088096158924e-02, 2.42013364152968944720e-02, 1.93995962848134140266e-02,
      1.43923539416612698838e-02, 9.23323415554584518705e-03, 3.98409624808451681005e-03};
  return kWeights;
}

V6Quad Scale(V6Quad h_vals, double scale) {
  for (auto &v : h_vals) {
    v *= scale;
  }
  return h_vals;
}

V6Quad Add(V6Quad lhs, V6Quad const &rhs) {
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    lhs[i] += rhs[i];
  }
  return lhs;
}

double ExtractTermV6(V6Quad const &h_vals, double p_value) {
  if (p_value == kV6Unseen) {
    return 0.0;
  }
  auto alpha = p_value - 1.0;
  if (std::abs(alpha) < kV6Qeps) {
    return 0.0;
  }
  auto const &nodes = V6Nodes();
  auto const &weights = V6Weights();
  double acc = 0.0;
  for (std::size_t i = 0; i < h_vals.size(); ++i) {
    acc += alpha * h_vals[i] / (1.0 + alpha * nodes[i]) * weights[i];
  }
  return acc;
}

bool GoesLeftV6(tree::ScalarTreeView const &tree, RegTree::FVec const &feat, bst_node_t nidx) {
  auto split_index = tree.SplitIndex(nidx);
  auto fvalue = feat.GetFvalue(split_index);
  auto missing = feat.IsMissing(split_index);
  auto const &cats = tree.GetCategoriesMatrix();
  bst_node_t next = RegTree::kInvalidNodeId;
  if (tree.HasCategoricalSplit()) {
    next = missing ? predictor::GetNextNode<true, true>(tree, nidx, fvalue, true, cats)
                   : predictor::GetNextNode<false, true>(tree, nidx, fvalue, false, cats);
  } else {
    next = missing ? predictor::GetNextNode<true, false>(tree, nidx, fvalue, true, cats)
                   : predictor::GetNextNode<false, false>(tree, nidx, fvalue, false, cats);
  }
  return next == tree.LeftChild(nidx);
}

double ChildWeightV6(tree::ScalarTreeView const &tree, bst_node_t parent, bst_node_t child) {
  auto parent_cover = tree.Stat(parent).sum_hess;
  CHECK_GT(parent_cover, 0.0f);
  return tree.Stat(child).sum_hess / parent_cover;
}

V6Quad TreeShapV6(tree::ScalarTreeView const &tree, RegTree::FVec const &feat, bst_node_t nidx,
                  V6Quad const &c_vals, double w_prod, std::vector<double> *p_vals, double *phi) {
  if (tree.IsLeaf(nidx)) {
    return Scale(c_vals, w_prod * tree.LeafValue(nidx));
  }

  auto split_index = tree.SplitIndex(nidx);
  auto left = tree.LeftChild(nidx);
  auto right = tree.RightChild(nidx);
  auto left_weight = ChildWeightV6(tree, nidx, left);
  auto right_weight = ChildWeightV6(tree, nidx, right);
  auto goes_left = GoesLeftV6(tree, feat, nidx);
  auto p_old = (*p_vals)[split_index];

  auto visit_child = [&](bst_node_t child, double child_weight, bool satisfies) {
    double p_e = 0.0;
    double p_up = 0.0;
    if (p_old == kV6Unseen) {
      p_e = satisfies ? 1.0 / child_weight : 0.0;
      p_up = 1.0;
    } else if (std::abs(p_old) < kV6Qeps) {
      p_e = 0.0;
      p_up = 0.0;
    } else {
      p_e = satisfies ? p_old / child_weight : 0.0;
      p_up = p_old;
    }

    auto c_child = c_vals;
    auto const &nodes = V6Nodes();
    auto alpha_e = p_e - 1.0;
    for (std::size_t i = 0; i < c_child.size(); ++i) {
      c_child[i] *= 1.0 + alpha_e * nodes[i];
    }

    if (p_old != kV6Unseen) {
      auto alpha_old = p_old - 1.0;
      if (std::abs(alpha_old) >= kV6Qeps) {
        for (std::size_t i = 0; i < c_child.size(); ++i) {
          c_child[i] /= 1.0 + alpha_old * nodes[i];
        }
      }
    }

    (*p_vals)[split_index] = p_e;
    auto h_child = TreeShapV6(tree, feat, child, c_child, w_prod * child_weight, p_vals, phi);
    (*p_vals)[split_index] = p_old;
    phi[split_index] += ExtractTermV6(h_child, p_e);
    phi[split_index] -= ExtractTermV6(h_child, p_up);
    return h_child;
  };

  auto left_h = visit_child(left, left_weight, goes_left);
  auto right_h = visit_child(right, right_weight, !goes_left);
  return Add(std::move(left_h), right_h);
}

void CalculateContributionsV6(tree::ScalarTreeView const &tree, RegTree::FVec const &feat,
                              std::vector<float> *mean_values, double *out_contribs) {
  out_contribs[feat.Size()] += (*mean_values)[0];

  if (tree.IsLeaf(RegTree::kRoot)) {
    return;
  }

  V6Quad c_init;
  c_init.fill(1.0);
  std::vector<double> p_vals(feat.Size(), kV6Unseen);
  TreeShapV6(tree, feat, RegTree::kRoot, c_init, 1.0, &p_vals, out_contribs);
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

void V6ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                  gbm::GBTreeModel const &model, bst_tree_t tree_end,
                  std::vector<float> const *tree_weights) {
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
  std::fill(contribs.begin(), contribs.end(), 0.0f);

  std::vector<std::vector<float>> mean_values(n_trees);
  common::ParallelFor(n_trees, n_threads, [&](auto i) {
    FillNodeMeanValues(model.trees[i]->HostScView(), &(mean_values[i]));
  });

  auto const n_groups = model.learner_model_param->num_output_group;
  CHECK_NE(n_groups, 0);
  auto const base_score = model.learner_model_param->BaseScore(DeviceOrd::CPU());
  auto const h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  std::vector<RegTree::FVec> feats_tloc(n_threads);
  std::vector<std::vector<double>> contribs_tloc(n_threads, std::vector<double>(ncolumns));

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
          std::fill(this_tree_contribs.begin(), this_tree_contribs.end(), 0.0);
          auto const sc_tree = model.trees[j]->HostScView();
          CalculateContributionsV6(sc_tree, feats, &mean_values[j], this_tree_contribs.data());
          auto const weight = tree_weights == nullptr ? 1.0f : (*tree_weights)[j];
          for (size_t ci = 0; ci < ncolumns; ++ci) {
            p_contribs[ci] += static_cast<float>(this_tree_contribs[ci] * weight);
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
