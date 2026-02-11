/**
 * Copyright 2017-2026, XGBoost Contributors
 */
#include "shap.h"

#include <algorithm>    // for fill
#include <limits>       // for numeric_limits
#include <type_traits>  // for remove_const_t
#include <vector>       // for vector

#include "../common/categorical.h"            // for IsCat
#include "../common/column_matrix.h"          // for ColumnMatrix
#include "../common/hist_util.h"              // for DispatchBinType, HistogramCuts
#include "../common/math.h"                   // for CheckNAN
#include "../common/threading_utils.h"        // for ParallelFor
#include "../data/gradient_index.h"           // for GHistIndexMatrix
#include "../gbm/gbtree_model.h"              // for GBTreeModel
#include "../predictor/predict_fn.h"          // for GetTreeLimit
#include "../predictor/treeshap.h"            // for CalculateContributions
#include "../tree/tree_view.h"                // for ScalarTreeView
#include "dmlc/omp.h"                         // for omp_get_thread_num
#include "xgboost/base.h"                     // for bst_omp_uint
#include "xgboost/logging.h"                  // for CHECK
#include "xgboost/multi_target_tree_model.h"  // for MTNotImplemented

namespace xgboost::interpretability {
namespace {
void ValidateTreeWeights(common::Span<float const> tree_weights, bst_tree_t tree_end) {
  if (tree_weights.empty()) {
    return;
  }
  CHECK_GE(tree_weights.size(), static_cast<std::size_t>(tree_end));
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

class GHistIndexMatrixView {
 private:
  GHistIndexMatrix const &page_;
  common::Span<FeatureType const> ft_;

  std::vector<std::uint32_t> const &ptrs_;
  std::vector<float> const &mins_;
  std::vector<float> const &values_;
  common::ColumnMatrix const &columns_;

 public:
  bst_idx_t const base_rowid;

 public:
  GHistIndexMatrixView(GHistIndexMatrix const &page, common::Span<FeatureType const> ft)
      : page_{page},
        ft_{ft},
        ptrs_{page.cut.Ptrs()},
        mins_{page.cut.MinValues()},
        values_{page.cut.Values()},
        columns_{page.Transpose()},
        base_rowid{page.base_rowid} {}

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
          out[fidx] = fvalue;
        }
      });
      n_non_missings += n_features;
    } else {
      for (bst_feature_t fidx = 0; fidx < n_features; ++fidx) {
        float fvalue = std::numeric_limits<float>::quiet_NaN();
        bool is_cat = common::IsCat(ft_, fidx);
        if (columns_.GetColumnType(fidx) == common::kSparseColumn) {
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
          out[fidx] = fvalue;
          n_non_missings++;
        }
      }
    }
    return n_non_missings;
  }

  [[nodiscard]] bst_idx_t Size() const { return page_.Size(); }
};
}  // namespace

namespace cpu_impl {
void ShapValues(Context const *ctx, DMatrix *p_fmat, HostDeviceVector<float> *out_contribs,
                gbm::GBTreeModel const &model, bst_tree_t tree_end,
                common::Span<float const> tree_weights, int condition, unsigned condition_feature) {
  CHECK(!model.learner_model_param->IsVectorLeaf()) << "Predict contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  MetaInfo const &info = p_fmat->Info();
  // number of valid trees
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  ValidateTreeWeights(tree_weights, tree_end);
  size_t const ncolumns = model.learner_model_param->num_feature + 1;
  // allocate space for (number of features + bias) times the number of rows
  std::vector<bst_float> &contribs = out_contribs->HostVector();
  contribs.resize(info.num_row_ * ncolumns * model.learner_model_param->num_output_group);
  // make sure contributions is zeroed, we could be reusing a previously allocated one
  std::fill(contribs.begin(), contribs.end(), 0);
  // initialize tree node mean values
  std::vector<std::vector<float>> mean_values(tree_end);
  for (bst_omp_uint i = 0; i < tree_end; ++i) {
    FillNodeMeanValues(model.trees[i]->HostScView(), &(mean_values[i]));
  }

  auto const n_groups = model.learner_model_param->num_output_group;
  CHECK_NE(n_groups, 0);
  auto const base_score = model.learner_model_param->BaseScore(DeviceOrd::CPU());
  auto const h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  auto const n_threads = ctx->Threads();
  std::vector<RegTree::FVec> feats_tloc(n_threads);
  std::vector<std::vector<bst_float>> contribs_tloc(n_threads, std::vector<bst_float>(ncolumns));

  auto device = ctx->Device().IsSycl() ? DeviceOrd::CPU() : ctx->Device();
  auto base_margin = info.base_margin_.View(device);

  if (p_fmat->PageExists<SparsePage>()) {
    for (auto const &page : p_fmat->GetBatches<SparsePage>()) {
      auto view = page.GetView();
      common::ParallelFor(view.Size(), n_threads, [&](auto i) {
        auto tid = omp_get_thread_num();
        auto &feats = feats_tloc[tid];
        if (feats.Size() == 0) {
          feats.Init(model.learner_model_param->num_feature);
        }
        auto &this_tree_contribs = contribs_tloc[tid];
        auto row_idx = page.base_rowid + i;
        feats.Fill(view[i]);
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
                  this_tree_contribs[ci] * (tree_weights.empty() ? 1 : tree_weights[j]);
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
    }
  } else {
    auto ft = p_fmat->Info().feature_types.ConstHostVector();
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx, {})) {
      GHistIndexMatrixView view{page, ft};
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
                  this_tree_contribs[ci] * (tree_weights.empty() ? 1 : tree_weights[j]);
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
    }
  }
}

void ApproxFeatureImportance(Context const *ctx, DMatrix *p_fmat,
                             HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                             bst_tree_t tree_end, common::Span<float const> tree_weights) {
  CHECK(!model.learner_model_param->IsVectorLeaf()) << "Predict contribution" << MTNotImplemented();
  CHECK(!p_fmat->Info().IsColumnSplit())
      << "Predict contribution support for column-wise data split is not yet implemented.";
  MetaInfo const &info = p_fmat->Info();
  tree_end = predictor::GetTreeLimit(model.trees, tree_end);
  ValidateTreeWeights(tree_weights, tree_end);
  size_t const ncolumns = model.learner_model_param->num_feature + 1;
  std::vector<bst_float> &contribs = out_contribs->HostVector();
  contribs.resize(info.num_row_ * ncolumns * model.learner_model_param->num_output_group);
  std::fill(contribs.begin(), contribs.end(), 0);
  std::vector<std::vector<float>> mean_values(tree_end);
  for (bst_omp_uint i = 0; i < tree_end; ++i) {
    FillNodeMeanValues(model.trees[i]->HostScView(), &(mean_values[i]));
  }

  auto const n_groups = model.learner_model_param->num_output_group;
  CHECK_NE(n_groups, 0);
  auto const base_score = model.learner_model_param->BaseScore(DeviceOrd::CPU());
  auto const h_tree_groups = model.TreeGroups(DeviceOrd::CPU());
  auto const n_threads = ctx->Threads();
  std::vector<RegTree::FVec> feats_tloc(n_threads);
  std::vector<std::vector<bst_float>> contribs_tloc(n_threads, std::vector<bst_float>(ncolumns));

  auto device = ctx->Device().IsSycl() ? DeviceOrd::CPU() : ctx->Device();
  auto base_margin = info.base_margin_.View(device);

  if (p_fmat->PageExists<SparsePage>()) {
    for (auto const &page : p_fmat->GetBatches<SparsePage>()) {
      auto view = page.GetView();
      common::ParallelFor(view.Size(), n_threads, [&](auto i) {
        auto tid = omp_get_thread_num();
        auto &feats = feats_tloc[tid];
        if (feats.Size() == 0) {
          feats.Init(model.learner_model_param->num_feature);
        }
        auto &this_tree_contribs = contribs_tloc[tid];
        auto row_idx = page.base_rowid + i;
        feats.Fill(view[i]);
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
                  this_tree_contribs[ci] * (tree_weights.empty() ? 1 : tree_weights[j]);
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
    }
  } else {
    auto ft = p_fmat->Info().feature_types.ConstHostVector();
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(ctx, {})) {
      GHistIndexMatrixView view{page, ft};
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
                  this_tree_contribs[ci] * (tree_weights.empty() ? 1 : tree_weights[j]);
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
    }
  }
}

void ShapInteractionValues(Context const *ctx, DMatrix *p_fmat,
                           HostDeviceVector<float> *out_contribs, gbm::GBTreeModel const &model,
                           bst_tree_t tree_end, common::Span<float const> tree_weights,
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
