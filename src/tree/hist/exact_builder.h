/**
 * Copyright 2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EXACT_BUILDER_H_
#define XGBOOST_TREE_HIST_EXACT_BUILDER_H_

#include <algorithm>  // for fill
#include <memory>     // for unique_ptr
#include <numeric>    // for iota
#include <vector>     // for vector

#include "../../common/exact_multinomial/leaf_solver.h"   // for ExactMultinomialLeafSolver
#include "../../common/exact_multinomial/packed_stats.h"  // for PackedMultinomialStats
#include "../../collective/allreduce.h"                    // for Allreduce
#include "../../collective/communicator-inl.h"             // for IsDistributed
#include "../../common/categorical.h"                      // for IsCat
#include "../../common/hist_util.h"                       // for HistogramCuts
#include "../../common/random.h"                          // for ColumnSampler
#include "../../common/timer.h"                           // for Monitor
#include "../../data/gradient_index.h"                    // for GHistIndexMatrix
#include "../common_row_partitioner.h"                    // for CommonRowPartitioner
#include "../constraints.h"                               // for FeatureInteractionConstraintHost
#include "../param.h"                                     // for TrainParam
#include "exact_evaluator.h"                              // for CheckExactTrainParam
#include "exact_histogram.h"                              // for ExactHistCollection
#include "exact_split.h"                                  // for EnumerateExactNode
#include "expand_entry.h"                                 // for MultiExpandEntry
#include "hist_param.h"                                   // for HistMakerTrainParam
#include "xgboost/base.h"                                 // for bst_node_t
#include "xgboost/data.h"                                 // for DMatrix, BatchParam
#include "xgboost/gradient.h"                             // for ExactHessian
#include "xgboost/tree_model.h"                           // for RegTree

namespace xgboost::tree {
/** @brief Defined in updater_quantile_hist.cc; shared so both builders bin identically. */
BatchParam HistBatch(TrainParam const* param);

/**
 * @brief Map a `K-1` dimensional Newton step onto the model's `K` outputs.
 *
 * The solve lives in the free coordinates with the last class pinned to zero, so the raw
 * embedding is `[w; 0]`. Softmax is invariant to a constant shift, so that vector is only
 * one representative of a whole line of equivalent outputs. Centering picks the
 * minimum-norm representative, which is the one whose L2 penalty matches
 * `R = lambda (I - 11^T/K)` -- the regularizer the solve actually used. Writing any other
 * representative would penalise the leaf differently from how it was computed.
 *
 * The result sums to zero, so repeated rounds cannot accumulate a drift along the gauge
 * direction.
 */
inline void CenteredLeafWeight(common::Span<double const> free_weight,
                               common::Span<float> out_weight) {
  auto n_classes = out_weight.size();
  CHECK_EQ(free_weight.size() + 1, n_classes);
  double mean = 0.0;
  for (auto v : free_weight) {
    mean += v;
  }
  // The reference class contributes 0 to the sum but still counts in the average.
  mean /= static_cast<double>(n_classes);
  for (std::size_t i = 0; i < free_weight.size(); ++i) {
    out_weight[i] = static_cast<float>(free_weight[i] - mean);
  }
  out_weight[n_classes - 1] = static_cast<float>(-mean);
}

/**
 * @brief Tree builder for exact multinomial training.
 *
 * Mirrors @ref MultiTargetHistBuilder's interface so that the generic `UpdateTree` driver
 * and the existing row partitioner, quantile bins and vector-leaf tree can be reused
 * unchanged. Only the statistics differ: every node carries one packed record holding the
 * `K-1` free gradients and the dense Hessian triangle, instead of `K` independent scalar
 * histograms.
 *
 * The scalar and existing multi-target builders are untouched; this class is selected only
 * when an exact Hessian sidecar is present.
 */
class ExactMultiTargetHistBuilder {
 public:
  ExactMultiTargetHistBuilder(Context const* ctx, TrainParam const* param,
                              HistMakerTrainParam const* hist_param,
                              std::shared_ptr<common::ColumnSampler> column_sampler,
                              common::Monitor* monitor)
      : monitor_{monitor},
        param_{param},
        hist_param_{hist_param},
        col_sampler_{std::move(column_sampler)},
        ctx_{ctx} {
    monitor_->Init(__func__);
  }

  /** @brief The per-row exact statistics for the current boosting round. */
  void SetExactHessian(ExactHessian const* hessian) { hessian_ = hessian; }

  void InitData(DMatrix* p_fmat, RegTree const* p_tree,
                linalg::MatrixView<GradientPair const> gpair) {
    monitor_->Start(__func__);
    CheckExactTrainParam(*param_);
    CHECK(hessian_) << "Exact multinomial training requires the exact Hessian sidecar.";
    CHECK(!hessian_->Empty()) << "The exact Hessian sidecar is empty.";
    CHECK_EQ(hessian_->NumRows(), gpair.Shape(0))
        << "The exact Hessian does not describe the current gradient.";

    n_classes_ = static_cast<bst_target_t>(gpair.Shape(1));
    CHECK_GE(n_classes_, 2);
    n_free_ = static_cast<bst_target_t>(n_classes_ - 1);
    CHECK_EQ(hessian_->n_free, n_free_);

    // Categorical splits need a partition search over category sets, which the exact
    // enumerator does not implement. Rejecting here is deliberate: silently scanning a
    // categorical feature as if it were ordered would produce a wrong split rather than a
    // slow one.
    auto feature_types = p_fmat->Info().feature_types.ConstHostSpan();
    if (!feature_types.empty()) {
      for (bst_feature_t fidx = 0; fidx < p_fmat->Info().num_col_; ++fidx) {
        CHECK(!common::IsCat(feature_types, fidx))
            << "multi_hessian=exact does not support categorical features yet (feature index "
            << fidx
            << " is categorical). Use numerical features, or set multi_hessian=diagonal.";
      }
    }

    p_last_fmat_ = p_fmat;
    bst_bin_t n_total_bins = 0;
    std::size_t page_idx = 0;
    for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }
      if (page_idx < partitioner_.size()) {
        partitioner_[page_idx].Reset(ctx_, page.Size(), page.base_rowid);
      } else {
        partitioner_.emplace_back(ctx_, page.Size(), page.base_rowid);
      }
      ++page_idx;
    }
    partitioner_.resize(page_idx);

    // Interaction constraints restrict which features may appear together on a root-to-leaf
    // path. That is a structural rule about the tree, independent of how curvature is
    // modelled, so the exact path enforces the same graph as the scalar path rather than
    // rejecting it.
    is_distributed_ = collective::IsDistributed();
    interaction_constraints_.Configure(*param_, p_fmat->Info().num_col_);
    hist_.Reset(n_total_bins, n_free_, hist_param_->MaxCachedHistNodes(ctx_->Device()));
    solver_ = std::make_unique<common::ExactMultinomialLeafSolver>(n_free_);
    this->ResizeThreadState(ctx_->Threads());
    node_total_.Reset(n_free_);
    free_weight_.assign(n_free_, 0.0);
    col_sampler_->Init(ctx_, p_fmat->Info().num_col_, p_fmat->Info().feature_weights,
                       param_->colsample_bynode, param_->colsample_bylevel,
                       param_->colsample_bytree);
    p_last_tree_ = p_tree;
    monitor_->Stop(__func__);
  }

  MultiExpandEntry InitRoot(DMatrix* p_fmat, linalg::MatrixView<GradientPair const> gpair,
                            RegTree* p_tree) {
    monitor_->Start(__func__);
    hist_.AllocateHistograms(std::vector<bst_node_t>{RegTree::kRoot});
    this->BuildNodeHist(p_fmat, RegTree::kRoot, gpair);
    this->AllreduceHist(std::vector<bst_node_t>{RegTree::kRoot});

    // Root weight from the joint Newton solve over every row.
    ExactNodeTotal(hist_[RegTree::kRoot], n_free_, hist_.TotalBins(), node_total_.Data());
    std::vector<float> weight(n_classes_, 0.0f);
    this->SolveCentered(node_total_.ConstView(),
                        common::Span<float>{weight.data(), weight.size()});
    auto root_curvature = ExactChildCurvature(node_total_.ConstView(), n_classes_);
    // SetRoot takes the final output, so eta is applied here; Expand applies it to children.
    for (auto& w : weight) {
      w *= param_->learning_rate;
    }
    linalg::Tensor<float, 1> root_weight{weight.cbegin(), weight.cend(), {n_classes_},
                                         DeviceOrd::CPU()};
    p_tree->SetRoot(root_weight.HostView(), static_cast<float>(root_curvature));

    std::vector<MultiExpandEntry> nodes{{RegTree::kRoot, 0}};
    this->EvaluateSplits(p_fmat, &nodes);
    monitor_->Stop(__func__);
    return nodes.front();
  }

  void UpdatePosition(DMatrix* p_fmat, RegTree const* p_tree,
                      std::vector<MultiExpandEntry> const& applied) {
    monitor_->Start(__func__);
    std::size_t page_id = 0;
    for (auto const& page : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      partitioner_.at(page_id).UpdatePosition(ctx_, page, applied, p_tree->HostMtView());
      ++page_id;
    }
    monitor_->Stop(__func__);
  }

  void BuildHistogram(DMatrix* p_fmat, RegTree const* p_tree,
                      std::vector<MultiExpandEntry> const& valid_candidates,
                      linalg::MatrixView<GradientPair const> gpair) {
    monitor_->Start(__func__);
    std::vector<bst_node_t> to_build;
    std::vector<bst_node_t> to_subtract;
    for (auto const& candidate : valid_candidates) {
      to_build.push_back(p_tree->LeftChild(candidate.nid));
      to_subtract.push_back(p_tree->RightChild(candidate.nid));
    }
    hist_.AllocateHistograms(common::Span<bst_node_t const>{to_build},
                             common::Span<bst_node_t const>{to_subtract});

    for (auto const& candidate : valid_candidates) {
      this->BuildNodeHist(p_fmat, p_tree->LeftChild(candidate.nid), gpair);
    }
    // Reduce the locally built children first, then derive each sibling by subtraction. The
    // parent is already global, so `parent - global_left` is the global right and needs no
    // second round of communication -- the same ordering the scalar path uses.
    this->AllreduceHist(to_build);
    for (auto const& candidate : valid_candidates) {
      auto left = p_tree->LeftChild(candidate.nid);
      auto right = p_tree->RightChild(candidate.nid);
      SubtractExactHist(hist_[right], common::Span<double const>{hist_[candidate.nid]},
                        common::Span<double const>{hist_[left]});
    }
    monitor_->Stop(__func__);
  }

  void EvaluateSplits(DMatrix* p_fmat, std::vector<MultiExpandEntry>* p_entries) {
    monitor_->Start(__func__);
    auto& entries = *p_entries;
    auto n_threads = ctx_->Threads();
    this->ResizeThreadState(n_threads);

    // A dense matrix has no missing values anywhere, which makes the backward enumeration
    // pass redundant. This is a property of the data, not a tolerance.
    //
    // The predicate is read from the DMatrix rather than from the page, because a dataset
    // can span several pages and only some of them need be dense. A page-level answer would
    // skip the backward pass for a feature that does have missing values elsewhere.
    auto may_have_missing = !p_fmat->IsDense();

    for (auto const& gmat : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      for (auto& entry : entries) {
        auto features = col_sampler_->GetFeatureSet(ctx_, entry.depth);
        // Drop features the interaction graph forbids at this node before enumerating.
        allowed_features_.clear();
        for (auto fidx : features->ConstHostSpan()) {
          if (interaction_constraints_.Query(entry.nid, fidx)) {
            allowed_features_.push_back(fidx);
          }
        }

        ExactNodeTotal(hist_[entry.nid], n_free_, hist_.TotalBins(), node_total_.Data());
        auto parent = node_total_.ConstView();
        // Constant for the node, so factorized once rather than once per candidate bin.
        auto parent_gain =
            ExactParentGain(solvers_.front().get(), *param_, n_classes_, parent);

        // One candidate slot per feature, filled in parallel and reduced in feature order.
        // Reducing by index rather than by completion keeps the chosen split independent of
        // thread scheduling, so repeated runs give identical trees.
        per_feature_.assign(allowed_features_.size(), ExactSplitCandidate{});
        auto hist_span = common::Span<double const>{hist_[entry.nid]};
        common::ParallelFor(allowed_features_.size(), n_threads, [&](std::size_t i) {
          auto tid = omp_get_thread_num();
          EnumerateExactFeature(solvers_[tid].get(), *param_, n_classes_, gmat.cut,
                                allowed_features_[i], hist_span, parent, parent_gain,
                                may_have_missing, &workspaces_[tid], &per_feature_[i]);
        });

        ExactSplitCandidate best;
        for (auto const& candidate : per_feature_) {
          best.Update(candidate);
        }
        // gamma lives in the same doubled units as the gain, so this is the scalar path's
        // comparison unchanged.
        if (best.valid && best.loss_chg <= static_cast<double>(param_->min_split_loss)) {
          best.valid = false;
        }

        entry.split.loss_chg = 0.0f;
        if (best.valid) {
          entry.split.loss_chg = static_cast<float>(best.loss_chg);
          entry.split.sindex = best.fidx;
          if (best.default_left) {
            entry.split.sindex |= (1U << 31);
          }
          entry.split.split_value = best.split_value;
          entry.split.is_cat = false;
        }
      }
      break;
    }
    monitor_->Stop(__func__);
  }

  void ApplyTreeSplit(MultiExpandEntry const& candidate, RegTree* p_tree) {
    monitor_->Start(__func__);
    ExactNodeTotal(hist_[candidate.nid], n_free_, hist_.TotalBins(), node_total_.Data());

    std::vector<float> base_weight(n_classes_, 0.0f);
    this->SolveCentered(node_total_.ConstView(),
                        common::Span<float>{base_weight.data(), base_weight.size()});

    // Children statistics come from the candidate's own split, re-derived from the parent
    // histogram so that the weights match the gain that selected this split.
    auto fidx = candidate.split.SplitIndex();
    auto default_left = candidate.split.DefaultLeft();
    ExactStatBuffer left_stats;
    ExactStatBuffer right_stats;
    left_stats.Reset(n_free_);
    right_stats.Reset(n_free_);
    // Cuts are identical across pages and the histogram already covers every page, so one
    // page supplies the bin range. See ReconstructExactSplitChildren for why the accumulation
    // order matters.
    auto const& gmat =
        *(p_last_fmat_->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_)).begin());
    ReconstructExactSplitChildren(gmat.cut, fidx, candidate.split.split_value, default_left,
                                  common::Span<double const>{hist_[candidate.nid]},
                                  common::Span<double const>{node_total_.Data()}, n_free_,
                                  &left_stats, &right_stats);

    std::vector<float> left_weight(n_classes_, 0.0f);
    std::vector<float> right_weight(n_classes_, 0.0f);
    this->SolveCentered(left_stats.ConstView(),
                        common::Span<float>{left_weight.data(), left_weight.size()});
    this->SolveCentered(right_stats.ConstView(),
                        common::Span<float>{right_weight.data(), right_weight.size()});

    auto left_curvature = ExactChildCurvature(left_stats.ConstView(), n_classes_);
    auto right_curvature = ExactChildCurvature(right_stats.ConstView(), n_classes_);

    ExpandBatch batch{param_->learning_rate};
    batch.Push(candidate.nid, fidx, candidate.split.split_value, default_left,
               common::Span<float const>{base_weight.data(), base_weight.size()},
               common::Span<float const>{left_weight.data(), left_weight.size()},
               common::Span<float const>{right_weight.data(), right_weight.size()},
               candidate.split.loss_chg, left_curvature, right_curvature);
    p_tree->Expand(ctx_, batch);
    CHECK(p_tree->IsMultiTarget());
    interaction_constraints_.Split(candidate.nid, fidx, p_tree->LeftChild(candidate.nid),
                                   p_tree->RightChild(candidate.nid));
    monitor_->Stop(__func__);
  }

  void LeafPartition(RegTree const& tree, linalg::MatrixView<GradientPair const> gpair,
                     std::vector<bst_node_t>* p_out_position) {
    monitor_->Start(__func__);
    p_out_position->resize(gpair.Shape(0));
    for (auto const& part : partitioner_) {
      part.LeafPartition(ctx_, tree.HostMtView(), gpair,
                         common::Span{p_out_position->data(), p_out_position->size()});
    }
    monitor_->Stop(__func__);
  }

  [[nodiscard]] ExactHistCollection const& Histogram() const { return hist_; }
  [[nodiscard]] bst_target_t NumClasses() const { return n_classes_; }

 private:
  /** @brief Give every thread its own enumeration workspace and solver. */
  void ResizeThreadState(std::int32_t n_threads) {
    auto wanted = static_cast<std::size_t>(std::max(n_threads, 1));
    if (workspaces_.size() == wanted && !solvers_.empty()) {
      return;
    }
    workspaces_.resize(wanted);
    for (auto& workspace : workspaces_) {
      workspace.Reset(n_free_);
    }
    solvers_.clear();
    for (std::size_t i = 0; i < wanted; ++i) {
      solvers_.emplace_back(std::make_unique<common::ExactMultinomialLeafSolver>(n_free_));
    }
  }

  /** @brief Solve the regularized Newton system and write the centered K outputs. */
  void SolveCentered(common::PackedMultinomialStats<double const> stats,
                     common::Span<float> out_weight) {
    auto reg = common::ExactL2::Centered(param_->reg_lambda, n_classes_);
    std::fill(free_weight_.begin(), free_weight_.end(), 0.0);
    if (!solver_->Solve(stats, reg,
                        common::Span<double>{free_weight_.data(), free_weight_.size()})) {
      // No usable curvature: the repository's scalar policy is a zero weight.
      std::fill(out_weight.begin(), out_weight.end(), 0.0f);
      return;
    }
    CenteredLeafWeight(common::Span<double const>{free_weight_.data(), free_weight_.size()},
                       out_weight);
  }

  /**
   * @brief Sum the given nodes' histograms across workers.
   *
   * The packed record is already a flat, fixed-stride array of doubles, and the cache lays
   * nodes out contiguously in allocation order, so the whole block reduces in one call with
   * no custom protocol. The only difference from the scalar path is the stride: bins carry
   * `ExactHistRecordSize(n_free)` doubles rather than the two of a GradientPairPrecise.
   */
  void AllreduceHist(std::vector<bst_node_t> const& nodes) {
    if (!is_distributed_ || nodes.empty()) {
      return;
    }
    auto stride = hist_.RecordsPerNode() * hist_.RecordSize();
    auto* first = hist_[nodes.front()].data();
    // Contiguity is what makes one call sufficient; assert it rather than assume it.
    for (std::size_t i = 1; i < nodes.size(); ++i) {
      CHECK_EQ(hist_[nodes[i]].data(), first + i * stride)
          << "exact histograms for a node batch are not contiguous";
    }
    auto rc = collective::Allreduce(
        ctx_, linalg::MakeVec(first, nodes.size() * stride), collective::Op::kSum);
    collective::SafeColl(rc);
  }

  /** @brief Accumulate one node's rows into its histogram. */
  void BuildNodeHist(DMatrix* p_fmat, bst_node_t nidx,
                     linalg::MatrixView<GradientPair const> gpair) {
    ZeroExactHist(hist_[nidx]);
    std::size_t page_id = 0;
    for (auto const& gmat : p_fmat->GetBatches<GHistIndexMatrix>(ctx_, HistBatch(param_))) {
      auto const& elem = partitioner_.at(page_id)[nidx];
      if (elem.Size() != 0) {
        common::Span<bst_idx_t const> rows{elem.begin(), elem.Size()};
        BuildExactHist(ctx_, hist_[nidx], n_free_, gmat, rows, gpair, *hessian_, &hist_buffer_);
      }
      ++page_id;
    }
  }

  common::Monitor* monitor_{nullptr};
  TrainParam const* param_{nullptr};
  HistMakerTrainParam const* hist_param_{nullptr};
  std::shared_ptr<common::ColumnSampler> col_sampler_;
  Context const* ctx_{nullptr};
  ExactHessian const* hessian_{nullptr};

  FeatureInteractionConstraintHost interaction_constraints_;
  std::vector<bst_feature_t> allowed_features_;
  ExactHistCollection hist_;
  ExactHistThreadBuffer hist_buffer_;
  std::unique_ptr<common::ExactMultinomialLeafSolver> solver_;
  std::vector<ExactEnumerateWorkspace> workspaces_;
  std::vector<std::unique_ptr<common::ExactMultinomialLeafSolver>> solvers_;
  std::vector<ExactSplitCandidate> per_feature_;
  ExactStatBuffer node_total_;
  std::vector<double> free_weight_;

  std::vector<CommonRowPartitioner> partitioner_;
  RegTree const* p_last_tree_{nullptr};
  DMatrix* p_last_fmat_{nullptr};

  bool is_distributed_{false};
  bst_target_t n_classes_{0};
  bst_target_t n_free_{0};
};
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EXACT_BUILDER_H_
