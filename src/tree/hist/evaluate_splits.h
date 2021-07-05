/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
#define XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_

#include <algorithm>
#include <memory>
#include <limits>
#include <utility>
#include <vector>

#include "../param.h"
#include "../constraints.h"
#include "../split_evaluator.h"
#include "../../common/random.h"
#include "../../common/hist_util.h"
#include "../../data/gradient_index.h"

namespace xgboost {
namespace tree {

template <typename GradientSumT, typename ExpandEntry> class HistEvaluator {
 private:
  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain{0.0f};
  };

 private:
  TrainParam param_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  TreeEvaluator tree_evaluator_;
  int32_t n_threads_ {0};
  FeatureInteractionConstraintHost interaction_constraints_;
  std::vector<NodeEntry> snode_;

  // if sum of statistics for non-missing values in the node
  // is equal to sum of statistics for all values:
  // then - there are no missing values
  // else - there are missing values
  bool static SplitContainsMissingValues(const GradStats e,
                                         const NodeEntry &snode) {
    if (e.GetGrad() == snode.stats.GetGrad() &&
        e.GetHess() == snode.stats.GetHess()) {
      return false;
    } else {
      return true;
    }
  }

  // Enumerate/Scan the split values of specific feature
  // Returns the sum of gradients corresponding to the data points that contains
  // a non-missing value for the particular feature fid.
  template <int d_step>
  GradStats EnumerateSplit(
      const GHistIndexMatrix &gmat, const common::GHistRow<GradientSumT> &hist,
      const NodeEntry &snode, SplitEntry *p_best, bst_feature_t fidx,
      bst_node_t nidx,
      TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");

    // aliases
    const std::vector<uint32_t> &cut_ptr = gmat.cut.Ptrs();
    const std::vector<bst_float> &cut_val = gmat.cut.Values();

    // statistics on both sides of split
    GradStats c;
    GradStats e;
    // best split so far
    SplitEntry best;

    // bin boundaries
    CHECK_LE(cut_ptr[fidx],
             static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    CHECK_LE(cut_ptr[fidx + 1],
             static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    // imin: index (offset) of the minimum value for feature fid
    //       need this for backward enumeration
    const auto imin = static_cast<int32_t>(cut_ptr[fidx]);
    // ibegin, iend: smallest/largest cut points for feature fid
    // use int to allow for value -1
    int32_t ibegin, iend;
    if (d_step > 0) {
      ibegin = static_cast<int32_t>(cut_ptr[fidx]);
      iend = static_cast<int32_t>(cut_ptr.at(fidx + 1));
    } else {
      ibegin = static_cast<int32_t>(cut_ptr[fidx + 1]) - 1;
      iend = static_cast<int32_t>(cut_ptr[fidx]) - 1;
    }

    for (int32_t i = ibegin; i != iend; i += d_step) {
      // start working
      // try to find a split
      e.Add(hist[i].GetGrad(), hist[i].GetHess());
      if (e.GetHess() >= param_.min_child_weight) {
        c.SetSubstract(snode.stats, e);
        if (c.GetHess() >= param_.min_child_weight) {
          bst_float loss_chg;
          bst_float split_pt;
          if (d_step > 0) {
            // forward enumeration: split at right bound of each bin
            loss_chg = static_cast<bst_float>(
                evaluator.CalcSplitGain(param_, nidx, fidx, GradStats{e},
                                        GradStats{c}) -
                snode.root_gain);
            split_pt = cut_val[i];
            best.Update(loss_chg, fidx, split_pt, d_step == -1, e, c);
          } else {
            // backward enumeration: split at left bound of each bin
            loss_chg = static_cast<bst_float>(
                evaluator.CalcSplitGain(param_, nidx, fidx, GradStats{c},
                                        GradStats{e}) -
                snode.root_gain);
            if (i == imin) {
              // for leftmost bin, left bound is the smallest feature value
              split_pt = gmat.cut.MinValues()[fidx];
            } else {
              split_pt = cut_val[i - 1];
            }
            best.Update(loss_chg, fidx, split_pt, d_step == -1, c, e);
          }
        }
      }
    }
    p_best->Update(best);

    return e;
  }

 public:
  void EvaluateSplits(const common::HistCollection<GradientSumT> &hist,
                      GHistIndexMatrix const &gidx, const RegTree &tree,
                      std::vector<ExpandEntry>* p_entries) {
    auto& entries = *p_entries;
    // All nodes are on the same level, so we can store the shared ptr.
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(
        entries.size());
    for (size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      auto nidx = entries[nidx_in_set].nid;
      features[nidx_in_set] =
          column_sampler_->GetFeatureSet(tree.GetDepth(nidx));
    }
    CHECK(!features.empty());
    const size_t grain_size =
        std::max<size_t>(1, features.front()->Size() / n_threads_);
    common::BlockedSpace2d space(entries.size(), [&](size_t nidx_in_set) {
      return features[nidx_in_set]->Size();
    }, grain_size);

    std::vector<ExpandEntry> tloc_candidates(omp_get_max_threads() * entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
      for (decltype(n_threads_) j = 0; j < n_threads_; ++j) {
        tloc_candidates[i * n_threads_ + j] = entries[i];
      }
    }
    auto evaluator = tree_evaluator_.GetEvaluator();

    common::ParallelFor2d(space, n_threads_, [&](size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      auto entry = &tloc_candidates[n_threads_ * nidx_in_set + tidx];
      auto best = &entry->split;
      auto nidx = entry->nid;
      auto histogram = hist[nidx];
      auto features_set = features[nidx_in_set]->ConstHostSpan();
      for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
        auto fidx = features_set[fidx_in_set];
        if (interaction_constraints_.Query(nidx, fidx)) {
          auto grad_stats = EnumerateSplit<+1>(gidx, histogram, snode_[nidx],
                                               best, fidx, nidx, evaluator);
          if (SplitContainsMissingValues(grad_stats, snode_[nidx])) {
            EnumerateSplit<-1>(gidx, histogram, snode_[nidx], best, fidx, nidx,
                               evaluator);
          }
        }
      }
    });

    for (unsigned nidx_in_set = 0; nidx_in_set < entries.size();
         ++nidx_in_set) {
      for (auto tidx = 0; tidx < n_threads_; ++tidx) {
        entries[nidx_in_set].split.Update(
            tloc_candidates[n_threads_ * nidx_in_set + tidx].split);
      }
    }
  }
  // Add splits to tree, handles all statistic
  void ApplyTreeSplit(ExpandEntry candidate, RegTree *p_tree) {
    auto evaluator = tree_evaluator_.GetEvaluator();
    RegTree &tree = *p_tree;

    GradStats parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
    auto base_weight =
        evaluator.CalcWeight(candidate.nid, param_, GradStats{parent_sum});

    auto left_weight = evaluator.CalcWeight(
        candidate.nid, param_, GradStats{candidate.split.left_sum});
    auto right_weight = evaluator.CalcWeight(
        candidate.nid, param_, GradStats{candidate.split.right_sum});

    tree.ExpandNode(candidate.nid, candidate.split.SplitIndex(),
                    candidate.split.split_value, candidate.split.DefaultLeft(),
                    base_weight, left_weight * param_.learning_rate,
                    right_weight * param_.learning_rate,
                    candidate.split.loss_chg, parent_sum.GetHess(),
                    candidate.split.left_sum.GetHess(),
                    candidate.split.right_sum.GetHess());

    // Set up child constraints
    auto left_child = tree[candidate.nid].LeftChild();
    auto right_child = tree[candidate.nid].RightChild();
    tree_evaluator_.AddSplit(candidate.nid, left_child, right_child,
                             tree[candidate.nid].SplitIndex(), left_weight,
                             right_weight);

    auto max_node = std::max(left_child, tree[candidate.nid].RightChild());
    max_node = std::max(candidate.nid, max_node);
    snode_.resize(tree.GetNodes().size());
    snode_.at(left_child).stats = candidate.split.left_sum;
    snode_.at(left_child).root_gain = evaluator.CalcGain(
        candidate.nid, param_, GradStats{candidate.split.left_sum});
    snode_.at(right_child).stats = candidate.split.right_sum;
    snode_.at(right_child).root_gain = evaluator.CalcGain(
        candidate.nid, param_, GradStats{candidate.split.right_sum});

    interaction_constraints_.Split(candidate.nid,
                                   tree[candidate.nid].SplitIndex(), left_child,
                                   right_child);
  }

  auto Evaluator() const { return tree_evaluator_.GetEvaluator(); }
  auto const& Stats() const { return snode_; }

  float InitRoot(GradStats const& root_sum) {
    snode_.resize(1);
    auto root_evaluator = tree_evaluator_.GetEvaluator();

    snode_[0].stats = GradStats{root_sum.GetGrad(), root_sum.GetHess()};
    snode_[0].root_gain = root_evaluator.CalcGain(RegTree::kRoot, param_,
                                                  GradStats{snode_[0].stats});
    auto weight = root_evaluator.CalcWeight(RegTree::kRoot, param_,
                                            GradStats{snode_[0].stats});
    return weight;
  }

 public:
  // The column sampler must be constructed by caller since we need to preserve the rng
  // for the entire training session.
  explicit HistEvaluator(TrainParam const &param, MetaInfo const &info,
                         int32_t n_threads,
                         std::shared_ptr<common::ColumnSampler> sampler,
                         bool skip_0_index = false)
      : param_{param}, column_sampler_{std::move(sampler)},
        tree_evaluator_{param, static_cast<bst_feature_t>(info.num_col_),
                        GenericParameter::kCpuId},
        n_threads_{n_threads} {
    interaction_constraints_.Configure(param, info.num_col_);
    column_sampler_->Init(info.num_col_, info.feature_weigths.HostVector(),
                          param_.colsample_bynode, param_.colsample_bylevel,
                          param_.colsample_bytree, skip_0_index);
  }
};
}      // namespace tree
}      // namespace xgboost
#endif  // XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
