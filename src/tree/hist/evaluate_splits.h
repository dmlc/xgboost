#ifndef XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
#define XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
#include "../param.h"
#include "../constraints.h"
#include "../split_evaluator.h"
#include "../../common/random.h"
#include "../../common/hist_util.h"
#include "../../data/gradient_index.h"

namespace xgboost {
namespace tree {
struct NodeEntry {
  /*! \brief statics for node entry */
  GradStats stats;
  /*! \brief loss of this node, without split */
  bst_float root_gain;
  /*! \brief weight calculated related to current data */
  float weight;
  /*! \brief current best solution */
  SplitEntry best;
  // constructor
  explicit NodeEntry(const TrainParam &) : root_gain(0.0f), weight(0.0f) {}
};

template <typename GradientSumT, typename ExpandEntry> class ApproxEvaluator {
  TrainParam param_;
  common::ColumnSampler column_sampler_;
  TreeEvaluator tree_evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;
  std::vector<NodeEntry> snode_;

 public:
  void EvaluateSplits(const common::HistCollection<GradientSumT> &hist,
                      GHistIndexMatrix const &gidx, const RegTree &tree,
                      std::vector<ExpandEntry *> entries) {
      const size_t grain_size = std::max<size_t>(
        1,
        column_sampler_.GetFeatureSet(tree.GetDepth(entries[0]->nid))->Size() /
        omp_get_max_threads());

    // All nodes are on the same level, so we can store the shared ptr.
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(
        entries.size());
    for (size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      auto nidx = entries[nidx_in_set]->nid;
      features[nidx_in_set] = column_sampler_.GetFeatureSet(tree.GetDepth(nidx));
    }
    common::BlockedSpace2d space(entries.size(), [&](size_t nidx_in_set) {
      return features[nidx_in_set]->Size();
    }, grain_size);

    auto num_threads = omp_get_max_threads();
    std::vector<ExpandEntry> tloc_candidates(omp_get_max_threads() * entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
      for (decltype(num_threads) j = 0; j < num_threads; ++j) {
        tloc_candidates[i * num_threads + j] = *entries[i];
      }
    }
    auto evaluator = tree_evaluator_.GetEvaluator();

    common::ParallelFor2d(space, num_threads, [&](size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      auto entry = &tloc_candidates[num_threads * nidx_in_set + tidx];
      auto best = &entry->split;
      auto nidx = entry->nid;
      auto histogram = hist[nidx];
      auto features_set = features[nidx_in_set]->ConstHostSpan();
      for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
        auto fidx = features_set[fidx_in_set];
        if (interaction_constraints_.Query(nidx, fidx)) {
          auto grad_stats = EnumerateSplit<common::GHistRow<GradientSumT>,
                                           NodeEntry, SplitEntry, +1>(
              gidx, histogram, snode_[nidx], best, nidx, fidx, param_,
              evaluator);
          if (SplitContainsMissingValues(grad_stats, snode_[nidx])) {
            EnumerateSplit<common::GHistRow<GradientSumT>, NodeEntry,
                           SplitEntry, -1>(gidx, histogram, snode_[nidx], best,
                                           nidx, fidx, param_, evaluator);
          }
        }
      }
    });

    for (unsigned nidx_in_set = 0; nidx_in_set < entries.size();
         ++nidx_in_set) {
      for (auto tidx = 0; tidx < num_threads; ++tidx) {
        entries[nidx_in_set]->split.Update(
            tloc_candidates[num_threads * nidx_in_set + tidx].split);
      }
    }
  }

  void ApplyTreeSplit(ExpandEntry candidate, TrainParam param,
                      RegTree *p_tree) {
    auto evaluator = tree_evaluator_.GetEvaluator();
    RegTree &tree = *p_tree;

    GradStats parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
    auto base_weight =
        evaluator.CalcWeight(candidate.nid, param, GradStats{parent_sum});

    auto left_weight =
        evaluator.CalcWeight(candidate.nid, param,
                             GradStats{candidate.split.left_sum}) *
        param.learning_rate;
    auto right_weight =
        evaluator.CalcWeight(candidate.nid, param,
                             GradStats{candidate.split.right_sum}) *
        param.learning_rate;

    tree.ExpandNode(candidate.nid, candidate.split.SplitIndex(),
                    candidate.split.split_value, candidate.split.DefaultLeft(),
                    base_weight, left_weight, right_weight,
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
        candidate.nid, param, GradStats{candidate.split.left_sum});
    snode_.at(right_child).stats = candidate.split.right_sum;
    snode_.at(right_child).root_gain = evaluator.CalcGain(
        candidate.nid, param, GradStats{candidate.split.right_sum});

    interaction_constraints_.Split(candidate.nid,
                                   tree[candidate.nid].SplitIndex(), left_child,
                                   right_child);
  }

  auto GetEvaluator() const { return tree_evaluator_.GetEvaluator(); }
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

  ApproxEvaluator() = default;
  explicit ApproxEvaluator(TrainParam param, MetaInfo const &info)
      : param_{std::move(param)}, tree_evaluator_{
                                      param,
                                      static_cast<bst_feature_t>(info.num_col_),
                                      GenericParameter::kCpuId} {
    column_sampler_.Init(info.num_col_, info.feature_weigths.HostVector(),
                         param_.colsample_bynode, param_.colsample_bylevel,
                         param_.colsample_bytree, false);
  }
};
}      // namespace tree
}      // namespace xgboost
#endif  // XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
