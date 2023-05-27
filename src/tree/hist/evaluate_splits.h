/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
#define XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_

#include <algorithm>                   // for copy
#include <cstddef>                     // for size_t
#include <limits>                      // for numeric_limits
#include <memory>                      // for shared_ptr
#include <numeric>                     // for accumulate
#include <utility>                     // for move
#include <vector>                      // for vector

#include "../../common/categorical.h"  // for CatBitField
#include "../../common/hist_util.h"    // for GHistRow, HistogramCuts
#include "../../common/linalg_op.h"    // for cbegin, cend, begin
#include "../../common/random.h"       // for ColumnSampler
#include "../constraints.h"            // for FeatureInteractionConstraintHost
#include "../param.h"                  // for TrainParam
#include "../split_evaluator.h"        // for TreeEvaluator
#include "expand_entry.h"              // for MultiExpandEntry
#include "xgboost/base.h"              // for bst_node_t, bst_target_t, bst_feature_t
#include "xgboost/context.h"           // for COntext
#include "xgboost/linalg.h"            // for Constants, Vector

namespace xgboost::tree {
class HistEvaluator {
 private:
  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain{0.0f};
  };

 private:
  Context const* ctx_;
  TrainParam const* param_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  TreeEvaluator tree_evaluator_;
  bool is_col_split_{false};
  FeatureInteractionConstraintHost interaction_constraints_;
  std::vector<NodeEntry> snode_;

  // if sum of statistics for non-missing values in the node
  // is equal to sum of statistics for all values:
  // then - there are no missing values
  // else - there are missing values
  bool static SplitContainsMissingValues(const GradStats e, const NodeEntry &snode) {
    if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
      return false;
    } else {
      return true;
    }
  }

  [[nodiscard]] bool IsValid(GradStats const &left, GradStats const &right) const {
    return left.GetHess() >= param_->min_child_weight &&
           right.GetHess() >= param_->min_child_weight;
  }

  /**
   * \brief Use learned direction with one-hot split. Other implementations (LGB) create a
   *        pseudo-category for missing value but here we just do a complete scan to avoid
   *        making specialized histogram bin.
   */
  void EnumerateOneHot(common::HistogramCuts const &cut, const common::GHistRow &hist,
                       bst_feature_t fidx, bst_node_t nidx,
                       TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                       SplitEntry *p_best) const {
    const std::vector<uint32_t> &cut_ptr = cut.Ptrs();
    const std::vector<bst_float> &cut_val = cut.Values();

    bst_bin_t ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
    bst_bin_t iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
    bst_bin_t n_bins = iend - ibegin;

    GradStats left_sum;
    GradStats right_sum;
    // best split so far
    SplitEntry best;
    best.is_cat = false;  // marker for whether it's updated or not.

    auto f_hist = hist.subspan(cut_ptr[fidx], n_bins);
    auto feature_sum = GradStats{
        std::accumulate(f_hist.data(), f_hist.data() + f_hist.size(), GradientPairPrecise{})};
    GradStats missing;
    auto const &parent = snode_[nidx];
    missing.SetSubstract(parent.stats, feature_sum);

    for (bst_bin_t i = ibegin; i != iend; i += 1) {
      auto split_pt = cut_val[i];

      // missing on left (treat missing as other categories)
      right_sum = GradStats{hist[i]};
      left_sum.SetSubstract(parent.stats, right_sum);
      if (IsValid(left_sum, right_sum)) {
        auto missing_left_chg =
            static_cast<float>(evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{left_sum},
                                                       GradStats{right_sum}) -
                               parent.root_gain);
        best.Update(missing_left_chg, fidx, split_pt, true, true, left_sum, right_sum);
      }

      // missing on right (treat missing as chosen category)
      right_sum.Add(missing);
      left_sum.SetSubstract(parent.stats, right_sum);
      if (IsValid(left_sum, right_sum)) {
        auto missing_right_chg =
            static_cast<float>(evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{left_sum},
                                                       GradStats{right_sum}) -
                               parent.root_gain);
        best.Update(missing_right_chg, fidx, split_pt, false, true, left_sum, right_sum);
      }
    }

    if (best.is_cat) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins + 1);
      best.cat_bits.resize(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      cat_bits.Set(best.split_value);
    }

    p_best->Update(best);
  }

  /**
   * \brief Enumerate with partition-based splits.
   *
   * The implementation is different from LightGBM. Firstly we don't have a
   * pseudo-cateogry for missing value, instead of we make 2 complete scans over the
   * histogram. Secondly, both scan directions generate splits in the same
   * order. Following table depicts the scan process, square bracket means the gradient in
   * missing values is resided on that partition:
   *
   *   | Forward  | Backward |
   *   |----------+----------|
   *   | [BCDE] A | E [ABCD] |
   *   | [CDE] AB | DE [ABC] |
   *   | [DE] ABC | CDE [AB] |
   *   | [E] ABCD | BCDE [A] |
   */
  template <int d_step>
  void EnumeratePart(common::HistogramCuts const &cut, common::Span<size_t const> sorted_idx,
                     common::GHistRow const &hist, bst_feature_t fidx, bst_node_t nidx,
                     TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                     SplitEntry *p_best) {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");

    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();
    auto const &parent = snode_[nidx];

    bst_bin_t f_begin = cut_ptr[fidx];
    bst_bin_t f_end = cut_ptr[fidx + 1];
    bst_bin_t n_bins_feature{f_end - f_begin};
    auto n_bins = std::min(param_->max_cat_threshold, n_bins_feature);

    // statistics on both sides of split
    GradStats left_sum;
    GradStats right_sum;
    // best split so far
    SplitEntry best;

    auto f_hist = hist.subspan(f_begin, n_bins_feature);
    bst_bin_t it_begin, it_end;
    if (d_step > 0) {
      it_begin = f_begin;
      it_end = it_begin + n_bins - 1;
    } else {
      it_begin = f_end - 1;
      it_end = it_begin - n_bins + 1;
    }

    bst_bin_t best_thresh{-1};
    for (bst_bin_t i = it_begin; i != it_end; i += d_step) {
      auto j = i - f_begin;  // index local to current feature
      if (d_step == 1) {
        right_sum.Add(f_hist[sorted_idx[j]].GetGrad(), f_hist[sorted_idx[j]].GetHess());
        left_sum.SetSubstract(parent.stats, right_sum);  // missing on left
      } else {
        left_sum.Add(f_hist[sorted_idx[j]].GetGrad(), f_hist[sorted_idx[j]].GetHess());
        right_sum.SetSubstract(parent.stats, left_sum);  // missing on right
      }
      if (IsValid(left_sum, right_sum)) {
        auto loss_chg = evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{left_sum},
                                                GradStats{right_sum}) -
                        parent.root_gain;
        // We don't have a numeric split point, nan here is a dummy split.
        if (best.Update(loss_chg, fidx, std::numeric_limits<float>::quiet_NaN(), d_step == 1, true,
                        left_sum, right_sum)) {
          best_thresh = i;
        }
      }
    }

    if (best_thresh != -1) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins_feature);
      best.cat_bits = decltype(best.cat_bits)(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      bst_bin_t partition = d_step == 1 ? (best_thresh - it_begin + 1) : (best_thresh - f_begin);
      CHECK_GT(partition, 0);
      std::for_each(sorted_idx.begin(), sorted_idx.begin() + partition, [&](size_t c) {
        auto cat = cut_val[c + f_begin];
        cat_bits.Set(cat);
      });
    }

    p_best->Update(best);
  }

  // Enumerate/Scan the split values of specific feature
  // Returns the sum of gradients corresponding to the data points that contains
  // a non-missing value for the particular feature fid.
  template <int d_step>
  GradStats EnumerateSplit(common::HistogramCuts const &cut, const common::GHistRow &hist,
                           bst_feature_t fidx, bst_node_t nidx,
                           TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                           SplitEntry *p_best) const {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");

    // aliases
    const std::vector<uint32_t> &cut_ptr = cut.Ptrs();
    const std::vector<bst_float> &cut_val = cut.Values();
    auto const &parent = snode_[nidx];

    // statistics on both sides of split
    GradStats left_sum;
    GradStats right_sum;
    // best split so far
    SplitEntry best;

    // bin boundaries
    CHECK_LE(cut_ptr[fidx], static_cast<uint32_t>(std::numeric_limits<bst_bin_t>::max()));
    CHECK_LE(cut_ptr[fidx + 1], static_cast<uint32_t>(std::numeric_limits<bst_bin_t>::max()));
    // imin: index (offset) of the minimum value for feature fid need this for backward
    //       enumeration
    const auto imin = static_cast<bst_bin_t>(cut_ptr[fidx]);
    // ibegin, iend: smallest/largest cut points for feature fid use int to allow for
    // value -1
    bst_bin_t ibegin, iend;
    if (d_step > 0) {
      ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
      iend = static_cast<bst_bin_t>(cut_ptr.at(fidx + 1));
    } else {
      ibegin = static_cast<bst_bin_t>(cut_ptr[fidx + 1]) - 1;
      iend = static_cast<bst_bin_t>(cut_ptr[fidx]) - 1;
    }

    for (bst_bin_t i = ibegin; i != iend; i += d_step) {
      // start working
      // try to find a split
      left_sum.Add(hist[i].GetGrad(), hist[i].GetHess());
      right_sum.SetSubstract(parent.stats, left_sum);
      if (IsValid(left_sum, right_sum)) {
        bst_float loss_chg;
        bst_float split_pt;
        if (d_step > 0) {
          // forward enumeration: split at right bound of each bin
          loss_chg =
              static_cast<float>(evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{left_sum},
                                                         GradStats{right_sum}) -
                                 parent.root_gain);
          split_pt = cut_val[i];  // not used for partition based
          best.Update(loss_chg, fidx, split_pt, d_step == -1, false, left_sum, right_sum);
        } else {
          // backward enumeration: split at left bound of each bin
          loss_chg =
              static_cast<float>(evaluator.CalcSplitGain(*param_, nidx, fidx, GradStats{right_sum},
                                                         GradStats{left_sum}) -
                                 parent.root_gain);
          if (i == imin) {
            split_pt = cut.MinValues()[fidx];
          } else {
            split_pt = cut_val[i - 1];
          }
          best.Update(loss_chg, fidx, split_pt, d_step == -1, false, right_sum, left_sum);
        }
      }
    }

    p_best->Update(best);
    return left_sum;
  }

  /**
   * @brief Gather the expand entries from all the workers.
   * @param entries Local expand entries on this worker.
   * @return Global expand entries gathered from all workers.
   */
  std::vector<CPUExpandEntry> Allgather(std::vector<CPUExpandEntry> const &entries) {
    auto const world = collective::GetWorldSize();
    auto const rank = collective::GetRank();
    auto const num_entries = entries.size();

    // First, gather all the primitive fields.
    std::vector<CPUExpandEntry> all_entries(num_entries * world);
    std::vector<uint32_t> cat_bits;
    std::vector<std::size_t> cat_bits_sizes;
    for (std::size_t i = 0; i < num_entries; i++) {
      all_entries[num_entries * rank + i].CopyAndCollect(entries[i], &cat_bits, &cat_bits_sizes);
    }
    collective::Allgather(all_entries.data(), all_entries.size() * sizeof(CPUExpandEntry));

    // Gather all the cat_bits.
    auto gathered = collective::AllgatherV(cat_bits, cat_bits_sizes);

    common::ParallelFor(num_entries * world, ctx_->Threads(), [&] (auto i) {
      // Copy the cat_bits back into all expand entries.
      all_entries[i].split.cat_bits.resize(gathered.sizes[i]);
      std::copy_n(gathered.result.cbegin() + gathered.offsets[i], gathered.sizes[i],
                  all_entries[i].split.cat_bits.begin());
    });

    return all_entries;
  }

 public:
  void EvaluateSplits(const common::HistCollection &hist, common::HistogramCuts const &cut,
                      common::Span<FeatureType const> feature_types, const RegTree &tree,
                      std::vector<CPUExpandEntry> *p_entries) {
    auto n_threads = ctx_->Threads();
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
        std::max<size_t>(1, features.front()->Size() / n_threads);
    common::BlockedSpace2d space(entries.size(), [&](size_t nidx_in_set) {
      return features[nidx_in_set]->Size();
    }, grain_size);

    std::vector<CPUExpandEntry> tloc_candidates(n_threads * entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
      for (decltype(n_threads) j = 0; j < n_threads; ++j) {
        tloc_candidates[i * n_threads + j] = entries[i];
      }
    }
    auto evaluator = tree_evaluator_.GetEvaluator();
    auto const& cut_ptrs = cut.Ptrs();

    common::ParallelFor2d(space, n_threads, [&](size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      auto entry = &tloc_candidates[n_threads * nidx_in_set + tidx];
      auto best = &entry->split;
      auto nidx = entry->nid;
      auto histogram = hist[nidx];
      auto features_set = features[nidx_in_set]->ConstHostSpan();
      for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
        auto fidx = features_set[fidx_in_set];
        bool is_cat = common::IsCat(feature_types, fidx);
        if (!interaction_constraints_.Query(nidx, fidx)) {
          continue;
        }
        if (is_cat) {
          auto n_bins = cut_ptrs.at(fidx + 1) - cut_ptrs[fidx];
          if (common::UseOneHot(n_bins, param_->max_cat_to_onehot)) {
            EnumerateOneHot(cut, histogram, fidx, nidx, evaluator, best);
          } else {
            std::vector<size_t> sorted_idx(n_bins);
            std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
            auto feat_hist = histogram.subspan(cut_ptrs[fidx], n_bins);
            // Sort the histogram to get contiguous partitions.
            std::stable_sort(sorted_idx.begin(), sorted_idx.end(), [&](size_t l, size_t r) {
              auto ret = evaluator.CalcWeightCat(*param_, feat_hist[l]) <
                         evaluator.CalcWeightCat(*param_, feat_hist[r]);
              return ret;
            });
            EnumeratePart<+1>(cut, sorted_idx, histogram, fidx, nidx, evaluator, best);
            EnumeratePart<-1>(cut, sorted_idx, histogram, fidx, nidx, evaluator, best);
          }
        } else {
          auto grad_stats = EnumerateSplit<+1>(cut, histogram, fidx, nidx, evaluator, best);
          if (SplitContainsMissingValues(grad_stats, snode_[nidx])) {
            EnumerateSplit<-1>(cut, histogram, fidx, nidx, evaluator, best);
          }
        }
      }
    });

    for (unsigned nidx_in_set = 0; nidx_in_set < entries.size();
         ++nidx_in_set) {
      for (auto tidx = 0; tidx < n_threads; ++tidx) {
        entries[nidx_in_set].split.Update(
            tloc_candidates[n_threads * nidx_in_set + tidx].split);
      }
    }

    if (is_col_split_) {
      // With column-wise data split, we gather the best splits from all the workers and update the
      // expand entries accordingly.
      auto all_entries = Allgather(entries);
      for (auto worker = 0; worker < collective::GetWorldSize(); ++worker) {
        for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
          entries[nidx_in_set].split.Update(
              all_entries[worker * entries.size() + nidx_in_set].split);
        }
      }
    }
  }

  // Add splits to tree, handles all statistic
  void ApplyTreeSplit(CPUExpandEntry const& candidate, RegTree *p_tree) {
    auto evaluator = tree_evaluator_.GetEvaluator();
    RegTree &tree = *p_tree;

    GradStats parent_sum = candidate.split.left_sum;
    parent_sum.Add(candidate.split.right_sum);
    auto base_weight = evaluator.CalcWeight(candidate.nid, *param_, GradStats{parent_sum});
    auto left_weight =
        evaluator.CalcWeight(candidate.nid, *param_, GradStats{candidate.split.left_sum});
    auto right_weight =
        evaluator.CalcWeight(candidate.nid, *param_, GradStats{candidate.split.right_sum});

    if (candidate.split.is_cat) {
      tree.ExpandCategorical(
          candidate.nid, candidate.split.SplitIndex(), candidate.split.cat_bits,
          candidate.split.DefaultLeft(), base_weight, left_weight * param_->learning_rate,
          right_weight * param_->learning_rate, candidate.split.loss_chg, parent_sum.GetHess(),
          candidate.split.left_sum.GetHess(), candidate.split.right_sum.GetHess());
    } else {
      tree.ExpandNode(candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
                      candidate.split.DefaultLeft(), base_weight,
                      left_weight * param_->learning_rate, right_weight * param_->learning_rate,
                      candidate.split.loss_chg, parent_sum.GetHess(),
                      candidate.split.left_sum.GetHess(), candidate.split.right_sum.GetHess());
    }

    // Set up child constraints
    auto left_child = tree[candidate.nid].LeftChild();
    auto right_child = tree[candidate.nid].RightChild();
    tree_evaluator_.AddSplit(candidate.nid, left_child, right_child,
                             tree[candidate.nid].SplitIndex(), left_weight,
                             right_weight);
    evaluator = tree_evaluator_.GetEvaluator();

    snode_.resize(tree.GetNodes().size());
    snode_.at(left_child).stats = candidate.split.left_sum;
    snode_.at(left_child).root_gain =
        evaluator.CalcGain(candidate.nid, *param_, GradStats{candidate.split.left_sum});
    snode_.at(right_child).stats = candidate.split.right_sum;
    snode_.at(right_child).root_gain =
        evaluator.CalcGain(candidate.nid, *param_, GradStats{candidate.split.right_sum});

    interaction_constraints_.Split(candidate.nid,
                                   tree[candidate.nid].SplitIndex(), left_child,
                                   right_child);
  }

  auto Evaluator() const { return tree_evaluator_.GetEvaluator(); }
  auto const& Stats() const { return snode_; }

  float InitRoot(GradStats const &root_sum) {
    snode_.resize(1);
    auto root_evaluator = tree_evaluator_.GetEvaluator();

    snode_[0].stats = GradStats{root_sum.GetGrad(), root_sum.GetHess()};
    snode_[0].root_gain =
        root_evaluator.CalcGain(RegTree::kRoot, *param_, GradStats{snode_[0].stats});
    auto weight = root_evaluator.CalcWeight(RegTree::kRoot, *param_, GradStats{snode_[0].stats});
    return weight;
  }

 public:
  // The column sampler must be constructed by caller since we need to preserve the rng
  // for the entire training session.
  explicit HistEvaluator(Context const *ctx, TrainParam const *param, MetaInfo const &info,
                         std::shared_ptr<common::ColumnSampler> sampler)
      : ctx_{ctx},
        param_{param},
        column_sampler_{std::move(sampler)},
        tree_evaluator_{*param, static_cast<bst_feature_t>(info.num_col_), Context::kCpuId},
        is_col_split_{info.IsColumnSplit()} {
    interaction_constraints_.Configure(*param, info.num_col_);
    column_sampler_->Init(ctx, info.num_col_, info.feature_weights.HostVector(),
                          param_->colsample_bynode, param_->colsample_bylevel,
                          param_->colsample_bytree);
  }
};

class HistMultiEvaluator {
  std::vector<double> gain_;
  linalg::Matrix<GradientPairPrecise> stats_;
  TrainParam const *param_;
  FeatureInteractionConstraintHost interaction_constraints_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  Context const *ctx_;
  bool is_col_split_{false};

 private:
  static double MultiCalcSplitGain(TrainParam const &param,
                                   linalg::VectorView<GradientPairPrecise const> left_sum,
                                   linalg::VectorView<GradientPairPrecise const> right_sum,
                                   linalg::VectorView<float> left_weight,
                                   linalg::VectorView<float> right_weight) {
    CalcWeight(param, left_sum, left_weight);
    CalcWeight(param, right_sum, right_weight);

    auto left_gain = CalcGainGivenWeight(param, left_sum, left_weight);
    auto right_gain = CalcGainGivenWeight(param, right_sum, right_weight);
    return left_gain + right_gain;
  }

  template <bst_bin_t d_step>
  bool EnumerateSplit(common::HistogramCuts const &cut, bst_feature_t fidx,
                      common::Span<common::GHistRow const> hist,
                      linalg::VectorView<GradientPairPrecise const> parent_sum, double parent_gain,
                      SplitEntryContainer<std::vector<GradientPairPrecise>> *p_best) const {
    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();
    auto const &min_val = cut.MinValues();

    auto sum = linalg::Empty<GradientPairPrecise>(ctx_, 2, hist.size());
    auto left_sum = sum.Slice(0, linalg::All());
    auto right_sum = sum.Slice(1, linalg::All());

    bst_bin_t ibegin, iend;
    if (d_step > 0) {
      ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
      iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
    } else {
      ibegin = static_cast<bst_bin_t>(cut_ptr[fidx + 1]) - 1;
      iend = static_cast<bst_bin_t>(cut_ptr[fidx]) - 1;
    }
    const auto imin = static_cast<bst_bin_t>(cut_ptr[fidx]);

    auto n_targets = hist.size();
    auto weight = linalg::Empty<float>(ctx_, 2, n_targets);
    auto left_weight = weight.Slice(0, linalg::All());
    auto right_weight = weight.Slice(1, linalg::All());

    for (bst_bin_t i = ibegin; i != iend; i += d_step) {
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto t_hist = hist[t];
        auto t_p = parent_sum(t);
        left_sum(t) += t_hist[i];
        right_sum(t) = t_p - left_sum(t);
      }

      if (d_step > 0) {
        auto split_pt = cut_val[i];
        auto loss_chg =
            MultiCalcSplitGain(*param_, right_sum, left_sum, right_weight, left_weight) -
            parent_gain;
        p_best->Update(loss_chg, fidx, split_pt, d_step == -1, false, left_sum, right_sum);
      } else {
        float split_pt;
        if (i == imin) {
          split_pt = min_val[fidx];
        } else {
          split_pt = cut_val[i - 1];
        }
        auto loss_chg =
            MultiCalcSplitGain(*param_, right_sum, left_sum, left_weight, right_weight) -
            parent_gain;
        p_best->Update(loss_chg, fidx, split_pt, d_step == -1, false, right_sum, left_sum);
      }
    }
    // return true if there's missing. Doesn't handle floating-point error well.
    if (d_step == +1) {
      return !std::equal(linalg::cbegin(left_sum), linalg::cend(left_sum),
                         linalg::cbegin(parent_sum));
    }
    return false;
  }

  /**
   * @brief Gather the expand entries from all the workers.
   * @param entries Local expand entries on this worker.
   * @return Global expand entries gathered from all workers.
   */
  std::vector<MultiExpandEntry> Allgather(std::vector<MultiExpandEntry> const &entries) {
    auto const world = collective::GetWorldSize();
    auto const rank = collective::GetRank();
    auto const num_entries = entries.size();

    // First, gather all the primitive fields.
    std::vector<MultiExpandEntry> all_entries(num_entries * world);
    std::vector<uint32_t> cat_bits;
    std::vector<std::size_t> cat_bits_sizes;
    std::vector<GradientPairPrecise> gradients;
    for (std::size_t i = 0; i < num_entries; i++) {
      all_entries[num_entries * rank + i].CopyAndCollect(entries[i], &cat_bits, &cat_bits_sizes,
                                                         &gradients);
    }
    collective::Allgather(all_entries.data(), all_entries.size() * sizeof(MultiExpandEntry));

    // Gather all the cat_bits.
    auto gathered_cat_bits = collective::AllgatherV(cat_bits, cat_bits_sizes);

    // Gather all the gradients.
    auto const num_gradients = gradients.size();
    std::vector<GradientPairPrecise> all_gradients(num_gradients * world);
    std::copy_n(gradients.cbegin(), num_gradients, all_gradients.begin() + num_gradients * rank);
    collective::Allgather(all_gradients.data(), all_gradients.size() * sizeof(GradientPairPrecise));

    auto const total_entries = num_entries * world;
    auto const gradients_per_entry = num_gradients / num_entries;
    auto const gradients_per_side = gradients_per_entry / 2;
    common::ParallelFor(total_entries, ctx_->Threads(), [&] (auto i) {
      // Copy the cat_bits back into all expand entries.
      all_entries[i].split.cat_bits.resize(gathered_cat_bits.sizes[i]);
      std::copy_n(gathered_cat_bits.result.cbegin() + gathered_cat_bits.offsets[i],
                  gathered_cat_bits.sizes[i], all_entries[i].split.cat_bits.begin());

      // Copy the gradients back into all expand entries.
      all_entries[i].split.left_sum.resize(gradients_per_side);
      std::copy_n(all_gradients.cbegin() + i * gradients_per_entry, gradients_per_side,
                  all_entries[i].split.left_sum.begin());
      all_entries[i].split.right_sum.resize(gradients_per_side);
      std::copy_n(all_gradients.cbegin() + i * gradients_per_entry + gradients_per_side,
                  gradients_per_side, all_entries[i].split.right_sum.begin());
    });

    return all_entries;
  }

 public:
  void EvaluateSplits(RegTree const &tree, common::Span<const common::HistCollection *> hist,
                      common::HistogramCuts const &cut, std::vector<MultiExpandEntry> *p_entries) {
    auto &entries = *p_entries;
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(entries.size());

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      auto nidx = entries[nidx_in_set].nid;
      features[nidx_in_set] = column_sampler_->GetFeatureSet(tree.GetDepth(nidx));
    }
    CHECK(!features.empty());

    std::int32_t n_threads = ctx_->Threads();
    std::size_t const grain_size = std::max<std::size_t>(1, features.front()->Size() / n_threads);
    common::BlockedSpace2d space(
        entries.size(), [&](std::size_t nidx_in_set) { return features[nidx_in_set]->Size(); },
        grain_size);

    std::vector<MultiExpandEntry> tloc_candidates(n_threads * entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
      for (std::int32_t j = 0; j < n_threads; ++j) {
        tloc_candidates[i * n_threads + j] = entries[i];
      }
    }
    common::ParallelFor2d(space, n_threads, [&](std::size_t nidx_in_set, common::Range1d r) {
      auto tidx = omp_get_thread_num();
      auto entry = &tloc_candidates[n_threads * nidx_in_set + tidx];
      auto best = &entry->split;
      auto parent_sum = stats_.Slice(entry->nid, linalg::All());
      std::vector<common::GHistRow> node_hist;
      for (auto t_hist : hist) {
        node_hist.push_back((*t_hist)[entry->nid]);
      }
      auto features_set = features[nidx_in_set]->ConstHostSpan();

      for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
        auto fidx = features_set[fidx_in_set];
        if (!interaction_constraints_.Query(entry->nid, fidx)) {
          continue;
        }
        auto parent_gain = gain_[entry->nid];
        bool missing =
            this->EnumerateSplit<+1>(cut, fidx, node_hist, parent_sum, parent_gain, best);
        if (missing) {
          this->EnumerateSplit<-1>(cut, fidx, node_hist, parent_sum, parent_gain, best);
        }
      }
    });

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      for (auto tidx = 0; tidx < n_threads; ++tidx) {
        entries[nidx_in_set].split.Update(tloc_candidates[n_threads * nidx_in_set + tidx].split);
      }
    }

    if (is_col_split_) {
      // With column-wise data split, we gather the best splits from all the workers and update the
      // expand entries accordingly.
      auto all_entries = Allgather(entries);
      for (auto worker = 0; worker < collective::GetWorldSize(); ++worker) {
        for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
          entries[nidx_in_set].split.Update(
              all_entries[worker * entries.size() + nidx_in_set].split);
        }
      }
    }
  }

  linalg::Vector<float> InitRoot(linalg::VectorView<GradientPairPrecise const> root_sum) {
    auto n_targets = root_sum.Size();
    stats_ = linalg::Constant(ctx_, GradientPairPrecise{}, 1, n_targets);
    gain_.resize(1);

    linalg::Vector<float> weight({n_targets}, ctx_->gpu_id);
    CalcWeight(*param_, root_sum, weight.HostView());
    auto root_gain = CalcGainGivenWeight(*param_, root_sum, weight.HostView());
    gain_.front() = root_gain;

    auto h_stats = stats_.HostView();
    std::copy(linalg::cbegin(root_sum), linalg::cend(root_sum), linalg::begin(h_stats));

    return weight;
  }

  void ApplyTreeSplit(MultiExpandEntry const &candidate, RegTree *p_tree) {
    auto n_targets = p_tree->NumTargets();
    auto parent_sum = stats_.Slice(candidate.nid, linalg::All());

    auto weight = linalg::Empty<float>(ctx_, 3, n_targets);
    auto base_weight = weight.Slice(0, linalg::All());
    CalcWeight(*param_, parent_sum, base_weight);

    auto left_weight = weight.Slice(1, linalg::All());
    auto left_sum =
        linalg::MakeVec(candidate.split.left_sum.data(), candidate.split.left_sum.size());
    CalcWeight(*param_, left_sum, param_->learning_rate, left_weight);

    auto right_weight = weight.Slice(2, linalg::All());
    auto right_sum =
        linalg::MakeVec(candidate.split.right_sum.data(), candidate.split.right_sum.size());
    CalcWeight(*param_, right_sum, param_->learning_rate, right_weight);

    p_tree->ExpandNode(candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
                       candidate.split.DefaultLeft(), base_weight, left_weight, right_weight);
    CHECK(p_tree->IsMultiTarget());
    auto left_child = p_tree->LeftChild(candidate.nid);
    CHECK_GT(left_child, candidate.nid);
    auto right_child = p_tree->RightChild(candidate.nid);
    CHECK_GT(right_child, candidate.nid);

    std::size_t n_nodes = p_tree->Size();
    gain_.resize(n_nodes);
    gain_[left_child] = CalcGainGivenWeight(*param_, left_sum, left_weight);
    gain_[right_child] = CalcGainGivenWeight(*param_, right_sum, right_weight);

    if (n_nodes >= stats_.Shape(0)) {
      stats_.Reshape(n_nodes * 2, stats_.Shape(1));
    }
    CHECK_EQ(stats_.Shape(1), n_targets);
    auto left_sum_stat = stats_.Slice(left_child, linalg::All());
    std::copy(candidate.split.left_sum.cbegin(), candidate.split.left_sum.cend(),
              linalg::begin(left_sum_stat));
    auto right_sum_stat = stats_.Slice(right_child, linalg::All());
    std::copy(candidate.split.right_sum.cbegin(), candidate.split.right_sum.cend(),
              linalg::begin(right_sum_stat));
  }

  explicit HistMultiEvaluator(Context const *ctx, MetaInfo const &info, TrainParam const *param,
                              std::shared_ptr<common::ColumnSampler> sampler)
      : param_{param},
        column_sampler_{std::move(sampler)},
        ctx_{ctx},
        is_col_split_{info.IsColumnSplit()} {
    interaction_constraints_.Configure(*param, info.num_col_);
    column_sampler_->Init(ctx, info.num_col_, info.feature_weights.HostVector(),
                          param_->colsample_bynode, param_->colsample_bylevel,
                          param_->colsample_bytree);
  }
};

/**
 * \brief CPU implementation of update prediction cache, which calculates the leaf value
 *        for the last tree and accumulates it to prediction vector.
 *
 * \param p_last_tree The last tree being updated by tree updater
 */
template <typename Partitioner>
void UpdatePredictionCacheImpl(Context const *ctx, RegTree const *p_last_tree,
                               std::vector<Partitioner> const &partitioner,
                               linalg::VectorView<float> out_preds) {
  auto const &tree = *p_last_tree;
  CHECK_EQ(out_preds.DeviceIdx(), Context::kCpuId);
  size_t n_nodes = p_last_tree->GetNodes().size();
  for (auto &part : partitioner) {
    CHECK_EQ(part.Size(), n_nodes);
    common::BlockedSpace2d space(
        part.Size(), [&](size_t node) { return part[node].Size(); }, 1024);
    common::ParallelFor2d(space, ctx->Threads(), [&](bst_node_t nidx, common::Range1d r) {
      if (!tree[nidx].IsDeleted() && tree[nidx].IsLeaf()) {
        auto const &rowset = part[nidx];
        auto leaf_value = tree[nidx].LeafValue();
        for (const size_t *it = rowset.begin + r.begin(); it < rowset.begin + r.end(); ++it) {
          out_preds(*it) += leaf_value;
        }
      }
    });
  }
}

template <typename Partitioner>
void UpdatePredictionCacheImpl(Context const *ctx, RegTree const *p_last_tree,
                               std::vector<Partitioner> const &partitioner,
                               linalg::MatrixView<float> out_preds) {
  CHECK_GT(out_preds.Size(), 0U);
  CHECK(p_last_tree);

  auto const &tree = *p_last_tree;
  if (!tree.IsMultiTarget()) {
    UpdatePredictionCacheImpl(ctx, p_last_tree, partitioner, out_preds.Slice(linalg::All(), 0));
    return;
  }

  auto const *mttree = tree.GetMultiTargetTree();
  auto n_nodes = mttree->Size();
  auto n_targets = tree.NumTargets();
  CHECK_EQ(out_preds.Shape(1), n_targets);
  CHECK_EQ(out_preds.DeviceIdx(), Context::kCpuId);

  for (auto &part : partitioner) {
    CHECK_EQ(part.Size(), n_nodes);
    common::BlockedSpace2d space(
        part.Size(), [&](size_t node) { return part[node].Size(); }, 1024);
    common::ParallelFor2d(space, ctx->Threads(), [&](bst_node_t nidx, common::Range1d r) {
      if (tree.IsLeaf(nidx)) {
        auto const &rowset = part[nidx];
        auto leaf_value = mttree->LeafValue(nidx);
        for (std::size_t const *it = rowset.begin + r.begin(); it < rowset.begin + r.end(); ++it) {
          for (std::size_t i = 0; i < n_targets; ++i) {
            out_preds(*it, i) += leaf_value(i);
          }
        }
      }
    });
  }
}
}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
