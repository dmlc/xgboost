/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
#define XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_

#include <algorithm>    // for copy
#include <array>        // for array
#include <cstddef>      // for size_t
#include <limits>       // for numeric_limits
#include <memory>       // for shared_ptr
#include <numeric>      // for iota
#include <type_traits>  // for is_same_v
#include <utility>      // for move
#include <vector>       // for vector

#include "../../common/categorical.h"  // for CatBitField
#include "../../common/hist_util.h"    // for GHistRow, HistogramCuts
#include "../../common/linalg_op.h"    // for cbegin, cend, begin
#include "../../common/random.h"       // for ColumnSampler
#include "../constraints.h"            // for FeatureInteractionConstraintHost
#include "../param.h"                  // for TrainParam
#include "../split_evaluator.h"        // for TreeEvaluator
#include "expand_entry.h"              // for MultiExpandEntry
#include "hist_cache.h"                // for BoundedHistCollection
#include "xgboost/base.h"              // for bst_node_t, bst_target_t, bst_feature_t
#include "xgboost/context.h"           // for Context
#include "xgboost/linalg.h"            // for Constants, Vector
#include "xgboost/tree_model.h"        // for RegTree

namespace xgboost::tree {
/** @brief CPU histogram split evaluation for scalar and vector tree candidates. */
template <typename ExpandEntryT>
class HistEvaluatorImpl {
  using SplitEntryT = decltype(ExpandEntryT{}.split);
  static constexpr bool kScalar = std::is_same_v<ExpandEntryT, CPUExpandEntry>;

  std::vector<double> gain_;
  linalg::Matrix<GradientPairPrecise> stats_;
  TrainParam const *param_;
  TreeEvaluator tree_evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  Context const *ctx_;

 private:
  // Enumerate/Scan the split values of specific feature
  template <bst_bin_t d_step, bool kSingleTarget>
  bool EnumerateSplitImpl(common::HistogramCuts const &cut, bst_feature_t fidx,
                          common::Span<common::ConstGHistRow> hist,
                          linalg::VectorView<GradientPairPrecise const> parent_sum,
                          double parent_gain, bst_node_t nidx,
                          TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                          SplitEntryT *p_best) const {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");
    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();
    CHECK_LE(cut_ptr[fidx], static_cast<uint32_t>(std::numeric_limits<bst_bin_t>::max()));
    CHECK_LE(cut_ptr[fidx + 1], static_cast<uint32_t>(std::numeric_limits<bst_bin_t>::max()));

    auto scan = [&](auto &sums) {
      // Construct views here to keep the one-target extent visible to the compiler.
      auto [left_sum, right_sum] = [&] {
        if constexpr (kSingleTarget) {
          return std::make_pair(linalg::MakeVec(sums.data(), 1),
                                linalg::MakeVec(sums.data() + 1, 1));
        } else {
          return std::make_pair(sums.Slice(0, linalg::All()), sums.Slice(1, linalg::All()));
        }
      }();
      SplitEntryT local_best;
      // Scalar candidates are cheap to merge. Reuse vector candidate storage across features.
      auto *best = kScalar ? &local_best : p_best;
      auto n_targets = kSingleTarget ? 1 : hist.size();
      // ibegin, iend: smallest/largest cut points for feature fid use int to allow for
      // value -1
      bst_bin_t ibegin, iend;
      if (d_step > 0) {
        ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
        iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
      } else {
        ibegin = static_cast<bst_bin_t>(cut_ptr[fidx + 1]) - 1;
        iend = static_cast<bst_bin_t>(cut_ptr[fidx]) - 1;
      }
      for (bst_bin_t i = ibegin; i != iend; i += d_step) {
        for (bst_target_t t = 0; t < n_targets; ++t) {
          auto t_hist = hist[t];
          auto t_p = parent_sum(t);
          left_sum(t) += t_hist[i];
          right_sum(t) = t_p - left_sum(t);
        }

        if (d_step > 0) {
          // forward enumeration: split at right bound of each bin
          auto split_pt = cut_val[i];
          auto loss_chg =
              evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
          best->Update(loss_chg, fidx, split_pt, d_step == -1, false, left_sum, right_sum);
        } else {
          // backward enumeration: split at left bound of each bin
          auto split_pt = common::HistogramCuts::NumericBinLowerBound(cut_ptr, cut_val, fidx, i);
          auto loss_chg =
              evaluator.CalcSplitGain(*param_, nidx, fidx, right_sum, left_sum) - parent_gain;
          best->Update(loss_chg, fidx, split_pt, d_step == -1, false, right_sum, left_sum);
        }
      }
      if constexpr (kScalar) {
        p_best->Update(local_best);
      }
      // if sum of statistics for non-missing values in the node
      // is equal to sum of statistics for all values:
      // then - there are no missing values
      // else - there are missing values
      // return true if there's missing. Doesn't handle floating-point error well.
      if (d_step == +1) {
        return !std::equal(linalg::cbegin(left_sum), linalg::cend(left_sum),
                           linalg::cbegin(parent_sum));
      }
      return false;
    };

    if constexpr (kSingleTarget) {
      std::array<GradientPairPrecise, 2> sums{};
      return scan(sums);
    } else {
      auto sums = linalg::Empty<GradientPairPrecise>(ctx_, 2, hist.size());
      return scan(sums);
    }
  }

  template <bst_bin_t d_step>
  bool EnumerateSplit(common::HistogramCuts const &cut, bst_feature_t fidx,
                      common::Span<common::ConstGHistRow> hist,
                      linalg::VectorView<GradientPairPrecise const> parent_sum, double parent_gain,
                      bst_node_t nidx, TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                      SplitEntryT *p_best) const {
    if (kScalar || hist.size() == 1) {
      return this->EnumerateSplitImpl<d_step, true>(cut, fidx, hist, parent_sum, parent_gain, nidx,
                                                    evaluator, p_best);
    }
    return this->EnumerateSplitImpl<d_step, false>(cut, fidx, hist, parent_sum, parent_gain, nidx,
                                                   evaluator, p_best);
  }

  /**
   * \brief Use learned direction with one-hot split. Other implementations (LGB) create a
   *        pseudo-category for missing value but here we just do a complete scan to avoid
   *        making specialized histogram bin.
   */
  void EnumerateOneHot(common::HistogramCuts const &cut, bst_feature_t fidx,
                       common::Span<common::ConstGHistRow> hist,
                       linalg::VectorView<GradientPairPrecise const> parent_sum, double parent_gain,
                       bst_node_t nidx, TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                       SplitEntryT *p_best) const {
    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();

    bst_bin_t ibegin = static_cast<bst_bin_t>(cut_ptr[fidx]);
    bst_bin_t iend = static_cast<bst_bin_t>(cut_ptr[fidx + 1]);
    bst_bin_t n_bins = iend - ibegin;
    auto n_targets = kScalar ? 1 : hist.size();

    std::array<GradientPairPrecise, 3> single_sum{};
    std::vector<GradientPairPrecise> sum(n_targets == 1 ? 0 : 3 * n_targets);
    auto *data = n_targets == 1 ? single_sum.data() : sum.data();
    auto left_sum = linalg::MakeVec(data, n_targets);
    auto right_sum = linalg::MakeVec(data + n_targets, n_targets);

    // Per-target missing gradient: parent_sum - sum_of_all_bins.
    auto missing = linalg::MakeVec(data + 2 * n_targets, n_targets);
    for (bst_target_t t = 0; t < n_targets; ++t) {
      auto f_hist = hist[t].subspan(cut_ptr[fidx], n_bins);
      GradientPairPrecise feature_sum{};
      for (bst_bin_t b = 0; b < n_bins; ++b) {
        feature_sum += f_hist[b];
      }
      missing(t) = parent_sum(t) - feature_sum;
    }

    SplitEntryT best;
    best.is_cat = false;

    for (bst_bin_t i = ibegin; i != iend; ++i) {
      auto split_pt = cut_val[i];

      // Evaluate missing values with the other categories, then with this category.
      for (bool default_left : {true, false}) {
        for (bst_target_t t = 0; t < n_targets; ++t) {
          right_sum(t) = hist[t][i];
          if (!default_left) {
            right_sum(t) += missing(t);
          }
          left_sum(t) = parent_sum(t) - right_sum(t);
        }
        auto gain = evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
        best.Update(gain, fidx, split_pt, default_left, true, left_sum, right_sum);
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
   * @brief Enumerate with partition-based splits.
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
  template <bst_bin_t d_step>
  void EnumeratePart(common::HistogramCuts const &cut, common::Span<size_t const> sorted_idx,
                     common::Span<common::ConstGHistRow> hist, bst_feature_t fidx, bst_node_t nidx,
                     TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator,
                     SplitEntryT *p_best) {
    static_assert(d_step == +1 || d_step == -1, "Invalid step.");
    auto n_targets = kScalar ? 1 : hist.size();

    auto const &cut_ptr = cut.Ptrs();
    auto const &cut_val = cut.Values();
    auto parent_sum = stats_.Slice(nidx, linalg::All());
    auto parent_gain = gain_[nidx];

    bst_bin_t f_begin = cut_ptr[fidx];
    bst_bin_t f_end = cut_ptr[fidx + 1];
    bst_bin_t n_bins_feature{f_end - f_begin};
    auto n_bins = std::min(param_->max_cat_threshold, n_bins_feature);

    bst_bin_t it_begin, it_end;
    if (d_step > 0) {
      it_begin = f_begin;
      it_end = it_begin + n_bins - 1;
    } else {
      it_begin = f_end - 1;
      it_end = it_begin - n_bins + 1;
    }

    std::array<GradientPairPrecise, 2> single_sum{};
    std::vector<GradientPairPrecise> sum(n_targets == 1 ? 0 : 2 * n_targets);
    auto *data = n_targets == 1 ? single_sum.data() : sum.data();
    auto left_sum = linalg::MakeVec(data, n_targets);
    auto right_sum = linalg::MakeVec(data + n_targets, n_targets);

    SplitEntryT best;
    bst_bin_t best_thresh{-1};
    for (bst_bin_t i = it_begin; i != it_end; i += d_step) {
      auto j = i - f_begin;  // index local to current feature
      for (bst_target_t t = 0; t < n_targets; ++t) {
        auto f_hist = hist[t].subspan(f_begin, n_bins_feature);
        if (d_step == 1) {
          right_sum(t) += f_hist[sorted_idx[j]];
          left_sum(t) = parent_sum(t) - right_sum(t);  // missing on left
        } else {
          left_sum(t) += f_hist[sorted_idx[j]];
          right_sum(t) = parent_sum(t) - left_sum(t);  // missing on right
        }
      }

      auto loss_chg =
          evaluator.CalcSplitGain(*param_, nidx, fidx, left_sum, right_sum) - parent_gain;
      // We don't have a numeric split point, nan here is a dummy split.
      if (best.Update(loss_chg, fidx, std::numeric_limits<float>::quiet_NaN(), d_step == 1, true,
                      left_sum, right_sum)) {
        best_thresh = i;
      }
    }

    if (best_thresh != -1) {
      auto n = common::CatBitField::ComputeStorageSize(n_bins_feature);
      best.cat_bits = decltype(best.cat_bits)(n, 0);
      common::CatBitField cat_bits{best.cat_bits};
      bst_bin_t partition = d_step == 1 ? (best_thresh - it_begin + 1) : (best_thresh - f_begin);
      CHECK_GT(partition, 0);
      std::for_each(sorted_idx.begin(), sorted_idx.begin() + partition, [&](std::size_t c) {
        auto cat = cut_val[c + f_begin];
        cat_bits.Set(cat);
      });
    }

    p_best->Update(best);
  }

 public:
  void EvaluateSplits(BoundedHistCollection const &hist, common::HistogramCuts const &cut,
                      common::Span<FeatureType const> feature_types,
                      std::vector<ExpandEntryT> *entries) {
    BoundedHistCollection const *hist_ptr = &hist;
    this->EvaluateSplits({&hist_ptr, 1}, cut, feature_types, entries);
  }

  void EvaluateSplits(common::Span<const BoundedHistCollection *> hist,
                      common::HistogramCuts const &cut,
                      common::Span<FeatureType const> feature_types,
                      std::vector<ExpandEntryT> *p_entries) {
    auto &entries = *p_entries;
    CHECK_EQ(stats_.Shape(1), hist.size());
    CHECK(!hist.empty());
    std::vector<std::shared_ptr<HostDeviceVector<bst_feature_t>>> features(entries.size());

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      features[nidx_in_set] = column_sampler_->GetFeatureSet(ctx_, entries[nidx_in_set].depth);
    }
    CHECK(!features.empty());

    std::int32_t n_threads = ctx_->Threads();
    std::size_t const grain_size = std::max<std::size_t>(1, features.front()->Size() / n_threads);
    common::BlockedSpace2d space{
        entries.size(), [&](std::size_t nidx_in_set) { return features[nidx_in_set]->Size(); },
        grain_size};

    std::vector<ExpandEntryT> tloc_candidates(n_threads * entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
      for (std::int32_t j = 0; j < n_threads; ++j) {
        tloc_candidates[i * n_threads + j] = entries[i];
      }
    }

    auto evaluator = tree_evaluator_.GetEvaluator();
    common::ParallelFor2d(
        space, n_threads,
        common::WithWorker([&](std::size_t nidx_in_set, common::Range1d r, common::Worker worker) {
          auto tidx = worker.Id();
          auto entry = &tloc_candidates[n_threads * nidx_in_set + tidx];
          auto best = &entry->split;

          auto parent_sum = stats_.Slice(entry->nid, linalg::All());
          auto n_targets = kScalar ? 1 : hist.size();
          std::conditional_t<kScalar, std::array<common::ConstGHistRow, 1>,
                             std::vector<common::ConstGHistRow>>
              node_hist;
          if constexpr (!kScalar) {
            node_hist.resize(n_targets);
          }
          for (std::size_t t = 0; t < n_targets; ++t) {
            node_hist[t] = (*hist[t])[entry->nid];
          }
          auto features_set = features[nidx_in_set]->ConstHostSpan();
          // Parent category weights are only needed for partition-based splits.
          std::array<float, 1> single_weight{};
          std::vector<float> base_weight;
          bool have_base_weight = false;

          for (auto fidx_in_set = r.begin(); fidx_in_set < r.end(); fidx_in_set++) {
            auto fidx = features_set[fidx_in_set];
            if (!interaction_constraints_.Query(entry->nid, fidx)) {
              continue;
            }
            auto parent_gain = gain_[entry->nid];
            bool is_cat = common::IsCat(feature_types, fidx);
            // Numeric split
            if (!is_cat) {
              bool missing = this->EnumerateSplit<+1>(cut, fidx, node_hist, parent_sum, parent_gain,
                                                      entry->nid, evaluator, best);
              if (missing) {
                this->EnumerateSplit<-1>(cut, fidx, node_hist, parent_sum, parent_gain, entry->nid,
                                         evaluator, best);
              }
              continue;
            }

            auto const &cut_ptr = cut.Ptrs();
            auto n_bins = cut_ptr.at(fidx + 1) - cut_ptr[fidx];
            // One hot split
            if (common::UseOneHot(n_bins, param_->max_cat_to_onehot)) {
              this->EnumerateOneHot(cut, fidx, node_hist, parent_sum, parent_gain, entry->nid,
                                    evaluator, best);
              continue;
            }

            // Partition split
            if (!have_base_weight) {
              if (n_targets != 1) {
                base_weight.resize(n_targets);
              }
              evaluator.CalcWeightCat(
                  *param_, parent_sum,
                  linalg::MakeVec(n_targets == 1 ? single_weight.data() : base_weight.data(),
                                  n_targets));
              have_base_weight = true;
            }
            auto h_bw = linalg::MakeVec(n_targets == 1 ? single_weight.data() : base_weight.data(),
                                        n_targets);
            std::vector<size_t> sorted_idx(n_bins);
            std::iota(sorted_idx.begin(), sorted_idx.end(), 0ul);
            linalg::Vector<GradientPairPrecise> grads({n_targets}, this->ctx_->Device());
            auto h_grads = grads.HostView();
            std::vector<float> child_w(n_targets, .0f);

            // Sort by s_c = w_p^T w_c
            // Project onto the parent's update direction to compare scores l_s < r_s
            // Since we care only about the ordering, no need to divide the norm.
            std::vector<double> scores(n_bins);
            for (std::size_t bin_idx = 0; bin_idx < n_bins; ++bin_idx) {
              for (decltype(n_targets) t_idx = 0; t_idx < n_targets; ++t_idx) {
                auto f_hist = node_hist[t_idx].subspan(cut_ptr[fidx], n_bins);
                h_grads(t_idx) = f_hist[bin_idx];
              }
              evaluator.CalcWeightCat(*param_, h_grads,
                                      linalg::MakeVec(child_w.data(), child_w.size()));
              double sc = .0;
              for (decltype(n_targets) t_idx = 0; t_idx < n_targets; ++t_idx) {
                sc += h_bw(t_idx) * child_w[t_idx];
              }
              // With one target a zero parent weight loses all category ordering.
              scores[bin_idx] = n_targets == 1 && h_bw(0) == 0.0f ? child_w[0] : sc;
            }

            std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                             [&](std::size_t l, std::size_t r) { return scores[l] < scores[r]; });

            this->EnumeratePart<+1>(cut, sorted_idx, node_hist, fidx, entry->nid, evaluator, best);
            this->EnumeratePart<-1>(cut, sorted_idx, node_hist, fidx, entry->nid, evaluator, best);
          }
        }));

    for (std::size_t nidx_in_set = 0; nidx_in_set < entries.size(); ++nidx_in_set) {
      for (auto tidx = 0; tidx < n_threads; ++tidx) {
        entries[nidx_in_set].split.Update(tloc_candidates[n_threads * nidx_in_set + tidx].split);
      }
    }
  }

  float InitRoot(GradStats const &root_sum) {
    GradientPairPrecise sum{root_sum.GetGrad(), root_sum.GetHess()};
    auto weight = this->InitRoot(linalg::MakeVec(&sum, 1));
    return weight(0);
  }

  linalg::Vector<float> InitRoot(linalg::VectorView<GradientPairPrecise const> root_sum) {
    auto n_targets = root_sum.Size();
    if constexpr (kScalar) {
      CHECK_EQ(n_targets, 1);
    }
    stats_ = linalg::Constant(ctx_, GradientPairPrecise{}, 1, n_targets);
    gain_.resize(1);

    linalg::Vector<float> weight({n_targets}, ctx_->Device());
    auto evaluator = tree_evaluator_.GetEvaluator();
    evaluator.CalcWeight(RegTree::kRoot, *param_, root_sum, weight.HostView());
    auto root_gain = evaluator.CalcGainGivenWeight(*param_, root_sum, weight.HostView());
    gain_.front() = root_gain;

    auto h_stats = stats_.HostView();
    std::copy(linalg::cbegin(root_sum), linalg::cend(root_sum), linalg::begin(h_stats));

    return weight;
  }

  // Add splits to tree, handles all statistic
  void ApplyTreeSplit(ExpandEntryT const &candidate, RegTree *p_tree) {
    // Use the split gradient's number of targets for intermediate weights
    // This may differ from p_tree->NumTargets() when using reduced gradient
    std::array<GradientPairPrecise, 2> single_sum{};
    auto [left_sum, right_sum] = [&] {
      if constexpr (kScalar) {
        single_sum[0] = {candidate.split.left_sum.GetGrad(), candidate.split.left_sum.GetHess()};
        single_sum[1] = {candidate.split.right_sum.GetGrad(), candidate.split.right_sum.GetHess()};
        return std::make_pair(linalg::MakeVec(single_sum.data(), 1),
                              linalg::MakeVec(single_sum.data() + 1, 1));
      } else {
        return std::make_pair(
            linalg::MakeVec(candidate.split.left_sum.data(), candidate.split.left_sum.size()),
            linalg::MakeVec(candidate.split.right_sum.data(), candidate.split.right_sum.size()));
      }
    }();
    auto n_split_targets = left_sum.Size();
    auto parent_sum = stats_.Slice(candidate.nid, linalg::All());

    auto evaluator = tree_evaluator_.GetEvaluator();
    std::array<float, 3> single_weight{};
    std::vector<float> weight(n_split_targets == 1 ? 0 : 3 * n_split_targets);
    auto *data = n_split_targets == 1 ? single_weight.data() : weight.data();
    auto base_weight = linalg::MakeVec(data, n_split_targets);
    auto left_weight = linalg::MakeVec(data + n_split_targets, n_split_targets);
    auto right_weight = linalg::MakeVec(data + 2 * n_split_targets, n_split_targets);
    evaluator.CalcWeight(candidate.nid, *param_, parent_sum, base_weight);
    evaluator.CalcSplitWeights(*param_, candidate.nid, candidate.split.SplitIndex(), left_sum,
                               right_sum, left_weight, right_weight);

    // Compute the loss_chg and sum hessians for parent and children
    float loss_chg = candidate.split.loss_chg;
    // Sum hessians across all targets for each child
    double left_sum_hess = 0.0, right_sum_hess = 0.0;
    for (std::size_t t = 0; t < n_split_targets; ++t) {
      left_sum_hess += left_sum(t).GetHess();
      right_sum_hess += right_sum(t).GetHess();
    }

    common::Span<CatWordT const> cat_bits;
    if (candidate.split.is_cat) {
      cat_bits = candidate.split.cat_bits;
    }
    SplitInfo split{candidate.nid, candidate.split.SplitIndex(), candidate.split.split_value,
                    candidate.split.DefaultLeft(), cat_bits};
    if constexpr (kScalar) {
      CHECK(!p_tree->IsMultiTarget());
      p_tree->Expand({split,
                      {base_weight(0), left_sum_hess + right_sum_hess},
                      {left_weight(0), left_sum_hess},
                      {right_weight(0), right_sum_hess},
                      loss_chg});
    } else {
      CHECK(p_tree->IsMultiTarget());
      p_tree->Expand(ctx_, ExpandBatch{{split,
                                        {base_weight.Values(), left_sum_hess + right_sum_hess},
                                        {left_weight.Values(), left_sum_hess},
                                        {right_weight.Values(), right_sum_hess},
                                        loss_chg}});
    }
    auto left_child = p_tree->LeftChild(candidate.nid);
    CHECK_GT(left_child, candidate.nid);
    auto right_child = p_tree->RightChild(candidate.nid);
    CHECK_GT(right_child, candidate.nid);

    // Set up child constraints
    tree_evaluator_.AddSplit(candidate.nid, left_child, right_child, candidate.split.SplitIndex(),
                             left_weight, right_weight);
    evaluator = tree_evaluator_.GetEvaluator();
    interaction_constraints_.Split(candidate.nid, candidate.split.SplitIndex(), left_child,
                                   right_child);

    std::size_t n_nodes = p_tree->Size();
    gain_.resize(n_nodes);
    gain_[left_child] = evaluator.CalcGainGivenWeight(*param_, left_sum, left_weight);
    gain_[right_child] = evaluator.CalcGainGivenWeight(*param_, right_sum, right_weight);

    if (n_nodes >= stats_.Shape(0)) {
      stats_.Reshape(n_nodes * 2, stats_.Shape(1));
    }
    CHECK_EQ(stats_.Shape(1), n_split_targets);
    auto left_sum_stat = stats_.Slice(left_child, linalg::All());
    std::copy(linalg::cbegin(left_sum), linalg::cend(left_sum), linalg::begin(left_sum_stat));
    auto right_sum_stat = stats_.Slice(right_child, linalg::All());
    std::copy(linalg::cbegin(right_sum), linalg::cend(right_sum), linalg::begin(right_sum_stat));
  }

  // The column sampler must be constructed by caller since we need to preserve the rng
  // for the entire training session.
  explicit HistEvaluatorImpl(Context const *ctx, TrainParam const *param, MetaInfo const &info,
                             std::shared_ptr<common::ColumnSampler> sampler)
      : HistEvaluatorImpl{ctx, info, param, 1, std::move(sampler)} {}

  explicit HistEvaluatorImpl(Context const *ctx, MetaInfo const &info, TrainParam const *param,
                             bst_target_t n_targets, std::shared_ptr<common::ColumnSampler> sampler)
      : param_{param},
        tree_evaluator_{*param, static_cast<bst_feature_t>(info.num_col_), DeviceOrd::CPU(),
                        n_targets},
        column_sampler_{std::move(sampler)},
        ctx_{ctx} {
    interaction_constraints_.Configure(*param, info.num_col_);
    column_sampler_->Init(ctx, info.num_col_, info.feature_weights, param_->colsample_bynode,
                          param_->colsample_bylevel, param_->colsample_bytree);
  }

  [[nodiscard]] double Gain(bst_node_t nidx) const { return gain_.at(nidx); }
  [[nodiscard]] auto Evaluator() const { return tree_evaluator_.GetEvaluator(); }
};

using HistEvaluator = HistEvaluatorImpl<CPUExpandEntry>;
using HistMultiEvaluator = HistEvaluatorImpl<MultiExpandEntry>;

}  // namespace xgboost::tree
#endif  // XGBOOST_TREE_HIST_EVALUATE_SPLITS_H_
