/*!
 * Copyright 2020 by XGBoost Contributors
 * \file updater_exact.cc
 */
#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>

#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"
#include "xgboost/span.h"
#include "xgboost/base.h"
#include "xgboost/json.h"

#include "param.h"
#include "updater_exact.h"

namespace xgboost {

namespace {
template <typename GradientT>
void SetSubstract(GradientT const &lhs, GradientT const &rhs, GradientT *out) {
  auto* out_gvec = &out->GetGrad()[0];
  auto* out_hvec = &out->GetHess()[0];
  auto const* l_gvec = &lhs.GetGrad()[0];
  auto const* l_hvec = &lhs.GetHess()[0];
  auto const* r_gvec = &rhs.GetGrad()[0];
  auto const* r_hvec = &rhs.GetHess()[0];
  size_t const size = lhs.GetGrad().Size();
  for (size_t i = 0; i < size; i++) {
    out_gvec[i] = l_gvec[i] - r_gvec[i];
    out_hvec[i] = l_hvec[i] - r_hvec[i];
  }
}

template <>
void SetSubstract<SingleGradientPair>(SingleGradientPair const &lhs,
                                      SingleGradientPair const &rhs,
                                      SingleGradientPair *out) {
  out->GetGrad().vec = lhs.GetGrad().vec - rhs.GetGrad().vec;
  out->GetHess().vec = lhs.GetHess().vec - rhs.GetHess().vec;
}
}  // anonymous namespace

namespace tree {

template <typename GradientT>
void MultiExact<GradientT>::InitData(DMatrix *data,
                                     common::Span<GradientPair const> gpairs, size_t targets) {
  monitor_.Start(__func__);
  this->targets_ = targets;
  this->positions_.clear();
  this->is_splitable_.clear();
  this->nodes_split_.clear();
  this->node_shift_ = 0;

  CHECK_EQ(gpairs.size(), data->Info().num_row_ * this->targets_);
  gpairs_ = std::vector<GradientT>(gpairs.size() / this->targets_,
                                   MakeGradientPair<GradientT>(this->targets_));
  CHECK_EQ(gpairs_.size(), data->Info().num_row_);
  is_splitable_.resize(param_.MaxNodes(), 1);

  tloc_scans_.resize(omp_get_max_threads());
  for (auto& scan : tloc_scans_) {
    scan.resize(param_.MaxNodes());
  }

  auto subsample = param_.subsample;

  // Get a vectorized veiw of gradients.
  for (size_t i = 0; i < data->Info().num_row_; ++i) {
    size_t beg = i * this->targets_;
    size_t end = beg + this->targets_;
    auto &vec = gpairs_[i];
    for (size_t j = beg; j < end; ++j) {
      vec.GetGrad()[j - beg] = gpairs[j].GetGrad();
      vec.GetHess()[j - beg] = gpairs[j].GetHess();
    }
  }

  if (subsample != 1.0) {
    size_t targets = this->targets_;
    std::bernoulli_distribution flip(subsample);
    auto &rnd = common::GlobalRandom();
    std::transform(gpairs_.begin(), gpairs_.end(), gpairs_.begin(),
                   [&flip, &rnd, targets](GradientT &g) {
                     if (!flip(rnd)) {
                       return MakeGradientPair<GradientT>(targets);
                     }
                     return g;
                   });
  }

  sampler_.Init(data->Info().num_col_, param_.colsample_bynode,
                param_.colsample_bylevel, param_.colsample_bytree);

  value_constraints_.Init(param_, targets_, &monotone_constriants_);
  monitor_.Stop(__func__);
}

template <typename GradientT>
void MultiExact<GradientT>::InitRoot(DMatrix *data, RegTree *tree) {
  monitor_.Start(__func__);
  GradientT root_sum {MakeGradientPair<GradientT>(tree->LeafSize())};
  root_sum =
      XGBOOST_PARALLEL_ACCUMULATE(gpairs_.cbegin(), gpairs_.cend(), root_sum,
                                  std::plus<GradientT>{});

  auto weight = value_constraints_.CalcWeight(root_sum, RegTree::kRoot, param_);
  tree->SetLeaf((weight * param_.learning_rate).vec, RegTree::kRoot,
                root_sum.GetHess().vec);

  positions_.resize(data->Info().num_row_);
  std::fill(positions_.begin(), positions_.end(), RegTree::kRoot);
  auto gain = MultiCalcGainGivenWeight(root_sum.GetGrad(),
                                       root_sum.GetHess(),
                                       weight, param_);
  SplitEntry root{RegTree::kRoot, root_sum, gain, param_};
  nodes_split_.push_back(root);

  auto p_feature_set = sampler_.GetFeatureSet(0);
  this->EvaluateSplit(data, p_feature_set->HostSpan());
  monitor_.Stop(__func__);
}

template <typename GradientT>
void MultiExact<GradientT>::EvaluateFeature(bst_feature_t fid,
                                            SparsePage::Inst const &column,
                                            std::vector<SplitEntry> *p_scans,
                                            std::vector<SplitEntry> *p_nodes) const {
  auto update_node = [fid, this](bool forward, SplitEntry const &scan,
                                 float fcond, float bcond, SplitEntry *node) {
    if (forward) {
      float loss_chg = value_constraints_.CalcSplitGain(
                           scan.candidate.left_sum, scan.candidate.right_sum,
                           node->nidx, fid, param_) -
                       node->root_gain;
      node->candidate.Update(loss_chg, fid, fcond, !forward,
                             scan.candidate.left_sum, scan.candidate.right_sum);
    } else {
      float loss_chg = value_constraints_.CalcSplitGain(
                           scan.candidate.right_sum, scan.candidate.left_sum,
                           node->nidx, fid, param_) -
                       node->root_gain;
      node->candidate.Update(loss_chg, fid, bcond, !forward,
                             scan.candidate.right_sum, scan.candidate.left_sum);
    }
  };

  auto search_kernel = [fid, this, p_scans, p_nodes, update_node](
                           Entry const *const beg, Entry const *const end,
                           bool const forward) {
    auto const inc = forward ? 1 : -1;
    auto& node_scans = *p_scans;
    size_t targets { this->targets_ };
    for (size_t i = node_shift_; i < nodes_split_.size(); ++i) {
      auto& scan = node_scans[i];
      scan.candidate.left_sum = MakeGradientPair<GradientT>(targets);
      scan.candidate.right_sum = MakeGradientPair<GradientT>(targets);
      scan.last_value = std::numeric_limits<float>::quiet_NaN();
    }

    auto &node_splits = *p_nodes;
    auto const* p_gpairs = gpairs_.data();
    auto const* p_positions = positions_.data();

    for (auto it = beg; it != end; it += inc) {
      bst_node_t const row_nidx = p_positions[it->index];
      if (is_splitable_[row_nidx] == 0 ||
          !interaction_constraints_.Query(row_nidx, fid)) {
        continue;
      }
      SplitEntry &scan = node_scans[row_nidx];
      if (AnyLT(scan.candidate.left_sum.GetHess(), param_.min_child_weight) ||
          it->fvalue == scan.last_value || std::isnan(scan.last_value)) {
        scan.Accumulate(p_gpairs[it->index], it->fvalue);
        continue;
      }

      SplitEntry &node = node_splits[row_nidx];
      SetSubstract(node.parent_sum, scan.candidate.left_sum,
                   &scan.candidate.right_sum);

      if (AnyLT(scan.candidate.right_sum.GetHess(), param_.min_child_weight)) {
        scan.Accumulate(p_gpairs[it->index], it->fvalue);
        continue;
      }

      float const cond = (it->fvalue + scan.last_value) * 0.5f;
      update_node(forward, scan, cond, cond, &node);
      scan.Accumulate(p_gpairs[it->index], it->fvalue);
    }

    // Try to use all statistic from current column.
    size_t const n_nodes = node_splits.size();
    for (size_t n = node_shift_; n < n_nodes; ++n) {
      auto &node = node_splits[n];
      auto &scan = node_scans[n];
      SetSubstract(node.parent_sum, scan.candidate.left_sum,
                   &scan.candidate.right_sum);
      if (AnyLT(scan.candidate.left_sum.GetHess(), param_.min_child_weight) ||
          AnyLT(scan.candidate.right_sum.GetHess(), param_.min_child_weight)) {
        continue;
      }
      bst_float const gap = std::abs(scan.last_value) + kRtEps;
      update_node(forward, scan, gap, -gap, &node);
    }
  };

  CHECK_LE(p_nodes->size(), param_.MaxNodes());
  if (NeedForward(column, param_)) {
    search_kernel(column.data(), column.data() + column.size(), true);
  }

  if (NeedBackward(column, param_)) {
    search_kernel(column.data() + column.size() - 1, column.data() - 1, false);
  }
}

template <typename GradientT>
void MultiExact<GradientT>::EvaluateSplit(
    DMatrix *data, common::Span<bst_feature_t const> features) {
  monitor_.Start(__func__);
  for (auto const &batch : data->GetBatches<SortedCSCPage>()) {
    CHECK_EQ(batch.Size(), data->Info().num_col_);
    std::vector<std::vector<SplitEntry>> tloc_splits(omp_get_max_threads());
    for (auto& s : tloc_splits) {
      s = nodes_split_;
    }

    dmlc::OMPException omp_handler;
#pragma omp parallel for schedule(dynamic)
    for (omp_ulong f = 0; f < features.size(); ++f) {  // NOLINT
      omp_handler.Run([&]() {
        auto fid = features[f];
        auto const &column = batch[fid];
        auto& splits = tloc_splits.at(omp_get_thread_num());
        auto& node_scans = tloc_scans_.at(omp_get_thread_num());
        this->EvaluateFeature(fid, column, &node_scans, &splits);
      });
    }
    omp_handler.Rethrow();

    for (auto const& splits : tloc_splits) {
      for (size_t i = node_shift_; i < splits.size(); ++i) {
        nodes_split_.at(splits.at(i).nidx).candidate.Update(splits.at(i).candidate);
      }
    }
  }
  monitor_.Stop(__func__);
}

template <typename GradientT>
size_t MultiExact<GradientT>::ExpandTree(RegTree *p_tree,
                                         std::vector<SplitEntry> *next) {
  auto& pending = *next;
  auto &tree = *p_tree;
  size_t max_node { 0 };
  auto const leaves = tree.GetNumLeaves();

  for (size_t n = node_shift_; n < nodes_split_.size(); ++n) {
    SplitEntry& split = nodes_split_.at(n);
    auto weight =
        value_constraints_.CalcWeight(split.parent_sum, split.nidx, param_);
    if (!split.IsValid(tree.GetDepth(split.nidx), leaves, param_)) {
      tree.SetLeaf((weight * param_.learning_rate).vec, split.nidx,
                   split.parent_sum.GetHess().vec);
      CHECK_EQ(is_splitable_[split.nidx], 1);
      is_splitable_[split.nidx] = 0;
      continue;
    }

    CHECK_NE(split.candidate.left_sum.GetGrad().Size(), 0);
    auto left_weight = value_constraints_.CalcWeight(split.candidate.left_sum,
                                                     split.nidx, param_);
    CHECK_NE(split.candidate.right_sum.GetGrad().Size(), 0);
    auto right_weight = value_constraints_.CalcWeight(split.candidate.right_sum,
                                                      split.nidx, param_);
    tree.ExpandNode(split.nidx,
                    split.candidate.SplitIndex(),
                    split.candidate.split_value,
                    split.candidate.DefaultLeft(),
                    weight.vec,
                    (left_weight * param_.learning_rate).vec,
                    (right_weight * param_.learning_rate).vec,
                    split.candidate.loss_chg,
                    split.parent_sum.GetHess().vec,
                    split.candidate.left_sum.GetHess().vec,
                    split.candidate.right_sum.GetHess().vec);
    auto left = tree[split.nidx].LeftChild();
    auto right = tree[split.nidx].RightChild();
    interaction_constraints_.Split(split.nidx, split.candidate.SplitIndex(), left, right);
    value_constraints_.Split(split.nidx, left, left_weight, right, right_weight,
                             split.candidate.SplitIndex());

    if (SplitEntry::ChildIsValid(tree.GetDepth(left), leaves, param_)) {
      auto gain = MultiCalcGainGivenWeight(split.candidate.left_sum.GetGrad(),
                                           split.candidate.left_sum.GetHess(),
                                           left_weight, param_);
      SplitEntry s { left, split.candidate.left_sum, gain, param_ };
      pending.push_back(s);
      CHECK_EQ(is_splitable_[left], 1);
      max_node = std::max(max_node, static_cast<size_t>(left));
    } else {
      is_splitable_[left] = 0;
    }
    if (SplitEntry::ChildIsValid(tree.GetDepth(right), leaves, param_)) {
      auto gain = MultiCalcGainGivenWeight(split.candidate.right_sum.GetGrad(),
                                           split.candidate.right_sum.GetHess(),
                                           right_weight, param_);
      SplitEntry s { right, split.candidate.right_sum, gain, param_ };
      pending.push_back(s);
      CHECK_EQ(is_splitable_[right], 1);
      max_node = std::max(max_node, static_cast<size_t>(right));
    } else {
      is_splitable_[right] = 0;
    }
  }
  return max_node;
}

template <typename GradientT>
void MultiExact<GradientT>::ApplySplit(DMatrix *m, RegTree *p_tree) {
  monitor_.Start(__func__);
  auto &tree = *p_tree;
  decltype(nodes_split_) pending;
  auto max_node = this->ExpandTree(p_tree, &pending);

  // Fill in non-missing values.
  std::vector<bst_feature_t> fsplits;
  for (size_t i = node_shift_; i < nodes_split_.size(); ++i) {
    auto const& split = nodes_split_.at(i);
    if (!tree[split.nidx].IsLeaf()) {
      fsplits.push_back(tree[split.nidx].SplitIndex());
    }
  }

  node_shift_ = nodes_split_.size();
  std::sort(fsplits.begin(), fsplits.end());
  fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());
  for (const auto &batch : m->GetBatches<SortedCSCPage>()) {
    for (auto fid : fsplits) {
      auto col = batch[fid];
      const auto ndata = static_cast<bst_omp_uint>(col.size());
#pragma omp parallel for schedule(static)
      for (omp_ulong j = 0; j < ndata; ++j) {
        const bst_uint ridx = col[j].index;
        bst_node_t nidx = positions_[ridx];
        const bst_float fvalue = col[j].fvalue;
        if (!tree[nidx].IsLeaf() && tree[nidx].SplitIndex() == fid) {
          if (fvalue < tree[nidx].SplitCond()) {
            positions_[ridx] = tree[nidx].LeftChild();
          } else {
            positions_[ridx] = tree[nidx].RightChild();
          }
        }
      }
    }
  }

  // Fill in the missing values.
#pragma omp parallel for schedule(static)
  for (omp_ulong r = 0; r < m->Info().num_row_; ++r) {
    auto nid = positions_[r];
    if (!tree[nid].IsLeaf()) {
      if (tree[nid].DefaultLeft()) {
        positions_[r] = tree[nid].LeftChild();
      } else {
        positions_[r] = tree[nid].RightChild();
      }
    }
  }

  if (nodes_split_.size() < max_node + 1) {
    nodes_split_.resize(max_node + 1);
    CHECK_LE(nodes_split_.size(), param_.MaxNodes());
  }
  for (auto split : pending) {
    nodes_split_.at(split.nidx) = split;
  }
  monitor_.Stop(__func__);
}

template <typename GradientT>
void MultiExact<GradientT>::UpdateTree(HostDeviceVector<GradientPair> *gpair,
                                       DMatrix *data, RegTree *tree) {
  this->InitData(data, gpair->ConstHostSpan(), tree->LeafSize());
  CHECK_NE(this->targets_, 0);

  this->InitRoot(data, tree);
  this->ApplySplit(data, tree);

  size_t depth { 1 };
  while (nodes_split_.size() - node_shift_ != 0) {
    auto p_feature_set = sampler_.GetFeatureSet(depth);
    this->EvaluateSplit(data, p_feature_set->HostSpan());
    this->ApplySplit(data, tree);
    depth++;
  }
}

template class MultiExact<SingleGradientPair>;
template class MultiExact<MultiGradientPair>;

class MultiExactUpdater : public TreeUpdater  {
  using SingleTargetExact = MultiExact<SingleGradientPair>;
  using MultiTargetExact = MultiExact<MultiGradientPair>;

  SingleTargetExact single_;
  MultiTargetExact multi_;

 public:
  explicit MultiExactUpdater(GenericParameter const *tparam)
      : single_{tparam}, multi_{tparam} {}
  char const *Name() const override { return single_.Name(); };
  void Configure(const Args &args) override {
    single_.Configure(args);
    multi_.Configure(args);
  }
  void LoadConfig(Json const& in) override {
    single_.LoadConfig(in);
    multi_.LoadConfig(in);
  }
  void SaveConfig(Json* p_out) const override {
    single_.SaveConfig(p_out);
    multi_.SaveConfig(p_out);
  }
  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* data,
              const std::vector<RegTree*>& trees) override {
    CHECK_NE(trees.size(), 0);
    if (trees.front()->Kind() == RegTree::kSingle) {
      single_.Update(gpair, data, trees);
    } else {
      multi_.Update(gpair, data, trees);
    }
  }
};

XGBOOST_REGISTER_TREE_UPDATER(MultiExact, "grow_colmaker")
    .describe("Grow tree with parallelization over columns.")
    .set_body([](GenericParameter const *tparam, LearnerModelParam const* mparam) {
       return new MultiExactUpdater(tparam);
    });

}  // namespace tree
}  // namespace xgboost
