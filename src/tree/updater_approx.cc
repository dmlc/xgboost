/*!
 * Copyright 2021-2022 XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#include <algorithm>
#include <unordered_map>
#include <memory>
#include <vector>
#include <set>

#include "common_row_partitioner.h"
#include "../common/random.h"
#include "../common/column_matrix.h"

#include "../data/gradient_index.h"
#include "constraints.h"
#include "driver.h"
#include "hist/evaluate_splits.h"
#include "hist/histogram.h"
#include "param.h"
#include "xgboost/base.h"
#include "xgboost/json.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"
#include "algorithm"

namespace xgboost {
namespace tree {
DMLC_REGISTRY_FILE_TAG(updater_approx);

namespace {
// Return the BatchParam used by DMatrix.
auto BatchSpec(TrainParam const &p, common::Span<float> hess, ObjInfo const task) {
  return BatchParam{p.max_bin, hess, !task.const_hess};
}

auto BatchSpec(TrainParam const &p, common::Span<float> hess) {
  return BatchParam{p.max_bin, hess, false};
}
}  // anonymous namespace

class GloablApproxBuilder {
 protected:
  TrainParam param_;
  std::shared_ptr<common::ColumnSampler> col_sampler_;
  HistEvaluator<CPUExpandEntry> evaluator_;
  HistogramBuilder<CPUExpandEntry> histogram_builder_;
  Context const *ctx_;
  ObjInfo const task_;

  std::vector<CommonRowPartitioner> partitioner_;
  // Pointer to last updated tree, used for update prediction cache.
  RegTree *p_last_tree_{nullptr};
  common::Monitor *monitor_;
  size_t n_batches_{0};
  // Cache for histogram cuts.
  common::HistogramCuts feature_values_;
  std::unordered_map<uint32_t, common::ColumnMatrix> column_matrix_;
  std::unordered_map<uint32_t, int32_t> split_conditions_;
  std::unordered_map<uint32_t, uint64_t> split_ind_;
  std::vector<uint16_t> child_node_ids_;

 public:
  void InitData(DMatrix *p_fmat, common::Span<float> hess) {
    monitor_->Start(__func__);

    n_batches_ = 0;
    int32_t n_total_bins = 0;
    partitioner_.clear();
    // Generating the GHistIndexMatrix is quite slow, is there a way to speed it up?
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(BatchSpec(param_, hess, task_))) {
      if (n_total_bins == 0) {
        n_total_bins = page.cut.TotalBins();
        feature_values_ = page.cut;
      } else {
        CHECK_EQ(n_total_bins, page.cut.TotalBins());
      }

      partitioner_.emplace_back(ctx_, page,
                                p_last_tree_, param_.max_depth,
                                param_.grow_policy == TrainParam::kLossGuide);
      n_batches_++;
    }

    histogram_builder_.Reset(n_total_bins, param_.max_bin,
                             ctx_->Threads(), n_batches_, param_.max_depth, rabit::IsDistributed());
    monitor_->Stop(__func__);
  }

  CPUExpandEntry InitRoot(DMatrix *p_fmat, std::vector<GradientPair> const &gpair,
                          common::Span<float> hess, RegTree *p_tree) {
    monitor_->Start(__func__);
    CPUExpandEntry best;
    best.nid = RegTree::kRoot;
    best.depth = 0;
    GradStats root_sum;
    for (auto const &g : gpair) {
      root_sum.Add(g);
    }
    rabit::Allreduce<rabit::op::Sum, double>(reinterpret_cast<double *>(&root_sum), 2);
    std::vector<CPUExpandEntry> nodes{best};
        this->BuildHistogram<true>(p_fmat, p_tree, {}, gpair,
                             hess, nodes, {});
    auto weight = evaluator_.InitRoot(root_sum);
    p_tree->Stat(RegTree::kRoot).sum_hess = root_sum.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    auto const &histograms = histogram_builder_.Histogram();
    auto ft = p_fmat->Info().feature_types.ConstHostSpan();
    evaluator_.EvaluateSplits(histograms, feature_values_, ft, *p_tree, &nodes);
    monitor_->Stop(__func__);

    return nodes.front();
  }

  void UpdatePredictionCache(DMatrix const *data, linalg::VectorView<float> out_preds) const {
    monitor_->Start(__func__);
    // Caching prediction seems redundant for approx tree method, as sketching takes up
    // majority of training time.
    CHECK_EQ(out_preds.Size(), data->Info().num_row_);
// <<<<<<< HEAD
    CHECK(p_last_tree_);

    auto evaluator = evaluator_.Evaluator();
    auto const &tree = *p_last_tree_;
    auto const &snode = evaluator_.Stats();
    size_t page_disp = 0;
    size_t i = 0;
    for (auto &part : partitioner_) {
      xgboost::tree::CommonRowPartitioner& prt =
        const_cast<xgboost::tree::CommonRowPartitioner&>(partitioner_[i++]);
      common::BlockedSpace2d space(1, [&](size_t node) {
        return prt.GetNodeAssignments().size();
      }, 1024);
      common::ParallelFor2d(space, ctx_->Threads(), [&](size_t node, common::Range1d r) {
        int tid = omp_get_thread_num();
        for (size_t it = r.begin(); it <  r.end(); ++it) {
          bst_float leaf_value;
          // if a node is marked as deleted by the pruner, traverse upward to locate
          // a non-deleted leaf.
          int nid = (~(static_cast<uint16_t>(1) << 15)) & prt.GetNodeAssignments()[it];
          if ((*p_last_tree_)[nid].IsDeleted()) {
            while ((*p_last_tree_)[nid].IsDeleted()) {
              nid = (*p_last_tree_)[nid].Parent();
            }
          }
          CHECK((*p_last_tree_)[nid].IsLeaf());
          leaf_value = (*p_last_tree_)[nid].LeafValue();
          out_preds(it + page_disp) += leaf_value;
        }
      });
      page_disp += prt.GetNodeAssignments().size();
    }
// =======
//     UpdatePredictionCacheImpl(ctx_, p_last_tree_, partitioner_, evaluator_, out_preds);
// >>>>>>> 0725fd60819f9758fbed6ee54f34f3696a2fb2f8
    monitor_->Stop(__func__);
  }

  template<bool is_root>
  void BuildHistogram(DMatrix *p_fmat, RegTree *p_tree,
                      std::vector<CPUExpandEntry> const &valid_candidates,
                      std::vector<GradientPair> const &gpair, common::Span<float> hess,
                      const std::vector<CPUExpandEntry>& nodes_to_build,
                      const std::vector<CPUExpandEntry>& nodes_to_sub) {
    monitor_->Start(__func__);

    std::vector<std::set<uint16_t>> merged_thread_ids_set(nodes_to_build.size());
    std::vector<std::vector<uint16_t>> merged_thread_ids(nodes_to_build.size());
    for (size_t nid = 0; nid < nodes_to_build.size(); ++nid) {
      const auto &entry = nodes_to_build[nid];
      for (auto & partitioner : partitioner_) {
        for (auto&  tid : partitioner.GetOptPartition().GetThreadIdsForNode(entry.nid)) {
          merged_thread_ids_set[nid].insert(tid);
        }
      }
      merged_thread_ids[nid].resize(merged_thread_ids_set[nid].size());
      std::copy(merged_thread_ids_set[nid].begin(),
                merged_thread_ids_set[nid].end(), merged_thread_ids[nid].begin());
    }
    size_t i = 0;
    for (auto const &page :
         p_fmat->GetBatches<GHistIndexMatrix>(BatchSpec(param_, hess))) {
      switch (page.index.GetBinTypeSize()) {
        case common::kUint8BinsTypeSize:
          histogram_builder_.template BuildHist<uint8_t, is_root> (i, page,
            p_tree, nodes_to_build, nodes_to_sub,
            gpair, &(partitioner_[i].GetOptPartition()),
            &(partitioner_[i].GetNodeAssignments()), &merged_thread_ids);
          break;
        case common::kUint16BinsTypeSize:
          histogram_builder_.template BuildHist<uint16_t, is_root> (i, page,
            p_tree, nodes_to_build, nodes_to_sub,
            gpair, &(partitioner_[i].GetOptPartition()),
            &(partitioner_[i].GetNodeAssignments()), &merged_thread_ids);
          break;
        case common::kUint32BinsTypeSize:
          histogram_builder_.template BuildHist<uint32_t, is_root> (i, page,
            p_tree, nodes_to_build, nodes_to_sub,
            gpair, &(partitioner_[i].GetOptPartition()),
            &(partitioner_[i].GetNodeAssignments()), &merged_thread_ids);
          break;
        default:
          CHECK(false);
      }
      i++;
    }
    monitor_->Stop(__func__);
  }

  void LeafPartition(RegTree const &tree, common::Span<float> hess,
                     std::vector<bst_node_t> *p_out_position) {
    monitor_->Start(__func__);
    if (!task_.UpdateTreeLeaf()) {
      return;
    }
    for (auto const &part : partitioner_) {
      part.LeafPartition(ctx_, tree, p_out_position);
    }
    monitor_->Stop(__func__);
  }

 public:
  explicit GloablApproxBuilder(TrainParam param, MetaInfo const &info, Context const *ctx,
                               std::shared_ptr<common::ColumnSampler> column_sampler, ObjInfo task,
                               common::Monitor *monitor)
      : param_{std::move(param)},
        col_sampler_{std::move(column_sampler)},
        evaluator_{param_, info, ctx->Threads(), col_sampler_},
        ctx_{ctx},
        task_{task},
        monitor_{monitor} {}

  void UpdateTree(DMatrix *p_fmat, std::vector<GradientPair> const &gpair, common::Span<float> hess,
                  RegTree *p_tree, HostDeviceVector<bst_node_t> *p_out_position) {
    p_last_tree_ = p_tree;
    this->InitData(p_fmat, hess);
    child_node_ids_.clear();
    child_node_ids_.emplace_back(0);
    split_conditions_.clear();
    split_ind_.clear();

    Driver<CPUExpandEntry> driver(param_);
    auto &tree = *p_tree;
    driver.Push({this->InitRoot(p_fmat, gpair, hess, p_tree)});
    auto expand_set = driver.Pop();
    int depth = 0;
    bool is_loss_guide = static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy) ==
                       TrainParam::kDepthWise ? false : true;

    /**
     * Note for update position
     * Root:
     *   Not applied: No need to update position as initialization has got all the rows ordered.
     *   Applied: Update position is run on applied nodes so the rows are partitioned.
     * Non-root:
     *   Not applied: That node is root of the subtree, same rule as root.
     *   Applied: Ditto
     */
    while (!expand_set.empty()) {
      child_node_ids_.clear();
      std::unordered_map<uint32_t, CPUExpandEntry> applied;
      std::unordered_map<uint32_t, bool> smalest_nodes_mask;
      depth = expand_set[0].depth + 1;
      // candidates that can be further splited.
      std::vector<CPUExpandEntry> valid_candidates;
      std::vector<CPUExpandEntry> applied_vec;
      bool is_applied = false;
      // candidates that can be applied.
      for (auto const &candidate : expand_set) {
        evaluator_.ApplyTreeSplit(candidate, p_tree);
        applied[candidate.nid] = candidate;
        applied_vec.push_back(candidate);
        CHECK_EQ(applied[candidate.nid].nid, candidate.nid);
        // num_leaves++;
        int left_child_nidx = tree[candidate.nid].LeftChild();
//        if (CPUExpandEntry::ChildIsValid(param_, p_tree->GetDepth(left_child_nidx), num_leaves)) {
        if (driver.IsChildValid(candidate)) {
          valid_candidates.emplace_back(candidate);
        } else {
          if (param_.grow_policy == TrainParam::kLossGuide) {
            smalest_nodes_mask[left_child_nidx] = true;
          }
          child_node_ids_.push_back(left_child_nidx);
          child_node_ids_.push_back(tree[candidate.nid].RightChild());
        }
      }

      std::vector<CPUExpandEntry> nodes_to_build;
      std::vector<CPUExpandEntry> nodes_to_sub;
      bool is_left_small = true;
      for (auto const &c : valid_candidates) {
        auto left_nidx = (*p_tree)[c.nid].LeftChild();
        auto right_nidx = (*p_tree)[c.nid].RightChild();
        CHECK_NE(left_nidx, right_nidx);
        auto fewer_right = c.split.right_sum.GetHess() < c.split.left_sum.GetHess();
        auto build_nidx = left_nidx;
        auto subtract_nidx = right_nidx;

        if (param_.grow_policy == TrainParam::kLossGuide) {
          smalest_nodes_mask[left_nidx] = true;
          if (fewer_right) {
            is_left_small = false;
            std::swap(build_nidx, subtract_nidx);
          }
        } else if (fewer_right) {
          std::swap(build_nidx, subtract_nidx);
          smalest_nodes_mask[right_nidx] = true;
        } else {
          smalest_nodes_mask[left_nidx] = true;
        }
        child_node_ids_.push_back(left_nidx);
        child_node_ids_.push_back(right_nidx);
        nodes_to_build.push_back(CPUExpandEntry{build_nidx, p_tree->GetDepth(build_nidx), {}});
        nodes_to_sub.push_back(CPUExpandEntry{subtract_nidx, p_tree->GetDepth(subtract_nidx), {}});
      }
      monitor_->Start("UpdatePosition");

      if (applied_vec.size()) {
        size_t i = 0;
        for (auto const &page :
             p_fmat->GetBatches<GHistIndexMatrix>(BatchSpec(param_, hess))) {
          const common::ColumnMatrix& column_matrix = page.Transpose();
          // clang-format off
          partitioner_.at(i).UpdatePositionDispatched({column_matrix.AnyMissing(),
            column_matrix.GetTypeSize(),
            is_loss_guide, page.cut.HasCategorical()},
            ctx_,
            page,
            applied_vec,
            p_tree,
            depth,
            &smalest_nodes_mask,
            is_loss_guide,
            &split_conditions_,
            &split_ind_, param_.max_depth,
            &child_node_ids_, is_left_small,
            true);
          // clang-format on
          i++;
        }
      }
      monitor_->Stop("UpdatePosition");

      std::vector<CPUExpandEntry> best_splits;
      if (!valid_candidates.empty()) {
        this->BuildHistogram<false>(p_fmat, p_tree, valid_candidates, gpair,
                             hess, nodes_to_build, nodes_to_sub);
        for (auto const &candidate : valid_candidates) {
          int left_child_nidx = tree[candidate.nid].LeftChild();
          int right_child_nidx = tree[candidate.nid].RightChild();
          CPUExpandEntry l_best{left_child_nidx, tree.GetDepth(left_child_nidx), {}};
          CPUExpandEntry r_best{right_child_nidx, tree.GetDepth(right_child_nidx), {}};
          best_splits.push_back(l_best);
          best_splits.push_back(r_best);
        }
        auto const &histograms = histogram_builder_.Histogram();
        auto ft = p_fmat->Info().feature_types.ConstHostSpan();
        monitor_->Start("EvaluateSplits");
        evaluator_.EvaluateSplits(histograms, feature_values_, ft, *p_tree, &best_splits);
        monitor_->Stop("EvaluateSplits");
      }
      driver.Push(best_splits.begin(), best_splits.end());
      expand_set = driver.Pop();
    }

    auto &h_position = p_out_position->HostVector();
    this->LeafPartition(tree, hess, &h_position);
  }
};

/**
 * \brief Implementation for the approx tree method.  It constructs quantile for every
 *        iteration.
 */
class GlobalApproxUpdater : public TreeUpdater {
  TrainParam param_;
  common::Monitor monitor_;
  // specializations for different histogram precision.
  std::unique_ptr<GloablApproxBuilder> pimpl_;
  // pointer to the last DMatrix, used for update prediction cache.
  DMatrix *cached_{nullptr};
  std::shared_ptr<common::ColumnSampler> column_sampler_ =
      std::make_shared<common::ColumnSampler>();
  ObjInfo task_;

 public:
  explicit GlobalApproxUpdater(GenericParameter const *ctx, ObjInfo task)
      : TreeUpdater(ctx), task_{task} {
    monitor_.Init(__func__);
  }

  void Configure(const Args &args) override { param_.UpdateAllowUnknown(args); }
  void LoadConfig(Json const &in) override {
    auto const &config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
  }
  void SaveConfig(Json *p_out) const override {
    auto &out = *p_out;
    out["train_param"] = ToJson(param_);
  }

  void InitData(TrainParam const &param, HostDeviceVector<GradientPair> const *gpair,
                std::vector<GradientPair> *sampled) {
    auto const &h_gpair = gpair->ConstHostVector();
    sampled->resize(h_gpair.size());
    std::copy(h_gpair.cbegin(), h_gpair.cend(), sampled->begin());
    auto &rnd = common::GlobalRandom();

    if (param.subsample != 1.0) {
      CHECK(param.sampling_method != TrainParam::kGradientBased)
          << "Gradient based sampling is not supported for approx tree method.";
      std::bernoulli_distribution coin_flip(param.subsample);
      std::transform(sampled->begin(), sampled->end(), sampled->begin(), [&](GradientPair &g) {
        if (coin_flip(rnd)) {
          return g;
        } else {
          return GradientPair{};
        }
      });
    }
  }

  char const *Name() const override { return "grow_histmaker"; }

  void Update(HostDeviceVector<GradientPair> *gpair, DMatrix *m,
              common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree *> &trees) override {
    float lr = param_.learning_rate;
    param_.learning_rate = lr / trees.size();

    pimpl_ = std::make_unique<GloablApproxBuilder>(param_, m->Info(), ctx_, column_sampler_, task_,
                                                   &monitor_);

    std::vector<GradientPair> h_gpair;
    InitData(param_, gpair, &h_gpair);
    // Obtain the hessian values for weighted sketching
    std::vector<float> hess(h_gpair.size());
    std::transform(h_gpair.begin(), h_gpair.end(), hess.begin(),
                   [](auto g) { return g.GetHess(); });

    cached_ = m;

    size_t t_idx = 0;
    for (auto p_tree : trees) {
      this->pimpl_->UpdateTree(m, h_gpair, hess, p_tree, &out_position[t_idx]);
      ++t_idx;
    }
    param_.learning_rate = lr;
  }

  bool UpdatePredictionCache(const DMatrix *data, linalg::VectorView<float> out_preds) override {
    if (data != cached_ || !pimpl_) {
      return false;
    }
    this->pimpl_->UpdatePredictionCache(data, out_preds);
    return true;
  }

  bool HasNodePosition() const override { return true; }
};

DMLC_REGISTRY_FILE_TAG(grow_histmaker);

XGBOOST_REGISTER_TREE_UPDATER(GlobalHistMaker, "grow_histmaker")
    .describe(
        "Tree constructor that uses approximate histogram construction "
        "for each node.")
    .set_body([](GenericParameter const *ctx, ObjInfo task) {
      return new GlobalApproxUpdater(ctx, task);
    });
}  // namespace tree
}  // namespace xgboost
