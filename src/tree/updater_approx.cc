/*!
 * Copyright 2021-2022 XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#include <algorithm>
#include <memory>
#include <vector>

#include "../common/random.h"
#include "../data/gradient_index.h"
#include "common_row_partitioner.h"
#include "constraints.h"
#include "driver.h"
#include "hist/evaluate_splits.h"
#include "hist/histogram.h"
#include "param.h"
#include "xgboost/base.h"
#include "xgboost/json.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"

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
      partitioner_.emplace_back(this->ctx_, page.Size(), page.base_rowid);
      n_batches_++;
    }

    histogram_builder_.Reset(n_total_bins, BatchSpec(param_, hess), ctx_->Threads(), n_batches_,
                             collective::IsDistributed());
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
    collective::Allreduce<collective::Operation::kSum>(reinterpret_cast<double *>(&root_sum), 2);
    std::vector<CPUExpandEntry> nodes{best};
    size_t i = 0;
    auto space = ConstructHistSpace(partitioner_, nodes);
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(BatchSpec(param_, hess))) {
      histogram_builder_.BuildHist(i, space, page, p_tree, partitioner_.at(i).Partitions(), nodes,
                                   {}, gpair);
      i++;
    }

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
    UpdatePredictionCacheImpl(ctx_, p_last_tree_, partitioner_, out_preds);
    monitor_->Stop(__func__);
  }

  void BuildHistogram(DMatrix *p_fmat, RegTree *p_tree,
                      std::vector<CPUExpandEntry> const &valid_candidates,
                      std::vector<GradientPair> const &gpair, common::Span<float> hess) {
    monitor_->Start(__func__);
    std::vector<CPUExpandEntry> nodes_to_build;
    std::vector<CPUExpandEntry> nodes_to_sub;

    for (auto const &c : valid_candidates) {
      auto left_nidx = (*p_tree)[c.nid].LeftChild();
      auto right_nidx = (*p_tree)[c.nid].RightChild();
      auto fewer_right = c.split.right_sum.GetHess() < c.split.left_sum.GetHess();

      auto build_nidx = left_nidx;
      auto subtract_nidx = right_nidx;
      if (fewer_right) {
        std::swap(build_nidx, subtract_nidx);
      }
      nodes_to_build.push_back(CPUExpandEntry{build_nidx, p_tree->GetDepth(build_nidx), {}});
      nodes_to_sub.push_back(CPUExpandEntry{subtract_nidx, p_tree->GetDepth(subtract_nidx), {}});
    }

    size_t i = 0;
    auto space = ConstructHistSpace(partitioner_, nodes_to_build);
    for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(BatchSpec(param_, hess))) {
      histogram_builder_.BuildHist(i, space, page, p_tree, partitioner_.at(i).Partitions(),
                                   nodes_to_build, nodes_to_sub, gpair);
      i++;
    }
    monitor_->Stop(__func__);
  }

  void LeafPartition(RegTree const &tree, common::Span<float const> hess,
                     std::vector<bst_node_t> *p_out_position) {
    monitor_->Start(__func__);
    if (!task_.UpdateTreeLeaf()) {
      return;
    }
    for (auto const &part : partitioner_) {
      part.LeafPartition(ctx_, tree, hess, p_out_position);
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

    Driver<CPUExpandEntry> driver(param_);
    auto &tree = *p_tree;
    driver.Push({this->InitRoot(p_fmat, gpair, hess, p_tree)});
    auto expand_set = driver.Pop();

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
      // candidates that can be further splited.
      std::vector<CPUExpandEntry> valid_candidates;
      // candidates that can be applied.
      std::vector<CPUExpandEntry> applied;
      for (auto const &candidate : expand_set) {
        evaluator_.ApplyTreeSplit(candidate, p_tree);
        applied.push_back(candidate);
        if (driver.IsChildValid(candidate)) {
          valid_candidates.emplace_back(candidate);
        }
      }

      monitor_->Start("UpdatePosition");
      size_t page_id = 0;
      for (auto const &page : p_fmat->GetBatches<GHistIndexMatrix>(BatchSpec(param_, hess))) {
        partitioner_.at(page_id).UpdatePosition(ctx_, page, applied, p_tree);
        page_id++;
      }
      monitor_->Stop("UpdatePosition");

      std::vector<CPUExpandEntry> best_splits;
      if (!valid_candidates.empty()) {
        this->BuildHistogram(p_fmat, p_tree, valid_candidates, gpair, hess);
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
