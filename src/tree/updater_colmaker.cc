/**
 * Copyright 2014-2026, XGBoost Contributors
 * \file updater_colmaker.cc
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */
#include <algorithm>
#include <cmath>
#include <vector>

#include "../collective/communicator-inl.h"  // for IsDistributed
#include "../common/error_msg.h"             // for NoCategorical
#include "../common/random.h"
#include "constraints.h"
#include "param.h"
#include "sample_position.h"  // for SamplePosition
#include "split_evaluator.h"
#include "tree_view.h"         // for ScalarTreeView
#include "xgboost/gradient.h"  // for GradientContainer
#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/parameter.h"
#include "xgboost/tree_updater.h"

namespace xgboost::tree {

DMLC_REGISTRY_FILE_TAG(updater_colmaker);

struct ColMakerTrainParam : XGBoostParameter<ColMakerTrainParam> {
  // speed optimization for dense column
  float opt_dense_col;
  // default direction choice
  int default_direction;

  DMLC_DECLARE_PARAMETER(ColMakerTrainParam) {
    DMLC_DECLARE_FIELD(opt_dense_col)
        .set_range(0.0f, 1.0f)
        .set_default(1.0f)
        .describe("EXP Param: speed optimization for dense column.");
    DMLC_DECLARE_FIELD(default_direction)
        .set_default(0)
        .add_enum("learn", 0)
        .add_enum("left", 1)
        .add_enum("right", 2)
        .describe("Default direction choice when encountering a missing value");
  }

  /*! \brief whether need forward small to big search: default right */
  inline bool NeedForwardSearch(float col_density, bool indicator) const {
    return default_direction == 2 ||
           (default_direction == 0 && (col_density < opt_dense_col) && !indicator);
  }
  /*! \brief whether need backward big to small search: default left */
  inline bool NeedBackwardSearch() const { return default_direction != 2; }
};

DMLC_REGISTER_PARAMETER(ColMakerTrainParam);

/*! \brief column-wise update to construct a tree */
class ColMaker : public TreeUpdater {
 public:
  explicit ColMaker(Context const *ctx) : TreeUpdater(ctx) {}
  void Configure(const Args &args) override { colmaker_param_.UpdateAllowUnknown(args); }

  void LoadConfig(Json const &in) override {
    auto const &config = get<Object const>(in);
    FromJson(config.at("colmaker_train_param"), &this->colmaker_param_);
  }
  void SaveConfig(Json *p_out) const override {
    auto &out = *p_out;
    out["colmaker_train_param"] = ToJson(colmaker_param_);
  }

  char const *Name() const override { return "grow_colmaker"; }

  void LazyGetColumnDensity(DMatrix *dmat) {
    // Finds densities if we don't already have them
    if (!column_densities_.empty()) {
      return;
    }

    auto const &info = dmat->Info();
    const auto num_cols = info.num_col_;
    const auto num_rows = info.num_row_;

    if (num_cols == 0 || num_rows == 0) {
      // No data, keep densities_ empty; callers should handle this gracefully.
      return;
    }

    std::vector<size_t> column_size(num_cols, 0);
    for (auto const &batch : dmat->GetBatches<SortedCSCPage>(ctx_)) {
      auto page = batch.GetView();
      auto const batch_size = batch.Size();
      // Safety check in case of inconsistent pages.
      CHECK_LE(batch_size, num_cols);
      for (auto i = 0u; i < batch_size; ++i) {
        column_size[i] += page[i].size();
      }
    }

    column_densities_.resize(column_size.size());
    for (size_t i = 0; i < column_densities_.size(); ++i) {
      size_t const nmiss = num_rows - column_size[i];
      column_densities_[i] =
          1.0f - (static_cast<float>(nmiss)) / static_cast<float>(num_rows);
    }
  }

  void Update(TrainParam const *param, GradientContainer *in_gpair, DMatrix *dmat,
              common::Span<HostDeviceVector<bst_node_t>> /*out_position*/,
              const std::vector<RegTree *> &trees) override {
    if (collective::IsDistributed()) {
      LOG(FATAL) << "Updater `grow_colmaker` or `exact` tree method doesn't "
                    "support distributed training.";
    }
    if (!dmat->SingleColBlock()) {
      LOG(FATAL) << "Updater `grow_colmaker` or `exact` tree method doesn't "
                    "support external memory training.";
    }
    if (dmat->Info().HasCategorical()) {
      LOG(FATAL) << error::NoCategorical("Updater `grow_colmaker` or `exact` tree method");
    }
    if (std::fabs(param->colsample_bynode - 1.0f) > 0.0f) {
      // Keep semantics but use a more explicit comparison.
      LOG(FATAL) << "column sample by node is not yet supported by the exact tree method";
    }

    this->LazyGetColumnDensity(dmat);
    // rescale learning rate according to size of trees
    interaction_constraints_.Configure(*param, dmat->Info().num_row_);

    auto gpair = in_gpair->FullGradOnly();
    CHECK_EQ(gpair->Shape(1), 1) << MTNotImplemented();

    // In case of empty trees vector, nothing to update.
    if (trees.empty()) {
      return;
    }

    auto const &host_gpair = gpair->Data()->ConstHostVector();
    for (auto *tree : trees) {
      CHECK(tree);
      CHECK(ctx_);
      Builder builder(*param, colmaker_param_, interaction_constraints_, ctx_, column_densities_);
      builder.Update(host_gpair, dmat, tree);
    }
  }

 protected:
  ColMakerTrainParam colmaker_param_;
  // SplitEvaluator that will be cloned for each Builder
  std::vector<float> column_densities_;

  FeatureInteractionConstraintHost interaction_constraints_;
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    GradStats stats;
    /*! \brief last feature value scanned */
    bst_float last_fvalue{0};
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    ThreadEntry() = default;
  };
  struct NodeEntry {
    /*! \brief statics for node entry */
    GradStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain{0.0f};
    /*! \brief weight calculated related to current data */
    bst_float weight{0.0f};
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    NodeEntry() = default;
  };
  // actual builder that runs the algorithm
  class Builder {
   public:
    // constructor
    explicit Builder(const TrainParam &param, const ColMakerTrainParam &colmaker_train_param,
                     FeatureInteractionConstraintHost _interaction_constraints, Context const *ctx,
                     const std::vector<float> &column_densities)
        : param_{param},
          colmaker_train_param_{colmaker_train_param},
          ctx_{ctx},
          tree_evaluator_{param_, column_densities.size(), DeviceOrd::CPU()},
          interaction_constraints_{std::move(_interaction_constraints)},
          column_densities_{column_densities} {
      // Ensure we have a valid context and at least one column for evaluator.
      CHECK(ctx_);
    }
    // update one tree, growing
    virtual void Update(const std::vector<GradientPair> &gpair, DMatrix *p_fmat, RegTree *p_tree) {
      std::vector<int> newnodes;
      this->InitData(gpair, *p_fmat);
      this->InitNewNode(qexpand_, gpair, *p_fmat, *p_tree);
      // We can check max_leaves too, but might break some grid searching pipelines.
      CHECK_GT(param_.max_depth, 0) << "exact tree method doesn't support unlimited depth.";

      for (int depth = 0; depth < param_.max_depth; ++depth) {
        this->FindSplit(depth, qexpand_, gpair, p_fmat, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, qexpand_, &newnodes);
        this->InitNewNode(newnodes, gpair, *p_fmat, *p_tree);

        for (auto nid : qexpand_) {
          if ((*p_tree)[nid].IsLeaf()) {
            continue;
          }
          int const cleft = (*p_tree)[nid].LeftChild();
          int const cright = (*p_tree)[nid].RightChild();

          tree_evaluator_.AddSplit(nid, cleft, cright, snode_[nid].best.SplitIndex(),
                                   snode_[cleft].weight, snode_[cright].weight);
          interaction_constraints_.Split(nid, snode_[nid].best.SplitIndex(), cleft, cright);
        }
        qexpand_ = newnodes;
        // if nothing left to be expand, break
        if (qexpand_.empty()) {
          break;
        }
      }
      // set all the rest expanding nodes to leaf
      for (const int nid : qexpand_) {
        (*p_tree)[nid].SetLeaf(snode_[nid].weight * param_.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->NumNodes(); ++nid) {
        auto &stat = p_tree->Stat(nid);
        stat.loss_chg = snode_[nid].best.loss_chg;
        stat.base_weight = snode_[nid].weight;
        stat.sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
      }
    }

   protected:
    // initialize temp data structure
    inline void InitData(const std::vector<GradientPair> &gpair, const DMatrix &fmat) {
      {
        // setup position
        position_.assign(gpair.size(), 0);
        CHECK_EQ(fmat.Info().num_row_, position_.size());
        // mark delete for the deleted datas
        for (size_t ridx = 0; ridx < position_.size(); ++ridx) {
          if (gpair[ridx].GetHess() < 0.0f) {
            position_[ridx] = ~position_[ridx];
          }
        }
        // mark subsample
        if (param_.subsample < 1.0f) {
          CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
              << "Only uniform sampling is supported, "
              << "gradient-based sampling is only support by the `hist` tree method.";
          std::bernoulli_distribution coin_flip(param_.subsample);
          auto &rnd = ctx_->Rng();
          for (size_t ridx = 0; ridx < position_.size(); ++ridx) {
            if (gpair[ridx].GetHess() < 0.0f) {
              continue;
            }
            if (!coin_flip(rnd)) {
              position_[ridx] = ~position_[ridx];
            }
          }
        }
      }
      {
        if (!column_sampler_) {
          column_sampler_ = common::MakeColumnSampler(ctx_);
        }
        column_sampler_->Init(ctx_, fmat.Info().num_col_, fmat.Info().feature_weights,
                              param_.colsample_bynode, param_.colsample_bylevel,
                              param_.colsample_bytree);
      }
      {
        // setup temp space for each thread
        // reserve a small space
        auto const nthreads = this->ctx_->Threads();
        stemp_.clear();
        stemp_.resize(nthreads);
        for (auto &vec : stemp_) {
          vec.clear();
          vec.reserve(256);
        }
        snode_.reserve(256);
      }
      {
        // expand query
        qexpand_.clear();
        qexpand_.reserve(256);
        qexpand_.push_back(0);
      }
    }
    /*!
     * \brief initialize the base_weight, root_gain,
     *  and NodeEntry for all the new nodes in qexpand
     */
    void InitNewNode(const std::vector<int> &qexpand, const std::vector<GradientPair> &gpair,
                     const DMatrix &fmat, RegTree const &tree) {
      auto const n_nodes = tree.NumNodes();
      auto sc_tree = tree.HostScView();
      {
        // setup statistics space for each tree node
        for (auto &i : stemp_) {
          i.assign(n_nodes, ThreadEntry());
        }
        snode_.assign(n_nodes, NodeEntry());
      }
      auto const &info = fmat.Info();
      // setup position
      common::ParallelFor(info.num_row_, ctx_->Threads(), [&](auto ridx) {
        auto const tid = omp_get_thread_num();
        if (position_[ridx] < 0) {
          return;
        }
        stemp_[tid][position_[ridx]].stats.Add(gpair[ridx]);
      });
      // sum the per thread statistics together
      for (int nid : qexpand) {
        GradStats stats;
        for (auto &s : stemp_) {
          stats.Add(s[nid].stats);
        }
        // update node statistics
        snode_[nid].stats = stats;
      }

      auto evaluator = tree_evaluator_.GetEvaluator();
      // calculating the weights
      for (bst_node_t nidx : qexpand) {
        bst_node_t const parentid = sc_tree.Parent(nidx);
        snode_[nidx].weight =
            static_cast<float>(evaluator.CalcWeight(parentid, param_, snode_[nidx].stats));
        snode_[nidx].root_gain =
            static_cast<float>(evaluator.CalcGain(parentid, param_, snode_[nidx].stats));
      }
    }
    /*! \brief update queue expand add in new leaves */
    void UpdateQueueExpand(RegTree const &tree, const std::vector<bst_node_t> &qexpand,
                           std::vector<int> *p_newnodes) {
      CHECK(p_newnodes);
      p_newnodes->clear();
      auto sc_tree = tree.HostScView();
      for (bst_node_t nidx : qexpand) {
        if (!sc_tree.IsLeaf(nidx)) {
          p_newnodes->push_back(sc_tree.LeftChild(nidx));
          p_newnodes->push_back(sc_tree.RightChild(nidx));
        }
      }
    }

    // update enumeration solution
    inline void UpdateEnumeration(
        int nid, GradientPair gstats, bst_float fvalue, int d_step, bst_uint fid,
        GradStats &c,                    // NOLINT
        std::vector<ThreadEntry> &temp,  // NOLINT(*)
        TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const {
      // get the statistics of nid
      ThreadEntry &e = temp[nid];
      // test if first hit, this is fine, because we set 0 during init
      if (e.stats.Empty()) {
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      } else {
        // try to find a split
        if (fvalue != e.last_fvalue && e.stats.sum_hess >= param_.min_child_weight) {
          c.SetSubstract(snode_[nid].stats, e.stats);
          if (c.sum_hess >= param_.min_child_weight) {
            bst_float loss_chg{0};
            auto const mid = (fvalue + e.last_fvalue) * 0.5f;
            bst_float const proposed_split = (mid == fvalue) ? e.last_fvalue : mid;
            if (d_step == -1) {
              loss_chg = static_cast<bst_float>(
                  evaluator.CalcSplitGain(param_, nid, fid, c, e.stats) - snode_[nid].root_gain);
              e.best.Update(loss_chg, fid, proposed_split, d_step == -1, false, c, e.stats);
            } else {
              loss_chg = static_cast<bst_float>(
                  evaluator.CalcSplitGain(param_, nid, fid, e.stats, c) - snode_[nid].root_gain);
              e.best.Update(loss_chg, fid, proposed_split, d_step == -1, false, e.stats, c);
            }
          }
        }
        // update the statistics
        e.stats.Add(gstats);
        e.last_fvalue = fvalue;
      }
    }
    // same as EnumerateSplit, with cacheline prefetch optimization
    void EnumerateSplit(const Entry *begin, const Entry *end, int d_step, bst_uint fid,
                        const std::vector<GradientPair> &gpair,
                        std::vector<ThreadEntry> &temp,  // NOLINT(*)
                        TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const {
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      for (auto nid : qexpand) {
        temp[nid].stats = GradStats();
      }
      // left statistics
      GradStats c;
      // local cache buffer for position and gradient pair
      constexpr int kBuffer = 32;
      int buf_position[kBuffer] = {};
      GradientPair buf_gpair[kBuffer] = {};

      // aligned ending position
      const Entry *align_end;
      if (d_step > 0) {
        align_end = begin + (end - begin) / kBuffer * kBuffer;
      } else {
        align_end = begin - (begin - end) / kBuffer * kBuffer;
      }
      int i;
      const Entry *it;
      const int align_step = d_step * kBuffer;
      // internal cached loop
      for (it = begin; it != align_end; it += align_step) {
        const Entry *p;
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          auto const idx = p->index;
          buf_position[i] = position_[idx];
          buf_gpair[i] = gpair[idx];
        }
        for (i = 0, p = it; i < kBuffer; ++i, p += d_step) {
          const int nid = buf_position[i];
          if (nid < 0 || !interaction_constraints_.Query(nid, fid)) {
            continue;
          }
          this->UpdateEnumeration(nid, buf_gpair[i], p->fvalue, d_step, fid, c, temp, evaluator);
        }
      }

      // finish up the ending piece
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        auto const idx = it->index;
        buf_position[i] = position_[idx];
        buf_gpair[i] = gpair[idx];
      }
      for (it = align_end, i = 0; it != end; ++i, it += d_step) {
        const int nid = buf_position[i];
        if (nid < 0 || !interaction_constraints_.Query(nid, fid)) {
          continue;
        }
        this->UpdateEnumeration(nid, buf_gpair[i], it->fvalue, d_step, fid, c, temp, evaluator);
      }
      // finish updating all statistics, check if it is possible to include all sum statistics
      for (int nid : qexpand) {
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode_[nid].stats, e.stats);
        if (e.stats.sum_hess >= param_.min_child_weight && c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          const bst_float gap = std::abs(e.last_fvalue) + kRtEps;
          const bst_float delta = d_step == +1 ? gap : -gap;
          if (d_step == -1) {
            loss_chg = static_cast<bst_float>(
                evaluator.CalcSplitGain(param_, nid, fid, c, e.stats) - snode_[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1, false, c, e.stats);
          } else {
            loss_chg = static_cast<bst_float>(
                evaluator.CalcSplitGain(param_, nid, fid, e.stats, c) - snode_[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1, false, e.stats, c);
          }
        }
      }
    }

    // update the solution candidate
    void UpdateSolution(SortedCSCPage const &batch, const std::vector<bst_feature_t> &feat_set,
                        const std::vector<GradientPair> &gpair) {
      // start enumeration
      const auto num_features = feat_set.size();
      if (num_features == 0) {
        return;
      }
      CHECK(this->ctx_);
      const int batch_size =
          std::max(static_cast<int>(num_features / this->ctx_->Threads() / 32), 1);
      auto page = batch.GetView();
      common::ParallelFor(
          num_features, ctx_->Threads(), common::Sched::Dyn(batch_size), [&](auto i) {
            auto evaluator = tree_evaluator_.GetEvaluator();
            bst_feature_t const fid = feat_set[i];
            int32_t const tid = omp_get_thread_num();
            auto c = page[fid];
            if (c.empty()) {
              return;
            }
            const bool ind = c.front().fvalue == c.back().fvalue;
            if (colmaker_train_param_.NeedForwardSearch(column_densities_[fid], ind)) {
              this->EnumerateSplit(c.data(), c.data() + c.size(), +1, fid, gpair, stemp_[tid],
                                   evaluator);
            }
            if (colmaker_train_param_.NeedBackwardSearch()) {
              this->EnumerateSplit(c.data() + c.size() - 1, c.data() - 1, -1, fid, gpair,
                                   stemp_[tid], evaluator);
            }
          });
    }

    // find splits at current level, do split per level
    void FindSplit(bst_node_t depth, const std::vector<int> &qexpand,
                   std::vector<GradientPair> const &gpair, DMatrix *p_fmat, RegTree *p_tree) {
      auto evaluator = tree_evaluator_.GetEvaluator();

      auto feat_set = column_sampler_->GetFeatureSet(depth);
      if (!feat_set) {
        return;
      }

      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>(ctx_)) {
        this->UpdateSolution(batch, feat_set->HostVector(), gpair);
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      this->SyncBestSolution(qexpand);
      // get the best result, we can synchronize the solution
      for (int nid : qexpand) {
        NodeEntry const &e = snode_[nid];
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > kRtEps) {
          bst_float const left_leaf_weight =
              evaluator.CalcWeight(nid, param_, e.best.left_sum) * param_.learning_rate;
          bst_float const right_leaf_weight =
              evaluator.CalcWeight(nid, param_, e.best.right_sum) * param_.learning_rate;
          p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value, e.best.DefaultLeft(),
                             e.weight, left_leaf_weight, right_leaf_weight, e.best.loss_chg,
                             e.stats.sum_hess, e.best.left_sum.GetHess(),
                             e.best.right_sum.GetHess(), 0);
        } else {
          (*p_tree)[nid].SetLeaf(e.weight * param_.learning_rate);
        }
      }
    }
    // reset position of each data points after split is created in the tree
    void ResetPosition(const std::vector<int> &qexpand, DMatrix *p_fmat, const RegTree &tree) {
      auto sc_tree = tree.HostScView();
      // set the positions in the nondefault
      this->SetNonDefaultPosition(qexpand, p_fmat, tree);
      // set rest of instances to default position
      // set default direct nodes to default
      // for leaf nodes that are not fresh, mark then to ~nid,
      // so that they are ignored in future statistics collection
      auto const num_row = p_fmat->Info().num_row_;
      common::ParallelFor(num_row, this->ctx_->Threads(), [&](auto ridx) {
        CHECK_LT(ridx, position_.size()) << "ridx exceed bound "
                                         << "ridx=" << ridx << " pos=" << position_.size();
        const bst_node_t nidx = SamplePosition::Decode(position_[ridx]);
        if (sc_tree.IsLeaf(nidx)) {
          // mark finish when it is not a fresh leaf
          if (sc_tree.RightChild(nidx) == -1) {
            position_[ridx] = ~nidx;
          }
        } else {
          // push to default branch
          if (sc_tree.DefaultLeft(nidx)) {
            this->SetEncodePosition(ridx, sc_tree.LeftChild(nidx));
          } else {
            this->SetEncodePosition(ridx, sc_tree.RightChild(nidx));
          }
        }
      });
    }
    // customization part
    // synchronize the best solution of each node
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (int nid : qexpand) {
        NodeEntry &e = snode_[nid];
        CHECK(this->ctx_);
        auto const nthreads = this->ctx_->Threads();
        for (int tid = 0; tid < nthreads; ++tid) {
          e.best.Update(stemp_[tid][nid].best);
        }
      }
    }
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand, DMatrix *p_fmat,
                                       const RegTree &tree) {
      // step 1, classify the non-default data into right places
      auto sc_tree = tree.HostScView();
      std::vector<unsigned> fsplits;
      fsplits.reserve(qexpand.size());
      for (int nid : qexpand) {
        if (!sc_tree.IsLeaf(nid)) {
          fsplits.push_back(sc_tree.SplitIndex(nid));
        }
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.erase(std::unique(fsplits.begin(), fsplits.end()), fsplits.end());
      if (fsplits.empty()) {
        return;
      }

      for (const auto &batch : p_fmat->GetBatches<SortedCSCPage>(ctx_)) {
        auto page = batch.GetView();
        for (auto fid : fsplits) {
          auto col = page[fid];
          common::ParallelFor(col.size(), this->ctx_->Threads(), [&](auto j) {
            const bst_uint ridx = col[j].index;
            bst_node_t nidx = SamplePosition::Decode(position_[ridx]);
            const float fvalue = col[j].fvalue;
            // go back to parent, correct those who are not default
            if (!sc_tree.IsLeaf(nidx) && sc_tree.SplitIndex(nidx) == fid) {
              if (fvalue < sc_tree.SplitCond(nidx)) {
                this->SetEncodePosition(ridx, sc_tree.LeftChild(nidx));
              } else {
                this->SetEncodePosition(ridx, sc_tree.RightChild(nidx));
              }
            }
          });
        }
      }
    }
    // utils to get/set position, with encoded format
    // encode the encoded position value for ridx
    void SetEncodePosition(bst_idx_t ridx, bst_node_t nidx) {
      const bool is_invalid = position_[ridx] < 0;
      position_[ridx] = SamplePosition::Encode(nidx, !is_invalid);
    }
    //  --data fields--
    const TrainParam &param_;
    const ColMakerTrainParam &colmaker_train_param_;
    // number of omp thread used during training
    Context const *ctx_;
    std::shared_ptr<common::ColumnSampler> column_sampler_;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position_;
    // PerThread x PerTreeNode: statistics for per thread construction
    std::vector<std::vector<ThreadEntry>> stemp_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode_;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
    TreeEvaluator tree_evaluator_;

    FeatureInteractionConstraintHost interaction_constraints_;
    const std::vector<float> &column_densities_;
  };
};

XGBOOST_REGISTER_TREE_UPDATER(ColMaker, "grow_colmaker")
    .describe("Grow tree with parallelization over columns.")
    .set_body([](Context const *ctx, auto) { return new ColMaker(ctx); });
}  // namespace xgboost::tree
