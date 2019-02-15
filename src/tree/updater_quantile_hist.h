/*!
 * Copyright 2017-2018 by Contributors
 * \file updater_quantile_hist.h
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Chen
 */
#ifndef XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
#define XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_

#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <memory>
#include <vector>
#include <string>
#include <queue>
#include <iomanip>
#include <utility>

#include "./param.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"

namespace xgboost {
namespace tree {

using xgboost::common::HistCutMatrix;
using xgboost::common::GHistIndexMatrix;
using xgboost::common::GHistIndexBlockMatrix;
using xgboost::common::GHistIndexRow;
using xgboost::common::HistCollection;
using xgboost::common::RowSetCollection;
using xgboost::common::GHistRow;
using xgboost::common::GHistBuilder;
using xgboost::common::ColumnMatrix;
using xgboost::common::Column;

/*! \brief construct a tree using quantized feature values */
class QuantileHistMaker: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override;

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix* data,
                             HostDeviceVector<bst_float>* out_preds) override;

 protected:
  // training parameter
  TrainParam param_;
  // quantized data matrix
  GHistIndexMatrix gmat_;
  // (optional) data matrix with feature grouping
  GHistIndexBlockMatrix gmatb_;
  // column accessor
  ColumnMatrix column_matrix_;
  bool is_gmat_initialized_;

  // data structure
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
    explicit NodeEntry(const TrainParam& param)
        : root_gain(0.0f), weight(0.0f) {}
  };
  // actual builder that runs the algorithm

  struct Builder {
   public:
    // constructor
    explicit Builder(const TrainParam& param,
                     std::unique_ptr<TreeUpdater> pruner,
                     std::unique_ptr<SplitEvaluator> spliteval)
      : param_(param), pruner_(std::move(pruner)),
        spliteval_(std::move(spliteval)), p_last_tree_(nullptr),
        p_last_fmat_(nullptr) {}
    // update one tree, growing
    virtual void Update(const GHistIndexMatrix& gmat,
                        const GHistIndexBlockMatrix& gmatb,
                        const ColumnMatrix& column_matrix,
                        HostDeviceVector<GradientPair>* gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree);

    inline void BuildHist(const std::vector<GradientPair>& gpair,
                          const RowSetCollection::Elem row_indices,
                          const GHistIndexMatrix& gmat,
                          const GHistIndexBlockMatrix& gmatb,
                          GHistRow hist,
                          bool sync_hist) {
      if (param_.enable_feature_grouping > 0) {
        hist_builder_.BuildBlockHist(gpair, row_indices, gmatb, hist);
      } else {
        hist_builder_.BuildHist(gpair, row_indices, gmat, hist);
      }
      if (sync_hist) {
        this->histred_.Allreduce(hist.data(), hist_builder_.GetNumBins());
      }
    }

    inline void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent) {
      hist_builder_.SubtractionTrick(self, sibling, parent);
    }

    bool UpdatePredictionCache(const DMatrix* data,
                               HostDeviceVector<bst_float>* p_out_preds);

   protected:
    /* tree growing policies */
    struct ExpandEntry {
      int nid;
      int depth;
      bst_float loss_chg;
      unsigned timestamp;
      ExpandEntry(int nid, int depth, bst_float loss_chg, unsigned tstmp)
              : nid(nid), depth(depth), loss_chg(loss_chg), timestamp(tstmp) {}
    };

    struct TreeGrowingPerfMonitor {
      enum timer_name {INIT_DATA, INIT_NEW_NODE, BUILD_HIST, EVALUATE_SPLIT, APPLY_SPLIT};

      double global_start;

      // performance counters
      double tstart;
      double time_init_data = 0;
      double time_init_new_node = 0;
      double time_build_hist = 0;
      double time_evaluate_split = 0;
      double time_apply_split = 0;

      inline void StartPerfMonitor() {
        global_start = dmlc::GetTime();
      }

      inline void EndPerfMonitor() {
        CHECK_GT(global_start, 0);
        double total_time = dmlc::GetTime() - global_start;
        LOG(INFO) << "\nInitData:          "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_init_data
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_init_data / total_time * 100 << "%)\n"
                  << "InitNewNode:       "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_init_new_node
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_init_new_node / total_time * 100 << "%)\n"
                  << "BuildHist:         "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_build_hist
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_build_hist / total_time * 100 << "%)\n"
                  << "EvaluateSplit:     "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_evaluate_split
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_evaluate_split / total_time * 100 << "%)\n"
                  << "ApplySplit:        "
                  << std::fixed << std::setw(6) << std::setprecision(4) << time_apply_split
                  << " (" << std::fixed << std::setw(5) << std::setprecision(2)
                  << time_apply_split / total_time * 100 << "%)\n"
                  << "========================================\n"
                  << "Total:             "
                  << std::fixed << std::setw(6) << std::setprecision(4) << total_time;
        // clear performance counters
        time_init_data = 0;
        time_init_new_node = 0;
        time_build_hist = 0;
        time_evaluate_split = 0;
        time_apply_split = 0;
      }

      inline void TickStart() {
        tstart = dmlc::GetTime();
      }

      inline void UpdatePerfTimer(const timer_name &timer_name) {
        CHECK_GT(tstart, 0);
        switch (timer_name) {
          case INIT_DATA:
            time_init_data += dmlc::GetTime() - tstart;
            break;
          case INIT_NEW_NODE:
            time_init_new_node += dmlc::GetTime() - tstart;
            break;
          case BUILD_HIST:
            time_build_hist += dmlc::GetTime() - tstart;
            break;
          case EVALUATE_SPLIT:
            time_evaluate_split += dmlc::GetTime() - tstart;
            break;
          case APPLY_SPLIT:
            time_apply_split += dmlc::GetTime() - tstart;
            break;
        }
        tstart = -1;
      }
    };

    // initialize temp data structure
    void InitData(const GHistIndexMatrix& gmat,
                  const std::vector<GradientPair>& gpair,
                  const DMatrix& fmat,
                  const RegTree& tree);

    void EvaluateSplit(const int nid,
                       const GHistIndexMatrix& gmat,
                       const HistCollection& hist,
                       const DMatrix& fmat,
                       const RegTree& tree);

    void ApplySplit(int nid,
                    const GHistIndexMatrix& gmat,
                    const ColumnMatrix& column_matrix,
                    const HistCollection& hist,
                    const DMatrix& fmat,
                    RegTree* p_tree);

    void ApplySplitDenseData(const RowSetCollection::Elem rowset,
                             const GHistIndexMatrix& gmat,
                             std::vector<RowSetCollection::Split>* p_row_split_tloc,
                             const Column& column,
                             bst_int split_cond,
                             bool default_left);

    void ApplySplitSparseData(const RowSetCollection::Elem rowset,
                              const GHistIndexMatrix& gmat,
                              std::vector<RowSetCollection::Split>* p_row_split_tloc,
                              const Column& column,
                              bst_uint lower_bound,
                              bst_uint upper_bound,
                              bst_int split_cond,
                              bool default_left);

    void AddNodeSplits(RegTree *p_tree);

    void CalculateNodeWeights(RegTree *p_tree);

    void CalculateWeight(int nid,
                         const RegTree &tree,
                         GHistRow hist);

    void InitNewNode(int nid,
                     const GHistIndexMatrix& gmat,
                     const std::vector<GradientPair>& gpair,
                     const DMatrix& fmat,
                     const RegTree& tree);

    // enumerate the split values of specific feature
    void EnumerateSplit(int d_step,
                        const GHistIndexMatrix& gmat,
                        const GHistRow& hist,
                        const NodeEntry& snode,
                        const MetaInfo& info,
                        SplitEntry* p_best,
                        bst_uint fid,
                        bst_uint nodeID);

    void ExpandWithDepthWidth(const GHistIndexMatrix &gmat,
                              const GHistIndexBlockMatrix &gmatb,
                              const ColumnMatrix &column_matrix,
                              DMatrix *p_fmat,
                              RegTree *p_tree,
                              const std::vector<GradientPair> &gpair_h);

    void BuildLocalHistograms(int *starting_index,
                              int *sync_count,
                              const GHistIndexMatrix &gmat,
                              const GHistIndexBlockMatrix &gmatb,
                              RegTree *p_tree,
                              const std::vector<GradientPair> &gpair_h);

    void SyncHistograms(int starting_index,
                        int sync_count,
                        RegTree *p_tree);

    void BuildNodeStats(const GHistIndexMatrix &gmat,
                        DMatrix *p_fmat,
                        RegTree *p_tree,
                        const std::vector<GradientPair> &gpair_h);

    void EvaluateSplits(const GHistIndexMatrix &gmat,
                        const ColumnMatrix &column_matrix,
                        DMatrix *p_fmat,
                        RegTree *p_tree,
                        int *num_leaves,
                        int depth,
                        unsigned *timestamp,
                        std::vector<ExpandEntry> *temp_qexpand_depth);

    void ExpandWithLossGuide(const GHistIndexMatrix& gmat,
                             const GHistIndexBlockMatrix& gmatb,
                             const ColumnMatrix& column_matrix,
                             DMatrix* p_fmat,
                             RegTree* p_tree,
                             const std::vector<GradientPair>& gpair_h);

    inline static bool LossGuide(ExpandEntry lhs, ExpandEntry rhs) {
      if (lhs.loss_chg == rhs.loss_chg) {
        return lhs.timestamp > rhs.timestamp;  // favor small timestamp
      } else {
        return lhs.loss_chg < rhs.loss_chg;  // favor large loss_chg
      }
    }

    //  --data fields--
    const TrainParam& param_;
    // number of omp thread used during training
    int nthread_;
    common::ColumnSampler column_sampler_;
    // the internal row sets
    RowSetCollection row_set_collection_;
    // the temp space for split
    std::vector<RowSetCollection::Split> row_split_tloc_;
    std::vector<SplitEntry> best_split_tloc_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode_;
    /*! \brief culmulative histogram of gradients. */
    HistCollection hist_;
    /*! \brief feature with least # of bins. to be used for dense specialization
               of InitNewNode() */
    uint32_t fid_least_bins_;
    /*! \brief local prediction cache; maps node id to leaf value */
    std::vector<float> leaf_value_cache_;

    GHistBuilder hist_builder_;
    GradStats root_stats;
    std::unique_ptr<TreeUpdater> pruner_;
    std::unique_ptr<SplitEvaluator> spliteval_;

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_;
    const DMatrix* p_last_fmat_;

    using ExpandQueue =
       std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                           std::function<bool(ExpandEntry, ExpandEntry)>>;

    std::unique_ptr<ExpandQueue> qexpand_loss_guided_;
    std::vector<ExpandEntry> qexpand_depth_wise_;
    // key is the node id which should be calculated by Subtraction Trick, value is the node which
    // provides the evidence for substracts
    std::unordered_map<int, int> nodes_for_subtraction_trick_;

    enum DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;

    TreeGrowingPerfMonitor perf_monitor;
    rabit::Reducer<GradStats, GradStats::Reduce> histred_;
  };

  std::unique_ptr<Builder> builder_;
  std::unique_ptr<TreeUpdater> pruner_;
  std::unique_ptr<SplitEvaluator> spliteval_;
};

}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
