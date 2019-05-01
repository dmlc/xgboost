/*!
 * Copyright 2017-2019 by Contributors
 * \file updater_quantile_hist.h
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Chen, Egor Smirnov
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
#include <deque>
#include <iomanip>
#include <unordered_map>
#include <utility>

#include "./param.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"

namespace xgboost {

/*!
 * \brief A C-style array with in-stack allocation. As long as the array is smaller than MaxStackSize, it will be allocated inside the stack. Otherwise, it will be heap-allocated.
 */
template<typename T, size_t MaxStackSize>
class MemStackAllocator {
 public:
  explicit MemStackAllocator(size_t required_size): required_size_(required_size) {
  }

  T* Get() {
    if (!ptr_) {
      if (MaxStackSize >= required_size_) {
        ptr_ = stack_mem_;
      } else {
        ptr_ =  reinterpret_cast<T*>(malloc(required_size_ * sizeof(T)));
        do_free_ = true;
      }
    }

    return ptr_;
  }

  ~MemStackAllocator() {
    if (do_free_) free(ptr_);
  }


 private:
  T* ptr_ = nullptr;
  bool do_free_ = false;
  size_t required_size_;
  T stack_mem_[MaxStackSize];
};

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


class RegTreeThreadSafe;

template<typename T, typename InitFunc, typename DeleteFunc>
class ThreadSafeStorage {
 public:
  ThreadSafeStorage(size_t nthread, InitFunc init, DeleteFunc del): nthread_(nthread),
    thread_local_storages_(nthread), mutexes_(nthread),
    init_func_(init), delete_func_(del) {
  }

  ~ThreadSafeStorage() {
    for (size_t i = 0; i < nthread_; ++i) {
      for (auto& it : thread_local_storages_[i])
        delete_func_(it);
    }
  }

  std::pair<T*, size_t> get() {
    const unsigned tid = omp_get_thread_num();
    std::lock_guard<std::mutex> lock(mutexes_[tid]);
    if (!thread_local_storages_[tid].size()) {
      return {init_func_(), tid};
    } else {
      T* ptr = thread_local_storages_[tid].back();
      thread_local_storages_[tid].pop_back();
      return {ptr, tid};
    }
  }

  std::pair<T*, size_t> get(size_t tid) {
    std::lock_guard<std::mutex> lock(mutexes_[tid]);
    if (!thread_local_storages_[tid].size()) {
      return {init_func_(), tid};
    } else {
      T* ptr = thread_local_storages_[tid].back();
      thread_local_storages_[tid].pop_back();
      return {ptr, tid};
    }
  }

  void release(std::pair<T*, size_t> pair) {
    const unsigned tid = pair.second;

    std::lock_guard<std::mutex> lock(mutexes_[tid]);
    thread_local_storages_[tid].push_back(pair.first);
  }

 private:
  size_t nthread_;
  std::vector<std::deque<T*>> thread_local_storages_;
  std::vector<std::mutex> mutexes_;
  InitFunc init_func_;
  DeleteFunc delete_func_;
};


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
 public:
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
        : root_gain(0.0f), weight(0.0f) {
    }
  };
  // actual builder that runs the algorithm

  struct Builder {
   public:
    using RowCollectionTLS = ThreadSafeStorage<RowSetCollection::Split,
      std::function<RowSetCollection::Split*()>,
      std::function<void(RowSetCollection::Split*)>>;

    using HistTLS = ThreadSafeStorage<tree::GradStats,
      std::function<tree::GradStats*()>,
      std::function<void(tree::GradStats*)>>;

    // constructor
    explicit Builder(const TrainParam& param,
                     std::unique_ptr<TreeUpdater> pruner,
                     std::unique_ptr<SplitEvaluator> spliteval)
      : prow_set_collection_tls_(nullptr), param_(param),
      pruner_(std::move(pruner)), spliteval_(std::move(spliteval)),
      p_last_tree_(nullptr), p_last_fmat_(nullptr) {  }
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
                          RegTreeThreadSafe* tree,
                          size_t parent_nid,
                          GHistRow sibling,
                          GHistRow parent,
                          int32_t this_nid,
                          int32_t another_nid,
                          bool sync_hist);

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
                  << std::fixed << std::setw(6) << std::setprecision(4) << total_time << std::endl;
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
        // CHECK_GT(tstart, 0); // TODO Fix
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
                       QuantileHistMaker::NodeEntry* snode,
                       int32_t depth);

    RegTree::Node ApplySplit(int nid,
                    const GHistIndexMatrix& gmat,
                    const ColumnMatrix& column_matrix,
                    const HistCollection& hist,
                    const DMatrix& fmat,
                    RegTreeThreadSafe* p_tree,
                    const QuantileHistMaker::NodeEntry& snode,
                    const RegTree::Node node);

    size_t ApplySplitDenseData(const RowSetCollection::Elem rowset,
                             const GHistIndexMatrix& gmat,
                             const Column& column,
                             bst_int split_cond,
                             bool default_left);

    size_t ApplySplitSparseData(const RowSetCollection::Elem rowset,
                              const GHistIndexMatrix& gmat,
                              const Column& column,
                              bst_uint lower_bound,
                              bst_uint upper_bound,
                              bst_int split_cond,
                              bool default_left);

    size_t MergeSplit(std::pair<RowSetCollection::Split*, size_t>* arr,
      std::pair<size_t, size_t>* sizes,
      size_t nblocks,
      size_t* rowset_begin);

    void InitNewNode(int nid,
                     const GHistIndexMatrix& gmat,
                     const std::vector<GradientPair>& gpair,
                     const DMatrix& fmat,
                     RegTreeThreadSafe* tree,
                     QuantileHistMaker::NodeEntry* snode,
                     int32_t parentid);

    // enumerate the split values of specific feature
    bool EnumerateSplit(int d_step,
                        const GHistIndexMatrix& gmat,
                        const GHistRow& hist,
                        const NodeEntry& snode,
                        const MetaInfo& info,
                        SplitEntry* p_best,
                        bst_uint fid,
                        bst_uint nodeID);

    void ExpandWithDepthWidthDistributed(const GHistIndexMatrix &gmat,
                              const GHistIndexBlockMatrix &gmatb,
                              const ColumnMatrix &column_matrix,
                              DMatrix *p_fmat,
                              RegTreeThreadSafe *p_tree,
                              const std::vector<GradientPair> &gpair_h);

    void ExpandWithDepthWidth(const GHistIndexMatrix &gmat,
                              const GHistIndexBlockMatrix &gmatb,
                              const ColumnMatrix &column_matrix,
                              DMatrix *p_fmat,
                              RegTreeThreadSafe *p_tree,
                              const std::vector<GradientPair> &gpair_h);


    void ExpandWithLossGuide(const GHistIndexMatrix& gmat,
                             const GHistIndexBlockMatrix& gmatb,
                             const ColumnMatrix& column_matrix,
                             DMatrix* p_fmat,
                             RegTreeThreadSafe* p_tree,
                             const std::vector<GradientPair>& gpair_h);

    void SyncHistograms(int starting_index,
                        int sync_count,
                        RegTreeThreadSafe *p_tree);

    inline void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent);

    void BuildNodeStat(const GHistIndexMatrix &gmat,
                        DMatrix *p_fmat,
                        RegTreeThreadSafe *p_tree,
                        const std::vector<GradientPair> &gpair_h,
                        int32_t nid);

    void CreateNewNodes(const GHistIndexMatrix &gmat,
        const ColumnMatrix &column_matrix,
        DMatrix *p_fmat,
        RegTreeThreadSafe *p_tree,
        int *num_leaves,
        int depth,
        unsigned *timestamp,
        std::vector<ExpandEntry> *temp_qexpand_depth,
        int32_t nid,
        std::mutex* mutex_add_nodes,
        const QuantileHistMaker::NodeEntry& snode,
        RegTree::Node node);

    inline static bool LossGuide(ExpandEntry lhs, ExpandEntry rhs) {
      if (lhs.loss_chg == rhs.loss_chg) {
        return lhs.timestamp > rhs.timestamp;  // favor small timestamp
      } else {
        return lhs.loss_chg < rhs.loss_chg;  // favor large loss_chg
      }
    }

    std::unique_ptr<RowCollectionTLS> prow_set_collection_tls_;
    std::unique_ptr<HistTLS> hist_tls_;

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

    GHistBuilder<HistTLS> hist_builder_;
    std::vector<GHistBuilder<HistTLS>> hist_builder_arr_;
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
