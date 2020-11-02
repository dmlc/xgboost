/*!
 * Copyright 2017-2018 by Contributors
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
#include <iomanip>
#include <unordered_map>
#include <utility>

#include "xgboost/data.h"
#include "xgboost/json.h"
#include "constraints.h"
#include "./param.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/timer.h"
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

using xgboost::common::GHistIndexMatrix;
using xgboost::common::GHistIndexBlockMatrix;
using xgboost::common::GHistIndexRow;
using xgboost::common::HistCollection;
using xgboost::common::RowSetCollection;
using xgboost::common::GHistRow;
using xgboost::common::GHistBuilder;
using xgboost::common::ColumnMatrix;
using xgboost::common::Column;

template <typename GradientSumT>
class HistSynchronizer;

template <typename GradientSumT>
class BatchHistSynchronizer;

template <typename GradientSumT>
class DistributedHistSynchronizer;

template <typename GradientSumT>
class HistRowsAdder;

template <typename GradientSumT>
class BatchHistRowsAdder;

template <typename GradientSumT>
class DistributedHistRowsAdder;

// training parameters specific to this algorithm
struct CPUHistMakerTrainParam
    : public XGBoostParameter<CPUHistMakerTrainParam> {
  bool single_precision_histogram = false;
  // declare parameters
  DMLC_DECLARE_PARAMETER(CPUHistMakerTrainParam) {
    DMLC_DECLARE_FIELD(single_precision_histogram).set_default(false).describe(
        "Use single precision to build histograms.");
  }
};

/*! \brief construct a tree using quantized feature values */
class QuantileHistMaker: public TreeUpdater {
 public:
  QuantileHistMaker() {
    updater_monitor_.Init("QuantileHistMaker");
  }
  void Configure(const Args& args) override;

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix* data,
                             HostDeviceVector<bst_float>* out_preds) override;

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    try {
      FromJson(config.at("cpu_hist_train_param"), &this->hist_maker_param_);
    } catch (std::out_of_range&) {
      // XGBoost model is from 1.1.x, so 'cpu_hist_train_param' is missing.
      // We add this compatibility check because it's just recently that we (developers) began
      // persuade R users away from using saveRDS() for model serialization. Hopefully, one day,
      // everyone will be using xgb.save().
      LOG(WARNING)
        << "Attempted to load internal configuration for a model file that was generated "
        << "by a previous version of XGBoost. A likely cause for this warning is that the model "
        << "was saved with saveRDS() in R or pickle.dump() in Python. We strongly ADVISE AGAINST "
        << "using saveRDS() or pickle.dump() so that the model remains accessible in current and "
        << "upcoming XGBoost releases. Please use xgb.save() instead to preserve models for the "
        << "long term. For more details and explanation, see "
        << "https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html";
      this->hist_maker_param_.UpdateAllowUnknown(Args{});
    }
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["cpu_hist_train_param"] = ToJson(hist_maker_param_);
  }

  char const* Name() const override {
    return "grow_quantile_histmaker";
  }

 protected:
  template <typename GradientSumT>
  friend class HistSynchronizer;
  template <typename GradientSumT>
  friend class BatchHistSynchronizer;
  template <typename GradientSumT>
  friend class DistributedHistSynchronizer;

  template <typename GradientSumT>
  friend class HistRowsAdder;
  template <typename GradientSumT>
  friend class BatchHistRowsAdder;
  template <typename GradientSumT>
  friend class DistributedHistRowsAdder;

  CPUHistMakerTrainParam hist_maker_param_;
  // training parameter
  TrainParam param_;
  // quantized data matrix
  GHistIndexMatrix gmat_;
  // (optional) data matrix with feature grouping
  GHistIndexBlockMatrix gmatb_;
  // column accessor
  ColumnMatrix column_matrix_;
  DMatrix const* p_last_dmat_ {nullptr};
  bool is_gmat_initialized_ {false};

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
    explicit NodeEntry(const TrainParam&)
        : root_gain(0.0f), weight(0.0f) {}
  };
  // actual builder that runs the algorithm

  template<typename GradientSumT>
  struct Builder {
   public:
    using GHistRowT = GHistRow<GradientSumT>;
    using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
    // constructor
    explicit Builder(const TrainParam& param,
                     std::unique_ptr<TreeUpdater> pruner,
                     FeatureInteractionConstraintHost int_constraints_,
                     DMatrix const* fmat)
      : param_(param),
        tree_evaluator_(param, fmat->Info().num_col_, GenericParameter::kCpuId),
        pruner_(std::move(pruner)),
        interaction_constraints_{std::move(int_constraints_)},
        p_last_tree_(nullptr), p_last_fmat_(fmat) {
      builder_monitor_.Init("Quantile::Builder");
    }
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
                          GHistRowT hist) {
      if (param_.enable_feature_grouping > 0) {
        hist_builder_.BuildBlockHist(gpair, row_indices, gmatb, hist);
      } else {
        hist_builder_.BuildHist(gpair, row_indices, gmat, hist,
                                data_layout_ != DataLayout::kSparseData);
      }
    }

    inline void SubtractionTrick(GHistRowT self,
                                 GHistRowT sibling,
                                 GHistRowT parent) {
      builder_monitor_.Start("SubtractionTrick");
      hist_builder_.SubtractionTrick(self, sibling, parent);
      builder_monitor_.Stop("SubtractionTrick");
    }

    bool UpdatePredictionCache(const DMatrix* data,
                               HostDeviceVector<bst_float>* p_out_preds);
    void SetHistSynchronizer(HistSynchronizer<GradientSumT>* sync);
    void SetHistRowsAdder(HistRowsAdder<GradientSumT>* adder);

   protected:
    friend class HistSynchronizer<GradientSumT>;
    friend class BatchHistSynchronizer<GradientSumT>;
    friend class DistributedHistSynchronizer<GradientSumT>;
    friend class HistRowsAdder<GradientSumT>;
    friend class BatchHistRowsAdder<GradientSumT>;
    friend class DistributedHistRowsAdder<GradientSumT>;

    /* tree growing policies */
    struct ExpandEntry {
      static const int kRootNid  = 0;
      static const int kEmptyNid = -1;
      int nid;
      int sibling_nid;
      int depth;
      bst_float loss_chg;
      unsigned timestamp;
      ExpandEntry(int nid, int sibling_nid, int depth, bst_float loss_chg,
                  unsigned tstmp)
          : nid(nid), sibling_nid(sibling_nid), depth(depth),
            loss_chg(loss_chg), timestamp(tstmp) {}

      bool IsValid(TrainParam const &param, int32_t num_leaves) const {
        bool ret = loss_chg <= kRtEps ||
                   (param.max_depth > 0 && this->depth == param.max_depth) ||
                   (param.max_leaves > 0 && num_leaves == param.max_leaves);
        return ret;
      }
    };

    // initialize temp data structure
    void InitData(const GHistIndexMatrix& gmat,
                  const std::vector<GradientPair>& gpair,
                  const DMatrix& fmat,
                  const RegTree& tree);

    void InitSampling(const std::vector<GradientPair>& gpair,
                      const DMatrix& fmat, std::vector<size_t>* row_indices);

    void EvaluateSplits(const std::vector<ExpandEntry>& nodes_set,
                        const GHistIndexMatrix& gmat,
                        const HistCollection<GradientSumT>& hist,
                        const RegTree& tree);

    void ApplySplit(std::vector<ExpandEntry> nodes,
                        const GHistIndexMatrix& gmat,
                        const ColumnMatrix& column_matrix,
                        const HistCollection<GradientSumT>& hist,
                        RegTree* p_tree);

    template <typename BinIdxType>
    void PartitionKernel(const size_t node_in_set, const size_t nid, common::Range1d range,
                         const int32_t split_cond,
                         const ColumnMatrix& column_matrix, const RegTree& tree);

    void AddSplitsToRowSet(const std::vector<ExpandEntry>& nodes, RegTree* p_tree);


    void FindSplitConditions(const std::vector<ExpandEntry>& nodes, const RegTree& tree,
                             const GHistIndexMatrix& gmat, std::vector<int32_t>* split_conditions);

    void InitNewNode(int nid,
                     const GHistIndexMatrix& gmat,
                     const std::vector<GradientPair>& gpair,
                     const DMatrix& fmat,
                     const RegTree& tree);

    // Enumerate the split values of specific feature
    // Returns the sum of gradients corresponding to the data points that contains a non-missing
    // value for the particular feature fid.
    template <int d_step>
    GradStats EnumerateSplit(
        const GHistIndexMatrix &gmat, const GHistRowT &hist,
        const NodeEntry &snode, SplitEntry *p_best, bst_uint fid,
        bst_uint nodeID,
        TreeEvaluator::SplitEvaluator<TrainParam> const &evaluator) const;

    // if sum of statistics for non-missing values in the node
    // is equal to sum of statistics for all values:
    // then - there are no missing values
    // else - there are missing values
    bool SplitContainsMissingValues(const GradStats e, const NodeEntry& snode);

    void ExpandWithDepthWise(const GHistIndexMatrix &gmat,
                             const GHistIndexBlockMatrix &gmatb,
                             const ColumnMatrix &column_matrix,
                             DMatrix *p_fmat,
                             RegTree *p_tree,
                             const std::vector<GradientPair> &gpair_h);

    void BuildLocalHistograms(const GHistIndexMatrix &gmat,
                              const GHistIndexBlockMatrix &gmatb,
                              RegTree *p_tree,
                              const std::vector<GradientPair> &gpair_h);

    void BuildHistogramsLossGuide(
                        ExpandEntry entry,
                        const GHistIndexMatrix &gmat,
                        const GHistIndexBlockMatrix &gmatb,
                        RegTree *p_tree,
                        const std::vector<GradientPair> &gpair_h);

    // Split nodes to 2 sets depending on amount of rows in each node
    // Histograms for small nodes will be built explicitly
    // Histograms for big nodes will be built by 'Subtraction Trick'
    void SplitSiblings(const std::vector<ExpandEntry>& nodes,
                   std::vector<ExpandEntry>* small_siblings,
                   std::vector<ExpandEntry>* big_siblings,
                   RegTree *p_tree);

    void ParallelSubtractionHist(const common::BlockedSpace2d& space,
                                 const std::vector<ExpandEntry>& nodes,
                                 const RegTree * p_tree);

    void BuildNodeStats(const GHistIndexMatrix &gmat,
                        DMatrix *p_fmat,
                        RegTree *p_tree,
                        const std::vector<GradientPair> &gpair_h);

    void EvaluateAndApplySplits(const GHistIndexMatrix &gmat,
                                const ColumnMatrix &column_matrix,
                                RegTree *p_tree,
                                int *num_leaves,
                                int depth,
                                unsigned *timestamp,
                                std::vector<ExpandEntry> *temp_qexpand_depth);

    void AddSplitsToTree(
              const GHistIndexMatrix &gmat,
              RegTree *p_tree,
              int *num_leaves,
              int depth,
              unsigned *timestamp,
              std::vector<ExpandEntry>* nodes_for_apply_split,
              std::vector<ExpandEntry>* temp_qexpand_depth);

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
    HistCollection<GradientSumT> hist_;
    /*! \brief culmulative local parent histogram of gradients. */
    HistCollection<GradientSumT> hist_local_worker_;
    TreeEvaluator tree_evaluator_;
    /*! \brief feature with least # of bins. to be used for dense specialization
               of InitNewNode() */
    uint32_t fid_least_bins_;
    /*! \brief local prediction cache; maps node id to leaf value */
    std::vector<float> leaf_value_cache_;

    GHistBuilder<GradientSumT> hist_builder_;
    std::unique_ptr<TreeUpdater> pruner_;
    FeatureInteractionConstraintHost interaction_constraints_;

    static constexpr size_t kPartitionBlockSize = 2048;
    common::PartitionBuilder<kPartitionBlockSize> partition_builder_;

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_;
    DMatrix const* const p_last_fmat_;

    using ExpandQueue =
       std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                           std::function<bool(ExpandEntry, ExpandEntry)>>;

    std::unique_ptr<ExpandQueue> qexpand_loss_guided_;
    std::vector<ExpandEntry> qexpand_depth_wise_;
    // key is the node id which should be calculated by Subtraction Trick, value is the node which
    // provides the evidence for substracts
    std::vector<ExpandEntry> nodes_for_subtraction_trick_;
    // list of nodes whose histograms would be built explicitly.
    std::vector<ExpandEntry> nodes_for_explicit_hist_build_;

    enum class DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;

    common::Monitor builder_monitor_;
    common::ParallelGHistBuilder<GradientSumT> hist_buffer_;
    rabit::Reducer<GradientPairT, GradientPairT::Reduce> histred_;
    std::unique_ptr<HistSynchronizer<GradientSumT>> hist_synchronizer_;
    std::unique_ptr<HistRowsAdder<GradientSumT>> hist_rows_adder_;
  };
  common::Monitor updater_monitor_;

  template<typename GradientSumT>
  void SetBuilder(std::unique_ptr<Builder<GradientSumT>>*, DMatrix *dmat);

  template<typename GradientSumT>
  void CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                         HostDeviceVector<GradientPair> *gpair,
                         DMatrix *dmat,
                         const std::vector<RegTree *> &trees);

 protected:
  std::unique_ptr<Builder<float>> float_builder_;
  std::unique_ptr<Builder<double>> double_builder_;

  std::unique_ptr<TreeUpdater> pruner_;
  FeatureInteractionConstraintHost int_constraint_;
};

template <typename GradientSumT>
class HistSynchronizer {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;

  virtual void SyncHistograms(BuilderT* builder,
                              int starting_index,
                              int sync_count,
                              RegTree *p_tree) = 0;
  virtual ~HistSynchronizer() = default;
};

template <typename GradientSumT>
class BatchHistSynchronizer: public HistSynchronizer<GradientSumT> {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;
  void SyncHistograms(BuilderT* builder,
                      int starting_index,
                      int sync_count,
                      RegTree *p_tree) override;
};

template <typename GradientSumT>
class DistributedHistSynchronizer: public HistSynchronizer<GradientSumT> {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;
  using ExpandEntryT = typename BuilderT::ExpandEntry;

  void SyncHistograms(BuilderT* builder, int starting_index,
                      int sync_count, RegTree *p_tree) override;

  void ParallelSubtractionHist(BuilderT* builder,
                               const common::BlockedSpace2d& space,
                               const std::vector<ExpandEntryT>& nodes,
                               const RegTree * p_tree);
};

template <typename GradientSumT>
class HistRowsAdder {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;

  virtual void AddHistRows(BuilderT* builder, int *starting_index,
                           int *sync_count, RegTree *p_tree) = 0;
  virtual ~HistRowsAdder() = default;
};

template <typename GradientSumT>
class BatchHistRowsAdder: public HistRowsAdder<GradientSumT> {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;
  void AddHistRows(BuilderT*, int *starting_index,
                   int *sync_count, RegTree *p_tree) override;
};

template <typename GradientSumT>
class DistributedHistRowsAdder: public HistRowsAdder<GradientSumT> {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;
  void AddHistRows(BuilderT*, int *starting_index,
                   int *sync_count, RegTree *p_tree) override;
};


}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
