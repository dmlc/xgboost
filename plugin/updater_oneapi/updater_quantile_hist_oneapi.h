/*!
 * Copyright 2017-2021 by Contributors
 * \file updater_quantile_hist_oneapi.h
 */
#ifndef XGBOOST_TREE_UPDATER_QUANTILE_HIST_ONEAPI_H_
#define XGBOOST_TREE_UPDATER_QUANTILE_HIST_ONEAPI_H_

#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <queue>

#include "hist_util_oneapi.h"
#include "row_set_oneapi.h"
#include "split_evaluator_oneapi.h"

#include "xgboost/data.h"
#include "xgboost/json.h"
#include "../../src/tree/constraints.h"
#include "../../src/common/random.h"

namespace xgboost {

/*!
 * \brief A C-style array with in-stack allocation.
          As long as the array is smaller than MaxStackSize, it will be allocated inside the stack. Otherwise, it will be heap-allocated.
          Temporary copy of implementation to remove dependency on updater_quantile_hist.h
 */
template<typename T, size_t MaxStackSize>
class MemStackAllocatorOneAPI {
 public:
  explicit MemStackAllocatorOneAPI(size_t required_size): required_size_(required_size) {
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

  ~MemStackAllocatorOneAPI() {
    if (do_free_) free(ptr_);
  }


 private:
  T* ptr_ = nullptr;
  bool do_free_ = false;
  size_t required_size_;
  T stack_mem_[MaxStackSize];
};

namespace tree {

using xgboost::common::HistCollectionOneAPI;
using xgboost::common::GHistBuilderOneAPI;
using xgboost::common::GHistIndexMatrixOneAPI;
using xgboost::common::GHistRowOneAPI;
using xgboost::common::RowSetCollectionOneAPI;

template <typename GradientSumT>
class HistSynchronizerOneAPI;

template <typename GradientSumT>
class BatchHistSynchronizerOneAPI;

template <typename GradientSumT>
class DistributedHistSynchronizerOneAPI;

template <typename GradientSumT>
class HistRowsAdderOneAPI;

template <typename GradientSumT>
class BatchHistRowsAdderOneAPI;

template <typename GradientSumT>
class DistributedHistRowsAdderOneAPI;

// training parameters specific to this algorithm
struct OneAPIHistMakerTrainParam
    : public XGBoostParameter<OneAPIHistMakerTrainParam> {
  bool single_precision_histogram = false;
  // declare parameters
  DMLC_DECLARE_PARAMETER(OneAPIHistMakerTrainParam) {
    DMLC_DECLARE_FIELD(single_precision_histogram).set_default(false).describe(
        "Use single precision to build histograms.");
  }
};

/*! \brief construct a tree using quantized feature values with DPC++ interface */
class QuantileHistMakerOneAPI: public TreeUpdater {
 public:
  explicit QuantileHistMakerOneAPI(ObjInfo task) : task_{task} {}
  void Configure(const Args& args) override;

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix* data,
                             linalg::VectorView<bst_float> out_preds) override;

  void LoadConfig(Json const& in) override {
    if (updater_backend_) {
      updater_backend_->LoadConfig(in);
    } else {
      auto const& config = get<Object const>(in);
      FromJson(config.at("train_param"), &this->param_);
    }
  }

  void SaveConfig(Json* p_out) const override {
    if (updater_backend_) {
      updater_backend_->SaveConfig(p_out);
    } else {
      auto& out = *p_out;
      out["train_param"] = ToJson(param_);
    }
  }

  char const* Name() const override {
    if (updater_backend_) {
        return updater_backend_->Name();
    } else {
        return "grow_quantile_histmaker";
    }
  }

 protected:
  // training parameter
  TrainParam param_;

  ObjInfo task_;
  std::unique_ptr<TreeUpdater> updater_backend_;
};

// data structure
template<typename GradType>
struct NodeEntry {
  /*! \brief statics for node entry */
  GradStatsOneAPI<GradType> stats;
  /*! \brief loss of this node, without split */
  GradType root_gain;
  /*! \brief weight calculated related to current data */
  GradType weight;
  /*! \brief current best solution */
  SplitEntryOneAPI<GradType> best;
  // constructor
  explicit NodeEntry(const TrainParam& param)
      : root_gain(0.0f), weight(0.0f) {}
};
// actual builder that runs the algorithm

/*! \brief construct a tree using quantized feature values with DPC++ backend on GPU*/
class GPUQuantileHistMakerOneAPI: public TreeUpdater {
 public:
  explicit GPUQuantileHistMakerOneAPI(ObjInfo task) : task_{task} {
    updater_monitor_.Init("GPUQuantileHistMakerOneAPI");
  }
  void Configure(const Args& args) override;

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix* data,
                             linalg::VectorView<bst_float> out_preds) override;

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    try {
      FromJson(config.at("oneapi_hist_train_param"), &this->hist_maker_param_);
    } catch (std::out_of_range& e) {
      // XGBoost model is from 1.1.x, so 'cpu_hist_train_param' is missing.
      // We add this compatibility check because it's just recently that we (developers) began
      // persuade R users away from using saveRDS() for model serialization. Hopefully, one day,
      // everyone will be using xgb.save().
      LOG(WARNING) << "Attempted to load interal configuration for a model file that was generated "
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
    out["oneapi_hist_train_param"] = ToJson(hist_maker_param_);
  }

  char const* Name() const override {
    return "grow_quantile_histmaker";
  }

 protected:
  template <typename GradientSumT>
  friend class HistSynchronizerOneAPI;
  template <typename GradientSumT>
  friend class BatchHistSynchronizerOneAPI;
  template <typename GradientSumT>
  friend class DistributedHistSynchronizerOneAPI;

  template <typename GradientSumT>
  friend class HistRowsAdderOneAPI;
  template <typename GradientSumT>
  friend class BatchHistRowsAdderOneAPI;
  template <typename GradientSumT>
  friend class DistributedHistRowsAdderOneAPI;

  OneAPIHistMakerTrainParam hist_maker_param_;
  // training parameter
  TrainParam param_;
  // quantized data matrix
  GHistIndexMatrixOneAPI gmat_;
  // (optional) data matrix with feature grouping
  // column accessor
  DMatrix const* p_last_dmat_ {nullptr};
  bool is_gmat_initialized_ {false};

  template<typename GradientSumT>
  struct Builder {
   public:
    using GHistRowT = GHistRowOneAPI<GradientSumT>;
    using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
    // constructor
    explicit Builder(sycl::queue qu,
                     const TrainParam& param,
                     std::unique_ptr<TreeUpdater> pruner,
                     FeatureInteractionConstraintHost int_constraints_,
                     DMatrix const* fmat)
      : qu_(qu), param_(param),
        tree_evaluator_(qu, param, fmat->Info().num_col_),
        pruner_(std::move(pruner)),
        interaction_constraints_{std::move(int_constraints_)},
        p_last_tree_(nullptr), p_last_fmat_(fmat) {
      builder_monitor_.Init("QuantileOneAPI::Builder");
      kernel_monitor_.Init("QuantileOneAPI::Kernels");
    }
    // update one tree, growing
    virtual void Update(const GHistIndexMatrixOneAPI& gmat,
                        HostDeviceVector<GradientPair>* gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree);

    inline void BuildHist(const std::vector<GradientPair>& gpair,
                          const USMVector<GradientPair>& gpair_device,
                          const RowSetCollectionOneAPI::Elem row_indices,
                          const GHistIndexMatrixOneAPI& gmat,
                          GHistRowT& hist,
                          GHistRowT& hist_buffer) {
      hist_builder_.BuildHist(gpair, gpair_device, row_indices, gmat, hist, data_layout_ != kSparseData, hist_buffer);
    }

    inline void SubtractionTrick(GHistRowT& self,
                                 GHistRowT& sibling,
                                 GHistRowT& parent) {
      builder_monitor_.Start("SubtractionTrick");
      hist_builder_.SubtractionTrick(self, sibling, parent);
      builder_monitor_.Stop("SubtractionTrick");
    }

    bool UpdatePredictionCache(const DMatrix* data,
                               linalg::VectorView<bst_float> p_out_preds);
    void SetHistSynchronizer(HistSynchronizerOneAPI<GradientSumT>* sync);
    void SetHistRowsAdder(HistRowsAdderOneAPI<GradientSumT>* adder);

   protected:
    friend class HistSynchronizerOneAPI<GradientSumT>;
    friend class BatchHistSynchronizerOneAPI<GradientSumT>;
    friend class DistributedHistSynchronizerOneAPI<GradientSumT>;
    friend class HistRowsAdderOneAPI<GradientSumT>;
    friend class BatchHistRowsAdderOneAPI<GradientSumT>;
    friend class DistributedHistRowsAdderOneAPI<GradientSumT>;

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

    struct SplitQuery {
      int nid;
      int fid;
      SplitEntryOneAPI<GradientSumT> best;
      const GradientPairT* hist;
    };

    // initialize temp data structure
    void InitData(const GHistIndexMatrixOneAPI& gmat,
                  const std::vector<GradientPair>& gpair,
                  const USMVector<GradientPair>& gpair_device,
                  const DMatrix& fmat,
                  const RegTree& tree);

    void InitSampling(const std::vector<GradientPair>& gpair,
                      const USMVector<GradientPair>& gpair_device,
                      const DMatrix& fmat, USMVector<size_t>& row_indices);

    void EvaluateSplits(const std::vector<ExpandEntry>& nodes_set,
                        const GHistIndexMatrixOneAPI& gmat,
                        const HistCollectionOneAPI<GradientSumT>& hist,
                        const RegTree& tree);

    // Enumerate the split values of specific feature
    // Returns the sum of gradients corresponding to the data points that contains a non-missing
    // value for the particular feature fid.
    template <int d_step>
    static GradStatsOneAPI<GradientSumT> EnumerateSplit(
        const uint32_t* cut_ptr,const bst_float* cut_val, const bst_float* cut_minval, const GradientPairT* hist_data,
        const NodeEntry<GradientSumT> &snode, SplitEntryOneAPI<GradientSumT>& p_best, bst_uint fid,
        bst_uint nodeID,
        typename TreeEvaluatorOneAPI<GradientSumT>::SplitEvaluator const &evaluator, const TrainParamOneAPI& param);

    static GradStatsOneAPI<GradientSumT> EnumerateSplit(sycl::ext::oneapi::sub_group& sg,
        const uint32_t* cut_ptr, const bst_float* cut_val, const GradientPairT* hist_data,
        const NodeEntry<GradientSumT> &snode, SplitEntryOneAPI<GradientSumT>& p_best, bst_uint fid,
        bst_uint nodeID,
        typename TreeEvaluatorOneAPI<GradientSumT>::SplitEvaluator const &evaluator, const TrainParamOneAPI& param);

    void ApplySplit(std::vector<ExpandEntry> nodes,
                        const GHistIndexMatrixOneAPI& gmat,
                        const HistCollectionOneAPI<GradientSumT>& hist,
                        RegTree* p_tree);

    template <typename BinIdxType>
    void PartitionKernel(const size_t node_in_set, const size_t nid, common::Range1d range,
                         const int32_t split_cond,
                         const GHistIndexMatrixOneAPI &gmat,
                         const RegTree& tree,
                         sycl::buffer<size_t, 1>& parts_size);

    void AddSplitsToRowSet(const std::vector<ExpandEntry>& nodes, RegTree* p_tree);


    void FindSplitConditions(const std::vector<ExpandEntry>& nodes, const RegTree& tree,
                             const GHistIndexMatrixOneAPI& gmat, std::vector<int32_t>* split_conditions);

    void InitNewNode(int nid,
                     const GHistIndexMatrixOneAPI& gmat,
                     const std::vector<GradientPair>& gpair,
                     const USMVector<GradientPair>& gpair_device,
                     const DMatrix& fmat,
                     const RegTree& tree);

    // if sum of statistics for non-missing values in the node
    // is equal to sum of statistics for all values:
    // then - there are no missing values
    // else - there are missing values
    static bool SplitContainsMissingValues(const GradStatsOneAPI<GradientSumT>& e, const NodeEntry<GradientSumT>& snode);

    void ExpandWithDepthWise(const GHistIndexMatrixOneAPI &gmat,
                             DMatrix *p_fmat,
                             RegTree *p_tree,
                             const std::vector<GradientPair> &gpair_h,
                             const USMVector<GradientPair> &gpair_device);

    void BuildLocalHistograms(const GHistIndexMatrixOneAPI &gmat,
                              RegTree *p_tree,
                              const std::vector<GradientPair> &gpair_h,
                              const USMVector<GradientPair> &gpair_device);

    void BuildHistogramsLossGuide(
                        ExpandEntry entry,
                        const GHistIndexMatrixOneAPI &gmat,
                        RegTree *p_tree,
                        const std::vector<GradientPair> &gpair_h,
                        const USMVector<GradientPair> &gpair_device);

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

    void BuildNodeStats(const GHistIndexMatrixOneAPI &gmat,
                        DMatrix *p_fmat,
                        RegTree *p_tree,
                        const std::vector<GradientPair> &gpair_h,
                        const USMVector<GradientPair> &gpair_device);

    void EvaluateAndApplySplits(const GHistIndexMatrixOneAPI &gmat,
                                RegTree *p_tree,
                                int *num_leaves,
                                int depth,
                                unsigned *timestamp,
                                std::vector<ExpandEntry> *temp_qexpand_depth);

    void AddSplitsToTree(
              const GHistIndexMatrixOneAPI &gmat,
              RegTree *p_tree,
              int *num_leaves,
              int depth,
              unsigned *timestamp,
              std::vector<ExpandEntry>* nodes_for_apply_split,
              std::vector<ExpandEntry>* temp_qexpand_depth);

    void ExpandWithLossGuide(const GHistIndexMatrixOneAPI& gmat,
                             DMatrix* p_fmat,
                             RegTree* p_tree,
                             const std::vector<GradientPair>& gpair_h,
                             const USMVector<GradientPair>& gpair_device);

    void ReduceHists(std::vector<int>& sync_ids, size_t nbins);

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
    RowSetCollectionOneAPI row_set_collection_;
    USMVector<SplitQuery> split_queries_device_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    USMVector<NodeEntry<GradientSumT>> snode_;
    /*! \brief culmulative histogram of gradients. */
    HistCollectionOneAPI<GradientSumT> hist_;
    /*! \brief culmulative local parent histogram of gradients. */
    HistCollectionOneAPI<GradientSumT> hist_local_worker_;
    TreeEvaluatorOneAPI<GradientSumT> tree_evaluator_;
    /*! \brief feature with least # of bins. to be used for dense specialization
               of InitNewNode() */
    uint32_t fid_least_bins_;

    GHistBuilderOneAPI<GradientSumT> hist_builder_;
    std::unique_ptr<TreeUpdater> pruner_;
    FeatureInteractionConstraintHost interaction_constraints_;

    common::PartitionBuilderOneAPI partition_builder_;

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

    enum DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;

    common::Monitor builder_monitor_;
    common::Monitor kernel_monitor_;
    common::ParallelGHistBuilderOneAPI<GradientSumT> hist_buffer_;
    rabit::Reducer<GradientPairT, GradientPairT::Reduce> histred_;
    std::unique_ptr<HistSynchronizerOneAPI<GradientSumT>> hist_synchronizer_;
    std::unique_ptr<HistRowsAdderOneAPI<GradientSumT>> hist_rows_adder_;

    sycl::queue qu_;
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

  sycl::queue qu_;

  ObjInfo task_;
};

template <typename GradientSumT>
class HistSynchronizerOneAPI {
 public:
  using BuilderT = GPUQuantileHistMakerOneAPI::Builder<GradientSumT>;

  virtual void SyncHistograms(BuilderT* builder,
                              std::vector<int>& sync_ids,
                              RegTree *p_tree) = 0;
  virtual ~HistSynchronizerOneAPI() = default;
};

template <typename GradientSumT>
class BatchHistSynchronizerOneAPI: public HistSynchronizerOneAPI<GradientSumT> {
 public:
  using BuilderT = GPUQuantileHistMakerOneAPI::Builder<GradientSumT>;
  void SyncHistograms(BuilderT* builder,
                      std::vector<int>& sync_ids,
                      RegTree *p_tree) override;
};

template <typename GradientSumT>
class DistributedHistSynchronizerOneAPI: public HistSynchronizerOneAPI<GradientSumT> {
 public:
  using BuilderT = GPUQuantileHistMakerOneAPI::Builder<GradientSumT>;
  using ExpandEntryT = typename BuilderT::ExpandEntry;

  void SyncHistograms(BuilderT* builder, std::vector<int>& sync_ids, RegTree *p_tree) override;

  void ParallelSubtractionHist(BuilderT* builder,
                               const std::vector<ExpandEntryT>& nodes,
                               const RegTree * p_tree);
};

template <typename GradientSumT>
class HistRowsAdderOneAPI {
 public:
  using BuilderT = GPUQuantileHistMakerOneAPI::Builder<GradientSumT>;

  virtual void AddHistRows(BuilderT* builder, std::vector<int>& sync_ids, RegTree *p_tree) = 0;
  virtual ~HistRowsAdderOneAPI() = default;
};

template <typename GradientSumT>
class BatchHistRowsAdderOneAPI: public HistRowsAdderOneAPI<GradientSumT> {
 public:
  using BuilderT = GPUQuantileHistMakerOneAPI::Builder<GradientSumT>;
  void AddHistRows(BuilderT*, std::vector<int>& sync_ids, RegTree *p_tree) override;
};

template <typename GradientSumT>
class DistributedHistRowsAdderOneAPI: public HistRowsAdderOneAPI<GradientSumT> {
 public:
  using BuilderT = GPUQuantileHistMakerOneAPI::Builder<GradientSumT>;
  void AddHistRows(BuilderT*, std::vector<int>& sync_ids, RegTree *p_tree) override;
};


}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_QUANTILE_HIST_ONEAPI_H_
