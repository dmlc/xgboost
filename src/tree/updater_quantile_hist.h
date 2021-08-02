/*!
 * Copyright 2017-2021 by Contributors
 * \file updater_quantile_hist.h
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Chen, Egor Smirnov
 */
#ifndef XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
#define XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_

#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <iomanip>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "xgboost/data.h"
#include "xgboost/json.h"

#include "hist/evaluate_splits.h"
#include "constraints.h"
#include "./param.h"
#include "./driver.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/timer.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/partition_builder.h"
#include "../common/column_matrix.h"

namespace xgboost {


struct RandomReplace {
 public:
  // similar value as for minstd_rand
  static constexpr uint64_t kBase = 16807;
  static constexpr uint64_t kMod = static_cast<uint64_t>(1) << 63;

  using EngineT = std::linear_congruential_engine<uint64_t, kBase, 0, kMod>;

  /*
    Right-to-left binary method: https://en.wikipedia.org/wiki/Modular_exponentiation
  */
  static uint64_t SimpleSkip(uint64_t exponent, uint64_t initial_seed,
                             uint64_t base, uint64_t mod) {
    CHECK_LE(exponent, mod);
    uint64_t result = 1;
    while (exponent > 0) {
      if (exponent % 2 == 1) {
        result = (result * base) % mod;
      }
      base = (base * base) % mod;
      exponent = exponent >> 1;
    }
    // with result we can now find the new seed
    return (result * initial_seed) % mod;
  }

  template<typename Condition, typename ContainerData>
  static void MakeIf(Condition condition, const typename ContainerData::value_type replace_value,
                     const uint64_t initial_seed, const size_t ibegin,
                     const size_t iend, ContainerData* gpair) {
    ContainerData& gpair_ref = *gpair;
    const uint64_t displaced_seed = SimpleSkip(ibegin, initial_seed, kBase, kMod);
    EngineT eng(displaced_seed);
    for (size_t i = ibegin; i < iend; ++i) {
      if (condition(i, eng)) {
        gpair_ref[i] = replace_value;
      }
    }
  }
};

namespace tree {

using xgboost::GHistIndexMatrix;
using xgboost::common::GHistIndexRow;
using xgboost::common::HistCollection;
using xgboost::common::RowSetCollection;
using xgboost::common::GHistRow;
using xgboost::common::GHistBuilder;
using xgboost::common::ColumnMatrix;
using xgboost::common::Column;

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

/* tree growing policies */
struct CPUExpandEntry {
  static const int kRootNid  = 0;
  static const int kEmptyNid = -1;
  int nid;
  int depth;
  SplitEntry split;

  CPUExpandEntry() = default;
  CPUExpandEntry(int nid, int depth, bst_float loss_chg)
      : nid(nid), depth(depth) {
    split.loss_chg = loss_chg;
  }

  bool IsValid(TrainParam const &param, int32_t num_leaves) const {
    bool invalid = split.loss_chg <= kRtEps ||
                   (param.max_depth > 0 && this->depth == param.max_depth) ||
                   (param.max_leaves > 0 && num_leaves == param.max_leaves);
    return !invalid;
  }

  bst_float GetLossChange() const {
    return split.loss_chg;
  }

  int GetNodeId() const {
    return nid;
  }

  int GetDepth() const {
    return depth;
  }
};

template <typename GradientSumT>
class HistogramBuilder;

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

  bool UpdatePredictionCache(const DMatrix *data,
                             VectorView<float> out_preds) override;

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
  // column accessor
  ColumnMatrix column_matrix_;
  DMatrix const* p_last_dmat_ {nullptr};
  bool is_gmat_initialized_ {false};

  // actual builder that runs the algorithm
  template<typename GradientSumT>
  struct Builder {
   public:
    using GHistRowT = GHistRow<GradientSumT>;
    using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
    // constructor
    explicit Builder(const size_t n_trees, const TrainParam &param,
                     std::unique_ptr<TreeUpdater> pruner, DMatrix const *fmat)
        : n_trees_(n_trees), param_(param), pruner_(std::move(pruner)),
          p_last_tree_(nullptr), p_last_fmat_(fmat), histogram_builder_{new HistogramBuilder<GradientSumT>} {
      builder_monitor_.Init("Quantile::Builder");
    }
    ~Builder();
    // update one tree, growing
    virtual void Update(const GHistIndexMatrix& gmat,
                        const ColumnMatrix& column_matrix,
                        HostDeviceVector<GradientPair>* gpair,
                        DMatrix* p_fmat,
                        RegTree* p_tree);

    inline void SubtractionTrick(GHistRowT self,
                                 GHistRowT sibling,
                                 GHistRowT parent) {
      builder_monitor_.Start("SubtractionTrick");
      // hist_builder_.SubtractionTrick(self, sibling, parent);
      builder_monitor_.Stop("SubtractionTrick");
    }

    bool UpdatePredictionCache(const DMatrix* data,
                               VectorView<float> out_preds);

   protected:
    // initialize temp data structure
    void InitData(const GHistIndexMatrix& gmat,
                  const DMatrix& fmat,
                  const RegTree& tree,
                  std::vector<GradientPair>* gpair);

    size_t GetNumberOfTrees();

    void InitSampling(const DMatrix& fmat,
                      std::vector<GradientPair>* gpair,
                      std::vector<size_t>* row_indices);

    template <bool any_missing>
    void ApplySplit(std::vector<CPUExpandEntry> nodes,
                        const GHistIndexMatrix& gmat,
                        const ColumnMatrix& column_matrix,
                        RegTree* p_tree);

    void AddSplitsToRowSet(const std::vector<CPUExpandEntry>& nodes, RegTree* p_tree);


    void FindSplitConditions(const std::vector<CPUExpandEntry>& nodes, const RegTree& tree,
                             const GHistIndexMatrix& gmat, std::vector<int32_t>* split_conditions);

    template <bool any_missing>
    void InitRoot(DMatrix* p_fmat,
                  RegTree *p_tree,
                  const std::vector<GradientPair> &gpair_h,
                  int *num_leaves, std::vector<CPUExpandEntry> *expand);

    // Split nodes to 2 sets depending on amount of rows in each node
    // Histograms for small nodes will be built explicitly
    // Histograms for big nodes will be built by 'Subtraction Trick'
    void SplitSiblings(const std::vector<CPUExpandEntry>& nodes,
                       std::vector<CPUExpandEntry>* nodes_to_evaluate,
                       RegTree *p_tree);

    void AddSplitsToTree(const std::vector<CPUExpandEntry>& expand,
                         RegTree *p_tree,
                         int *num_leaves,
                         std::vector<CPUExpandEntry>* nodes_for_apply_split);

    template <bool any_missing>
    void ExpandTree(const GHistIndexMatrix& gmat,
                    const ColumnMatrix& column_matrix,
                    DMatrix* p_fmat,
                    RegTree* p_tree,
                    const std::vector<GradientPair>& gpair_h);

    //  --data fields--
    const size_t n_trees_;
    const TrainParam& param_;
    // number of omp thread used during training
    int nthread_;
    std::shared_ptr<common::ColumnSampler> column_sampler_{
        std::make_shared<common::ColumnSampler>()};

    std::vector<size_t> unused_rows_;
    // the internal row sets
    RowSetCollection row_set_collection_;
    std::vector<GradientPair> gpair_local_;

    /*! \brief feature with least # of bins. to be used for dense specialization
               of InitNewNode() */
    uint32_t fid_least_bins_;

    std::unique_ptr<TreeUpdater> pruner_;
    std::unique_ptr<HistEvaluator<GradientSumT, CPUExpandEntry>> evaluator_;

    static constexpr size_t kPartitionBlockSize = 2048;
    common::PartitionBuilder<kPartitionBlockSize> partition_builder_;

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_;
    DMatrix const* const p_last_fmat_;
    DMatrix* p_last_fmat_mutable_;

    // key is the node id which should be calculated by Subtraction Trick, value is the node which
    // provides the evidence for subtraction
    std::vector<CPUExpandEntry> nodes_for_subtraction_trick_;
    // list of nodes whose histograms would be built explicitly.
    std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;

    enum class DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;
    std::unique_ptr<HistogramBuilder<GradientSumT>> histogram_builder_;

    common::Monitor builder_monitor_;
  };
  common::Monitor updater_monitor_;

  template<typename GradientSumT>
  void SetBuilder(const size_t n_trees, std::unique_ptr<Builder<GradientSumT>>*, DMatrix *dmat);

  template<typename GradientSumT>
  void CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                         HostDeviceVector<GradientPair> *gpair,
                         DMatrix *dmat,
                         GHistIndexMatrix const& gmat,
                         const std::vector<RegTree *> &trees);

 protected:
  std::unique_ptr<Builder<float>> float_builder_;
  std::unique_ptr<Builder<double>> double_builder_;

  std::unique_ptr<TreeUpdater> pruner_;
};

template <typename GradientSumT> class HistogramBuilder {
  using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
  using GHistRowT = GHistRow<GradientSumT>;

  /*! \brief culmulative histogram of gradients. */
  HistCollection<GradientSumT> hist_;
  /*! \brief culmulative local parent histogram of gradients. */
  HistCollection<GradientSumT> hist_local_worker_;
  GHistBuilder<GradientSumT> builder_;
  common::ParallelGHistBuilder<GradientSumT> buffer_;
  rabit::Reducer<GradientPairT, GradientPairT::Reduce> reducer_;
  int32_t n_threads_;

 public:
  void Reset(uint32_t n_bins, int32_t n_threads) {
    CHECK_GE(n_threads, 1);
    n_threads_ = n_threads;
    hist_.Init(n_bins);
    hist_local_worker_.Init(n_bins);
    buffer_.Init(n_bins);
    builder_ = GHistBuilder<GradientSumT>(n_threads, n_bins);
  }

  template <bool any_missing>
  void BuildLocalHistograms(
      DMatrix *p_fmat, RegTree *p_tree,
      std::vector<CPUExpandEntry> nodes_for_explicit_hist_build,
      RowSetCollection const &row_set_collection,
      const std::vector<GradientPair> &gpair_h) {

    const size_t n_nodes = nodes_for_explicit_hist_build.size();

    // create space of size (# rows in each node)
    common::BlockedSpace2d space(
        n_nodes,
        [&](size_t node) {
          const int32_t nid = nodes_for_explicit_hist_build[node].nid;
          return row_set_collection[nid].Size();
        },
        256);

    std::vector<GHistRowT> target_hists(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
      const int32_t nid = nodes_for_explicit_hist_build[i].nid;
      target_hists[i] = hist_[nid];
    }

    CHECK_LE(this->n_threads_, 16);
    buffer_.Reset(this->n_threads_, n_nodes, space, target_hists);

    // Parallel processing by nodes and data in each node
    for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>()) {
      common::ParallelFor2d(
          space, this->n_threads_, [&](size_t nid_in_set, common::Range1d r) {
            const auto tid = static_cast<unsigned>(omp_get_thread_num());
            const int32_t nid = nodes_for_explicit_hist_build[nid_in_set].nid;

            auto start_of_row_set = row_set_collection[nid].begin;
            auto rid_set = RowSetCollection::Elem(
                start_of_row_set + r.begin(), start_of_row_set + r.end(), nid);
            builder_.template BuildHist<any_missing>(
                gpair_h, rid_set, gmat,
                buffer_.GetInitializedHist(tid, nid_in_set));
          });
    }
  }

  void AddHistRowsLocal(
      int *starting_index, int *sync_count,
      std::vector<CPUExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<CPUExpandEntry> const &nodes_for_subtraction_trick) {
    for (auto const &entry : nodes_for_explicit_hist_build) {
      int nid = entry.nid;
      this->hist_.AddHistRow(nid);
      (*starting_index) = std::min(nid, (*starting_index));
    }
    (*sync_count) = nodes_for_explicit_hist_build.size();

    for (auto const &node : nodes_for_subtraction_trick) {
      this->hist_.AddHistRow(node.nid);
    }
    this->hist_.AllocateAllData();
  };

  void AddHistRowsDistributed(
      int *starting_index, int *sync_count,
      std::vector<CPUExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<CPUExpandEntry> const &nodes_for_subtraction_trick,
      RegTree *p_tree) {
    const size_t explicit_size = nodes_for_explicit_hist_build.size();
    const size_t subtaction_size = nodes_for_subtraction_trick.size();
    std::vector<int> merged_node_ids(explicit_size + subtaction_size);
    for (size_t i = 0; i < explicit_size; ++i) {
      merged_node_ids[i] = nodes_for_explicit_hist_build[i].nid;
    }
    for (size_t i = 0; i < subtaction_size; ++i) {
      merged_node_ids[explicit_size + i] = nodes_for_subtraction_trick[i].nid;
    }
    std::sort(merged_node_ids.begin(), merged_node_ids.end());
    int n_left = 0;
    for (auto const &nid : merged_node_ids) {
      if ((*p_tree)[nid].IsLeftChild()) {
        this->hist_.AddHistRow(nid);
        (*starting_index) = std::min(nid, (*starting_index));
        n_left++;
        this->hist_local_worker_.AddHistRow(nid);
      }
    }
    for (auto const &nid : merged_node_ids) {
      if (!((*p_tree)[nid].IsLeftChild())) {
        this->hist_.AddHistRow(nid);
        this->hist_local_worker_.AddHistRow(nid);
      }
    }
    this->hist_.AllocateAllData();
    this->hist_local_worker_.AllocateAllData();
    (*sync_count) = std::max(1, n_left);
  }

  void
  BuildHist(DMatrix *p_fmat, RegTree *p_tree,
            RowSetCollection const &row_set_collection,
            std::vector<CPUExpandEntry> const &nodes_for_explicit_hist_build,
            std::vector<CPUExpandEntry> const &nodes_for_subtraction_trick,
            std::vector<GradientPair> const &gpair) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    if (rabit::IsDistributed()) {
      this->AddHistRowsDistributed(&starting_index, &sync_count,
                                   nodes_for_explicit_hist_build,
                                   nodes_for_subtraction_trick, p_tree);
    } else {
      this->AddHistRowsLocal(&starting_index, &sync_count,
                             nodes_for_explicit_hist_build,
                             nodes_for_subtraction_trick);
    }

    if (p_fmat->IsDense()) {
      BuildLocalHistograms<false>(p_fmat, p_tree, nodes_for_explicit_hist_build,
                                  row_set_collection, gpair);
    } else {
      BuildLocalHistograms<true>(p_fmat, p_tree, nodes_for_explicit_hist_build,
                                 row_set_collection, gpair);
    }
    if (rabit::IsDistributed()) {
      this->SyncHistogramDistributed(p_tree, nodes_for_explicit_hist_build,
                                     nodes_for_subtraction_trick,
                                     starting_index, sync_count);
    } else {
      this->SyncHistogramLocal(p_tree, nodes_for_explicit_hist_build,
                               nodes_for_subtraction_trick, starting_index,
                               sync_count);
    }
  }

  void SyncHistogramDistributed(
      RegTree *p_tree,
      std::vector<CPUExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<CPUExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count) {
    const size_t nbins = builder_.GetNumBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        1024);
    common::ParallelFor2d(
        space, n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once
          buffer_.ReduceHist(node, r.begin(), r.end());
          // Store posible parent node
          auto this_local = hist_local_worker_[entry.nid];
          common::CopyHist(this_local, this_hist, r.begin(), r.end());

          if (!(*p_tree)[entry.nid].IsRoot()) {
            const size_t parent_id = (*p_tree)[entry.nid].Parent();
            const int subtraction_node_id =
                nodes_for_subtraction_trick[node].nid;
            auto parent_hist = this->hist_local_worker_[parent_id];
            auto sibling_hist = this->hist_[subtraction_node_id];
            common::SubtractionHist(sibling_hist, parent_hist, this_hist,
                                    r.begin(), r.end());
            // Store posible parent node
            auto sibling_local = hist_local_worker_[subtraction_node_id];
            common::CopyHist(sibling_local, sibling_hist, r.begin(), r.end());
          }
        });

    reducer_.Allreduce(this->hist_[starting_index].data(),
                       builder_.GetNumBins() * sync_count);

    ParallelSubtractionHist(space, nodes_for_explicit_hist_build,
                            nodes_for_subtraction_trick, p_tree);

    common::BlockedSpace2d space2(
        nodes_for_subtraction_trick.size(), [&](size_t) { return nbins; },
        1024);
    ParallelSubtractionHist(space2, nodes_for_subtraction_trick,
                            nodes_for_explicit_hist_build, p_tree);
  }

  void SyncHistogramLocal(
      RegTree *p_tree,
      std::vector<CPUExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<CPUExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count) {
    const size_t nbins = this->builder_.GetNumBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        1024);

    common::ParallelFor2d(
        space, this->n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once
          this->buffer_.ReduceHist(node, r.begin(), r.end());

          if (!(*p_tree)[entry.nid].IsRoot()) {
            const size_t parent_id = (*p_tree)[entry.nid].Parent();
            const int subtraction_node_id =
                nodes_for_subtraction_trick[node].nid;
            auto parent_hist = this->hist_[parent_id];
            auto sibling_hist = this->hist_[subtraction_node_id];
            common::SubtractionHist(sibling_hist, parent_hist, this_hist,
                                    r.begin(), r.end());
          }
        });
  }

  void
  ParallelSubtractionHist(const common::BlockedSpace2d &space,
                          const std::vector<CPUExpandEntry> &nodes,
                          const std::vector<CPUExpandEntry> &subtraction_nodes,
                          const RegTree *p_tree) {
    common::ParallelFor2d(
        space, this->n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes[node];
          if (!((*p_tree)[entry.nid].IsLeftChild())) {
            auto this_hist = this->hist_[entry.nid];

            if (!(*p_tree)[entry.nid].IsRoot()) {
              const int subtraction_node_id = subtraction_nodes[node].nid;
              auto parent_hist = hist_[(*p_tree)[entry.nid].Parent()];
              auto sibling_hist = hist_[subtraction_node_id];
              SubtractionHist(this_hist, parent_hist, sibling_hist, r.begin(),
                              r.end());
            }
          }
        });
  }

  HistCollection<GradientSumT> const& Histogram() {
    return hist_;
  }
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
