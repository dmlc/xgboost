/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_updater.h
 */
#ifndef PLUGIN_SYCL_TREE_HIST_UPDATER_H_
#define PLUGIN_SYCL_TREE_HIST_UPDATER_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/tree_updater.h>
#pragma GCC diagnostic pop

#include <vector>
#include <memory>
#include <queue>
#include <utility>

#include "../common/partition_builder.h"
#include "split_evaluator.h"
#include "hist_synchronizer.h"
#include "hist_row_adder.h"

#include "../../src/common/random.h"
#include "../data.h"

namespace xgboost {
namespace sycl {
namespace tree {

// data structure
template<typename GradType>
struct NodeEntry {
  /*! \brief statics for node entry */
  GradStats<GradType> stats;
  /*! \brief loss of this node, without split */
  GradType root_gain;
  /*! \brief weight calculated related to current data */
  GradType weight;
  /*! \brief current best solution */
  SplitEntry<GradType> best;
  // constructor
  explicit NodeEntry(const xgboost::tree::TrainParam& param)
      : root_gain(0.0f), weight(0.0f) {}
};

template<typename GradientSumT>
class HistUpdater {
 public:
  template <MemoryType memory_type = MemoryType::shared>
  using GHistRowT = common::GHistRow<GradientSumT, memory_type>;
  using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;

  explicit HistUpdater(const Context* ctx,
                       ::sycl::queue* qu,
                       const xgboost::tree::TrainParam& param,
                       FeatureInteractionConstraintHost int_constraints_,
                       DMatrix const* fmat)
    : ctx_(ctx), qu_(qu), param_(param),
      tree_evaluator_(qu, param, fmat->Info().num_col_),
      interaction_constraints_{std::move(int_constraints_)},
      p_last_tree_(nullptr), p_last_fmat_(fmat) {
    builder_monitor_.Init("SYCL::Quantile::HistUpdater");
    kernel_monitor_.Init("SYCL::Quantile::HistUpdater");
    if (param.max_depth > 0) {
      snode_device_.Resize(qu, 1u << (param.max_depth + 1));
    }
    has_fp64_support_ = qu_->get_device().has(::sycl::aspect::fp64);
    const auto sub_group_sizes =
      qu_->get_device().get_info<::sycl::info::device::sub_group_sizes>();
    sub_group_size_ = sub_group_sizes.back();
  }

  // update one tree, growing
  void Update(xgboost::tree::TrainParam const *param,
              const common::GHistIndexMatrix &gmat,
              const HostDeviceVector<GradientPair>& gpair,
              DMatrix *p_fmat,
              xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
              RegTree *p_tree);

  bool UpdatePredictionCache(const DMatrix* data,
                             linalg::MatrixView<float> p_out_preds);

  void SetHistSynchronizer(HistSynchronizer<GradientSumT>* sync);
  void SetHistRowsAdder(HistRowsAdder<GradientSumT>* adder);

 protected:
  friend class BatchHistSynchronizer<GradientSumT>;
  friend class DistributedHistSynchronizer<GradientSumT>;

  friend class BatchHistRowsAdder<GradientSumT>;
  friend class DistributedHistRowsAdder<GradientSumT>;

  struct SplitQuery {
    bst_node_t nid;
    size_t fid;
    const GradientPairT* hist;
  };

  void InitSampling(const HostDeviceVector<GradientPair>& gpair,
                    USMVector<size_t, MemoryType::on_device>* row_indices);

  void EvaluateSplits(const std::vector<ExpandEntry>& nodes_set,
                      const common::GHistIndexMatrix& gmat,
                      const RegTree& tree);

  // Enumerate the split values of specific feature
  // Returns the sum of gradients corresponding to the data points that contains a non-missing
  // value for the particular feature fid.
  static void EnumerateSplit(const ::sycl::sub_group& sg,
      const uint32_t* cut_ptr, const bst_float* cut_val, const GradientPairT* hist_data,
      const NodeEntry<GradientSumT> &snode, SplitEntry<GradientSumT>* p_best, bst_uint fid,
      bst_uint nodeID,
      typename TreeEvaluator<GradientSumT>::SplitEvaluator const &evaluator,
      float min_child_weight);

  void ApplySplit(std::vector<ExpandEntry> nodes,
                      const common::GHistIndexMatrix& gmat,
                      RegTree* p_tree);

  void AddSplitsToRowSet(const std::vector<ExpandEntry>& nodes, RegTree* p_tree);

  void InitData(const common::GHistIndexMatrix& gmat,
                const HostDeviceVector<GradientPair>& gpair,
                const DMatrix& fmat,
                const RegTree& tree);

  inline ::sycl::event BuildHist(
                        const HostDeviceVector<GradientPair>& gpair,
                        const common::RowSetCollection::Elem row_indices,
                        const common::GHistIndexMatrix& gmat,
                        GHistRowT<MemoryType::on_device>* hist,
                        GHistRowT<MemoryType::on_device>* hist_buffer,
                        ::sycl::event event_priv) {
    return hist_builder_.BuildHist(gpair, row_indices, gmat, hist,
                                   data_layout_ != kSparseData, hist_buffer, event_priv);
  }

  void InitNewNode(int nid,
                   const common::GHistIndexMatrix& gmat,
                   const HostDeviceVector<GradientPair>& gpair,
                   const RegTree& tree);

  // Split nodes to 2 sets depending on amount of rows in each node
  // Histograms for small nodes will be built explicitly
  // Histograms for big nodes will be built by 'Subtraction Trick'
  void SplitSiblings(const std::vector<ExpandEntry>& nodes,
                  std::vector<ExpandEntry>* small_siblings,
                  std::vector<ExpandEntry>* big_siblings,
                  RegTree *p_tree);

  void BuildNodeStats(const common::GHistIndexMatrix &gmat,
                      RegTree *p_tree,
                      const HostDeviceVector<GradientPair>& gpair);

  void EvaluateAndApplySplits(const common::GHistIndexMatrix &gmat,
                              RegTree *p_tree,
                              int *num_leaves,
                              int depth,
                              std::vector<ExpandEntry> *temp_qexpand_depth);

  void AddSplitsToTree(
            const common::GHistIndexMatrix &gmat,
            RegTree *p_tree,
            int *num_leaves,
            int depth,
            std::vector<ExpandEntry>* nodes_for_apply_split,
            std::vector<ExpandEntry>* temp_qexpand_depth);

  void ExpandWithDepthWise(const common::GHistIndexMatrix &gmat,
                            RegTree *p_tree,
                            const HostDeviceVector<GradientPair>& gpair);

  void BuildLocalHistograms(const common::GHistIndexMatrix &gmat,
                            RegTree *p_tree,
                            const HostDeviceVector<GradientPair>& gpair);

  void BuildHistogramsLossGuide(
                      ExpandEntry entry,
                      const common::GHistIndexMatrix &gmat,
                      RegTree *p_tree,
                      const HostDeviceVector<GradientPair>& gpair);

  void ExpandWithLossGuide(const common::GHistIndexMatrix& gmat,
                           RegTree* p_tree,
                           const HostDeviceVector<GradientPair>& gpair);

  void ReduceHists(const std::vector<int>& sync_ids, size_t nbins);

  inline static bool LossGuide(ExpandEntry lhs, ExpandEntry rhs) {
    if (lhs.GetLossChange() == rhs.GetLossChange()) {
      return lhs.GetNodeId() > rhs.GetNodeId();  // favor small timestamp
    } else {
      return lhs.GetLossChange() < rhs.GetLossChange();  // favor large loss_chg
    }
  }

  //  --data fields--
  const Context* ctx_;
  bool has_fp64_support_;
  size_t sub_group_size_;

  // the internal row sets
  common::RowSetCollection row_set_collection_;

  const xgboost::tree::TrainParam& param_;
  std::shared_ptr<xgboost::common::ColumnSampler> column_sampler_;

  std::vector<SplitQuery> split_queries_host_;
  USMVector<SplitQuery, MemoryType::on_device> split_queries_device_;

  USMVector<SplitEntry<GradientSumT>, MemoryType::on_device> best_splits_device_;
  std::vector<SplitEntry<GradientSumT>> best_splits_host_;

  TreeEvaluator<GradientSumT> tree_evaluator_;
  FeatureInteractionConstraintHost interaction_constraints_;

  // back pointers to tree and data matrix
  const RegTree* p_last_tree_;
  DMatrix const* const p_last_fmat_;

  using ExpandQueue =
      std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                          std::function<bool(ExpandEntry, ExpandEntry)>>;

  std::unique_ptr<ExpandQueue> qexpand_loss_guided_;
  std::vector<ExpandEntry> qexpand_depth_wise_;

  enum DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
  DataLayout data_layout_;

  constexpr static size_t kBufferSize = 2048;
  common::GHistBuilder<GradientSumT> hist_builder_;
  common::ParallelGHistBuilder<GradientSumT> hist_buffer_;
  /*! \brief culmulative histogram of gradients. */
  common::HistCollection<GradientSumT, MemoryType::on_device> hist_;
  /*! \brief culmulative local parent histogram of gradients. */
  common::HistCollection<GradientSumT, MemoryType::on_device> hist_local_worker_;

  /*! \brief TreeNode Data: statistics for each constructed node */
  std::vector<NodeEntry<GradientSumT>> snode_host_;
  USMVector<NodeEntry<GradientSumT>, MemoryType::on_device> snode_device_;

  xgboost::common::Monitor builder_monitor_;
  xgboost::common::Monitor kernel_monitor_;

  /*! \brief feature with least # of bins. to be used for dense specialization
              of InitNewNode() */
  uint32_t fid_least_bins_;

  uint64_t seed_ = 0;

  common::PartitionBuilder partition_builder_;

  // key is the node id which should be calculated by Subtraction Trick, value is the node which
  // provides the evidence for substracts
  std::vector<ExpandEntry> nodes_for_subtraction_trick_;
  // list of nodes whose histograms would be built explicitly.
  std::vector<ExpandEntry> nodes_for_explicit_hist_build_;

  std::unique_ptr<HistSynchronizer<GradientSumT>> hist_synchronizer_;
  std::unique_ptr<HistRowsAdder<GradientSumT>> hist_rows_adder_;

  std::vector<GradientPairT> reduce_buffer_;
  ::sycl::queue* qu_;
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_HIST_UPDATER_H_
