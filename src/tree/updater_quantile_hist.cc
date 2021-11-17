/*!
 * Copyright 2017-2021 by Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <dmlc/timer.h>
#include <rabit/rabit.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "xgboost/logging.h"
#include "xgboost/tree_updater.h"

#include "constraints.h"
#include "param.h"
#include "./updater_quantile_hist.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"
#include "../common/threading_utils.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist);

DMLC_REGISTER_PARAMETER(CPUHistMakerTrainParam);

void QuantileHistMaker::Configure(const Args& args) {
  // initialize pruner
  if (!pruner_) {
    pruner_.reset(TreeUpdater::Create("prune", tparam_));
  }
  pruner_->Configure(args);
  param_.UpdateAllowUnknown(args);
  hist_maker_param_.UpdateAllowUnknown(args);
}

template<typename GradientSumT>
void QuantileHistMaker::SetBuilder(const size_t n_trees,
                                   std::unique_ptr<Builder<GradientSumT>>* builder,
                                   DMatrix *dmat) {
  builder->reset(
      new Builder<GradientSumT>(n_trees, param_, std::move(pruner_), dmat));
}

template<typename GradientSumT>
void QuantileHistMaker::CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                                          HostDeviceVector<GradientPair> *gpair,
                                          DMatrix *dmat,
                                          GHistIndexMatrix const& gmat,
                                          const std::vector<RegTree *> &trees) {
  for (auto tree : trees) {
    builder->Update(gmat, column_matrix_, gpair, dmat, tree);
  }
}

void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  auto it = dmat->GetBatches<GHistIndexMatrix>(
                    BatchParam{GenericParameter::kCpuId, param_.max_bin})
                .begin();
  auto p_gmat = it.Page();
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("GmatInitialization");
    column_matrix_.Init(*p_gmat, param_.sparse_threshold);
    updater_monitor_.Stop("GmatInitialization");
    // A proper solution is puting cut matrix in DMatrix, see:
    // https://github.com/dmlc/xgboost/issues/5143
    is_gmat_initialized_ = true;
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();

  // build tree
  const size_t n_trees = trees.size();
  if (hist_maker_param_.single_precision_histogram) {
    if (!float_builder_) {
      this->SetBuilder(n_trees, &float_builder_, dmat);
    }
    CallBuilderUpdate(float_builder_, gpair, dmat, *p_gmat, trees);
  } else {
    if (!double_builder_) {
      SetBuilder(n_trees, &double_builder_, dmat);
    }
    CallBuilderUpdate(double_builder_, gpair, dmat, *p_gmat, trees);
  }

  param_.learning_rate = lr;

  p_last_dmat_ = dmat;
}

bool QuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data, VectorView<float> out_preds) {
  if (hist_maker_param_.single_precision_histogram && float_builder_) {
      return float_builder_->UpdatePredictionCache(data, out_preds);
  } else if (double_builder_) {
      return double_builder_->UpdatePredictionCache(data, out_preds);
  } else {
      return false;
  }
}


template <typename GradientSumT>
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::InitRoot(
    DMatrix *p_fmat, RegTree *p_tree, const std::vector<GradientPair> &gpair_h,
    int *num_leaves, std::vector<CPUExpandEntry> *expand) {
  CPUExpandEntry node(CPUExpandEntry::kRootNid, p_tree->GetDepth(0), 0.0f);

  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(node);

  this->histogram_builder_->BuildHist(p_fmat, p_tree, row_set_collection_,
                                      nodes_for_explicit_hist_build_,
                                      nodes_for_subtraction_trick_, gpair_h);

  {
    auto nid = CPUExpandEntry::kRootNid;
    GHistRowT hist = this->histogram_builder_->Histogram()[nid];
    GradientPairT grad_stat;
    if (data_layout_ == DataLayout::kDenseDataZeroBased ||
        data_layout_ == DataLayout::kDenseDataOneBased) {
      auto const &gmat = *(p_fmat
                               ->GetBatches<GHistIndexMatrix>(BatchParam{
                                   GenericParameter::kCpuId, param_.max_bin})
                               .begin());
      const std::vector<uint32_t> &row_ptr = gmat.cut.Ptrs();
      const uint32_t ibegin = row_ptr[fid_least_bins_];
      const uint32_t iend = row_ptr[fid_least_bins_ + 1];
      auto begin = hist.data();
      for (uint32_t i = ibegin; i < iend; ++i) {
        const GradientPairT et = begin[i];
        grad_stat.Add(et.GetGrad(), et.GetHess());
      }
    } else {
      const RowSetCollection::Elem e = row_set_collection_[nid];
      for (const size_t *it = e.begin; it < e.end; ++it) {
        grad_stat.Add(gpair_h[*it].GetGrad(), gpair_h[*it].GetHess());
      }
      rabit::Allreduce<rabit::op::Sum, GradientSumT>(
          reinterpret_cast<GradientSumT *>(&grad_stat), 2);
    }

    auto weight = evaluator_->InitRoot(GradStats{grad_stat});
    p_tree->Stat(RegTree::kRoot).sum_hess = grad_stat.GetHess();
    p_tree->Stat(RegTree::kRoot).base_weight = weight;
    (*p_tree)[RegTree::kRoot].SetLeaf(param_.learning_rate * weight);

    std::vector<CPUExpandEntry> entries{node};
    builder_monitor_.Start("EvaluateSplits");
    for (auto const &gmat : p_fmat->GetBatches<GHistIndexMatrix>(
             BatchParam{GenericParameter::kCpuId, param_.max_bin})) {
      evaluator_->EvaluateSplits(histogram_builder_->Histogram(), gmat, *p_tree, &entries);
    }
    builder_monitor_.Stop("EvaluateSplits");
    node = entries.front();
  }

  expand->push_back(node);
  ++(*num_leaves);
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToTree(
          const std::vector<CPUExpandEntry>& expand,
          RegTree *p_tree,
          int *num_leaves,
          std::vector<CPUExpandEntry>* nodes_for_apply_split) {
  for (auto const& entry : expand) {
    if (entry.IsValid(param_, *num_leaves)) {
      nodes_for_apply_split->push_back(entry);
      evaluator_->ApplyTreeSplit(entry, p_tree);
      (*num_leaves)++;
    }
  }
}

// Split nodes to 2 sets depending on amount of rows in each node
// Histograms for small nodes will be built explicitly
// Histograms for big nodes will be built by 'Subtraction Trick'
// Exception: in distributed setting, we always build the histogram for the left child node
//    and use 'Subtraction Trick' to built the histogram for the right child node.
//    This ensures that the workers operate on the same set of tree nodes.
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::SplitSiblings(
    const std::vector<CPUExpandEntry> &nodes_for_apply_split,
    std::vector<CPUExpandEntry> *nodes_to_evaluate, RegTree *p_tree) {
  builder_monitor_.Start("SplitSiblings");
  for (auto const& entry : nodes_for_apply_split) {
    int nid = entry.nid;

    const int cleft = (*p_tree)[nid].LeftChild();
    const int cright = (*p_tree)[nid].RightChild();
    const CPUExpandEntry left_node = CPUExpandEntry(cleft, p_tree->GetDepth(cleft), 0.0);
    const CPUExpandEntry right_node = CPUExpandEntry(cright, p_tree->GetDepth(cright), 0.0);
    nodes_to_evaluate->push_back(left_node);
    nodes_to_evaluate->push_back(right_node);
    if (row_set_collection_[cleft].Size() < row_set_collection_[cright].Size()) {
      nodes_for_explicit_hist_build_.push_back(left_node);
      nodes_for_subtraction_trick_.push_back(right_node);
    } else {
      nodes_for_explicit_hist_build_.push_back(right_node);
      nodes_for_subtraction_trick_.push_back(left_node);
    }
  }
  CHECK_EQ(nodes_for_subtraction_trick_.size(), nodes_for_explicit_hist_build_.size());
  builder_monitor_.Stop("SplitSiblings");
}

template<typename GradientSumT>
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::ExpandTree(
    const GHistIndexMatrix& gmat,
    const ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  builder_monitor_.Start("ExpandTree");
  int num_leaves = 0;

  Driver<CPUExpandEntry> driver(static_cast<TrainParam::TreeGrowPolicy>(param_.grow_policy));
  std::vector<CPUExpandEntry> expand;
  InitRoot<any_missing>(p_fmat, p_tree, gpair_h, &num_leaves, &expand);
  driver.Push(expand[0]);

  int32_t depth = 0;
  while (!driver.IsEmpty()) {
    expand = driver.Pop();
    depth = expand[0].depth + 1;
    std::vector<CPUExpandEntry> nodes_for_apply_split;
    std::vector<CPUExpandEntry> nodes_to_evaluate;
    nodes_for_explicit_hist_build_.clear();
    nodes_for_subtraction_trick_.clear();

    AddSplitsToTree(expand, p_tree, &num_leaves, &nodes_for_apply_split);

    if (nodes_for_apply_split.size() != 0) {
      ApplySplit<any_missing>(nodes_for_apply_split, gmat, column_matrix, p_tree);
      SplitSiblings(nodes_for_apply_split, &nodes_to_evaluate, p_tree);

      if (depth < param_.max_depth) {
        this->histogram_builder_->BuildHist(
            p_fmat, p_tree, row_set_collection_, nodes_for_explicit_hist_build_,
            nodes_for_subtraction_trick_, gpair_h);
      } else {
        int starting_index = std::numeric_limits<int>::max();
        int sync_count = 0;
        this->histogram_builder_->AddHistRows(
            &starting_index, &sync_count, nodes_for_explicit_hist_build_,
            nodes_for_subtraction_trick_, p_tree);
      }

      builder_monitor_.Start("EvaluateSplits");
      evaluator_->EvaluateSplits(this->histogram_builder_->Histogram(), gmat,
                                 *p_tree, &nodes_to_evaluate);
      builder_monitor_.Stop("EvaluateSplits");

      for (size_t i = 0; i < nodes_for_apply_split.size(); ++i) {
        CPUExpandEntry left_node = nodes_to_evaluate.at(i * 2 + 0);
        CPUExpandEntry right_node = nodes_to_evaluate.at(i * 2 + 1);
        driver.Push(left_node);
        driver.Push(right_node);
      }
    }
  }
  builder_monitor_.Stop("ExpandTree");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::Update(
    const GHistIndexMatrix &gmat,
    const ColumnMatrix &column_matrix,
    HostDeviceVector<GradientPair> *gpair,
    DMatrix *p_fmat, RegTree *p_tree) {
  builder_monitor_.Start("Update");

  std::vector<GradientPair>* gpair_ptr = &(gpair->HostVector());
  // in case 'num_parallel_trees != 1' no posibility to change initial gpair
  if (GetNumberOfTrees() != 1) {
    gpair_local_.resize(gpair_ptr->size());
    gpair_local_ = *gpair_ptr;
    gpair_ptr = &gpair_local_;
  }
  p_last_fmat_mutable_ = p_fmat;

  this->InitData(gmat, *p_fmat, *p_tree, gpair_ptr);

  if (column_matrix.AnyMissing()) {
    ExpandTree<true>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
  } else {
    ExpandTree<false>(gmat, column_matrix, p_fmat, p_tree, *gpair_ptr);
  }
  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});

  builder_monitor_.Stop("Update");
}

template<typename GradientSumT>
bool QuantileHistMaker::Builder<GradientSumT>::UpdatePredictionCache(
    const DMatrix* data,
    VectorView<float> out_preds) {
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_ ||
      p_last_fmat_ != p_last_fmat_mutable_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");

  CHECK_GT(out_preds.Size(), 0U);

  size_t n_nodes = row_set_collection_.end() - row_set_collection_.begin();

  common::BlockedSpace2d space(n_nodes, [&](size_t node) {
    return row_set_collection_[node].Size();
  }, 1024);
  CHECK_EQ(out_preds.DeviceIdx(), GenericParameter::kCpuId);
  common::ParallelFor2d(space, this->nthread_, [&](size_t node, common::Range1d r) {
    const RowSetCollection::Elem rowset = row_set_collection_[node];
    if (rowset.begin != nullptr && rowset.end != nullptr) {
      int nid = rowset.node_id;
      bst_float leaf_value;
      // if a node is marked as deleted by the pruner, traverse upward to locate
      // a non-deleted leaf.
      if ((*p_last_tree_)[nid].IsDeleted()) {
        while ((*p_last_tree_)[nid].IsDeleted()) {
          nid = (*p_last_tree_)[nid].Parent();
        }
        CHECK((*p_last_tree_)[nid].IsLeaf());
      }
      leaf_value = (*p_last_tree_)[nid].LeafValue();

      for (const size_t* it = rowset.begin + r.begin(); it < rowset.begin + r.end(); ++it) {
        out_preds[*it] += leaf_value;
      }
    }
  });

  builder_monitor_.Stop("UpdatePredictionCache");
  return true;
}

template<typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitSampling(const DMatrix& fmat,
                                                std::vector<GradientPair>* gpair,
                                                std::vector<size_t>* row_indices) {
  const auto& info = fmat.Info();
  auto& rnd = common::GlobalRandom();
  std::vector<GradientPair>& gpair_ref = *gpair;

#if XGBOOST_CUSTOMIZE_GLOBAL_PRNG
  std::bernoulli_distribution coin_flip(param_.subsample);
  for (size_t i = 0; i < info.num_row_; ++i) {
    if (!(gpair_ref[i].GetHess() >= 0.0f && coin_flip(rnd)) || gpair_ref[i].GetGrad() == 0.0f) {
      gpair_ref[i] = GradientPair(0);
    }
  }
#else
  const size_t nthread = this->nthread_;
  uint64_t initial_seed = rnd();

  const size_t discard_size = info.num_row_ / nthread;
  std::bernoulli_distribution coin_flip(param_.subsample);

  dmlc::OMPException exc;
  #pragma omp parallel num_threads(nthread)
  {
    exc.Run([&]() {
      const size_t tid = omp_get_thread_num();
      const size_t ibegin = tid * discard_size;
      const size_t iend = (tid == (nthread - 1)) ?
                          info.num_row_ : ibegin + discard_size;
      RandomReplace::MakeIf([&](size_t i, RandomReplace::EngineT& eng) {
        return !(gpair_ref[i].GetHess() >= 0.0f && coin_flip(eng));
      }, GradientPair(0), initial_seed, ibegin, iend, &gpair_ref);
    });
  }
  exc.Rethrow();
#endif  // XGBOOST_CUSTOMIZE_GLOBAL_PRNG
}
template<typename GradientSumT>
size_t QuantileHistMaker::Builder<GradientSumT>::GetNumberOfTrees() {
  return n_trees_;
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::InitData(
    const GHistIndexMatrix &gmat, const DMatrix &fmat, const RegTree &tree,
    std::vector<GradientPair> *gpair) {
  CHECK((param_.max_depth > 0 || param_.max_leaves > 0))
      << "max_depth or max_leaves cannot be both 0 (unlimited); "
      << "at least one should be a positive quantity.";
  if (param_.grow_policy == TrainParam::kDepthWise) {
    CHECK(param_.max_depth > 0) << "max_depth cannot be 0 (unlimited) "
                                << "when grow_policy is depthwise.";
  }
  builder_monitor_.Start("InitData");
  const auto& info = fmat.Info();

  {
    // initialize the row set
    row_set_collection_.Clear();
    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    // initialize histogram builder
    dmlc::OMPException exc;
#pragma omp parallel
    {
      exc.Run([&]() {
        this->nthread_ = omp_get_num_threads();
      });
    }
    exc.Rethrow();
    this->histogram_builder_->Reset(nbins, param_.max_bin, this->nthread_);

    std::vector<size_t>& row_indices = *row_set_collection_.Data();
    row_indices.resize(info.num_row_);
    size_t* p_row_indices = row_indices.data();
    // mark subsample and build list of member rows

    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      builder_monitor_.Start("InitSampling");
      InitSampling(fmat, gpair, &row_indices);
      builder_monitor_.Stop("InitSampling");
      CHECK_EQ(row_indices.size(), info.num_row_);
      // We should check that the partitioning was done correctly
      // and each row of the dataset fell into exactly one of the categories
    }
    common::MemStackAllocator<bool, 128> buff(this->nthread_);
    bool* p_buff = buff.Get();
    std::fill(p_buff, p_buff + this->nthread_, false);

    const size_t block_size = info.num_row_ / this->nthread_ + !!(info.num_row_ % this->nthread_);

    #pragma omp parallel num_threads(this->nthread_)
    {
      exc.Run([&]() {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(info.num_row_));

        for (size_t i = ibegin; i < iend; ++i) {
          if ((*gpair)[i].GetHess() < 0.0f) {
            p_buff[tid] = true;
            break;
          }
        }
      });
    }
    exc.Rethrow();

    bool has_neg_hess = false;
    for (int32_t tid = 0; tid < this->nthread_; ++tid) {
      if (p_buff[tid]) {
        has_neg_hess = true;
      }
    }

    if (has_neg_hess) {
      size_t j = 0;
      for (size_t i = 0; i < info.num_row_; ++i) {
        if ((*gpair)[i].GetHess() >= 0.0f) {
          p_row_indices[j++] = i;
        }
      }
      row_indices.resize(j);
    } else {
      #pragma omp parallel num_threads(this->nthread_)
      {
        exc.Run([&]() {
          const size_t tid = omp_get_thread_num();
          const size_t ibegin = tid * block_size;
          const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
              static_cast<size_t>(info.num_row_));
          for (size_t i = ibegin; i < iend; ++i) {
            p_row_indices[i] = i;
          }
        });
      }
      exc.Rethrow();
    }
  }

  row_set_collection_.Init();

  {
    /* determine layout of data */
    const size_t nrow = info.num_row_;
    const size_t ncol = info.num_col_;
    const size_t nnz = info.num_nonzero_;
    // number of discrete bins for feature 0
    const uint32_t nbins_f0 = gmat.cut.Ptrs()[1] - gmat.cut.Ptrs()[0];
    if (nrow * ncol == nnz) {
      // dense data with zero-based indexing
      data_layout_ = DataLayout::kDenseDataZeroBased;
    } else if (nbins_f0 == 0 && nrow * (ncol - 1) == nnz) {
      // dense data with one-based indexing
      data_layout_ = DataLayout::kDenseDataOneBased;
    } else {
      // sparse data
      data_layout_ = DataLayout::kSparseData;
    }
  }
  // store a pointer to the tree
  p_last_tree_ = &tree;
  if (data_layout_ == DataLayout::kDenseDataOneBased) {
    evaluator_.reset(new HistEvaluator<GradientSumT, CPUExpandEntry>{
        param_, info, this->nthread_, column_sampler_, true});
  } else {
    evaluator_.reset(new HistEvaluator<GradientSumT, CPUExpandEntry>{
        param_, info, this->nthread_, column_sampler_, false});
  }

  if (data_layout_ == DataLayout::kDenseDataZeroBased
      || data_layout_ == DataLayout::kDenseDataOneBased) {
    /* specialized code for dense data:
       choose the column that has a least positive number of discrete bins.
       For dense data (with no missing value),
       the sum of gradient histogram is equal to snode[nid] */
    const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
    const auto nfeature = static_cast<bst_uint>(row_ptr.size() - 1);
    uint32_t min_nbins_per_feature = 0;
    for (bst_uint i = 0; i < nfeature; ++i) {
      const uint32_t nbins = row_ptr[i + 1] - row_ptr[i];
      if (nbins > 0) {
        if (min_nbins_per_feature == 0 || min_nbins_per_feature > nbins) {
          min_nbins_per_feature = nbins;
          fid_least_bins_ = i;
        }
      }
    }
    CHECK_GT(min_nbins_per_feature, 0U);
  }

  builder_monitor_.Stop("InitData");
}

template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::FindSplitConditions(
                                                     const std::vector<CPUExpandEntry>& nodes,
                                                     const RegTree& tree,
                                                     const GHistIndexMatrix& gmat,
                                                     std::vector<int32_t>* split_conditions) {
  const size_t n_nodes = nodes.size();
  split_conditions->resize(n_nodes);

  for (size_t i = 0; i < nodes.size(); ++i) {
    const int32_t nid = nodes[i].nid;
    const bst_uint fid = tree[nid].SplitIndex();
    const bst_float split_pt = tree[nid].SplitCond();
    const uint32_t lower_bound = gmat.cut.Ptrs()[fid];
    const uint32_t upper_bound = gmat.cut.Ptrs()[fid + 1];
    int32_t split_cond = -1;
    // convert floating-point split_pt into corresponding bin_id
    // split_cond = -1 indicates that split_pt is less than all known cut points
    CHECK_LT(upper_bound,
             static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
    for (uint32_t bound = lower_bound; bound < upper_bound; ++bound) {
      if (split_pt == gmat.cut.Values()[bound]) {
        split_cond = static_cast<int32_t>(bound);
      }
    }
    (*split_conditions)[i] = split_cond;
  }
}
template <typename GradientSumT>
void QuantileHistMaker::Builder<GradientSumT>::AddSplitsToRowSet(
                                               const std::vector<CPUExpandEntry>& nodes,
                                               RegTree* p_tree) {
  const size_t n_nodes = nodes.size();
  for (unsigned int i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes[i].nid;
    const size_t n_left = partition_builder_.GetNLeftElems(i);
    const size_t n_right = partition_builder_.GetNRightElems(i);
    CHECK_EQ((*p_tree)[nid].LeftChild() + 1, (*p_tree)[nid].RightChild());
    row_set_collection_.AddSplit(nid, (*p_tree)[nid].LeftChild(),
        (*p_tree)[nid].RightChild(), n_left, n_right);
  }
}

template <typename GradientSumT>
template <bool any_missing>
void QuantileHistMaker::Builder<GradientSumT>::ApplySplit(const std::vector<CPUExpandEntry> nodes,
                                            const GHistIndexMatrix& gmat,
                                            const ColumnMatrix& column_matrix,
                                            RegTree* p_tree) {
  builder_monitor_.Start("ApplySplit");
  // 1. Find split condition for each split
  const size_t n_nodes = nodes.size();
  std::vector<int32_t> split_conditions;
  FindSplitConditions(nodes, *p_tree, gmat, &split_conditions);
  // 2.1 Create a blocked space of size SUM(samples in each node)
  common::BlockedSpace2d space(n_nodes, [&](size_t node_in_set) {
    int32_t nid = nodes[node_in_set].nid;
    return row_set_collection_[nid].Size();
  }, kPartitionBlockSize);
  // 2.2 Initialize the partition builder
  // allocate buffers for storage intermediate results by each thread
  partition_builder_.Init(space.Size(), n_nodes, [&](size_t node_in_set) {
    const int32_t nid = nodes[node_in_set].nid;
    const size_t size = row_set_collection_[nid].Size();
    const size_t n_tasks = size / kPartitionBlockSize + !!(size % kPartitionBlockSize);
    return n_tasks;
  });
  // 2.3 Split elements of row_set_collection_ to left and right child-nodes for each node
  // Store results in intermediate buffers from partition_builder_
  common::ParallelFor2d(space, this->nthread_, [&](size_t node_in_set, common::Range1d r) {
    size_t begin = r.begin();
    const int32_t nid = nodes[node_in_set].nid;
    const size_t task_id = partition_builder_.GetTaskIdx(node_in_set, begin);
    partition_builder_.AllocateForTask(task_id);
      switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        partition_builder_.template Partition<uint8_t, any_missing>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix,
                  *p_tree, row_set_collection_[nid].begin);
        break;
      case common::kUint16BinsTypeSize:
        partition_builder_.template Partition<uint16_t, any_missing>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix,
                  *p_tree, row_set_collection_[nid].begin);
        break;
      case common::kUint32BinsTypeSize:
        partition_builder_.template Partition<uint32_t, any_missing>(node_in_set, nid, r,
                  split_conditions[node_in_set], column_matrix,
                  *p_tree, row_set_collection_[nid].begin);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    });
  // 3. Compute offsets to copy blocks of row-indexes
  // from partition_builder_ to row_set_collection_
  partition_builder_.CalculateRowOffsets();

  // 4. Copy elements from partition_builder_ to row_set_collection_ back
  // with updated row-indexes for each tree-node
  common::ParallelFor2d(space, this->nthread_, [&](size_t node_in_set, common::Range1d r) {
    const int32_t nid = nodes[node_in_set].nid;
    partition_builder_.MergeToArray(node_in_set, r.begin(),
        const_cast<size_t*>(row_set_collection_[nid].begin));
  });
  // 5. Add info about splits into row_set_collection_
  AddSplitsToRowSet(nodes, p_tree);
  builder_monitor_.Stop("ApplySplit");
}

template struct QuantileHistMaker::Builder<float>;
template struct QuantileHistMaker::Builder<double>;

XGBOOST_REGISTER_TREE_UPDATER(FastHistMaker, "grow_fast_histmaker")
.describe("(Deprecated, use grow_quantile_histmaker instead.)"
          " Grow tree using quantized histogram.")
.set_body(
    []() {
      LOG(WARNING) << "grow_fast_histmaker is deprecated, "
                   << "use grow_quantile_histmaker instead.";
      return new QuantileHistMaker();
    });

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker")
.describe("Grow tree using quantized histogram.")
.set_body(
    []() {
      return new QuantileHistMaker();
    });
}  // namespace tree
}  // namespace xgboost
