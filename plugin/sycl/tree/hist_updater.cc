/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_updater.cc
 */

#include "hist_updater.h"

#include <oneapi/dpl/random>

#include <functional>

#include "../../src/tree/common_row_partitioner.h"

#include "../common/hist_util.h"
#include "../../src/collective/allreduce.h"

namespace xgboost {
namespace sycl {
namespace tree {

using ::sycl::ext::oneapi::plus;
using ::sycl::ext::oneapi::minimum;
using ::sycl::ext::oneapi::maximum;

template <typename GradientSumT>
void HistUpdater<GradientSumT>::ReduceHists(const std::vector<int>& sync_ids,
                                            size_t nbins) {
  if (reduce_buffer_.size() < sync_ids.size() * nbins) {
    reduce_buffer_.resize(sync_ids.size() * nbins);
  }
  for (size_t i = 0; i < sync_ids.size(); i++) {
    auto& this_hist = hist_[sync_ids[i]];
    const GradientPairT* psrc = reinterpret_cast<const GradientPairT*>(this_hist.DataConst());
    qu_->memcpy(reduce_buffer_.data() + i * nbins, psrc, nbins*sizeof(GradientPairT)).wait();
  }

  auto buffer_vec = linalg::MakeVec(reinterpret_cast<GradientSumT*>(reduce_buffer_.data()),
                                    2 * nbins * sync_ids.size());
  auto rc = collective::Allreduce(ctx_, buffer_vec, collective::Op::kSum);
  SafeColl(rc);

  for (size_t i = 0; i < sync_ids.size(); i++) {
    auto& this_hist = hist_[sync_ids[i]];
    GradientPairT* psrc = reinterpret_cast<GradientPairT*>(this_hist.Data());
    qu_->memcpy(psrc, reduce_buffer_.data() + i * nbins, nbins*sizeof(GradientPairT)).wait();
  }
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::SetHistSynchronizer(
    HistSynchronizer<GradientSumT> *sync) {
  hist_synchronizer_.reset(sync);
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::SetHistRowsAdder(
    HistRowsAdder<GradientSumT> *adder) {
  hist_rows_adder_.reset(adder);
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::BuildHistogramsLossGuide(
    ExpandEntry entry,
    const common::GHistIndexMatrix &gmat,
    RegTree *p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  nodes_for_explicit_hist_build_.clear();
  nodes_for_subtraction_trick_.clear();
  nodes_for_explicit_hist_build_.push_back(entry);

  if (!(*p_tree)[entry.nid].IsRoot()) {
    auto sibling_id = entry.GetSiblingId(p_tree);
    nodes_for_subtraction_trick_.emplace_back(sibling_id, p_tree->GetDepth(sibling_id));
  }

  std::vector<int> sync_ids;
  hist_rows_adder_->AddHistRows(this, &sync_ids, p_tree);
  qu_->wait_and_throw();
  BuildLocalHistograms(gmat, p_tree, gpair);
  hist_synchronizer_->SyncHistograms(this, sync_ids, p_tree);
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::BuildLocalHistograms(
    const common::GHistIndexMatrix &gmat,
    RegTree *p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  builder_monitor_.Start("BuildLocalHistograms");
  const size_t n_nodes = nodes_for_explicit_hist_build_.size();
  ::sycl::event event;

  for (size_t i = 0; i < n_nodes; i++) {
    const int32_t nid = nodes_for_explicit_hist_build_[i].nid;

    if (row_set_collection_[nid].Size() > 0) {
      event = BuildHist(gpair, row_set_collection_[nid], gmat, &(hist_[nid]),
                        &(hist_buffer_.GetDeviceBuffer()), event);
    } else {
      common::InitHist(qu_, &(hist_[nid]), hist_[nid].Size(), &event);
    }
  }
  qu_->wait_and_throw();
  builder_monitor_.Stop("BuildLocalHistograms");
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::BuildNodeStats(
    const common::GHistIndexMatrix &gmat,
    RegTree *p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  builder_monitor_.Start("BuildNodeStats");
  for (auto const& entry : qexpand_depth_wise_) {
    int nid = entry.nid;
    this->InitNewNode(nid, gmat, gpair, *p_tree);
    // add constraints
    if (!(*p_tree)[nid].IsLeftChild() && !(*p_tree)[nid].IsRoot()) {
      // it's a right child
      auto parent_id = (*p_tree)[nid].Parent();
      auto left_sibling_id = (*p_tree)[parent_id].LeftChild();
      auto parent_split_feature_id = snode_host_[parent_id].best.SplitIndex();
      tree_evaluator_.AddSplit(
          parent_id, left_sibling_id, nid, parent_split_feature_id,
          snode_host_[left_sibling_id].weight, snode_host_[nid].weight);
      interaction_constraints_.Split(parent_id, parent_split_feature_id,
                                     left_sibling_id, nid);
    }
  }
  builder_monitor_.Stop("BuildNodeStats");
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::AddSplitsToTree(
    const common::GHistIndexMatrix &gmat,
    RegTree *p_tree,
    int *num_leaves,
    int depth,
    std::vector<ExpandEntry>* nodes_for_apply_split,
    std::vector<ExpandEntry>* temp_qexpand_depth) {
  builder_monitor_.Start("AddSplitsToTree");
  auto evaluator = tree_evaluator_.GetEvaluator();
  for (auto const& entry : qexpand_depth_wise_) {
    const auto lr = param_.learning_rate;
    int nid = entry.nid;

    if (snode_host_[nid].best.loss_chg < kRtEps ||
        (param_.max_depth > 0 && depth == param_.max_depth) ||
        (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_host_[nid].weight * lr);
    } else {
      nodes_for_apply_split->push_back(entry);

      NodeEntry<GradientSumT>& e = snode_host_[nid];
      bst_float left_leaf_weight =
          evaluator.CalcWeight(nid, GradStats<GradientSumT>{e.best.left_sum}) * lr;
      bst_float right_leaf_weight =
          evaluator.CalcWeight(nid, GradStats<GradientSumT>{e.best.right_sum}) * lr;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.GetHess(),
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());

      int left_id = (*p_tree)[nid].LeftChild();
      int right_id = (*p_tree)[nid].RightChild();
      temp_qexpand_depth->push_back(ExpandEntry(left_id,  p_tree->GetDepth(left_id)));
      temp_qexpand_depth->push_back(ExpandEntry(right_id, p_tree->GetDepth(right_id)));
      // - 1 parent + 2 new children
      (*num_leaves)++;
    }
  }
  builder_monitor_.Stop("AddSplitsToTree");
}


template<typename GradientSumT>
void HistUpdater<GradientSumT>::EvaluateAndApplySplits(
    const common::GHistIndexMatrix &gmat,
    RegTree *p_tree,
    int *num_leaves,
    int depth,
    std::vector<ExpandEntry> *temp_qexpand_depth) {
  EvaluateSplits(qexpand_depth_wise_, gmat, *p_tree);

  std::vector<ExpandEntry> nodes_for_apply_split;
  AddSplitsToTree(gmat, p_tree, num_leaves, depth,
                  &nodes_for_apply_split, temp_qexpand_depth);
  ApplySplit(nodes_for_apply_split, gmat, p_tree);
}

// Split nodes to 2 sets depending on amount of rows in each node
// Histograms for small nodes will be built explicitly
// Histograms for big nodes will be built by 'Subtraction Trick'
// Exception: in distributed setting, we always build the histogram for the left child node
//    and use 'Subtraction Trick' to built the histogram for the right child node.
//    This ensures that the workers operate on the same set of tree nodes.
template <typename GradientSumT>
void HistUpdater<GradientSumT>::SplitSiblings(
    const std::vector<ExpandEntry> &nodes,
    std::vector<ExpandEntry> *small_siblings,
    std::vector<ExpandEntry> *big_siblings,
    RegTree *p_tree) {
  builder_monitor_.Start("SplitSiblings");
  for (auto const& entry : nodes) {
    int nid = entry.nid;
    RegTree::Node &node = (*p_tree)[nid];
    if (node.IsRoot()) {
      small_siblings->push_back(entry);
    } else {
      const int32_t left_id = (*p_tree)[node.Parent()].LeftChild();
      const int32_t right_id = (*p_tree)[node.Parent()].RightChild();

      if (nid == left_id && row_set_collection_[left_id ].Size() <
                            row_set_collection_[right_id].Size()) {
        small_siblings->push_back(entry);
      } else if (nid == right_id && row_set_collection_[right_id].Size() <=
                                    row_set_collection_[left_id ].Size()) {
        small_siblings->push_back(entry);
      } else {
        big_siblings->push_back(entry);
      }
    }
  }
  builder_monitor_.Stop("SplitSiblings");
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::ExpandWithDepthWise(
    const common::GHistIndexMatrix &gmat,
    RegTree *p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  int num_leaves = 0;

  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.emplace_back(ExpandEntry::kRootNid,
                                   p_tree->GetDepth(ExpandEntry::kRootNid));
  ++num_leaves;
  for (int depth = 0; depth < param_.max_depth + 1; depth++) {
    std::vector<int> sync_ids;
    std::vector<ExpandEntry> temp_qexpand_depth;
    SplitSiblings(qexpand_depth_wise_, &nodes_for_explicit_hist_build_,
                  &nodes_for_subtraction_trick_, p_tree);
    hist_rows_adder_->AddHistRows(this, &sync_ids, p_tree);
    BuildLocalHistograms(gmat, p_tree, gpair);
    hist_synchronizer_->SyncHistograms(this, sync_ids, p_tree);
    BuildNodeStats(gmat, p_tree, gpair);

    EvaluateAndApplySplits(gmat, p_tree, &num_leaves, depth,
                   &temp_qexpand_depth);

    // clean up
    qexpand_depth_wise_.clear();
    nodes_for_subtraction_trick_.clear();
    nodes_for_explicit_hist_build_.clear();
    if (temp_qexpand_depth.empty()) {
      break;
    } else {
      qexpand_depth_wise_ = temp_qexpand_depth;
      temp_qexpand_depth.clear();
    }
  }
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::ExpandWithLossGuide(
    const common::GHistIndexMatrix& gmat,
    RegTree* p_tree,
    const HostDeviceVector<GradientPair>& gpair) {
  builder_monitor_.Start("ExpandWithLossGuide");
  int num_leaves = 0;
  const auto lr = param_.learning_rate;

  ExpandEntry node(ExpandEntry::kRootNid, p_tree->GetDepth(ExpandEntry::kRootNid));
  BuildHistogramsLossGuide(node, gmat, p_tree, gpair);

  this->InitNewNode(ExpandEntry::kRootNid, gmat, gpair, *p_tree);

  this->EvaluateSplits({node}, gmat, *p_tree);
  node.split.loss_chg = snode_host_[ExpandEntry::kRootNid].best.loss_chg;

  qexpand_loss_guided_->push(node);
  ++num_leaves;

  while (!qexpand_loss_guided_->empty()) {
    const ExpandEntry candidate = qexpand_loss_guided_->top();
    const int nid = candidate.nid;
    qexpand_loss_guided_->pop();
    if (!candidate.IsValid(param_, num_leaves)) {
      (*p_tree)[nid].SetLeaf(snode_host_[nid].weight * lr);
    } else {
      auto evaluator = tree_evaluator_.GetEvaluator();
      NodeEntry<GradientSumT>& e = snode_host_[nid];
      bst_float left_leaf_weight =
          evaluator.CalcWeight(nid, GradStats<GradientSumT>{e.best.left_sum}) * lr;
      bst_float right_leaf_weight =
          evaluator.CalcWeight(nid, GradStats<GradientSumT>{e.best.right_sum}) * lr;
      p_tree->ExpandNode(nid, e.best.SplitIndex(), e.best.split_value,
                         e.best.DefaultLeft(), e.weight, left_leaf_weight,
                         right_leaf_weight, e.best.loss_chg, e.stats.GetHess(),
                         e.best.left_sum.GetHess(), e.best.right_sum.GetHess());

      this->ApplySplit({candidate}, gmat, p_tree);

      const int cleft = (*p_tree)[nid].LeftChild();
      const int cright = (*p_tree)[nid].RightChild();

      ExpandEntry left_node(cleft, p_tree->GetDepth(cleft));
      ExpandEntry right_node(cright, p_tree->GetDepth(cright));

      if (row_set_collection_[cleft].Size() < row_set_collection_[cright].Size()) {
        BuildHistogramsLossGuide(left_node, gmat, p_tree, gpair);
      } else {
        BuildHistogramsLossGuide(right_node, gmat, p_tree, gpair);
      }

      this->InitNewNode(cleft, gmat, gpair, *p_tree);
      this->InitNewNode(cright, gmat, gpair, *p_tree);
      bst_uint featureid = snode_host_[nid].best.SplitIndex();
      tree_evaluator_.AddSplit(nid, cleft, cright, featureid,
                               snode_host_[cleft].weight, snode_host_[cright].weight);
      interaction_constraints_.Split(nid, featureid, cleft, cright);

      this->EvaluateSplits({left_node, right_node}, gmat, *p_tree);
      left_node.split.loss_chg = snode_host_[cleft].best.loss_chg;
      right_node.split.loss_chg = snode_host_[cright].best.loss_chg;

      qexpand_loss_guided_->push(left_node);
      qexpand_loss_guided_->push(right_node);

      ++num_leaves;  // give two and take one, as parent is no longer a leaf
    }
  }
  builder_monitor_.Stop("ExpandWithLossGuide");
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::Update(
    xgboost::tree::TrainParam const *param,
    const common::GHistIndexMatrix &gmat,
    const HostDeviceVector<GradientPair>& gpair,
    DMatrix *p_fmat,
    xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
    RegTree *p_tree) {
  builder_monitor_.Start("Update");

  tree_evaluator_.Reset(qu_, param_, p_fmat->Info().num_col_);
  interaction_constraints_.Reset();

  this->InitData(gmat, gpair, *p_fmat, *p_tree);
  if (param_.grow_policy == xgboost::tree::TrainParam::kLossGuide) {
    ExpandWithLossGuide(gmat, p_tree, gpair);
  } else {
    ExpandWithDepthWise(gmat, p_tree, gpair);
  }

  for (int nid = 0; nid < p_tree->NumNodes(); ++nid) {
    p_tree->Stat(nid).loss_chg = snode_host_[nid].best.loss_chg;
    p_tree->Stat(nid).base_weight = snode_host_[nid].weight;
    p_tree->Stat(nid).sum_hess = static_cast<float>(snode_host_[nid].stats.GetHess());
  }

  builder_monitor_.Stop("Update");
}

template<typename GradientSumT>
bool HistUpdater<GradientSumT>::UpdatePredictionCache(
    const DMatrix* data,
    linalg::MatrixView<float> out_preds) {
  CHECK(out_preds.Device().IsSycl());
  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }
  builder_monitor_.Start("UpdatePredictionCache");
  CHECK_GT(out_preds.Size(), 0U);

  size_t n_nodes = row_set_collection_.Size();
  std::vector<::sycl::event> events(n_nodes);
  for (size_t node = 0; node < n_nodes; node++) {
    const common::RowSetCollection::Elem& rowset = row_set_collection_[node];
    if (rowset.begin != nullptr && rowset.end != nullptr && rowset.Size() != 0) {
      int nid = rowset.node_id;
      // if a node is marked as deleted by the pruner, traverse upward to locate
      // a non-deleted leaf.
      if ((*p_last_tree_)[nid].IsDeleted()) {
        while ((*p_last_tree_)[nid].IsDeleted()) {
          nid = (*p_last_tree_)[nid].Parent();
        }
        CHECK((*p_last_tree_)[nid].IsLeaf());
      }
      bst_float leaf_value = (*p_last_tree_)[nid].LeafValue();
      const size_t* rid = rowset.begin;
      const size_t num_rows = rowset.Size();

      events[node] = qu_->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for<>(::sycl::range<1>(num_rows), [=](::sycl::item<1> pid) {
          size_t row_id = rid[pid.get_id(0)];
          float& val = const_cast<float&>(out_preds(row_id));
          val += leaf_value;
        });
      });
    }
  }
  qu_->wait();

  builder_monitor_.Stop("UpdatePredictionCache");
  return true;
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::InitSampling(
      const HostDeviceVector<GradientPair>& gpair,
      USMVector<size_t, MemoryType::on_device>* row_indices) {
  const size_t num_rows = row_indices->Size();
  auto* row_idx = row_indices->Data();
  const auto* gpair_ptr = gpair.ConstDevicePointer();
  uint64_t num_samples = 0;
  const auto subsample = param_.subsample;
  ::sycl::event event;

  {
    ::sycl::buffer<uint64_t, 1> flag_buf(&num_samples, 1);
    uint64_t seed = seed_;
    seed_ += num_rows;

   /*
    * oneDLP bernoulli_distribution implicitly uses double.
    * In this case the device doesn't have fp64 support,
    * we generate bernoulli distributed random values from uniform distribution
    */
    if (has_fp64_support_) {
      // Use oneDPL bernoulli_distribution for better perf
      event = qu_->submit([&](::sycl::handler& cgh) {
        auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(num_rows)),
                                            [=](::sycl::item<1> pid) {
          uint64_t i = pid.get_id(0);
          // Create minstd_rand engine
          oneapi::dpl::minstd_rand engine(seed, i);
          oneapi::dpl::bernoulli_distribution coin_flip(subsample);
          auto bernoulli_rnd = coin_flip(engine);

          if (gpair_ptr[i].GetHess() >= 0.0f && bernoulli_rnd) {
            AtomicRef<uint64_t> num_samples_ref(flag_buf_acc[0]);
            row_idx[num_samples_ref++] = i;
          }
        });
      });
    } else {
      // Use oneDPL uniform, as far as bernoulli_distribution uses fp64
      event = qu_->submit([&](::sycl::handler& cgh) {
        auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(num_rows)),
                                            [=](::sycl::item<1> pid) {
          uint64_t i = pid.get_id(0);
          oneapi::dpl::minstd_rand engine(seed, i);
          oneapi::dpl::uniform_real_distribution<float> distr;
          const float rnd = distr(engine);
          const bool bernoulli_rnd = rnd < subsample ? 1 : 0;

          if (gpair_ptr[i].GetHess() >= 0.0f && bernoulli_rnd) {
            AtomicRef<uint64_t> num_samples_ref(flag_buf_acc[0]);
            row_idx[num_samples_ref++] = i;
          }
        });
      });
    }
    /* After calling a destructor for flag_buf,  content will be copyed to num_samples */
  }

  row_indices->Resize(qu_, num_samples, 0, &event);
  qu_->wait();
}

template<typename GradientSumT>
void HistUpdater<GradientSumT>::InitData(
                                const common::GHistIndexMatrix& gmat,
                                const HostDeviceVector<GradientPair>& gpair,
                                const DMatrix& fmat,
                                const RegTree& tree) {
  CHECK((param_.max_depth > 0 || param_.max_leaves > 0))
      << "max_depth or max_leaves cannot be both 0 (unlimited); "
      << "at least one should be a positive quantity.";
  if (param_.grow_policy == xgboost::tree::TrainParam::kDepthWise) {
    CHECK(param_.max_depth > 0) << "max_depth cannot be 0 (unlimited) "
                                << "when grow_policy is depthwise.";
  }
  builder_monitor_.Start("InitData");
  const auto& info = fmat.Info();

  if (!column_sampler_) {
    column_sampler_ = xgboost::common::MakeColumnSampler(ctx_);
  }

  // initialize the row set
  {
    row_set_collection_.Clear();

    // initialize histogram collection
    uint32_t nbins = gmat.cut.Ptrs().back();
    hist_.Init(qu_, nbins);
    hist_local_worker_.Init(qu_, nbins);

    hist_buffer_.Init(qu_, nbins);
    size_t buffer_size = kBufferSize;
    hist_buffer_.Reset(kBufferSize);

    // initialize histogram builder
    hist_builder_ = common::GHistBuilder<GradientSumT>(qu_, nbins);

    USMVector<size_t, MemoryType::on_device>* row_indices = &(row_set_collection_.Data());
    row_indices->Resize(qu_, info.num_row_);
    size_t* p_row_indices = row_indices->Data();
    // mark subsample and build list of member rows
    if (param_.subsample < 1.0f) {
      CHECK_EQ(param_.sampling_method, xgboost::tree::TrainParam::kUniform)
        << "Only uniform sampling is supported, "
        << "gradient-based sampling is only support by GPU Hist.";
      InitSampling(gpair, row_indices);
    } else {
      int has_neg_hess = 0;
      const GradientPair* gpair_ptr = gpair.ConstDevicePointer();
      ::sycl::event event;
      {
        ::sycl::buffer<int, 1> flag_buf(&has_neg_hess, 1);
        event = qu_->submit([&](::sycl::handler& cgh) {
          auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::read_write>(cgh);
          cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(info.num_row_)),
                                            [=](::sycl::item<1> pid) {
            const size_t idx = pid.get_id(0);
            p_row_indices[idx] = idx;
            if (gpair_ptr[idx].GetHess() < 0.0f) {
              AtomicRef<int> has_neg_hess_ref(flag_buf_acc[0]);
              has_neg_hess_ref.fetch_max(1);
            }
          });
        });
      }

      if (has_neg_hess) {
        size_t max_idx = 0;
        {
          ::sycl::buffer<size_t, 1> flag_buf(&max_idx, 1);
          event = qu_->submit([&](::sycl::handler& cgh) {
            cgh.depends_on(event);
            auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::read_write>(cgh);
            cgh.parallel_for<>(::sycl::range<1>(::sycl::range<1>(info.num_row_)),
                                                [=](::sycl::item<1> pid) {
              const size_t idx = pid.get_id(0);
              if (gpair_ptr[idx].GetHess() >= 0.0f) {
                AtomicRef<size_t> max_idx_ref(flag_buf_acc[0]);
                p_row_indices[max_idx_ref++] = idx;
              }
            });
          });
        }
        row_indices->Resize(qu_, max_idx, 0, &event);
      }
      qu_->wait_and_throw();
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
      data_layout_ = kDenseDataZeroBased;
    } else if (nbins_f0 == 0 && nrow * (ncol - 1) == nnz) {
      // dense data with one-based indexing
      data_layout_ = kDenseDataOneBased;
    } else {
      // sparse data
      data_layout_ = kSparseData;
    }
  }

  // store a pointer to the tree
  p_last_tree_ = &tree;
  column_sampler_->Init(ctx_, info.num_col_, info.feature_weights.ConstHostVector(),
                        param_.colsample_bynode, param_.colsample_bylevel,
                        param_.colsample_bytree);
  if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
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

  std::fill(snode_host_.begin(), snode_host_.end(),  NodeEntry<GradientSumT>(param_));

  {
    if (param_.grow_policy == xgboost::tree::TrainParam::kLossGuide) {
      qexpand_loss_guided_.reset(new ExpandQueue(LossGuide));
    } else {
      qexpand_depth_wise_.clear();
    }
  }
  builder_monitor_.Stop("InitData");
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::AddSplitsToRowSet(
                                                const std::vector<ExpandEntry>& nodes,
                                                RegTree* p_tree) {
  const size_t n_nodes = nodes.size();
  for (size_t i = 0; i < n_nodes; ++i) {
    const int32_t nid = nodes[i].nid;
    const size_t n_left = partition_builder_.GetNLeftElems(i);
    const size_t n_right = partition_builder_.GetNRightElems(i);

    row_set_collection_.AddSplit(nid, (*p_tree)[nid].LeftChild(),
        (*p_tree)[nid].RightChild(), n_left, n_right);
  }
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::ApplySplit(
                      const std::vector<ExpandEntry> nodes,
                      const common::GHistIndexMatrix& gmat,
                      RegTree* p_tree) {
  using CommonRowPartitioner = xgboost::tree::CommonRowPartitioner;
  builder_monitor_.Start("ApplySplit");

  const size_t n_nodes = nodes.size();
  std::vector<int32_t> split_conditions(n_nodes);
  CommonRowPartitioner::FindSplitConditions(nodes, *p_tree, gmat, &split_conditions);

  partition_builder_.Init(qu_, n_nodes, [&](size_t node_in_set) {
    const int32_t nid = nodes[node_in_set].nid;
    return row_set_collection_[nid].Size();
  });

  ::sycl::event event;
  partition_builder_.Partition(gmat, nodes, row_set_collection_,
                               split_conditions, p_tree, &event);
  qu_->wait_and_throw();

  for (size_t node_in_set = 0; node_in_set < n_nodes; node_in_set++) {
    const int32_t nid = nodes[node_in_set].nid;
    size_t* data_result = const_cast<size_t*>(row_set_collection_[nid].begin);
    partition_builder_.MergeToArray(node_in_set, data_result, &event);
  }
  qu_->wait_and_throw();

  AddSplitsToRowSet(nodes, p_tree);

  builder_monitor_.Stop("ApplySplit");
}

template <typename GradientSumT>
void HistUpdater<GradientSumT>::InitNewNode(int nid,
                                            const common::GHistIndexMatrix& gmat,
                                            const HostDeviceVector<GradientPair>& gpair,
                                            const RegTree& tree) {
  builder_monitor_.Start("InitNewNode");

  snode_host_.resize(tree.NumNodes(), NodeEntry<GradientSumT>(param_));
  {
    if (tree[nid].IsRoot()) {
      GradStats<GradientSumT> grad_stat;
      if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
        const std::vector<uint32_t>& row_ptr = gmat.cut.Ptrs();
        const uint32_t ibegin = row_ptr[fid_least_bins_];
        const uint32_t iend = row_ptr[fid_least_bins_ + 1];
        const auto* hist = reinterpret_cast<GradStats<GradientSumT>*>(hist_[nid].Data());

        std::vector<GradStats<GradientSumT>> ets(iend - ibegin);
        qu_->memcpy(ets.data(), hist + ibegin,
                   (iend - ibegin) * sizeof(GradStats<GradientSumT>)).wait_and_throw();
        for (const auto& et : ets) {
          grad_stat += et;
        }
      } else {
        const common::RowSetCollection::Elem e = row_set_collection_[nid];
        const size_t* row_idxs = e.begin;
        const size_t size = e.Size();
        const GradientPair* gpair_ptr = gpair.ConstDevicePointer();

        ::sycl::buffer<GradStats<GradientSumT>> buff(&grad_stat, 1);
        qu_->submit([&](::sycl::handler& cgh) {
          auto reduction = ::sycl::reduction(buff, cgh, ::sycl::plus<>());
          cgh.parallel_for<>(::sycl::range<1>(size), reduction,
                            [=](::sycl::item<1> pid, auto& sum) {
            size_t i = pid.get_id(0);
            size_t row_idx = row_idxs[i];
            if constexpr (std::is_same<GradientPair::ValueT, GradientSumT>::value) {
              sum += gpair_ptr[row_idx];
            } else {
              sum += GradStats<GradientSumT>(gpair_ptr[row_idx].GetGrad(),
                                             gpair_ptr[row_idx].GetHess());
            }
          });
        }).wait_and_throw();
      }
      auto rc = collective::Allreduce(
                      ctx_, linalg::MakeVec(reinterpret_cast<GradientSumT*>(&grad_stat), 2),
                      collective::Op::kSum);
      SafeColl(rc);
      snode_host_[nid].stats = grad_stat;
    } else {
      int parent_id = tree[nid].Parent();
      if (tree[nid].IsLeftChild()) {
        snode_host_[nid].stats = snode_host_[parent_id].best.left_sum;
      } else {
        snode_host_[nid].stats = snode_host_[parent_id].best.right_sum;
      }
    }
  }

  // calculating the weights
  {
    auto evaluator = tree_evaluator_.GetEvaluator();
    bst_uint parentid = tree[nid].Parent();
    snode_host_[nid].weight = evaluator.CalcWeight(parentid, snode_host_[nid].stats);
    snode_host_[nid].root_gain = evaluator.CalcGain(parentid, snode_host_[nid].stats);
  }
  builder_monitor_.Stop("InitNewNode");
}

// nodes_set - set of nodes to be processed in parallel
template<typename GradientSumT>
void HistUpdater<GradientSumT>::EvaluateSplits(
                        const std::vector<ExpandEntry>& nodes_set,
                        const common::GHistIndexMatrix& gmat,
                        const RegTree& tree) {
  builder_monitor_.Start("EvaluateSplits");

  const size_t n_nodes_in_set = nodes_set.size();

  using FeatureSetType = std::shared_ptr<HostDeviceVector<bst_feature_t>>;

  // Generate feature set for each tree node
  size_t pos = 0;
  for (size_t nid_in_set = 0; nid_in_set < n_nodes_in_set; ++nid_in_set) {
    const bst_node_t nid = nodes_set[nid_in_set].nid;
    FeatureSetType features_set = column_sampler_->GetFeatureSet(tree.GetDepth(nid));
    for (size_t idx = 0; idx < features_set->Size(); idx++) {
      const size_t fid = features_set->ConstHostVector()[idx];
      if (interaction_constraints_.Query(nid, fid)) {
        auto this_hist = hist_[nid].DataConst();
        if (pos < split_queries_host_.size()) {
          split_queries_host_[pos] = SplitQuery{nid, fid, this_hist};
        } else {
          split_queries_host_.push_back({nid, fid, this_hist});
        }
        ++pos;
      }
    }
  }
  const size_t total_features = pos;

  split_queries_device_.Resize(qu_, total_features);
  auto event = qu_->memcpy(split_queries_device_.Data(), split_queries_host_.data(),
                          total_features * sizeof(SplitQuery));

  auto evaluator = tree_evaluator_.GetEvaluator();
  SplitQuery* split_queries_device = split_queries_device_.Data();
  const uint32_t* cut_ptr = gmat.cut.cut_ptrs_.ConstDevicePointer();
  const bst_float* cut_val = gmat.cut.cut_values_.ConstDevicePointer();

  snode_device_.ResizeNoCopy(qu_, snode_host_.size());
  event = qu_->memcpy(snode_device_.Data(), snode_host_.data(),
                     snode_host_.size() * sizeof(NodeEntry<GradientSumT>), event);
  const NodeEntry<GradientSumT>* snode = snode_device_.Data();

  const float min_child_weight = param_.min_child_weight;

  best_splits_device_.ResizeNoCopy(qu_, total_features);
  if (best_splits_host_.size() < total_features) best_splits_host_.resize(total_features);
  SplitEntry<GradientSumT>* best_splits = best_splits_device_.Data();

  event = qu_->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event);
    cgh.parallel_for<>(::sycl::nd_range<2>(::sycl::range<2>(total_features, sub_group_size_),
                                           ::sycl::range<2>(1, sub_group_size_)),
                       [=](::sycl::nd_item<2> pid) {
      int i = pid.get_global_id(0);
      auto sg = pid.get_sub_group();
      int nid = split_queries_device[i].nid;
      int fid = split_queries_device[i].fid;
      const GradientPairT* hist_data = split_queries_device[i].hist;

      best_splits[i] = snode[nid].best;
      EnumerateSplit(sg, cut_ptr, cut_val, hist_data, snode[nid],
                     &(best_splits[i]), fid, nid, evaluator, min_child_weight);
    });
  });
  event = qu_->memcpy(best_splits_host_.data(), best_splits,
                     total_features * sizeof(SplitEntry<GradientSumT>), event);

  qu_->wait();
  for (size_t i = 0; i < total_features; i++) {
    int nid = split_queries_host_[i].nid;
    snode_host_[nid].best.Update(best_splits_host_[i]);
  }

  builder_monitor_.Stop("EvaluateSplits");
}

// Enumerate the split values of specific feature.
// Returns the sum of gradients corresponding to the data points that contains a non-missing value
// for the particular feature fid.
template <typename GradientSumT>
void HistUpdater<GradientSumT>::EnumerateSplit(
    const ::sycl::sub_group& sg,
    const uint32_t* cut_ptr,
    const bst_float* cut_val,
    const GradientPairT* hist_data,
    const NodeEntry<GradientSumT>& snode,
    SplitEntry<GradientSumT>* p_best,
    bst_uint fid,
    bst_uint nodeID,
    typename TreeEvaluator<GradientSumT>::SplitEvaluator const &evaluator,
    float min_child_weight) {
  SplitEntry<GradientSumT> best;

  int32_t ibegin = static_cast<int32_t>(cut_ptr[fid]);
  int32_t iend = static_cast<int32_t>(cut_ptr[fid + 1]);

  GradStats<GradientSumT> sum(0, 0);

  int32_t sub_group_size = sg.get_local_range().size();
  const size_t local_id = sg.get_local_id()[0];

  /* TODO(razdoburdin)
   * Currently the first additions are fast and the last are slow.
   * Maybe calculating of reduce overgroup in seprate kernel and reusing it here can be faster
   */
  for (int32_t i = ibegin + local_id; i < iend; i += sub_group_size) {
    sum.Add(::sycl::inclusive_scan_over_group(sg, hist_data[i].GetGrad(), std::plus<>()),
            ::sycl::inclusive_scan_over_group(sg, hist_data[i].GetHess(), std::plus<>()));

    if (sum.GetHess() >= min_child_weight) {
      GradStats<GradientSumT> c = snode.stats - sum;
      if (c.GetHess() >= min_child_weight) {
        bst_float loss_chg = evaluator.CalcSplitGain(nodeID, fid, sum, c) - snode.root_gain;
        bst_float split_pt = cut_val[i];
        best.Update(loss_chg, fid, split_pt, false, sum, c);
      }
    }

    const bool last_iter = i + sub_group_size >= iend;
    if (!last_iter) {
      size_t end = i - local_id + sub_group_size;
      if (end > iend) end = iend;
      for (size_t j = i + 1; j < end; ++j) {
        sum.Add(hist_data[j].GetGrad(), hist_data[j].GetHess());
      }
    }
  }

  bst_float total_loss_chg = ::sycl::reduce_over_group(sg, best.loss_chg, maximum<>());
  bst_feature_t total_split_index = ::sycl::reduce_over_group(sg,
                                                              best.loss_chg == total_loss_chg ?
                                                              best.SplitIndex() :
                                                              (1U << 31) - 1U, minimum<>());
  if (best.loss_chg == total_loss_chg &&
      best.SplitIndex() == total_split_index) p_best->Update(best);
}

template class HistUpdater<float>;
template class HistUpdater<double>;

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost
