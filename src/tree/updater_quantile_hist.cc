/*!
 * Copyright 2017-2019 by Contributors
 * \file updater_quantile_hist.cc
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Checn, Egor Smirnov
 */
#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <xgboost/logging.h>
#include <xgboost/tree_updater.h>

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <queue>
#include <iomanip>
#include <numeric>
#include <string>
#include <utility>

#include "./param.h"
#include "./updater_quantile_hist.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/column_matrix.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist);

void QuantileHistMaker::Init(const std::vector<std::pair<std::string, std::string> >& args) {
  // initialize pruner
  if (!pruner_) {
    pruner_.reset(TreeUpdater::Create("prune"));
  }
  pruner_->Init(args);
  param_.InitAllowUnknown(args);
  is_gmat_initialized_ = false;

  // initialise the split evaluator
  if (!spliteval_) {
    spliteval_.reset(SplitEvaluator::Create(param_.split_evaluator));
  }

  spliteval_->Init(args);
}

void QuantileHistMaker::Update(HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               const std::vector<RegTree *> &trees) {
  // omp_set_nested(1);
  if (is_gmat_initialized_ == false) {
    double tstart = dmlc::GetTime();
    gmat_.Init(dmat, static_cast<uint32_t>(param_.max_bin));
    column_matrix_.Init(gmat_, param_.sparse_threshold);
    if (param_.enable_feature_grouping > 0) {
      gmatb_.Init(gmat_, column_matrix_, param_);
    }
    is_gmat_initialized_ = true;
    LOG(INFO) << "Generating gmat: " << dmlc::GetTime() - tstart << " sec";
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();
  // build tree
  if (!builder_) {
    builder_.reset(new Builder(
        param_,
        std::move(pruner_),
        std::unique_ptr<SplitEvaluator>(spliteval_->GetHostClone())));
  }
  for (auto tree : trees) {
    builder_->Update(gmat_, gmatb_, column_matrix_, gpair, dmat, tree);
  }
  param_.learning_rate = lr;
}

bool QuantileHistMaker::UpdatePredictionCache(
    const DMatrix* data,
    HostDeviceVector<bst_float>* out_preds) {
  if (!builder_ || param_.subsample < 1.0f) {
    return false;
  } else {
    return builder_->UpdatePredictionCache(data, out_preds);
  }
}

void QuantileHistMaker::Builder::BuildNodeStat(
    const GHistIndexMatrix &gmat,
    DMatrix *p_fmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h,
    int32_t nid) {

  auto parent_id = (*p_tree)[nid].Parent();

  this->InitNewNode(nid, gmat, gpair_h, *p_fmat, p_tree, &(snode_[nid]), parent_id);
  // add constraints
  if (!(*p_tree)[nid].IsLeftChild() && !(*p_tree)[nid].IsRoot()) {
    // it's a right child
    auto left_sibling_id = (*p_tree)[parent_id].LeftChild();
    auto parent_split_feature_id = snode_[parent_id].best.SplitIndex();

    {
      spliteval_->AddSplit(parent_id, left_sibling_id, nid, parent_split_feature_id,
          snode_[left_sibling_id].weight, snode_[nid].weight);
    }
  }
}

void QuantileHistMaker::Builder::BuildNodeStatBatch(
    const GHistIndexMatrix &gmat,
    DMatrix *p_fmat,
    RegTree *p_tree,
    const std::vector<GradientPair> &gpair_h,
    const std::vector<ExpandEntry>& nodes) {
  perf_monitor.TickStart();
  for (const auto& node : nodes) {
    const int32_t nid = node.nid;
    const int32_t sibling_nid = node.sibling_nid;
    BuildNodeStat(gmat, p_fmat, p_tree, gpair_h, nid);
    if (sibling_nid > -1) {
      BuildNodeStat(gmat, p_fmat, p_tree, gpair_h, sibling_nid);
    }
  }
  perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::INIT_NEW_NODE);
}

template<typename RowIdxType, typename IdxType>
std::pair<size_t, size_t> PartitionDenseLeftDefaultKernel(const RowIdxType* rid,
  const IdxType* idx, const IdxType offset, const int32_t split_cond,
  const size_t istart, const size_t iend, RowIdxType* p_left, RowIdxType* p_right) {
  size_t ileft = 0;
  size_t iright = 0;

  for (size_t i = istart; i < iend; i++) {
    if ( idx[rid[i]] == std::numeric_limits<uint32_t>::max() ||
        static_cast<int32_t>(idx[rid[i]] + offset) <= split_cond) {
      p_left[ileft++] = rid[i];
    } else {
      p_right[iright++] = rid[i];
    }
  }
  return { ileft, iright };
}

template<typename RowIdxType, typename IdxType>
std::pair<size_t, size_t> PartitionDenseRightDefaultKernel(const RowIdxType* rid,
  const IdxType* idx, const IdxType offset, const int32_t split_cond,
  const size_t istart, const size_t iend, RowIdxType* p_left, RowIdxType* p_right) {
  size_t ileft = 0;
  size_t iright = 0;

  for (size_t i = istart; i < iend; i++) {
    if (idx[rid[i]] == std::numeric_limits<uint32_t>::max() ||
      static_cast<int32_t>(idx[rid[i]] + offset) > split_cond) {
      p_right[iright++] = rid[i];
    } else {
      p_left[ileft++] = rid[i];
    }
  }
  return { ileft, iright };
}

template<typename RowIdxType, typename IdxType>
std::pair<size_t, size_t> PartitionSparseKernel(const RowIdxType* rowid,
    const IdxType* idx, const int32_t split_cond, const size_t ibegin,
    const size_t iend, RowIdxType* p_left, RowIdxType* p_right,
    Column column, bool default_left) {
  size_t ileft = 0;
  size_t iright = 0;

  if (ibegin < iend) {  // ensure that [ibegin, iend) is nonempty range
    // search first nonzero row with index >= rowid[ibegin]
    const size_t* p = std::lower_bound(column.GetRowData(),
                                       column.GetRowData() + column.Size(),
                                       rowid[ibegin]);
    if (p != column.GetRowData() + column.Size() && *p <= rowid[iend - 1]) {
      size_t cursor = p - column.GetRowData();

      for (size_t i = ibegin; i < iend; ++i) {
        const size_t rid = rowid[i];
        while (cursor < column.Size()
               && column.GetRowIdx(cursor) < rid
               && column.GetRowIdx(cursor) <= rowid[iend - 1]) {
          ++cursor;
        }
        if (cursor < column.Size() && column.GetRowIdx(cursor) == rid) {
          const uint32_t rbin = column.GetFeatureBinIdx(cursor);
          if (static_cast<int32_t>(rbin + column.GetBaseIdx()) <= split_cond) {
            p_left[ileft++] = rid;
          } else {
            p_right[iright++] = rid;
          }
          ++cursor;
        } else {
          // missing value
          if (default_left) {
            p_left[ileft++] = rid;
          } else {
            p_right[iright++] = rid;
          }
        }
      }
    } else {  // all rows in [ibegin, iend) have missing values
      if (default_left) {
        for (size_t i = ibegin; i < iend; ++i) {
          const size_t rid = rowid[i];
          p_left[ileft++] = rid;
        }
      } else {
        for (size_t i = ibegin; i < iend; ++i) {
          const size_t rid = rowid[i];
          p_right[iright++] = rid;
        }
      }
    }
  }
  return {ileft, iright};
}


int32_t QuantileHistMaker::Builder::FindSplitCond(int32_t nid,
                                                  RegTree *p_tree,
                                                  const GHistIndexMatrix &gmat) {
  bst_float left_leaf_weight  = spliteval_->ComputeWeight(nid,
      snode_[nid].best.left_sum) * param_.learning_rate;
  bst_float right_leaf_weight = spliteval_->ComputeWeight(nid,
      snode_[nid].best.right_sum) * param_.learning_rate;
  p_tree->ExpandNode(nid, snode_[nid].best.SplitIndex(), snode_[nid].best.split_value,
                  snode_[nid].best.DefaultLeft(), snode_[nid].weight, left_leaf_weight,
                  right_leaf_weight, snode_[nid].best.loss_chg, snode_[nid].stats.sum_hess);

  RegTree::Node node = (*p_tree)[nid];
  // Categorize member rows
  const bst_uint fid = node.SplitIndex();
  const bst_float split_pt = node.SplitCond();
  const uint32_t lower_bound = gmat.cut.row_ptr[fid];
  const uint32_t upper_bound = gmat.cut.row_ptr[fid + 1];
  int32_t split_cond = -1;
  // convert floating-point split_pt into corresponding bin_id
  // split_cond = -1 indicates that split_pt is less than all known cut points
  CHECK_LT(upper_bound,
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  for (uint32_t i = lower_bound; i < upper_bound; ++i) {
    if (split_pt == gmat.cut.cut[i]) {
      split_cond = static_cast<int32_t>(i);
    }
  }
  return split_cond;
}

void QuantileHistMaker::Builder::CreateNewNodesBatch(
    const std::vector<ExpandEntry>& nodes,
    const GHistIndexMatrix &gmat,
    const ColumnMatrix &column_matrix,
    DMatrix *p_fmat,
    RegTree *p_tree,
    int *num_leaves,
    int depth,
    unsigned *timestamp,
    std::vector<ExpandEntry> *temp_qexpand_depth) {
  perf_monitor.TickStart();
  const size_t block_size = 2048;

  // inputs for tasks
  std::vector<int32_t> nids_to_split;
  std::vector<int32_t> cond_to_split;
  std::vector<size_t> n_blocks_vec;
  std::vector<std::pair<size_t, size_t>> tasks;
  size_t* buffer = buffer_for_partition_.data();
  size_t cur_buff_offset = 0;

  // buffers to store partial results
  std::vector<std::vector<std::pair<size_t*, size_t*>>> buffers_by_nids;
  std::vector<std::vector<std::pair<size_t, size_t>>> sizes_by_nids;

  auto create_nodes = [&](int32_t this_nid) {
    if (snode_[this_nid].best.loss_chg < kRtEps ||
        (param_.max_depth > 0 && depth == param_.max_depth) ||
        (param_.max_leaves > 0 && (*num_leaves) == param_.max_leaves)) {
      (*p_tree)[this_nid].SetLeaf(snode_[this_nid].weight * param_.learning_rate);
    } else {
      nids_to_split.push_back(this_nid);
      cond_to_split.push_back(FindSplitCond(this_nid, p_tree, gmat));

      const size_t nrows = row_set_collection_[this_nid].Size();
      const size_t n_blocks = nrows / block_size + !!(nrows % block_size);
      n_blocks_vec.push_back(n_blocks);

      buffers_by_nids.resize(buffers_by_nids.size() + 1);

      for (size_t i = 0; i < n_blocks; ++i) {
        const size_t istart = i*block_size;
        const size_t iend = (i == n_blocks-1) ? nrows : istart + block_size;
        buffers_by_nids.back().push_back({ buffer + cur_buff_offset,
            buffer + cur_buff_offset + (iend-istart) });
        cur_buff_offset += 2*(iend-istart);
        tasks.emplace_back(cond_to_split.size() - 1,  i);
      }
      sizes_by_nids.emplace_back(n_blocks);
    }
  };
  for (const auto& node : nodes) {
    const int32_t nid = node.nid;
    const int32_t sibling_nid = node.sibling_nid;
    create_nodes(nid);

    if (sibling_nid > -1) {
      create_nodes(sibling_nid);
    }
  }

  // buffer to store # of rows in left part for each row-block
  std::vector<size_t> left_sizes;

  const int32_t size = tasks.size();
  #pragma omp parallel
  {
    // compute partial partitions
    #pragma omp for schedule(guided)
    for (int32_t i = 0; i < size; ++i) {
      const size_t node_idx = tasks[i].first;
      const size_t iblock   = tasks[i].second;
      const int32_t split_cond = cond_to_split[node_idx];
      const int32_t nid = nids_to_split[node_idx];
      const bst_uint fid = (*p_tree)[nid].SplitIndex();

      const size_t nrows = row_set_collection_[nid].Size();
      const size_t nblocks = n_blocks_vec[node_idx];
      const size_t istart = iblock*block_size;
      const size_t iend = (iblock == nblocks-1) ? nrows : istart + block_size;

      const bool default_left = (*p_tree)[nid].DefaultLeft();
      const Column column = column_matrix.GetColumn(fid);

      const uint32_t* idx = column.GetIndex();
      const size_t* rid = row_set_collection_[nid].begin;

      if (column.GetType() == xgboost::common::kDenseColumn) {
        if (default_left) {
          sizes_by_nids[node_idx][iblock] = PartitionDenseLeftDefaultKernel<size_t, uint32_t>(
              rid, idx, column.GetBaseIdx(), split_cond, istart, iend,
              buffers_by_nids[node_idx][iblock].first, buffers_by_nids[node_idx][iblock].second);
        } else {
          sizes_by_nids[node_idx][iblock] = PartitionDenseRightDefaultKernel<size_t, uint32_t>(
              rid, idx, column.GetBaseIdx(), split_cond, istart, iend,
              buffers_by_nids[node_idx][iblock].first, buffers_by_nids[node_idx][iblock].second);
        }
      } else {
        sizes_by_nids[node_idx][iblock] = PartitionSparseKernel<size_t, uint32_t>(
          rid, idx, split_cond, istart, iend, buffers_by_nids[node_idx][iblock].first,
          buffers_by_nids[node_idx][iblock].second, column, default_left);
      }
    }

    // calculate sizes of left parts in each block
    #pragma omp single
    {
      for (size_t inode = 0; inode < nids_to_split.size(); ++inode) {
        size_t nLeft = 0;
        for (auto& size : sizes_by_nids[inode]) {
          nLeft += size.first;
        }
        left_sizes.push_back(nLeft);
      }
    }

    // merge partial results to one
    #pragma omp for schedule(guided)
    for (int32_t i = 0; i < size; ++i) {
      const size_t node_idx = tasks[i].first;
      const size_t iblock   = tasks[i].second;

      const int32_t nid = nids_to_split[node_idx];
      auto* rid = const_cast<size_t*>(row_set_collection_[nid].begin);

      size_t iLeft = 0;
      size_t iRight = 0;

      const size_t nLeft = left_sizes[node_idx];

      for (size_t j = 0; j < iblock; ++j) {
        iLeft += sizes_by_nids[node_idx][j].first;
        iRight += sizes_by_nids[node_idx][j].second;
      }

      memcpy(rid + iLeft, buffers_by_nids[node_idx][iblock].first,
            sizes_by_nids[node_idx][iblock].first * sizeof(rid[0]));
      memcpy(rid + nLeft + iRight, buffers_by_nids[node_idx][iblock].second,
            sizes_by_nids[node_idx][iblock].second * sizeof(rid[0]));
    }
  }
  // register new nodes
  for (size_t i = 0; i < nids_to_split.size(); ++i) {
    const int32_t nid = nids_to_split[i];
    const size_t nLeft = left_sizes[i];
    RegTree::Node node = (*p_tree)[nid];

    const int32_t left_id = node.LeftChild();
    const int32_t right_id = node.RightChild();
    row_set_collection_.AddSplit(nid, nLeft, left_id, right_id);

    if (row_set_collection_[left_id].Size() < row_set_collection_[right_id].Size()) {
        temp_qexpand_depth->push_back(ExpandEntry(left_id, right_id, nid,
            depth + 1, 0.0, (*timestamp)++));
    } else {
        temp_qexpand_depth->push_back(ExpandEntry(right_id, left_id, nid,
            depth + 1, 0.0, (*timestamp)++));
    }
  }

  perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::APPLY_SPLIT);
}

std::tuple<float*, GradStats*>  QuantileHistMaker::Builder::GetHistBuffer(
    std::vector<uint8_t>* hist_is_init, std::vector<GradStats>* grad_stats,
    size_t block_id, size_t nthread, size_t tid,
    std::vector<float*>* data_hist, size_t hist_size) {

  const size_t n_hist_for_current_node = hist_is_init->size();
  const size_t hist_id = ((n_hist_for_current_node == nthread) ? tid : block_id);

  float* local_data_hist = (*data_hist)[hist_id];
  if (!(*hist_is_init)[hist_id]) {
    std::fill(local_data_hist, local_data_hist + hist_size, 0.0f);
    (*hist_is_init)[hist_id] = true;
  }

  return std::make_tuple(local_data_hist, &(*grad_stats)[hist_id]);
}

void QuantileHistMaker::Builder::BuildHistsBatch(const std::vector<ExpandEntry>& nodes,
    RegTree* p_tree, const GHistIndexMatrix &gmat, const std::vector<GradientPair>& gpair,
    bool sync_hist, std::vector<std::vector<float*>>* hist_buffers,
    std::vector<std::vector<uint8_t>>* hist_is_init) {
  perf_monitor.TickStart();
  const size_t block_size_rows = 256;
  const size_t nthread = static_cast<size_t>(this->nthread_);
  const size_t nbins = gmat.cut.row_ptr.back();
  const size_t hist_size = 2  * nbins;

  hist_buffers->resize(nodes.size());
  hist_is_init->resize(nodes.size());

  // input data for tasks
  int32_t n_tasks = 0;
  std::vector<int32_t> task_nid;
  std::vector<int32_t> task_node_idx;
  std::vector<int32_t> task_block_idx;

  // result vector
  std::vector<std::vector<GradStats>> grad_stats(nodes.size());

  size_t i_hist = 0;

  // prepare tasks for parallel exection
  for (size_t i = 0; i < nodes.size(); ++i) {
    const int32_t nid = nodes[i].nid;
    const int32_t sibling_nid = nodes[i].sibling_nid;
    hist_.AddHistRow(nid);
    if (sibling_nid > -1) {
      hist_.AddHistRow(sibling_nid);
    }
    const size_t nrows = row_set_collection_[nid].Size();
    const size_t n_local_blocks = nrows / block_size_rows + !!(nrows % block_size_rows);
    const size_t n_local_histograms = std::min(nthread, n_local_blocks);

    for (size_t j = 0; j < n_local_blocks; ++j) {
      task_nid.push_back(nid);
      task_node_idx.push_back(i);
      task_block_idx.push_back(j);
    }
    n_tasks += n_local_blocks;

    (*hist_buffers)[i].clear();
    for (size_t j = 0; j < n_local_histograms; j++) {
      (*hist_buffers)[i].push_back(reinterpret_cast<float*>(hist_buff_[i_hist++].data()));
    }
    (*hist_is_init)[i].clear();
    (*hist_is_init)[i].resize(n_local_histograms, false);
    grad_stats[i].resize(n_local_histograms);
  }
  const float* const pgh = reinterpret_cast<const float*>(gpair.data());

  // execute tasks in parallel
  #pragma omp parallel for schedule(guided)
  for (int32_t itask = 0; itask < n_tasks; ++itask) {
    const size_t tid = omp_get_thread_num();
    const int32_t nid = task_nid[itask];
    const int32_t block_id = task_block_idx[itask];
    const int32_t node_idx = task_node_idx[itask];

    float* data_local_hist;
    GradStats* grad_stat;
    std::tie(data_local_hist, grad_stat) = GetHistBuffer(&(*hist_is_init)[node_idx],
        &grad_stats[node_idx], block_id, nthread, tid,
        &(*hist_buffers)[node_idx], hist_size);

    const size_t* row_ptr = gmat.row_ptr.data();
    const size_t* rid =  row_set_collection_[nid].begin;

    const size_t nrows = row_set_collection_[nid].Size();
    const size_t istart = block_id * block_size_rows;
    const size_t iend = (((block_id+1)*block_size_rows > nrows) ? nrows : istart + block_size_rows);

    if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
      common::BuildHistLocalDense(istart, iend, nrows, rid, gmat.index.data(), pgh,
        row_ptr, data_local_hist, grad_stat);
    } else {
      common::BuildHistLocalSparse(istart, iend, nrows, rid, gmat.index.data(), pgh,
        row_ptr, data_local_hist, grad_stat);
    }
  }

  // merge grad stats
  if (!sync_hist) {
    for (size_t inode = 0; inode < nodes.size(); ++inode) {
      const int32_t nid = nodes[inode].nid;
      const int32_t sibling_nid = nodes[inode].sibling_nid;

      GradStats grad_stat;

      for (size_t ihist = 0; ihist < (*hist_is_init)[inode].size(); ++ihist) {
        if ((*hist_is_init)[inode][ihist]) {
          grad_stat.Add(grad_stats[inode][ihist]);
        }
      }
      if (snode_.size() <= size_t(nid)) {
        snode_.resize(nid + 1, NodeEntry(param_));
      }
      snode_[nid].stats = grad_stat;

      if (sibling_nid > -1) {
        if (snode_.size() <= size_t(sibling_nid)) {
          snode_.resize(sibling_nid + 1, NodeEntry(param_));
        }

        auto node = (*p_tree)[nid];
        int32_t parent_nid = node.Parent();

        auto& st = snode_[parent_nid].stats;
        snode_[sibling_nid].stats.SetSubstract(st, grad_stat);
      }
    }
  }

  perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::BUILD_HIST);
}

void QuantileHistMaker::Builder::ReduceHistograms(float* hist_data, float* sibling_hist_data,
    float* parent_hist_data, size_t fid, size_t inode,
    const std::vector<std::vector<uint8_t>>& hist_is_init, const GHistIndexMatrix &gmat,
    const std::vector<std::vector<float*>>& hist_buffers) {
  const std::vector<uint32_t>& cut_ptr = gmat.cut.row_ptr;
  const size_t ibegin = 2 * cut_ptr[fid];
  const size_t iend = 2 * cut_ptr[fid + 1];

  bool is_init = false;
  for (size_t ihist = 0; ihist < hist_is_init[inode].size(); ++ihist) {
    float* partial_data = hist_buffers[inode][ihist];
    if (hist_is_init[inode][ihist] && is_init) {
      for (size_t i = ibegin; i < iend; ++i) {
        hist_data[i] += partial_data[i];
      }
    } else if (hist_is_init[inode][ihist]) {
      for (size_t i = ibegin; i < iend; ++i) {
        hist_data[i] = partial_data[i];
      }
      is_init = true;
    }
  }

  if (sibling_hist_data) {
    for (size_t i = ibegin; i < iend; ++i) {
      sibling_hist_data[i] = parent_hist_data[i] - hist_data[i];
    }
  }
}

void QuantileHistMaker::Builder::ExpandWithDepthWidth(
  const GHistIndexMatrix &gmat,
  const GHistIndexBlockMatrix &gmatb,
  const ColumnMatrix &column_matrix,
  DMatrix* p_fmat,
  RegTree* p_tree,
  const std::vector<GradientPair> &gpair_h) {
  unsigned timestamp = 0;
  int num_leaves = 0;

  // in depth_wise growing, we feed loss_chg with 0.0 since it is not used anyway
  qexpand_depth_wise_.emplace_back(0, -1, -1, p_tree->GetDepth(0), 0.0, timestamp++);
  ++num_leaves;

  for (int depth = 0; depth < param_.max_depth + 1; depth++) {
    std::vector<ExpandEntry> temp_qexpand_depth;

    // buffer to store partial histograms
    std::vector<std::vector<float*>> hist_buffers;
    // uint8_t is used instead of bool due to read/write
    // to std::vector<bool> - thread unsafe
    std::vector<std::vector<uint8_t>> hist_is_init;

    BuildHistsBatch(qexpand_depth_wise_, p_tree, gmat, gpair_h, false,
        &hist_buffers, &hist_is_init);
    BuildNodeStatBatch(gmat, p_fmat, p_tree, gpair_h, qexpand_depth_wise_);
    EvaluateSplitsBatch(qexpand_depth_wise_, gmat, *p_fmat, hist_is_init, hist_buffers, p_tree);
    CreateNewNodesBatch(qexpand_depth_wise_, gmat, column_matrix, p_fmat, p_tree,
        &num_leaves, depth, &timestamp, &temp_qexpand_depth);

    num_leaves += temp_qexpand_depth.size();

    // clean up
    qexpand_depth_wise_.clear();
    nodes_for_subtraction_trick_.clear();
    if (temp_qexpand_depth.empty()) {
      break;
    } else {
      qexpand_depth_wise_ = temp_qexpand_depth;
      temp_qexpand_depth.clear();
    }
  }
}

void QuantileHistMaker::Builder::ExpandWithLossGuide(
    const GHistIndexMatrix& gmat,
    const GHistIndexBlockMatrix& gmatb,
    const ColumnMatrix& column_matrix,
    DMatrix* p_fmat,
    RegTree* p_tree,
    const std::vector<GradientPair>& gpair_h) {
  unsigned timestamp = 0;
  int num_leaves = 0;

  std::vector<std::vector<float*>> hist_buffers;
  std::vector<std::vector<uint8_t>> hist_is_init;

  for (int nid = 0; nid < p_tree->param.num_roots; ++nid) {
    std::vector<ExpandEntry> nodes_to_build{ExpandEntry(
        0, -1, -1, p_tree->GetDepth(0), 0.0, timestamp++)};

    BuildHistsBatch(nodes_to_build, p_tree, gmat, gpair_h, false, &hist_buffers, &hist_is_init);
    BuildNodeStatBatch(gmat, p_fmat, p_tree, gpair_h, nodes_to_build);
    EvaluateSplitsBatch(nodes_to_build, gmat, *p_fmat, hist_is_init, hist_buffers, p_tree);

    qexpand_loss_guided_->push(ExpandEntry(nid, -1, -1, p_tree->GetDepth(nid),
                               snode_[nid].best.loss_chg,
                               timestamp++));
    ++num_leaves;
  }

  while (!qexpand_loss_guided_->empty()) {
    const ExpandEntry candidate = qexpand_loss_guided_->top();
    const int32_t nid = candidate.nid;
    qexpand_loss_guided_->pop();

    std::vector<ExpandEntry> nodes_to_build{candidate};
    std::vector<ExpandEntry> successors;

    CreateNewNodesBatch(nodes_to_build, gmat, column_matrix, p_fmat, p_tree,
        &num_leaves, candidate.depth, &timestamp, &successors);

    if (!successors.empty()) {
      BuildHistsBatch(successors, p_tree, gmat, gpair_h, false, &hist_buffers, &hist_is_init);
      BuildNodeStatBatch(gmat, p_fmat, p_tree, gpair_h, successors);
      EvaluateSplitsBatch(successors, gmat, *p_fmat, hist_is_init, hist_buffers, p_tree);

      const int32_t cleft = (*p_tree)[nid].LeftChild();
      const int32_t cright = (*p_tree)[nid].RightChild();

      qexpand_loss_guided_->push(ExpandEntry(cleft, -1, nid, p_tree->GetDepth(cleft),
                                 snode_[cleft].best.loss_chg,
                                 timestamp++));
      qexpand_loss_guided_->push(ExpandEntry(cright, -1, nid, p_tree->GetDepth(cright),
                                 snode_[cright].best.loss_chg,
                                 timestamp++));
      ++num_leaves;  // give two and take one, as parent is no longer a leaf
    }
  }
}

void QuantileHistMaker::Builder::Update(const GHistIndexMatrix& gmat,
                                        const GHistIndexBlockMatrix& gmatb,
                                        const ColumnMatrix& column_matrix,
                                        HostDeviceVector<GradientPair>* gpair,
                                        DMatrix* p_fmat,
                                        RegTree* p_tree) {
  perf_monitor.StartPerfMonitor();

  const std::vector<GradientPair>& gpair_h = gpair->ConstHostVector();
  spliteval_->Reset();

  perf_monitor.TickStart();
  this->InitData(gmat, gpair_h, *p_fmat, *p_tree);
  perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::INIT_DATA);

  if (param_.grow_policy == TrainParam::kLossGuide) {
    ExpandWithLossGuide(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
  } else {
    ExpandWithDepthWidth(gmat, gmatb, column_matrix, p_fmat, p_tree, gpair_h);
  }

  for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
    p_tree->Stat(nid).loss_chg = snode_[nid].best.loss_chg;
    p_tree->Stat(nid).base_weight = snode_[nid].weight;
    p_tree->Stat(nid).sum_hess = static_cast<float>(snode_[nid].stats.sum_hess);
  }

  pruner_->Update(gpair, p_fmat, std::vector<RegTree*>{p_tree});

  perf_monitor.EndPerfMonitor();
}

bool QuantileHistMaker::Builder::UpdatePredictionCache(
      const DMatrix* data,
      HostDeviceVector<bst_float>* p_out_preds) {
  std::vector<bst_float>& out_preds = p_out_preds->HostVector();

  // p_last_fmat_ is a valid pointer as long as UpdatePredictionCache() is called in
  // conjunction with Update().
  if (!p_last_fmat_ || !p_last_tree_ || data != p_last_fmat_) {
    return false;
  }

  if (leaf_value_cache_.empty()) {
    leaf_value_cache_.resize(p_last_tree_->param.num_nodes,
                             std::numeric_limits<float>::infinity());
  }

  CHECK_GT(out_preds.size(), 0U);

  const size_t block_size = 2048;
  const size_t n_nodes = row_set_collection_.end() - row_set_collection_.begin();
  std::vector<RowSetCollection::Elem> tasks_elem;
  std::vector<size_t> tasks_iblock;
  std::vector<size_t> tasks_nblock;

  for (size_t k = 0; k < n_nodes; ++k) {
    const size_t nrows = row_set_collection_[k].Size();
    const size_t nblocks = nrows / block_size + !!(nrows % block_size);

    for (size_t i = 0; i < nblocks; ++i) {
      tasks_elem.push_back(row_set_collection_[k]);
      tasks_iblock.push_back(i);
      tasks_nblock.push_back(nblocks);
    }
  }

  #pragma omp parallel for schedule(guided)
  for (int32_t k = 0; k < tasks_elem.size(); ++k) {
    const RowSetCollection::Elem rowset = tasks_elem[k];
    if (rowset.begin != nullptr && rowset.end != nullptr && rowset.node_id != -1) {
      const size_t nrows = rowset.Size();
      const size_t iblock  = tasks_iblock[k];
      const size_t nblocks = tasks_nblock[k];

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

      const size_t istart = iblock*block_size;
      const size_t iend = (iblock == nblocks-1) ? nrows : istart + block_size;

      for (size_t it = istart; it < iend; ++it) {
        out_preds[rowset.begin[it]] += leaf_value;
      }
    }
  }

  return true;
}

void QuantileHistMaker::Builder::InitData(const GHistIndexMatrix& gmat,
                                          const std::vector<GradientPair>& gpair,
                                          const DMatrix& fmat,
                                          const RegTree& tree) {
  CHECK_EQ(tree.param.num_nodes, tree.param.num_roots)
      << "ColMakerHist: can only grow new tree";
  CHECK((param_.max_depth > 0 || param_.max_leaves > 0))
      << "max_depth or max_leaves cannot be both 0 (unlimited); "
      << "at least one should be a positive quantity.";
  if (param_.grow_policy == TrainParam::kDepthWise) {
    CHECK(param_.max_depth > 0) << "max_depth cannot be 0 (unlimited) "
                                << "when grow_policy is depthwise.";
  }
  const auto& info = fmat.Info();

  {
    // initialize the row set
    row_set_collection_.Clear();
    // clear local prediction cache
    leaf_value_cache_.clear();
    // initialize histogram collection
    uint32_t nbins = gmat.cut.row_ptr.back();
    hist_.Init(nbins);
    hist_buff_.Init(nbins);

    // initialize histogram builder
    #pragma omp parallel
    {
      this->nthread_ = omp_get_num_threads();
    }

    const auto nthread = static_cast<bst_omp_uint>(this->nthread_);
    row_split_tloc_.resize(nthread);
    hist_builder_.Init(this->nthread_, nbins);

    CHECK_EQ(info.root_index_.size(), 0U);
    std::vector<size_t>& row_indices = row_set_collection_.row_indices_;
    row_indices.resize(info.num_row_);
    auto* p_row_indices = row_indices.data();
    // mark subsample and build list of member rows

    if (param_.subsample < 1.0f) {
      std::bernoulli_distribution coin_flip(param_.subsample);
      auto& rnd = common::GlobalRandom();
      size_t j = 0;
      for (size_t i = 0; i < info.num_row_; ++i) {
        if (gpair[i].GetHess() >= 0.0f && coin_flip(rnd)) {
          p_row_indices[j++] = i;
        }
      }
      row_indices.resize(j);
    } else {
      MemStackAllocator<bool, 128> buff(this->nthread_);
      bool* p_buff = buff.Get();
      std::fill(p_buff, p_buff + this->nthread_, false);

      const size_t block_size = info.num_row_ / this->nthread_ + !!(info.num_row_ % this->nthread_);

      #pragma omp parallel num_threads(this->nthread_)
      {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(info.num_row_));

        for (size_t i = ibegin; i < iend; ++i) {
          if (gpair[i].GetHess() < 0.0f) {
            p_buff[tid] = true;
            break;
          }
        }
      }

      bool has_neg_hess = false;
      for (int32_t tid = 0; tid < this->nthread_; ++tid) {
        if (p_buff[tid]) {
          has_neg_hess = true;
        }
      }

      if (has_neg_hess) {
        size_t j = 0;
        for (size_t i = 0; i < info.num_row_; ++i) {
          if (gpair[i].GetHess() >= 0.0f) {
            p_row_indices[j++] = i;
          }
        }
        row_indices.resize(j);
      } else {
        #pragma omp parallel num_threads(this->nthread_)
        {
          const size_t tid = omp_get_thread_num();
          const size_t ibegin = tid * block_size;
          const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
              static_cast<size_t>(info.num_row_));
          for (size_t i = ibegin; i < iend; ++i) {
           p_row_indices[i] = i;
          }
        }
      }
    }
  }
  row_set_collection_.Init();
  buffer_for_partition_.reserve(2 * info.num_row_);

  {
    /* determine layout of data */
    const size_t nrow = info.num_row_;
    const size_t ncol = info.num_col_;
    const size_t nnz = info.num_nonzero_;
    // number of discrete bins for feature 0
    const uint32_t nbins_f0 = gmat.cut.row_ptr[1] - gmat.cut.row_ptr[0];
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
  {
    // store a pointer to the tree
    p_last_tree_ = &tree;
    // store a pointer to training data
    p_last_fmat_ = &fmat;
  }
  if (data_layout_ == kDenseDataOneBased) {
    column_sampler_.Init(info.num_col_, param_.colsample_bynode, param_.colsample_bylevel,
            param_.colsample_bytree, true);
  } else {
    column_sampler_.Init(info.num_col_, param_.colsample_bynode, param_.colsample_bylevel,
            param_.colsample_bytree,  false);
  }
  if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
    /* specialized code for dense data:
       choose the column that has a least positive number of discrete bins.
       For dense data (with no missing value),
       the sum of gradient histogram is equal to snode[nid] */
    const std::vector<uint32_t>& row_ptr = gmat.cut.row_ptr;
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
  {
    snode_.reserve(256);
    snode_.clear();
  }
  {
    if (param_.grow_policy == TrainParam::kLossGuide) {
      qexpand_loss_guided_.reset(new ExpandQueue(LossGuide));
    } else {
      qexpand_depth_wise_.clear();
    }
  }
}

void QuantileHistMaker::Builder::EvaluateSplitsBatch(
        const std::vector<ExpandEntry>& nodes,
        const GHistIndexMatrix& gmat,
        const DMatrix& fmat,
        const std::vector<std::vector<uint8_t>>& hist_is_init,
        const std::vector<std::vector<float*>>& hist_buffers,
        RegTree* p_tree) {
  perf_monitor.TickStart();
  const MetaInfo& info = fmat.Info();
  // prepare tasks
  std::vector<std::pair<int32_t, size_t>> tasks;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto p_feature_set = column_sampler_.GetFeatureSet(nodes[i].depth);
    const auto& feature_set = p_feature_set->HostVector();
    const auto nfeature = static_cast<bst_uint>(feature_set.size());
    for (size_t j = 0; j < nfeature; ++j) {
      tasks.push_back({i, feature_set[j]});
    }
  }

  // partial results
  std::vector<std::pair<SplitEntry, SplitEntry>> splits(tasks.size());
  // parallel enumeration
  #pragma omp parallel for schedule(guided)
  for (int32_t i = 0; i < tasks.size(); ++i) {
    const int32_t  node_idx    = tasks[i].first;
    const size_t   fid         = tasks[i].second;
    const int32_t  nid         = nodes[node_idx].nid;
    const int32_t  sibling_nid = nodes[node_idx].sibling_nid;
    const int32_t  parent_nid  = nodes[node_idx].parent_nid;

    float* hist_data         = reinterpret_cast<float*>(hist_[nid].data());
    float* sibling_hist_data = sibling_nid > -1 ?
        reinterpret_cast<float*>(hist_[sibling_nid].data()) : nullptr;
    float* parent_hist_data  = sibling_nid > -1 ?
        reinterpret_cast<float*>(hist_[parent_nid].data()) : nullptr;

    // reduce needed part of a hist here to have it in cache before enumeratation
    ReduceHistograms(hist_data, sibling_hist_data, parent_hist_data, fid, node_idx,
        hist_is_init, gmat, hist_buffers);

    if (spliteval_->CheckFeatureConstraint(nid, fid)) {
      auto& snode = snode_[nid];
      bool compute_backward = this->EnumerateSplit(+1, gmat, hist_[nid], snode,
          info, &splits[i].first, fid, nid);

      if (compute_backward) {
        this->EnumerateSplit(-1, gmat, hist_[nid], snode, info,
            &splits[i].first, fid, nid);
      }
    }

    if (sibling_nid > -1 && spliteval_->CheckFeatureConstraint(sibling_nid, fid)) {
      auto& snode = snode_[sibling_nid];
      bool compute_backward = this->EnumerateSplit(+1, gmat, hist_[sibling_nid], snode,
          info, &splits[i].second, fid, sibling_nid);
      if (compute_backward) {
        this->EnumerateSplit(-1, gmat, hist_[sibling_nid], snode, info,
            &splits[i].second, fid, sibling_nid);
      }
    }
  }

  // choice of the best splits
  for (size_t i = 0; i < splits.size(); ++i) {
    const int32_t  node_idx = tasks[i].first;
    const int32_t  nid = nodes[node_idx].nid;
    const int32_t  sibling_nid = nodes[node_idx].sibling_nid;
    auto& snode = snode_[nid];
    snode.best.Update(splits[i].first);
    if (sibling_nid > -1) {
      auto& snode = snode_[sibling_nid];
      snode.best.Update(splits[i].second);
    }
  }

  perf_monitor.UpdatePerfTimer(TreeGrowingPerfMonitor::timer_name::EVALUATE_SPLIT);
}

void QuantileHistMaker::Builder::InitNewNode(int nid,
                                             const GHistIndexMatrix& gmat,
                                             const std::vector<GradientPair>& gpair,
                                             const DMatrix& fmat,
                                             RegTree* tree,
                                             QuantileHistMaker::NodeEntry* snode,
                                             int32_t parentid) {
  if (param_.enable_feature_grouping > 0 || rabit::IsDistributed()) {
    auto& stats = snode_[nid].stats;
    if (data_layout_ == kDenseDataZeroBased || data_layout_ == kDenseDataOneBased) {
      /* specialized code for dense data
         For dense data (with no missing value),
         the sum of gradient histogram is equal to snode[nid] */
      GHistRow hist = hist_[nid];
      const std::vector<uint32_t>& row_ptr = gmat.cut.row_ptr;

      const uint32_t ibegin = row_ptr[fid_least_bins_];
      const uint32_t iend = row_ptr[fid_least_bins_ + 1];
      for (uint32_t i = ibegin; i < iend; ++i) {
        const tree::GradStats et = hist[i];
        stats.Add(et.sum_grad, et.sum_hess);
      }
    } else {
      const RowSetCollection::Elem e = row_set_collection_[nid];
      for (const size_t* it = e.begin; it < e.end; ++it) {
        stats.Add(gpair[*it]);
      }
    }
  }

  // calculating the weights
  {
    snode->weight = static_cast<float>(
      spliteval_->ComputeWeight(parentid, snode->stats));
    snode->root_gain = static_cast<float>(
      spliteval_->ComputeScore(parentid, snode->stats,
      snode->weight));
  }
}

// enumerate the split values of specific feature
bool QuantileHistMaker::Builder::EnumerateSplit(int d_step,
                                                const GHistIndexMatrix& gmat,
                                                const GHistRow& hist,
                                                const NodeEntry& snode,
                                                const MetaInfo& info,
                                                SplitEntry* p_best,
                                                bst_uint fid,
                                                bst_uint nodeID) {
  CHECK(d_step == +1 || d_step == -1);

  // aliases
  const std::vector<uint32_t>& cut_ptr = gmat.cut.row_ptr;
  const std::vector<bst_float>& cut_val = gmat.cut.cut;

  // statistics on both sides of split
  GradStats c;
  GradStats e;
  // best split so far
  SplitEntry best;

  // bin boundaries
  CHECK_LE(cut_ptr[fid],
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(cut_ptr[fid + 1],
           static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
  // imin: index (offset) of the minimum value for feature fid
  //       need this for backward enumeration
  const auto imin = static_cast<int32_t>(cut_ptr[fid]);
  // ibegin, iend: smallest/largest cut points for feature fid
  // use int to allow for value -1
  int32_t ibegin, iend;
  if (d_step > 0) {
    ibegin = static_cast<int32_t>(cut_ptr[fid]);
    iend = static_cast<int32_t>(cut_ptr[fid + 1]);
  } else {
    ibegin = static_cast<int32_t>(cut_ptr[fid + 1]) - 1;
    iend = static_cast<int32_t>(cut_ptr[fid]) - 1;
  }

  if (d_step == 1) {
    for (int32_t i = ibegin; i < iend; i++) {
      e.Add(hist[i].GetGrad(), hist[i].GetHess());
      if (e.sum_hess >= param_.min_child_weight) {
        c.SetSubstract(snode.stats, e);
        if (c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          bst_float split_pt;
          {
            loss_chg = static_cast<bst_float>(spliteval_->ComputeSplitScore(nodeID,
              fid, e, c) - snode.root_gain);
          }

          split_pt = cut_val[i];
          best.Update(loss_chg, fid, split_pt, d_step == -1, e, c);
        }
      }
    }
    p_best->Update(best);

    if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
      return false;
    }
  } else {
    for (int32_t i = ibegin; i != iend; i--) {
      e.Add(hist[i].GetGrad(), hist[i].GetHess());
      if (e.sum_hess >= param_.min_child_weight) {
        c.SetSubstract(snode.stats, e);
        if (c.sum_hess >= param_.min_child_weight) {
          bst_float loss_chg;
          bst_float split_pt;

          // backward enumeration: split at left bound of each bin
          {
            loss_chg = static_cast<bst_float>(
                spliteval_->ComputeSplitScore(nodeID, fid, c, e) -
                snode.root_gain);
          }

          if (i == imin) {
            // for leftmost bin, left bound is the smallest feature value
            split_pt = gmat.cut.min_val[fid];
          } else {
            split_pt = cut_val[i - 1];
          }
          best.Update(loss_chg, fid, split_pt, d_step == -1, c, e);
        }
      }
    }
    p_best->Update(best);

    if (e.GetGrad() == snode.stats.GetGrad() && e.GetHess() == snode.stats.GetHess()) {
      return false;
    }
  }

  return true;
}

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
