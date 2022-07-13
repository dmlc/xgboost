
/*!
 * Copyright 2022 by Contributors
 * \file opt_partition_builder.h
 * \brief Quick Utility to compute subset of rows
 */
#ifndef XGBOOST_COMMON_OPT_PARTITION_BUILDER_H_
#define XGBOOST_COMMON_OPT_PARTITION_BUILDER_H_

#include <xgboost/data.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <utility>
#include <memory>
#include "xgboost/tree_model.h"
#include "../common/column_matrix.h"

namespace xgboost {
namespace common {

struct Slice {
  uint32_t* addr {nullptr};
  uint32_t b {0};
  uint32_t e {0};
  uint32_t Size() const {
    return e - b;
  }
};
// The builder is required for samples partition to left and rights children for set of nodes
// template by number of rows
class OptPartitionBuilder {
 public:
  std::vector<uint16_t> empty;
  std::vector<std::vector<Slice>> threads_addr;
  std::unordered_map<uint32_t, std::vector<uint16_t>> threads_id_for_nodes;
  std::vector<std::vector<uint16_t>> node_id_for_threads;
  std::vector<std::vector<uint32_t>> threads_rows_nodes_wise;
  std::vector<std::unordered_map<uint32_t, uint32_t>> threads_nodes_count;
  std::vector<std::vector<uint32_t>> threads_nodes_count_vec;
  std::vector<std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>>> nodes_count;
  std::vector<Slice> partitions;
  std::vector<std::vector<uint32_t>> vec_rows;
  std::vector<std::vector<uint32_t>> vec_rows_remain;
  std::vector<std::unordered_map<uint32_t, size_t> > states;
  const RegTree* p_tree;
  // can be common for all threads!
  std::vector<std::unordered_map<uint32_t, bool>> default_flags;
  const uint8_t* data_hash;
  std::vector<uint8_t>* missing_ptr;
  size_t* row_ind_ptr;
  std::vector<uint32_t> row_set_collection_vec;
  uint32_t gmat_n_rows;
  uint32_t base_rowid;
  uint32_t* row_indices_ptr;
  size_t n_threads = 0;
  uint32_t summ_size = 0;
  uint32_t summ_size_remain = 0;
  uint32_t max_depth = 0;

  const std::vector<Slice> &GetSlices(const uint32_t tid) const {
    return threads_addr[tid];
  }

  const std::vector<uint16_t> &GetNodes(const uint32_t tid) const {
    return node_id_for_threads[tid];
  }

  const std::vector<uint16_t> &GetThreadIdsForNode(const uint32_t nid) const {
    if (threads_id_for_nodes.find(nid) == threads_id_for_nodes.end()) {
      return empty;
    } else {
      const std::vector<uint16_t> & res = threads_id_for_nodes.at(nid);
      return res;
    }
  }

  void Init(GHistIndexMatrix const& gmat, const ColumnMatrix& column_matrix,
            const RegTree* p_tree_local, size_t nthreads, size_t max_depth,
            bool is_lossguide) {
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        Init<BinTypeMap<kUint8BinsTypeSize>::Type>(gmat, column_matrix,
            p_tree_local, nthreads, max_depth, is_lossguide);
        break;
      case common::kUint16BinsTypeSize:
        Init<BinTypeMap<kUint16BinsTypeSize>::Type>(gmat, column_matrix,
            p_tree_local, nthreads, max_depth, is_lossguide);
        break;
      default:
        Init<BinTypeMap<kUint32BinsTypeSize>::Type>(gmat, column_matrix,
            p_tree_local, nthreads, max_depth, is_lossguide);
        break;
    }
  }

  template <typename BinIdxType>
  void Init(GHistIndexMatrix const& gmat, const ColumnMatrix& column_matrix,
            const RegTree* p_tree_local, size_t nthreads, size_t max_depth,
            bool is_lossguide) {
    gmat_n_rows = gmat.row_ptr.size() - 1;
    base_rowid = gmat.base_rowid;
    p_tree = p_tree_local;
    if ((states.size() == 0 && column_matrix.AnyMissing()) ||
        (data_hash != reinterpret_cast<const uint8_t*>(column_matrix.GetIndexData())
        && column_matrix.AnyMissing()) ||
        (missing_ptr != column_matrix.GetMissing() && column_matrix.AnyMissing()) ||
        (row_ind_ptr != column_matrix.GetRowId())) {
      missing_ptr = const_cast<std::vector<uint8_t>*>(column_matrix.GetMissing());
      row_ind_ptr = const_cast<size_t*>(column_matrix.GetRowId());
      states.clear();
      default_flags.clear();
      states.resize(nthreads);
      default_flags.resize(nthreads);
    }
    data_hash = reinterpret_cast<const uint8_t*>(column_matrix.GetIndexData());
    n_threads = nthreads;
    this->max_depth = max_depth;
    vec_rows.resize(nthreads);
    if (is_lossguide) {
      partitions.resize(1 << (max_depth + 2));
      vec_rows_remain.resize(nthreads);
    }
    threads_nodes_count.clear();
    threads_nodes_count.resize(n_threads);
    threads_nodes_count_vec.clear();
    threads_nodes_count_vec.resize(n_threads);
    if (vec_rows[0].size() == 0) {
      size_t chunck_size = common::GetBlockSize(gmat_n_rows, nthreads);
    #pragma omp parallel num_threads(n_threads)
      {
        size_t tid = omp_get_thread_num();
        if (vec_rows[tid].size() == 0) {
          vec_rows[tid].resize(chunck_size + 2, 0);
          if (is_lossguide) {
            vec_rows_remain[tid].resize(chunck_size + 2, 0);
          }
        }
      }
    }
    threads_addr.clear();
    threads_id_for_nodes.clear();
    node_id_for_threads.clear();
    nodes_count.clear();
    UpdateRootThreadWork();
  }

  template<bool is_loss_guided, bool all_dense, bool any_cat, typename Predicate>
  void CommonPartition(size_t tid, const size_t row_indices_begin,
                       const size_t row_indices_end, uint16_t* nodes_ids,
                       std::unordered_map<uint32_t, int32_t>* split_conditions,
                       std::unordered_map<uint32_t, uint64_t>* split_ind,
                       std::unordered_map<uint32_t, bool>* smalest_nodes_mask,
                       const ColumnMatrix& column_matrix,
                       const std::vector<uint32_t>& split_nodes, Predicate&& pred, size_t depth) {
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        CommonPartition<BinTypeMap<kUint8BinsTypeSize>::Type, is_loss_guided, all_dense, any_cat>(
                           tid, row_indices_begin, row_indices_end,
                           column_matrix.template GetIndexData<uint8_t>(),
                           nodes_ids, split_conditions, split_ind, smalest_nodes_mask,
                           column_matrix, split_nodes, std::forward<Predicate>(pred), depth);
        break;
      case common::kUint16BinsTypeSize:
        CommonPartition<BinTypeMap<kUint16BinsTypeSize>::Type, is_loss_guided, all_dense, any_cat>(
                           tid, row_indices_begin, row_indices_end,
                           column_matrix.template GetIndexData<uint16_t>(),
                           nodes_ids, split_conditions, split_ind, smalest_nodes_mask,
                           column_matrix, split_nodes, std::forward<Predicate>(pred), depth);
        break;
      default:
        CommonPartition<BinTypeMap<kUint32BinsTypeSize>::Type, is_loss_guided, all_dense, any_cat>(
                           tid, row_indices_begin, row_indices_end,
                           column_matrix.template GetIndexData<uint32_t>(),
                           nodes_ids, split_conditions, split_ind, smalest_nodes_mask,
                           column_matrix, split_nodes, std::forward<Predicate>(pred), depth);
        break;
    }
  }

  template<typename BufferType>
  BufferType GetBufferItem(const std::vector<BufferType>& buffer,
                           const uint32_t item_idx) const {
    return (buffer.data())[item_idx];
  }

  bool GetBufferItem(const std::vector<bool>& buffer,
                     const uint32_t item_idx) const {
    return buffer[item_idx];
  }

  template<typename BufferType>
  BufferType GetBufferItem(const std::unordered_map<uint32_t, BufferType>& map_buffer,
                           const uint32_t item_idx) {
    return map_buffer.find(item_idx) != map_buffer.end() ?
                                        map_buffer.at(item_idx) : 0;
  }

  template<typename BinIdxType, bool is_loss_guided,
           bool all_dense, bool any_cat,
           typename SplitConditionsBufferType,
           typename SplitIndBufferType,
           typename SmalestNodesMaskType,
           typename Predicate>
  void CommonPartition(size_t tid, const size_t row_indices_begin,
                       const size_t row_indices_end, const BinIdxType* numa, uint16_t* nodes_ids,
                       SplitConditionsBufferType* split_conditions,
                       SplitIndBufferType* split_ind,
                       SmalestNodesMaskType* smalest_nodes_mask,
                       const ColumnMatrix& column_matrix,
                       const std::vector<uint32_t>& split_nodes, Predicate&& pred, size_t depth) {
    CHECK_EQ(data_hash, reinterpret_cast<const uint8_t*>(column_matrix.GetIndexData()));
    const auto& column_list = column_matrix.GetColumnViewList();
    uint32_t rows_count = 0;
    uint32_t rows_left_count = 0;
    uint32_t* rows = vec_rows[tid].data();
    uint32_t* rows_left = nullptr;
    if (is_loss_guided) {
      rows_left = vec_rows_remain[tid].data();
    }
    auto& split_ind_data = *split_ind;
    auto& split_conditions_data = *split_conditions;
    auto& smalest_nodes_mask_data = *smalest_nodes_mask;
    const BinIdxType* columnar_data = numa;

    if (!all_dense && row_indices_begin < row_indices_end) {
      const uint32_t first_row_id = !is_loss_guided ? row_indices_begin :
                                                      row_indices_ptr[row_indices_begin];
      for (const auto& nid : split_nodes) {
        states[tid][nid] = column_list[split_ind_data[nid]]->GetInitialState(first_row_id);
        default_flags[tid][nid] = (*p_tree)[nid].DefaultLeft();
      }
    }
    for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
      const uint32_t i = !is_loss_guided ? ii : row_indices_ptr[ii];
      const uint32_t nid = nodes_ids[i];
      if ((*p_tree)[nid].IsLeaf()) {
        continue;
      }
      const int32_t sc = GetBufferItem(split_conditions_data, nid);

      if (any_cat) {
        uint64_t si = GetBufferItem(split_ind_data, nid);
        const int32_t cmp_value = static_cast<int32_t>(columnar_data[si + i]);
        nodes_ids[i] = pred(i, cmp_value, nid, sc) ? (*p_tree)[nid].LeftChild() :
                       (*p_tree)[nid].RightChild();
      } else if (all_dense) {
        uint64_t si = GetBufferItem(split_ind_data, nid);
        const int32_t cmp_value = static_cast<int32_t>(columnar_data[si + i]);
        nodes_ids[i] = cmp_value <= sc ? (*p_tree)[nid].LeftChild() : (*p_tree)[nid].RightChild();
      } else {
        uint64_t si = GetBufferItem(split_ind_data, nid);
        int32_t cmp_value = column_list[si]->template GetBinIdx<BinIdxType, int32_t>(i,
                                                                        &(states[tid][nid]));

        if (cmp_value == Column::kMissingId) {
          nodes_ids[i] = default_flags[tid][nid]
                         ? (*p_tree)[nid].LeftChild()
                         : (*p_tree)[nid].RightChild();
        } else {
          nodes_ids[i] = cmp_value <= sc ? (*p_tree)[nid].LeftChild() :
                         (*p_tree)[nid].RightChild();
        }
      }
      const uint16_t check_node_id = nodes_ids[i];
      uint32_t inc = GetBufferItem(smalest_nodes_mask_data, check_node_id);
      rows[1 + rows_count] = i;
      rows_count += inc;
      if (is_loss_guided) {
        rows_left[1 + rows_left_count] = i;
        rows_left_count += !static_cast<bool>(inc);
      } else {
        threads_nodes_count[tid][check_node_id] += inc;
      }
    }

    rows[0] = rows_count;
    if (is_loss_guided) {
      rows_left[0] = rows_left_count;
    }
  }

  size_t DepthSize(GHistIndexMatrix const& gmat,
                   const std::vector<uint16_t>& compleate_trees_depth_wise,
                   bool is_lossguided) {
    if (is_lossguided) {
      CHECK_GT(compleate_trees_depth_wise.size(), 0);
      size_t max_nid = std::max(compleate_trees_depth_wise[0],
                                compleate_trees_depth_wise[1]);
      partitions.resize(max_nid + 1);
      CHECK_LT((*p_tree)[compleate_trees_depth_wise[0]].Parent(), partitions.size());
      return partitions[(*p_tree)[compleate_trees_depth_wise[0]].Parent()].Size();
    } else {
      return gmat.row_ptr.size() - 1;
    }
  }
  size_t DepthBegin(const std::vector<uint16_t>& compleate_trees_depth_wise,
                    bool is_lossguided) {
    if (is_lossguided) {
      CHECK_GT(compleate_trees_depth_wise.size(), 0);
      size_t max_nid = std::max(compleate_trees_depth_wise[0],
                                compleate_trees_depth_wise[1]);
      partitions.resize(max_nid + 1);
      CHECK_LT((*p_tree)[compleate_trees_depth_wise[0]].Parent(), partitions.size());
      return partitions[(*p_tree)[compleate_trees_depth_wise[0]].Parent()].b;
    } else {
      return 0;
    }
  }

  void ResizeRowsBuffer(size_t nrows) {
    row_set_collection_vec.resize(nrows);
    row_indices_ptr = row_set_collection_vec.data();
  }

  uint32_t* GetRowsBuffer() const {
    return row_indices_ptr;
  }

  size_t GetPartitionSize(size_t nid) const {
    return partitions[nid].Size();
  }
  void SetSlice(size_t nid, uint32_t begin, uint32_t size) {
    if (partitions.size()) {
      CHECK_LT(nid, partitions.size());

      partitions[nid].b = begin;
      partitions[nid].e = begin + size;
    }
  }
  void UpdateRowBuffer(const std::vector<uint16_t>& compleate_trees_depth_wise,
                       GHistIndexMatrix const& gmat, size_t n_features, size_t depth,
                       const std::vector<uint16_t>& node_ids_, bool is_loss_guided) {
    summ_size = 0;
    summ_size_remain = 0;
    for (uint32_t i = 0; i < n_threads; ++i) {
      summ_size += vec_rows[i][0];
    }

    const bool hist_fit_to_l2 = 1024*1024*0.8 > 16*gmat.cut.Ptrs().back();
    const size_t n_bins = gmat.cut.Ptrs().back();
    threads_id_for_nodes.clear();
    nodes_count.clear();
    node_id_for_threads.clear();
    node_id_for_threads.resize(n_threads);
    if (is_loss_guided) {
      const int cleft = compleate_trees_depth_wise[0];
      const int cright = compleate_trees_depth_wise[1];
      const int parent_id = (*p_tree)[cleft].Parent();
      CHECK_LT(parent_id, partitions.size());
      const size_t parent_begin = partitions[parent_id].b;
      const size_t parent_size = partitions[parent_id].Size();
      for (uint32_t i = 0; i < n_threads; ++i) {
        summ_size_remain += vec_rows_remain[i][0];
      }
      CHECK_EQ(summ_size + summ_size_remain, parent_size);
      SetSlice(cleft, partitions[parent_id].b, summ_size);
      SetSlice(cright, partitions[parent_id].b + summ_size, summ_size_remain);

      #pragma omp parallel num_threads(n_threads)
      {
        uint32_t tid = omp_get_thread_num();
        uint32_t thread_displace = parent_begin;
        for (size_t id = 0; id < tid; ++id) {
          thread_displace += vec_rows[id][0];
        }
        CHECK_LE(thread_displace + vec_rows[tid][0], parent_begin + summ_size);
        std::copy(vec_rows[tid].data() + 1,
                  vec_rows[tid].data() + 1 + vec_rows[tid][0],
                  row_indices_ptr + thread_displace);
        uint32_t thread_displace_left = parent_begin + summ_size;
        for (size_t id = 0; id < tid; ++id) {
          thread_displace_left += vec_rows_remain[id][0];
        }
        CHECK_LE(thread_displace_left + vec_rows_remain[tid][0], parent_begin + parent_size);
        std::copy(vec_rows_remain[tid].data() + 1,
                  vec_rows_remain[tid].data() + 1 + vec_rows_remain[tid][0],
                  row_indices_ptr + thread_displace_left);
      }
    } else if (n_features*summ_size / n_threads <
               (static_cast<size_t>(1) << (depth + 1))*n_bins ||
               (depth >= 1 && !hist_fit_to_l2)) {
      threads_rows_nodes_wise.resize(n_threads);
      nodes_count.resize(n_threads);
      #pragma omp parallel num_threads(n_threads)
      {
        size_t tid = omp_get_thread_num();
        if (threads_rows_nodes_wise[tid].size() == 0) {
          threads_rows_nodes_wise[tid].resize(vec_rows[tid].size(), 0);
        }
        std::unordered_map<uint32_t, uint32_t> nc;

        std::vector<uint32_t> unique_node_ids(threads_nodes_count[tid].size(), 0);
        size_t i = 0;
        for (const auto& tnc : threads_nodes_count[tid]) {
          CHECK_LT(i, unique_node_ids.size());
          unique_node_ids[i++] = tnc.first;
        }
        std::sort(unique_node_ids.begin(), unique_node_ids.end());
        size_t cummulative_summ = 0;
        std::unordered_map<uint32_t, uint32_t> counts;
        for (const auto& uni : unique_node_ids) {
          nodes_count[tid][uni].first = cummulative_summ;
          counts[uni] = cummulative_summ;
          nodes_count[tid][uni].second = nodes_count[tid][uni].first +
                                         threads_nodes_count[tid][uni];
          cummulative_summ += threads_nodes_count[tid][uni];
        }
        for (size_t i = 0; i < vec_rows[tid][0]; ++i) {
          const uint32_t row_id = vec_rows[tid][i + 1];
          const uint16_t check_node_id = node_ids_[row_id];
          const uint32_t nod_id = check_node_id;
          threads_rows_nodes_wise[tid][counts[nod_id]++] = row_id;
        }
      }
    }
  }
  void UpdateThreadsWork(const std::vector<uint16_t>& compleate_trees_depth_wise,
                         GHistIndexMatrix const& gmat,
                         size_t n_features, size_t depth, bool is_loss_guided,
                         bool is_left_small = true,
                         bool check_is_left_small = false) {
    const size_t n_bins = gmat.cut.Ptrs().back();
    threads_addr.clear();
    threads_addr.resize(n_threads);
    const bool hist_fit_to_l2 = 1024*1024*0.8 > 16*gmat.cut.Ptrs().back();
    if (is_loss_guided) {
      const int cleft = compleate_trees_depth_wise[0];
      const int cright = compleate_trees_depth_wise[1];
      uint32_t min_node_size = std::min(summ_size, summ_size_remain);
      uint32_t min_node_id = summ_size <= summ_size_remain ? cleft : cright;
      if (check_is_left_small) {
        min_node_id = is_left_small ? cleft : cright;
        min_node_size = is_left_small ? summ_size : summ_size_remain;
      }
      uint32_t thread_size = std::max(common::GetBlockSize(min_node_size, n_threads),
                             std::min(min_node_size, static_cast<uint32_t>(512)));
      for (size_t tid = 0; tid <  n_threads; ++tid) {
        uint32_t th_begin = thread_size * tid;
        uint32_t th_end = std::min(th_begin + thread_size, min_node_size);
        if (th_end > th_begin) {
          CHECK_LT(min_node_id, partitions.size());
          threads_addr[tid].push_back({row_indices_ptr, partitions[min_node_id].b + th_begin,
                                       partitions[min_node_id].b + th_end});
          threads_id_for_nodes[min_node_id].push_back(tid);
          node_id_for_threads[tid].push_back(min_node_id);
        }
      }
    } else if (n_features*summ_size / n_threads <
               (static_cast<size_t>(1) << (depth + 1))*n_bins
               || (depth >= 1 && !hist_fit_to_l2)) {
      uint32_t block_size = std::max(common::GetBlockSize(summ_size, n_threads),
                                     std::min(summ_size, static_cast<uint32_t>(512)));
      uint32_t node_id = 0;
      uint32_t curr_thread_size = block_size;
      uint32_t curr_node_disp = 0;
      uint32_t curr_thread_id = 0;
      for (uint32_t i = 0; i < n_threads; ++i) {
        while (curr_thread_size != 0) {
          const uint32_t curr_thread_node_size = threads_nodes_count[curr_thread_id%
                                                                     n_threads][node_id];
          if (curr_thread_node_size == 0) {
            ++curr_thread_id;
            node_id = curr_thread_id / n_threads;
          } else if (curr_thread_node_size > 0 && curr_thread_node_size <= curr_thread_size) {
            const uint32_t begin = nodes_count[curr_thread_id%n_threads][node_id].first;
            CHECK_EQ(nodes_count[curr_thread_id%n_threads][node_id].first + curr_thread_node_size,
                    nodes_count[curr_thread_id%n_threads][node_id].second);
            threads_addr[i].push_back({
              threads_rows_nodes_wise[curr_thread_id%n_threads].data(), begin,
              begin + curr_thread_node_size
            });
            CHECK_LT(i, node_id_for_threads.size());
            if (threads_id_for_nodes[node_id].size() != 0) {
              if (threads_id_for_nodes[node_id].back() != i) {
                threads_id_for_nodes[node_id].push_back(i);
                node_id_for_threads[i].push_back(node_id);
              }
            } else {
              threads_id_for_nodes[node_id].push_back(i);
              node_id_for_threads[i].push_back(node_id);
            }
            threads_nodes_count[curr_thread_id%n_threads][node_id] = 0;
            curr_thread_size -= curr_thread_node_size;
            ++curr_thread_id;
            node_id = curr_thread_id / n_threads;
          } else {
            const uint32_t begin = nodes_count[curr_thread_id%n_threads][node_id].first;
            CHECK_EQ(nodes_count[curr_thread_id%n_threads][node_id].first + curr_thread_node_size,
                    nodes_count[curr_thread_id%n_threads][node_id].second);
            CHECK_LT(i, threads_addr.size());
            threads_addr[i].push_back({
              threads_rows_nodes_wise[curr_thread_id%n_threads].data(), begin,
              begin + curr_thread_size
            });
            CHECK_LT(i, node_id_for_threads.size());
            if (threads_id_for_nodes[node_id].size() != 0) {
              if (threads_id_for_nodes[node_id].back() != i) {
                threads_id_for_nodes[node_id].push_back(i);
                node_id_for_threads[i].push_back(node_id);
              }
            } else {
              threads_id_for_nodes[node_id].push_back(i);
              node_id_for_threads[i].push_back(node_id);
            }
            threads_nodes_count[curr_thread_id%n_threads][node_id] -= curr_thread_size;
            nodes_count[curr_thread_id%n_threads][node_id].first += curr_thread_size;
            curr_thread_size = 0;
          }
        }
        curr_thread_size = std::min(block_size,
                                    summ_size > block_size*(i+1) ?
                                    summ_size - block_size*(i+1) : 0);
      }
    } else {
      uint32_t block_size = common::GetBlockSize(summ_size, n_threads);
      uint32_t curr_vec_rowsid = 0;
      uint32_t curr_vec_rowssize = vec_rows[curr_vec_rowsid][0];
      uint32_t curr_thread_size = block_size;
      for (uint32_t i = 0; i < n_threads; ++i) {
        std::vector<uint32_t> borrowed_work;
        while (curr_thread_size != 0) {
          if (curr_vec_rowssize > curr_thread_size) {
            threads_addr[i].push_back({
              vec_rows[curr_vec_rowsid].data(),
              1 + vec_rows[curr_vec_rowsid][0] - curr_vec_rowssize,
              1 + vec_rows[curr_vec_rowsid][0] - curr_vec_rowssize + curr_thread_size});
            borrowed_work.push_back(curr_vec_rowsid);
            curr_vec_rowssize -= curr_thread_size;
            curr_thread_size = 0;
          } else if (curr_vec_rowssize == curr_thread_size) {
            threads_addr[i].push_back({
              vec_rows[curr_vec_rowsid].data(),
              1 + vec_rows[curr_vec_rowsid][0] - curr_vec_rowssize,
              1 + vec_rows[curr_vec_rowsid][0] - curr_vec_rowssize + curr_thread_size});
            borrowed_work.push_back(curr_vec_rowsid);
            curr_vec_rowsid += (curr_vec_rowsid < (n_threads - 1));
            curr_vec_rowssize = vec_rows[curr_vec_rowsid][0];
            curr_thread_size = 0;
          } else {
            threads_addr[i].push_back({vec_rows[curr_vec_rowsid].data(),
                                      1 + vec_rows[curr_vec_rowsid][0] - curr_vec_rowssize,
                                      1 + vec_rows[curr_vec_rowsid][0]});
            borrowed_work.push_back(curr_vec_rowsid);
            curr_thread_size -= curr_vec_rowssize;
            curr_vec_rowsid += (curr_vec_rowsid < (n_threads - 1));
            curr_vec_rowssize = vec_rows[curr_vec_rowsid][0];
          }
        }
        curr_thread_size = std::min(block_size,
                                    summ_size > block_size*(i+1) ?
                                    summ_size - block_size*(i+1) : 0);
        for (const auto& borrowed_tid : borrowed_work) {
          for (const auto& node_id : compleate_trees_depth_wise) {
            if (threads_nodes_count[borrowed_tid][node_id] != 0) {
              if (threads_id_for_nodes[node_id].size() != 0) {
                if (threads_id_for_nodes[node_id].back() != i) {
                  threads_id_for_nodes[node_id].push_back(i);
                  node_id_for_threads[i].push_back(node_id);
                }
              } else {
                threads_id_for_nodes[node_id].push_back(i);
                node_id_for_threads[i].push_back(node_id);
            }
            }
          }
        }
      }
    }
    threads_nodes_count.clear();
    threads_nodes_count.resize(n_threads);
    threads_nodes_count_vec.clear();
    threads_nodes_count_vec.resize(n_threads);
    nodes_count.clear();
  }
  void UpdateRootThreadWork() {
    threads_addr.clear();
    threads_addr.resize(n_threads);
    threads_id_for_nodes.clear();
    node_id_for_threads.clear();
    node_id_for_threads.resize(n_threads);
    const uint32_t n_rows = gmat_n_rows;
    const uint32_t block_size = common::GetBlockSize(n_rows, n_threads);
    for (uint32_t tid = 0; tid < n_threads; ++tid) {
      const uint32_t begin = tid * block_size;
      const uint32_t end = std::min(begin + block_size, n_rows);
      if (end > begin) {
        threads_addr[tid].push_back({nullptr, begin, end});
        threads_id_for_nodes[0].push_back(tid);
        node_id_for_threads[tid].push_back(0);
      }
    }
  }
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_OPT_PARTITION_BUILDER_H_
