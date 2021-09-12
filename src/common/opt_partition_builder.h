
/*!
 * Copyright 2021 by Contributors
 * \file opt_partition_builder.h
 * \brief Quick Utility to compute subset of rows
 */
#ifndef XGBOOST_COMMON_OPT_PARTITION_BUILDER_H_
#define XGBOOST_COMMON_OPT_PARTITION_BUILDER_H_

#include <xgboost/data.h>
#include <algorithm>
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
  uint32_t Size() {
    return e - b;
  }
};
// The builder is required for samples partition to left and rights children for set of nodes
// template by number of rows
class OptPartitionBuilder {
 public:
  std::vector<std::vector<Slice>> threads_addr;
  std::vector<std::vector<uint16_t>> threads_id_for_nodes;
  std::vector<std::vector<uint16_t>> node_id_for_threads;
  std::vector<std::vector<uint32_t>> threads_rows_nodes_wise;
  std::vector<std::vector<uint32_t>> threads_nodes_count;
  std::vector<std::vector<int>> nodes_count;
  std::vector<Slice> partitions;
  std::vector<std::vector<uint32_t>> vec_rows;
  std::vector<std::vector<uint32_t>> vec_rows_remain;
  std::vector<std::unique_ptr<const Column<uint8_t> >> columns8;
  std::vector<const DenseColumn<uint8_t, true>*> dcolumns8;
  std::vector<const SparseColumn<uint8_t>*> scolumns8;
  std::vector<std::unique_ptr<const Column<uint16_t> >> columns16;
  std::vector<const DenseColumn<uint16_t, true>*> dcolumns16;
  std::vector<const SparseColumn<uint16_t>*> scolumns16;
  std::vector<std::unique_ptr<const Column<uint32_t> >> columns32;
  std::vector<const DenseColumn<uint32_t, true>*> dcolumns32;
  std::vector<const SparseColumn<uint32_t>*> scolumns32;
  std::vector<std::vector<size_t> > states;
  const RegTree* p_tree;
  // can be common for all threads!
  std::vector<std::vector<bool>> default_flags;
  const uint8_t* data_hash;
  std::vector<uint32_t> row_set_collection_vec;
  uint32_t gmat_n_rows;
  uint32_t* row_indices_ptr;
  size_t n_threads = 0;
  uint32_t summ_size = 0;
  uint32_t summ_size_remain = 0;
  uint32_t max_depth = 0;

  template<typename BinIdxType>
  std::vector<std::unique_ptr<const Column<BinIdxType> >>& GetColumnsRef() {
    const BinIdxType dummy = 0;
    return GetColumnsRefImpl(&dummy);
  }

  std::vector<std::unique_ptr<const Column<uint8_t> >>& GetColumnsRefImpl(const uint8_t* dummy) {
      return columns8;
  }

  std::vector<std::unique_ptr<const Column<uint16_t> >>& GetColumnsRefImpl(const uint16_t* dummy) {
      return columns16;
  }

  std::vector<std::unique_ptr<const Column<uint32_t> >>& GetColumnsRefImpl(const uint32_t* dummy) {
      return columns32;
  }

  template<typename BinIdxType>
  std::vector<const DenseColumn<BinIdxType, true>*>& GetDenseColumnsRef() {
    const BinIdxType dummy = 0;
    return GetDenseColumnsRefImpl(&dummy);
  }

  std::vector<const DenseColumn<uint8_t, true>*>& GetDenseColumnsRefImpl(const uint8_t* dummy) {
      return dcolumns8;
  }

  std::vector<const DenseColumn<uint16_t, true>*>& GetDenseColumnsRefImpl(const uint16_t* dummy) {
      return dcolumns16;
  }

  std::vector<const DenseColumn<uint32_t, true>*>& GetDenseColumnsRefImpl(const uint32_t* dummy) {
      return dcolumns32;
  }

  template<typename BinIdxType>
  std::vector<const SparseColumn<BinIdxType>*>& GetSparseColumnsRef() {
    const BinIdxType dummy = 0;
    return GetSparseColumnsRefImpl(&dummy);
  }

  std::vector<const SparseColumn<uint8_t>*>& GetSparseColumnsRefImpl(const uint8_t* dummy) {
      return scolumns8;
  }

  std::vector<const SparseColumn<uint16_t>*>& GetSparseColumnsRefImpl(const uint16_t* dummy) {
      return scolumns16;
  }

  std::vector<const SparseColumn<uint32_t>*>& GetSparseColumnsRefImpl(const uint32_t* dummy) {
      return scolumns32;
  }

  template <typename BinIdxType>
  void Init(GHistIndexMatrix const& gmat, const ColumnMatrix& column_matrix,
            const RegTree* p_tree_local, size_t nthreads, size_t max_depth,
            size_t n_rows, bool is_lossguide) {
    gmat_n_rows = gmat.row_ptr.size() - 1;
    p_tree = p_tree_local;
    if ((states.size() == 0 && !gmat.IsDense()) ||
        (data_hash != reinterpret_cast<const uint8_t*>(column_matrix.GetIndexData())
        && !gmat.IsDense())) {
      states.clear();
      default_flags.clear();
      states.resize(nthreads);
      default_flags.resize(nthreads);
      GetColumnsRef<BinIdxType>().resize(column_matrix.GetNumFeature());
      GetDenseColumnsRef<BinIdxType>().resize(column_matrix.GetNumFeature());
      GetSparseColumnsRef<BinIdxType>().resize(column_matrix.GetNumFeature());
      for (size_t fid = 0; fid < column_matrix.GetNumFeature(); ++fid) {
        GetColumnsRef<BinIdxType>()[fid] = column_matrix.GetColumn<BinIdxType, true>(fid);
        GetDenseColumnsRef<BinIdxType>()[fid] = dynamic_cast<const DenseColumn<BinIdxType, true>*>(
                                                GetColumnsRef<BinIdxType>()[fid].get());
        GetSparseColumnsRef<BinIdxType>()[fid] =
          dynamic_cast<const SparseColumn<BinIdxType>*>(GetColumnsRef<BinIdxType>()[fid].get());
      }
      for (size_t tid = 0; tid < nthreads; ++tid) {
        states[tid].resize(1 << (max_depth + 1), 0);
        default_flags[tid].resize(1 << (max_depth + 1), false);
      }
    }
    data_hash = reinterpret_cast<const uint8_t*>(column_matrix.GetIndexData());
    n_threads = nthreads;
    this->max_depth = max_depth;
    vec_rows.resize(nthreads);
    if (is_lossguide) {
      partitions.resize(1 << (max_depth + 2));
      vec_rows_remain.resize(nthreads);
    } else {
      threads_nodes_count.clear();
      threads_nodes_count.resize(n_threads);
    }
    if (vec_rows[0].size() == 0) {
      size_t chunck_size = n_rows / nthreads + !!(n_rows % nthreads);
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
  }

  template<typename BinIdxType, bool is_loss_guided, bool all_dense = true>
  void CommonPartition(size_t tid, const size_t row_indices_begin,
                       const size_t row_indices_end, const BinIdxType* numa, uint16_t* nodes_ids,
                       std::vector<int32_t>* split_conditions, std::vector<uint64_t>* split_ind,
                       const std::vector<bool>& smalest_nodes_mask,
                       const std::vector<GradientPair> &gpair_h,
                       std::vector<uint16_t>* curr_level_nodes,
                       const ColumnMatrix& column_matrix,
                       const std::vector<uint32_t>& split_nodes) {
    std::vector<std::unique_ptr<
                const Column<BinIdxType> > >& columns = GetColumnsRef<BinIdxType>();
    const auto& dense_columns = GetDenseColumnsRef<BinIdxType>();
    const auto& sparse_columns = GetSparseColumnsRef<BinIdxType>();
    uint32_t count = 0;
    uint32_t count2 = 0;
    uint32_t* rows = vec_rows[tid].data();
    uint32_t* rows_left = nullptr;
    if (is_loss_guided) {
      rows_left = vec_rows_remain[tid].data();
    }
    if (!is_loss_guided) {
      threads_nodes_count[tid].resize(1 << (max_depth + 1), 0);
    }
    uint32_t* nodes_count = threads_nodes_count[tid].data();
    uint16_t* curr_level_nodes_data = curr_level_nodes->data();
    uint64_t* split_ind_data = split_ind->data();
    int32_t* split_conditions_data = split_conditions->data();
    const BinIdxType* columnar_data = numa;

    if (!all_dense) {
      std::vector<size_t>& local_states = states[tid];
      std::vector<bool>& local_default_flags = default_flags[tid];
      const uint32_t first_row_id = !is_loss_guided ? row_indices_begin :
                                                      row_indices_ptr[row_indices_begin];
      for (const auto& nid : split_nodes) {
        if (columns[split_ind_data[nid]]->GetType() == common::kDenseColumn) {
          local_states[nid] = dense_columns[split_ind_data[nid]]->GetInitialState(first_row_id);
        } else {
          local_states[nid] = sparse_columns[split_ind_data[nid]]->GetInitialState(first_row_id);
        }
        local_default_flags[nid] = (*p_tree)[nid].DefaultLeft();
      }
    }
    const float* pgh = reinterpret_cast<const float*>(gpair_h.data());
    for (size_t ii = row_indices_begin; ii < row_indices_end; ++ii) {
      const uint32_t i = !is_loss_guided ? ii : row_indices_ptr[ii];
      const uint32_t nid = nodes_ids[i];

      if ((static_cast<uint16_t>(1) << 15 & nid)) {
        continue;
      }
      const int32_t sc = split_conditions_data[nid];

      if (all_dense) {
        const int32_t cmp_value = static_cast<int32_t>(columnar_data[split_ind_data[nid] + i]);
        nodes_ids[i] = (curr_level_nodes_data[2*nid + !(cmp_value <= sc)]);
      } else {
        std::vector<size_t>& local_states = states[tid];
        std::vector<bool>& local_default_flags = default_flags[tid];
        int32_t cmp_value = 0;
        if (columns[split_ind_data[nid]]->GetType() == common::kDenseColumn) {
          cmp_value = dense_columns[split_ind_data[nid]]->GetBinIdx(i, nullptr);
        } else {
          cmp_value = sparse_columns[split_ind_data[nid]]->GetBinIdx(i, &local_states[nid]);
        }
        if (cmp_value == Column<BinIdxType>::kMissingId) {
          const bool default_left = local_default_flags[nid];
          if (default_left) {
            nodes_ids[i] = (curr_level_nodes_data[2*nid]);
          } else {
            nodes_ids[i] = (curr_level_nodes_data[2*nid + 1]);
          }
        } else {
          nodes_ids[i] = (curr_level_nodes_data[2*nid + !(cmp_value <= sc)]);
        }
      }
      const uint16_t check_node_id = (~(static_cast<uint16_t>(1) << 15)) & nodes_ids[i];

      uint32_t inc = static_cast<uint32_t>(smalest_nodes_mask[check_node_id]);
      rows[1 + count] = i;
      count += inc;
      if (is_loss_guided) {
        rows_left[1 + count2] = i;
        count2 += !static_cast<bool>(inc);
      } else {
        nodes_count[check_node_id] += inc;
      }
    }
    rows[0] = count;
    if (is_loss_guided) {
      rows_left[0] = count2;
    }
  }

  size_t DepthSize(GHistIndexMatrix const& gmat,
                   const std::vector<uint16_t>& compleate_trees_depth_wise,
                   const RegTree* p_tree, bool is_lossguided) {
    if (is_lossguided) {
      return partitions[(*p_tree)[compleate_trees_depth_wise[0]].Parent()].Size();
    } else {
      return gmat.row_ptr.size() - 1;
    }
  }
  size_t DepthBegin(const std::vector<uint16_t>& compleate_trees_depth_wise,
                    const RegTree* p_tree, bool is_lossguided) {
    if (is_lossguided) {
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

  void SetSlice(size_t nid, uint32_t begin, uint32_t size) {
    if (partitions.size()) {
      partitions[nid].b = begin;
      partitions[nid].e = begin + size;
    }
  }
  void UpdateRowBuffer(const std::vector<uint16_t>& compleate_trees_depth_wise,
                       const RegTree* p_tree,
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
    const size_t inc = (is_loss_guided == true);
    threads_id_for_nodes.resize(1 << (max_depth + inc));
    node_id_for_threads.clear();
    node_id_for_threads.resize(n_threads);
    if (is_loss_guided) {
      const int cleft = compleate_trees_depth_wise[0];
      const int cright = compleate_trees_depth_wise[1];
      const int parent_id = (*p_tree)[cleft].Parent();
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
    } else if (n_features*summ_size / n_threads < (1 << (depth + 1))*n_bins ||
               (depth >= 1 && !hist_fit_to_l2)) {
      threads_rows_nodes_wise.resize(n_threads);
      nodes_count.resize(n_threads);
      #pragma omp parallel num_threads(n_threads)
      {
        size_t tid = omp_get_thread_num();
        if (threads_rows_nodes_wise[tid].size() == 0) {
          threads_rows_nodes_wise[tid].resize(vec_rows[tid].size(), 0);
        }
        nodes_count[tid].resize((1 << (max_depth)) + 1, 0);
        for (size_t i = 1; i < (1 << (max_depth)); ++i) {
          nodes_count[tid][i + 1] += nodes_count[tid][i] + threads_nodes_count[tid][i-1];
        }
        for (size_t i = 0; i < vec_rows[tid][0]; ++i) {
          const uint32_t row_id = vec_rows[tid][i + 1];
          const uint32_t nod_id = node_ids_[row_id];
          threads_rows_nodes_wise[tid][nodes_count[tid][nod_id + 1]++] = row_id;
        }
      }
    }
  }
  void UpdateThreadsWork(const std::vector<uint16_t>& compleate_trees_depth_wise,
                         GHistIndexMatrix const& gmat,
                         size_t n_features, size_t depth, bool is_loss_guided) {
    const size_t n_bins = gmat.cut.Ptrs().back();
    threads_addr.clear();
    threads_addr.resize(n_threads);
    const bool hist_fit_to_l2 = 1024*1024*0.8 > 16*gmat.cut.Ptrs().back();
    if (is_loss_guided) {
      const int cleft = compleate_trees_depth_wise[0];
      const int cright = compleate_trees_depth_wise[1];
      // template!
      const uint32_t min_node_size = std::min(summ_size, summ_size_remain);
      const uint32_t min_node_id = summ_size <= summ_size_remain ? cleft : cright;
      uint32_t thread_size = std::max(static_cast<uint32_t>(min_node_size/n_threads +
                                      !!(min_node_size%n_threads)),
                                      std::min(min_node_size, static_cast<uint32_t>(512)));
      for (size_t tid = 0; tid <  n_threads; ++tid) {
        uint32_t th_begin = thread_size * tid;
        uint32_t th_end = std::min(th_begin + thread_size, min_node_size);
        if (th_end > th_begin) {
          threads_addr[tid].push_back({row_indices_ptr, partitions[min_node_id].b + th_begin,
                                       partitions[min_node_id].b + th_end});
          CHECK_LT(min_node_id, threads_id_for_nodes.size());
          threads_id_for_nodes[min_node_id].push_back(tid);
          node_id_for_threads[tid].push_back(min_node_id);
        }
      }
    } else if (n_features*summ_size / n_threads < (1 << (depth + 1))*n_bins
               || (depth >= 1 && !hist_fit_to_l2)) {
      uint32_t block_size = std::max(static_cast<uint32_t>(summ_size/n_threads +
                                     !!(summ_size%n_threads)),
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
            const uint32_t begin = nodes_count[curr_thread_id%n_threads][node_id];
            CHECK_EQ(nodes_count[curr_thread_id%n_threads][node_id] + curr_thread_node_size,
                    nodes_count[curr_thread_id%n_threads][node_id+1]);
            threads_addr[i].push_back({
              threads_rows_nodes_wise[curr_thread_id%n_threads].data(), begin,
              begin + curr_thread_node_size
            });
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
            const uint32_t begin = nodes_count[curr_thread_id%n_threads][node_id];
            CHECK_EQ(nodes_count[curr_thread_id%n_threads][node_id] + curr_thread_node_size,
                    nodes_count[curr_thread_id%n_threads][node_id+1]);

            threads_addr[i].push_back({
              threads_rows_nodes_wise[curr_thread_id%n_threads].data(), begin,
              begin + curr_thread_size
            });
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
            nodes_count[curr_thread_id%n_threads][node_id] += curr_thread_size;
            curr_thread_size = 0;
          }
        }
        curr_thread_size = std::min(block_size,
                                    summ_size > block_size*(i+1) ?
                                    summ_size - block_size*(i+1) : 0);
      }
    } else {
      uint32_t block_size = summ_size/n_threads + !!(summ_size%n_threads);
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
    nodes_count.clear();
  }
  // template for uint32_t
  void UpdateRootThreadWork(bool is_dense) {
    threads_addr.clear();
    threads_addr.resize(n_threads);
    threads_id_for_nodes.clear();
    threads_id_for_nodes.resize(1);
    node_id_for_threads.clear();
    node_id_for_threads.resize(n_threads);
    const uint32_t n_rows = gmat_n_rows;
    const uint32_t block_size = n_rows / n_threads + !!(n_rows % n_threads);
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
