/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_HISTOGRAM_H_

#include <algorithm>
#include <limits>
#include <vector>

#include <memory>
#include "rabit/rabit.h"
#include "xgboost/tree_model.h"
#include "../../common/hist_util.h"
#include "../../common/column_matrix.h"
#include "../../common/opt_partition_builder.h"
#include "../../common/random.h"

#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

namespace xgboost {
namespace tree {

struct Prefetch1 {
 public:
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kPrefetchOffset = 10;

 private:
  static constexpr size_t kNoPrefetchSize1 =
      kPrefetchOffset + kCacheLineSize /
      sizeof(decltype(GHistIndexMatrix::row_ptr)::value_type);

 public:
  static size_t NoPrefetchSize(size_t rows) {
    return std::min(rows, kNoPrefetchSize1);
  }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return Prefetch1::kCacheLineSize / sizeof(T);
  }
};

template<bool do_prefetch,
         typename BinIdxType,
         bool feature_blocking,
         bool is_root,
         bool any_missing,
         bool is_single>
inline void RowsWiseBuildHist(const BinIdxType* gradient_index,
                              const uint32_t* rows,
                              const size_t* row_ptr,
                              const uint32_t row_begin,
                              const uint32_t row_end,
                              const size_t n_features,
                              uint16_t* nodes_ids,
                              const std::vector<std::vector<uint64_t>>& offsets640,
                              const uint16_t* nodes_mapping_ids,
                              const float* pgh, const size_t ib) {
  const uint32_t two {2};
  for (size_t ri = row_begin; ri < row_end; ++ri) {
    const size_t i = is_root ? ri : rows[ri];
    const size_t icol_start = any_missing ? row_ptr[i] : i * n_features;
    const size_t icol_end = any_missing ? row_ptr[i + 1] : icol_start + n_features;
    const size_t row_sizes = any_missing ? icol_end - icol_start : n_features;
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    const size_t idx_gh = two * i;

    const uint32_t nid = is_root ? 0 : nodes_mapping_ids[nodes_ids[i]];
    if (do_prefetch) {
      const size_t icol_start_prefetch = any_missing ?
                                         row_ptr[rows[ri + Prefetch1::kPrefetchOffset]] :
                                         rows[ri + Prefetch1::kPrefetchOffset] * n_features;
      const size_t icol_end_prefetch = any_missing ?
                                       row_ptr[rows[ri + Prefetch1::kPrefetchOffset] + 1] :
                                       (icol_start_prefetch + n_features);

      PREFETCH_READ_T0(pgh + two * rows[ri + Prefetch1::kPrefetchOffset]);
      PREFETCH_READ_T0(0 + rows[ri + Prefetch1::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_end_prefetch;
          j += Prefetch1::GetPrefetchStep<BinIdxType>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    } else if (is_root) {
      nodes_ids[i] = 0;
    }
    const uint64_t* offsets64 = offsets640[nid].data();
    const size_t begin = feature_blocking ? ib*13 : 0;
    const size_t end = feature_blocking ? std::min(begin + 13, n_features) : row_sizes;
    const double pgh_d[] = {pgh[idx_gh], pgh[idx_gh + 1]};
    for (size_t jb = begin;  jb < end; ++jb) {
      if (is_single) {
        float* hist_local = reinterpret_cast<float*>(
          offsets64[jb] + (static_cast<size_t>(gr_index_local[jb])) * 8);
        *(hist_local) +=  pgh[idx_gh];
        *(hist_local + 1) +=  pgh[idx_gh + 1];
      } else {
        double* hist_local = reinterpret_cast<double*>(
          offsets64[jb] + (static_cast<size_t>(gr_index_local[jb])) * 16);
        *(hist_local) +=  pgh_d[0];
        *(hist_local + 1) +=  pgh_d[1];
      }
    }
  }
}

template<typename GradientSumT, typename BinIdxType,
         bool read_by_column, bool feature_blocking,
         bool is_root, bool any_missing, bool column_sampling>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                          const uint32_t* rows,
                          const uint32_t row_begin,
                          const uint32_t row_end,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features, const BinIdxType* numa,
                          uint16_t* nodes_ids, const std::vector<std::vector<uint64_t>>& offsets640,
                          const common::ColumnMatrix *column_matrix,
                          const uint16_t* nodes_mapping_ids, const std::vector<int>& fids) {
  constexpr bool kIsSingle = static_cast<bool>(sizeof(GradientSumT) == 4);

  if (read_by_column) {
    using DenseColumnT = common::DenseColumn<BinIdxType, any_missing>;
    const float* pgh = reinterpret_cast<const float*>(gpair.data());
    const size_t n_columns = column_sampling ? fids.size() : n_features;
    for (size_t cid = 0; cid < n_columns; ++cid) {
      const size_t local_cid = column_sampling ? fids[cid] : cid;
      const auto column_ptr = column_matrix->GetColumn<BinIdxType, any_missing>(local_cid);
      const DenseColumnT* column = dynamic_cast<const DenseColumnT*>(column_ptr.get());
      CHECK_NE(column, static_cast<const DenseColumnT*>(nullptr));
      const BinIdxType* gr_index_local = column->GetFeatureBinIdxPtr().data();
      const size_t base_idx = column->GetBaseIdx();
      for (size_t ii = row_begin; ii < row_end; ++ii) {
        const size_t row_id = is_root ? ii : rows[ii];
        if (is_root && (cid == 0)) {
          nodes_ids[row_id] = 0;
        }
        if (!any_missing || (any_missing && !column->IsMissing(row_id))) {
          const uint32_t nid = is_root ? 0 :nodes_mapping_ids[nodes_ids[row_id]];
          const size_t idx_gh = row_id << 1;
          const uint64_t* offsets64 = offsets640[nid].data();
          const double pgh_d[2] = {pgh[idx_gh], pgh[idx_gh + 1]};
          if (kIsSingle) {
            float* hist_local = reinterpret_cast<float*>(
              offsets64[local_cid] + (static_cast<size_t>(gr_index_local[row_id])) * 8 +
              (any_missing ? base_idx * 8 : 0));
            *(hist_local) +=  pgh[idx_gh];
            *(hist_local + 1) +=  pgh[idx_gh + 1];
          } else {
            double* hist_local = reinterpret_cast<double*>(
              offsets64[local_cid] + (static_cast<size_t>(gr_index_local[row_id])) * 16 +
              (any_missing ? base_idx * 16 : 0));
            *(hist_local) +=  pgh_d[0];
            *(hist_local + 1) +=  pgh_d[1];
          }
        }
      }
    }
  } else {
    const size_t row_size = row_end - row_begin;
    const size_t* row_ptr =  gmat.row_ptr.data();
    const float* pgh = reinterpret_cast<const float*>(gpair.data());
    const BinIdxType* gradient_index = any_missing ? gmat.index.Data<BinIdxType>() : numa;

    size_t feature_block_size = n_features;
    if (feature_blocking) {
      feature_block_size = 13;
    }
    const size_t nb = n_features / feature_block_size + !!(n_features % feature_block_size);
    const size_t size_with_prefetch = (is_root || feature_blocking) ? 0 :
                                      ((row_size > Prefetch1::kPrefetchOffset) ?
                                      (row_size - Prefetch1::kPrefetchOffset) : 0);
    for (size_t ib = 0; ib < nb; ++ib) {
      RowsWiseBuildHist<true, BinIdxType,
                        feature_blocking,
                        is_root, any_missing,
                        kIsSingle> (gradient_index, rows, row_ptr, row_begin,
                                    row_begin + size_with_prefetch,
                                    n_features, nodes_ids,
                                    offsets640, nodes_mapping_ids, pgh, ib);
      RowsWiseBuildHist<false, BinIdxType,
                        feature_blocking,
                        is_root, any_missing,
                        kIsSingle> (gradient_index, rows, row_ptr, row_begin + size_with_prefetch,
                                    row_end, n_features, nodes_ids,
                                    offsets640, nodes_mapping_ids, pgh, ib);
    }
  }
}

template <typename GradientSumT, typename ExpandEntry> class HistogramBuilder {
  using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
  using GHistRowT = common::GHistRow<GradientSumT>;
  common::Monitor builder_monitor_;
  /*! \brief culmulative histogram of gradients. */
  common::HistCollection<GradientSumT> hist_;
  /*! \brief culmulative local parent histogram of gradients. */
  common::HistCollection<GradientSumT> hist_local_worker_;
  common::GHistBuilder<GradientSumT> builder_;
  common::ParallelGHistBuilder<GradientSumT> buffer_;
  std::vector<std::vector<std::vector<uint64_t>>> offsets64_;
  rabit::Reducer<GradientPairT, GradientPairT::Reduce> reducer_;
  int32_t max_bin_ {-1};
  int32_t n_threads_ {-1};
  int32_t max_depth_ {1};
  float colsample_bytree_ {1.0};
  float colsample_bylevel_ {1.0};
  float colsample_bynode_ {1.0};
  size_t n_bins_ {0};
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  // Whether XGBoost is running in distributed environment.
  bool is_distributed_ {false};

 public:
  std::vector<std::vector<std::vector<GradientSumT>>> histograms_buffer;
  std::vector<std::vector<std::vector<GradientSumT>>>* GetHistBuffer() {
    return &histograms_buffer;
  }
  /**
   * \param total_bins       Total number of bins across all features
   * \param max_bin_per_feat Maximum number of bins per feature, same as the `max_bin`
   *                         training parameter.
   * \param n_threads        Number of threads.
   * \param is_distributed   Mostly used for testing to allow injecting parameters instead
   *                         of using global rabit variable.
   */
  void Reset(uint32_t total_bins,
             int32_t max_bin_per_feat, int32_t n_threads,
             int32_t n_features, int32_t max_depth,
             float colsample_bytree, float colsample_bylevel, float colsample_bynode,
             std::shared_ptr<common::ColumnSampler> column_sampler,
             const uint32_t* offsets = nullptr, const bool is_dense = true,
             bool is_distributed = rabit::IsDistributed()) {
    CHECK_GE(n_threads, 1);
    n_threads_ = n_threads;
    CHECK_GE(max_bin_per_feat, 2);
    max_bin_ = max_bin_per_feat;
    hist_.Init(total_bins);
    hist_local_worker_.Init(total_bins);
    buffer_.Init(total_bins);
    builder_ = common::GHistBuilder<GradientSumT>(n_threads, total_bins);
    is_distributed_ = is_distributed;
    max_depth_ = std::max(max_depth, 1);
    colsample_bytree_ = colsample_bytree;
    colsample_bylevel_ = colsample_bylevel;
    colsample_bynode_ = colsample_bynode;
    column_sampler_ = column_sampler;
    if (histograms_buffer.size() == 0) {
      n_bins_ = total_bins;
      histograms_buffer.resize(n_threads);
      offsets64_.resize(n_threads);
      #pragma omp parallel num_threads(n_threads)
      {
        const size_t tid = omp_get_thread_num();
        histograms_buffer[tid].resize((1 << (max_depth_ - 1)));
        offsets64_[tid].resize((1 << (max_depth_ - 1)));
      }
    }
  }

  template<typename BinIdxType, bool any_missing, bool hist_fit_to_l2, typename BuildHistFunc>
  BuildHistFunc GetBuildHistSrategy(int depth, const bool any_sparse_column) {
    // now column sampling supported only for missings due to fid_least_bins_ set
    if (any_missing
        && (colsample_bytree_ < 0.1 || colsample_bylevel_ < 0.1)
        && !any_sparse_column
        && colsample_bynode_ == 1) {
      if (depth == 0) {
        return BuildHistKernel<GradientSumT, BinIdxType,
                               /*read_by_column*/   true,
                               /*feature_blocking*/ false,
                               /*is_root*/          true,
                               /*any_missing*/      any_missing,
                               /*column_sampling*/  true>;
      } else {
        return BuildHistKernel<GradientSumT, BinIdxType,
                               /*read_by_column*/   true,
                               /*feature_blocking*/ false,
                               /*is_root*/          false,
                               /*any_missing*/      any_missing,
                               /*column_sampling*/  true>;
      }
    } else {
      if (depth == 0) {
        return BuildHistKernel<GradientSumT, BinIdxType,
                               /*read_by_column*/   !hist_fit_to_l2 && !any_missing,
                               /*feature_blocking*/ false,
                               /*is_root*/          true,
                               /*any_missing*/      any_missing,
                               /*column_sampling*/  false>;
      } else {
        if (depth == 1) {  // make 1st depth 33% faster for hist not fitted to L2 cache
          return BuildHistKernel<GradientSumT, BinIdxType,
                                 /*read_by_column*/   !hist_fit_to_l2 && !any_missing,
                                 /*feature_blocking*/ false,
                                 /*is_root*/          false,
                                 /*any_missing*/      any_missing,
                                 /*column_sampling*/  false>;
        } else {
          return BuildHistKernel<GradientSumT, BinIdxType,
                                 /*read_by_column*/   false,
                                 /*feature_blocking*/ !hist_fit_to_l2 && !any_missing,
                                 /*is_root*/          false,
                                 /*any_missing*/      any_missing,
                                 /*column_sampling*/  false>;
        }
      }
    }
  }

  template <typename BinIdxType, bool any_missing, bool hist_fit_to_l2>
  void BuildLocalHistograms(DMatrix *p_fmat, const GHistIndexMatrix &gmat,
                            const std::vector<GradientPair> &gpair_h,
                            int depth,
                            const common::ColumnMatrix& column_matrix,
                            const common::OptPartitionBuilder* p_opt_partition_builder,
                            // template?
                            const std::vector<uint16_t>* p_nodes_mapping,
                            std::vector<uint16_t>* node_ids) {
    // std::string depth_str = std::to_string(depth);
    const std::vector<uint16_t>& nodes_mapping = *p_nodes_mapping;
    const common::OptPartitionBuilder& opt_partition_builder = *p_opt_partition_builder;
    std::vector<uint16_t>& node_ids_ = *node_ids;
    using BuildHistFunc = void(*)(const std::vector<GradientPair>&,
                                  const uint32_t*,
                                  const uint32_t,
                                  const uint32_t,
                                  const GHistIndexMatrix&,
                                  const size_t,
                                  const BinIdxType*, uint16_t*,
                                  const std::vector<std::vector<uint64_t>>&,
                                  const common::ColumnMatrix*, const uint16_t*,
                                  const std::vector<int>&);
    BuildHistFunc build_hist_func = GetBuildHistSrategy<BinIdxType,
                                                        any_missing,
                                                        hist_fit_to_l2,
                                                        BuildHistFunc>(
                                                          depth, column_matrix.AnySparseColumn());
    builder_monitor_.Start("BuildLocalHistograms");
    const size_t n_features = gmat.cut.Ptrs().size() - 1;
    int nthreads = this->n_threads_;

    std::vector<int> fids;
    // now column sampling supported only for missings due to fid_least_bins_ set
    if (any_missing
        && (colsample_bytree_ < 0.1 || colsample_bylevel_ < 0.1)
        && !column_matrix.AnySparseColumn()
        && colsample_bynode_ == 1) {
      const size_t n_sampled_features = column_sampler_->GetFeatureSet(depth)->Size();
      fids.resize(n_sampled_features, 0);
      for (size_t i = 0; i < n_sampled_features; ++i) {
        fids[i] = column_sampler_->GetFeatureSet(depth)->ConstHostVector()[i];
      }
    }
    // Parallel processing by nodes and data in each node
    for (auto const &gmat_local : p_fmat->GetBatches<GHistIndexMatrix>(
             BatchParam{GenericParameter::kCpuId, max_bin_})) {
      #pragma omp parallel num_threads(nthreads)
      {
        size_t tid = omp_get_thread_num();
        const BinIdxType* numa = tid < nthreads/2 ?  gmat_local.index.Data<BinIdxType>() :
                                                    gmat_local.index.SecondData<BinIdxType>();
        const std::vector<common::Slice>& local_slices =
          opt_partition_builder.threads_addr[tid];
        const size_t thread_size = opt_partition_builder.node_id_for_threads[tid].size();
        for (size_t nid = 0; nid < thread_size; ++nid) {
          const size_t node_id = opt_partition_builder.node_id_for_threads[tid][nid];
          const int32_t mapped_node_id = nodes_mapping.data()[node_id];
          if (offsets64_[tid][mapped_node_id].size() == 0) {
            offsets64_[tid][mapped_node_id].resize(n_features, 0);
            histograms_buffer[tid][mapped_node_id].resize(n_bins_*2, 0);
            uint64_t* offsets640 = offsets64_[tid][mapped_node_id].data();
            const uint32_t* offsets = gmat_local.index.Offset();
            for (size_t i = 0; i < n_features; ++i) {
              offsets640[i] = !any_missing ?
                reinterpret_cast<uint64_t>(histograms_buffer[tid][mapped_node_id].data()) +
                sizeof(GradientSumT)*2*static_cast<uint64_t>(offsets[i]) :
                reinterpret_cast<uint64_t>(histograms_buffer[tid][mapped_node_id].data());
            }
          }
        }
        for (const common::Slice& slice : local_slices) {
          const uint32_t* rows = slice.addr;
          build_hist_func(gpair_h, rows, slice.b, slice.e, gmat_local, n_features,
                          numa, node_ids_.data(), offsets64_[tid],
                          &column_matrix, nodes_mapping.data(), fids);
        }
      }
    }

    builder_monitor_.Stop("BuildLocalHistograms");
  }


  void
  AddHistRows(int *starting_index, int *sync_count,
              std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
              std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
              RegTree *p_tree) {
    if (is_distributed_) {
      this->AddHistRowsDistributed(starting_index, sync_count,
                                   nodes_for_explicit_hist_build,
                                   nodes_for_subtraction_trick, p_tree);
    } else {
      this->AddHistRowsLocal(starting_index, sync_count,
                             nodes_for_explicit_hist_build,
                             nodes_for_subtraction_trick);
    }
  }

  /* Main entry point of this class, build histogram for tree nodes. */
  template <typename BinIdxType, bool any_missing, bool hist_fit_to_l2>
  void BuildHist(DMatrix *p_fmat, const GHistIndexMatrix &gmat, RegTree *p_tree,
                 std::vector<GradientPair> const &gpair,
                 int depth, const common::ColumnMatrix& column_matrix,
                 std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                 std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
                 const common::OptPartitionBuilder* p_opt_partition_builder,
                 // template?
                 std::vector<uint16_t>* p_nodes_mapping,
                 std::vector<uint16_t>* node_ids) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    this->AddHistRows(&starting_index, &sync_count,
                      nodes_for_explicit_hist_build,
                      nodes_for_subtraction_trick, p_tree);
    if (!any_missing
        || (any_missing && (colsample_bytree_ < 0.1
        || colsample_bylevel_ < 0.1)
        && !column_matrix.AnySparseColumn() && colsample_bynode_ == 1)) {
      BuildLocalHistograms<BinIdxType,
                           any_missing, hist_fit_to_l2>(p_fmat, gmat, gpair, depth, column_matrix,
                                                        p_opt_partition_builder,
                                                        p_nodes_mapping, node_ids);
    } else {
      BuildLocalHistograms<uint32_t,
                           any_missing, hist_fit_to_l2>(p_fmat, gmat, gpair, depth, column_matrix,
                                                        p_opt_partition_builder,
                                                        p_nodes_mapping, node_ids);
    }

    if (is_distributed_) {
      this->template SyncHistograms<true>(gmat, p_tree, nodes_for_explicit_hist_build,
                                     nodes_for_subtraction_trick,
                                     starting_index, sync_count,
                                     p_opt_partition_builder, p_nodes_mapping);
    } else if ((gmat.IsDense() && depth == 0)
               || colsample_bylevel_ != 1 || colsample_bynode_ != 1 || colsample_bytree_ != 1) {
      this->template SyncHistograms<false>(gmat, p_tree, nodes_for_explicit_hist_build,
                               nodes_for_subtraction_trick,
                               starting_index, sync_count,
                               p_opt_partition_builder, p_nodes_mapping);
    }
  }

  template<bool is_distributed>
  void SyncHistograms(
      const GHistIndexMatrix &gmat,
      RegTree *p_tree,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count,
      const common::OptPartitionBuilder* p_opt_partition_builder,
      const std::vector<uint16_t>* p_nodes_mapping) {
    const std::vector<uint16_t>& nodes_mapping = *p_nodes_mapping;
    const common::OptPartitionBuilder& opt_partition_builder = *p_opt_partition_builder;
    const size_t nbins = builder_.GetNumBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        512);

    common::ParallelFor2d(
        space, n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once

          GradientSumT* dest_hist = reinterpret_cast<GradientSumT*>(this_hist.data());
          if (opt_partition_builder.threads_id_for_nodes[entry.nid].size() != 0) {
            const int32_t node_id = nodes_mapping.data()[entry.nid];
            const size_t first_thread_id =
              opt_partition_builder.threads_id_for_nodes[entry.nid][0];
            GradientSumT* hist0 =  histograms_buffer[first_thread_id][node_id].data();
            common::ReduceHist(dest_hist,
                               hist0,
                               &histograms_buffer,
                               node_id,
                               opt_partition_builder.threads_id_for_nodes[entry.nid],
                               2 * r.begin(), 2 * r.end());
          } else {
            common::ClearHist(dest_hist, 2 * r.begin(), 2 * r.end());
          }
          // Store posible parent node
          if (is_distributed) {
            auto this_local = hist_local_worker_[entry.nid];
            common::CopyHist(this_local, this_hist, r.begin(), r.end());
          }

          if (!(*p_tree)[entry.nid].IsRoot()) {
            const size_t parent_id = (*p_tree)[entry.nid].Parent();
            const int subtraction_node_id =
                nodes_for_subtraction_trick[node].nid;
            GradientSumT* parent_hist = nullptr;
            if (is_distributed) {
              parent_hist = reinterpret_cast<GradientSumT*>(
                this->hist_local_worker_[parent_id].data());
            } else {
              parent_hist = reinterpret_cast<GradientSumT*>(
                this->hist_[parent_id].data());
            }
            GradientSumT* largest_hist =
              reinterpret_cast<GradientSumT*>(this->hist_[subtraction_node_id].data());
            // subtric large
            common::SubtractionHist(largest_hist, parent_hist, dest_hist,
                                    2 * r.begin(), 2 * r.end());
            // Store posible parent node
            if (is_distributed) {
              auto sibling_local = hist_local_worker_[subtraction_node_id];
              common::CopyHist(sibling_local, this->hist_[subtraction_node_id], r.begin(), r.end());
            }
          }
        });

    if (is_distributed) {
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
  }

 public:
  /* Getters for tests. */
  common::HistCollection<GradientSumT> const& Histogram() {
    return hist_;
  }
  auto& Buffer() { return buffer_; }

 private:
  void
  ParallelSubtractionHist(const common::BlockedSpace2d &space,
                          const std::vector<ExpandEntry> &nodes,
                          const std::vector<ExpandEntry> &subtraction_nodes,
                          const RegTree *p_tree) {
    common::ParallelFor2d(
      space, this->n_threads_, [&](size_t node, common::Range1d r) {
        const auto &entry = nodes[node];
        if (!((*p_tree)[entry.nid].IsLeftChild())) {
          auto this_hist = this->hist_[entry.nid];
          GradientSumT* dest_hist = reinterpret_cast<GradientSumT*>(this_hist.data());
          if (!(*p_tree)[entry.nid].IsRoot()) {
            const int subtraction_node_id = subtraction_nodes[node].nid;
            GradientSumT* parent_hist = reinterpret_cast<GradientSumT*>(
              this->hist_[(*p_tree)[entry.nid].Parent()].data());
            GradientSumT* largest_hist = reinterpret_cast<GradientSumT*>(
              this->hist_[subtraction_node_id].data());
            common::SubtractionHist(dest_hist, parent_hist, largest_hist,
                                  2 * r.begin(), 2 * r.end());
          }
        }
      });
  }

  // Add a tree node to histogram buffer in local training environment.
  void AddHistRowsLocal(
      int *starting_index, int *sync_count,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick) {
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
  }

  void AddHistRowsDistributed(
      int *starting_index, int *sync_count,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
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
};
}      // namespace tree
}      // namespace xgboost
#endif  // XGBOOST_TREE_HIST_HISTOGRAM_H_
