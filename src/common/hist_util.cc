/*!
 * Copyright 2017-2020 by Contributors
 * \file hist_util.cc
 */
#include <dmlc/timer.h>
#include <dmlc/omp.h>

#include <rabit/rabit.h>
#include <numeric>
#include <vector>

#include "xgboost/base.h"
#include "../common/common.h"
#include "hist_util.h"
#include "random.h"
#include "column_matrix.h"
#include "quantile.h"
#include "./../tree/updater_quantile_hist.h"

#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

namespace xgboost {
namespace common {

template<typename BinIdxType>
void GHistIndexMatrix::SetIndexDataForDense(common::Span<BinIdxType> index_data_span,
                                    size_t batch_threads, const SparsePage& batch,
                                    size_t rbegin, common::Span<const uint32_t> offsets_span,
                                    size_t nbins) {
  const xgboost::Entry* data_ptr = batch.data.HostVector().data();
  const std::vector<bst_row_t>& offset_vec = batch.offset.HostVector();
  const size_t batch_size = batch.Size();
  CHECK_LT(batch_size, offset_vec.size());
  BinIdxType* index_data = index_data_span.data();
  const uint32_t* offsets = offsets_span.data();
  #pragma omp parallel for num_threads(batch_threads) schedule(static)
    for (omp_ulong i = 0; i < batch_size; ++i) {
      const int tid = omp_get_thread_num();
      size_t ibegin = row_ptr[rbegin + i];
      size_t iend = row_ptr[rbegin + i + 1];
      const size_t size = offset_vec[i + 1] - offset_vec[i];
      SparsePage::Inst inst = {data_ptr + offset_vec[i], size};
      CHECK_EQ(ibegin + inst.size(), iend);
      for (bst_uint j = 0; j < inst.size(); ++j) {
        uint32_t idx = cut.SearchBin(inst[j]);
        index_data[ibegin + j] = static_cast<BinIdxType>(idx - offsets[j]);
        ++hit_count_tloc_[tid * nbins + idx];
      }
    }
}
template void GHistIndexMatrix::SetIndexDataForDense(common::Span<uint8_t> index_data_span,
                                             size_t batch_threads, const SparsePage& batch,
                                             size_t rbegin,
                                             common::Span<const uint32_t> offsets_span,
                                             size_t nbins);
template void GHistIndexMatrix::SetIndexDataForDense(common::Span<uint16_t> index_data_span,
                                             size_t batch_threads, const SparsePage& batch,
                                             size_t rbegin,
                                             common::Span<const uint32_t> offsets_span,
                                             size_t nbins);
template void GHistIndexMatrix::SetIndexDataForDense(common::Span<uint32_t> index_data_span,
                                             size_t batch_threads, const SparsePage& batch,
                                             size_t rbegin,
                                             common::Span<const uint32_t> offsets_span,
                                             size_t nbins);

void GHistIndexMatrix::SetIndexDataForSparse(common::Span<uint32_t> index_data_span,
                                                 size_t batch_threads,
                                                 const SparsePage& batch, size_t rbegin,
                                                 size_t nbins) {
  const xgboost::Entry* data_ptr = batch.data.HostVector().data();
  const std::vector<bst_row_t>& offset_vec = batch.offset.HostVector();
  const size_t batch_size = batch.Size();
  CHECK_LT(batch_size, offset_vec.size());
  uint32_t* index_data = index_data_span.data();
  #pragma omp parallel for num_threads(batch_threads) schedule(static)
    for (omp_ulong i = 0; i < batch_size; ++i) {
      const int tid = omp_get_thread_num();
      size_t ibegin = row_ptr[rbegin + i];
      size_t iend = row_ptr[rbegin + i + 1];
      const size_t size = offset_vec[i + 1] - offset_vec[i];
      SparsePage::Inst inst = {data_ptr + offset_vec[i], size};
      CHECK_EQ(ibegin + inst.size(), iend);
      for (bst_uint j = 0; j < inst.size(); ++j) {
        uint32_t idx = cut.SearchBin(inst[j]);
        index_data[ibegin + j] = idx;
        ++hit_count_tloc_[tid * nbins + idx];
      }
    }
}

void GHistIndexMatrix::ResizeIndex(const size_t rbegin, const SparsePage& batch,
                                   const size_t n_offsets, const size_t n_index,
                                   const bool isDense) {
  if ((max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint8_t>::max())) && isDense) {
    index.SetBinTypeSize(kUint8BinsTypeSize);
    index.Resize((sizeof(uint8_t)) * n_index);
  } else if ((max_num_bins - 1 > static_cast<int>(std::numeric_limits<uint8_t>::max())  &&
    max_num_bins - 1 <= static_cast<int>(std::numeric_limits<uint16_t>::max())) && isDense) {
    index.SetBinTypeSize(kUint16BinsTypeSize);
    index.Resize((sizeof(uint16_t)) * n_index);
  } else {
    index.SetBinTypeSize(kUint32BinsTypeSize);
    index.Resize((sizeof(uint32_t)) * n_index);
  }
}

HistogramCuts::HistogramCuts() {
  monitor_.Init(__FUNCTION__);
  cut_ptrs_.HostVector().emplace_back(0);
}

// Dispatch to specific builder.
void HistogramCuts::Build(DMatrix* dmat, uint32_t const max_num_bins) {
  auto const& info = dmat->Info();
  size_t const total = info.num_row_ * info.num_col_;
  size_t const nnz = info.num_nonzero_;
  float const sparsity = static_cast<float>(nnz) / static_cast<float>(total);
  // Use a small number to avoid calling `dmat->GetColumnBatches'.
  float constexpr kSparsityThreshold = 0.0005;
  // FIXME(trivialfis): Distributed environment is not supported.
  if (sparsity < kSparsityThreshold && (!rabit::IsDistributed())) {
    LOG(INFO) << "Building quantile cut on a sparse dataset.";
    SparseCuts cuts(this);
    cuts.Build(dmat, max_num_bins);
  } else {
    LOG(INFO) << "Building quantile cut on a dense dataset or distributed environment.";
    DenseCuts cuts(this);
    cuts.Build(dmat, max_num_bins);
  }
  LOG(INFO) << "Total number of hist bins: " << cut_ptrs_.HostVector().back();
}

bool CutsBuilder::UseGroup(DMatrix* dmat) {
  auto& info = dmat->Info();
  return CutsBuilder::UseGroup(info);
}

bool CutsBuilder::UseGroup(MetaInfo const& info) {
  size_t const num_groups = info.group_ptr_.size() == 0 ?
                            0 : info.group_ptr_.size() - 1;
  // Use group index for weights?
  bool const use_group_ind = num_groups != 0 &&
                             (info.weights_.Size() != info.num_row_);
  return use_group_ind;
}

void SparseCuts::SingleThreadBuild(SparsePage const& page, MetaInfo const& info,
                                   uint32_t max_num_bins,
                                   bool const use_group_ind,
                                   uint32_t beg_col, uint32_t end_col,
                                   uint32_t thread_id) {
  CHECK_GE(end_col, beg_col);
  constexpr float kFactor = 8;

  // Data groups, used in ranking.
  std::vector<bst_uint> const& group_ptr = info.group_ptr_;
  auto &local_min_vals = p_cuts_->min_vals_.HostVector();
  auto &local_cuts = p_cuts_->cut_values_.HostVector();
  auto &local_ptrs = p_cuts_->cut_ptrs_.HostVector();
  local_min_vals.resize(end_col - beg_col, 0);

  for (uint32_t col_id = beg_col; col_id < page.Size() && col_id < end_col; ++col_id) {
    // Using a local variable makes things easier, but at the cost of memory trashing.
    WQSketch sketch;
    common::Span<xgboost::Entry const> const column = page[col_id];
    uint32_t const n_bins = std::min(static_cast<uint32_t>(column.size()),
                                     max_num_bins);
    if (n_bins == 0) {
      // cut_ptrs_ is initialized with a zero, so there's always an element at the back
      local_ptrs.emplace_back(local_ptrs.back());
      continue;
    }

    sketch.Init(info.num_row_, 1.0 / (n_bins * kFactor));
    for (auto const& entry : column) {
      uint32_t weight_ind = 0;
      if (use_group_ind) {
        auto row_idx = entry.index;
        uint32_t group_ind =
            this->SearchGroupIndFromRow(group_ptr, page.base_rowid + row_idx);
        weight_ind = group_ind;
      } else {
        weight_ind = entry.index;
      }
      sketch.Push(entry.fvalue, info.GetWeight(weight_ind));
    }

    WQSketch::SummaryContainer out_summary;
    sketch.GetSummary(&out_summary);
    WQSketch::SummaryContainer summary;
    summary.Reserve(n_bins + 1);
    summary.SetPrune(out_summary, n_bins + 1);

    // Can be use data[1] as the min values so that we don't need to
    // store another array?
    float mval = summary.data[0].value;
    local_min_vals[col_id - beg_col]  = mval - (fabs(mval) + 1e-5);

    this->AddCutPoint(summary, max_num_bins);

    bst_float cpt = (summary.size > 0) ?
                    summary.data[summary.size - 1].value :
                    local_min_vals[col_id - beg_col];
    cpt += fabs(cpt) + 1e-5;
    local_cuts.emplace_back(cpt);

    local_ptrs.emplace_back(local_cuts.size());
  }
}

std::vector<size_t> SparseCuts::LoadBalance(SparsePage const& page,
                                            size_t const nthreads) {
  /* Some sparse datasets have their mass concentrating on small
   * number of features.  To avoid wating for a few threads running
   * forever, we here distirbute different number of columns to
   * different threads according to number of entries. */
  size_t const total_entries = page.data.Size();
  size_t const entries_per_thread = common::DivRoundUp(total_entries, nthreads);

  std::vector<size_t> cols_ptr(nthreads+1, 0);
  size_t count {0};
  size_t current_thread {1};

  for (size_t col_id = 0; col_id < page.Size(); ++col_id) {
    auto const column = page[col_id];
    cols_ptr[current_thread]++;  // add one column to thread
    count += column.size();
    if (count > entries_per_thread + 1) {
      current_thread++;
      count = 0;
      cols_ptr[current_thread] = cols_ptr[current_thread-1];
    }
  }
  // Idle threads.
  for (; current_thread < cols_ptr.size() - 1; ++current_thread) {
    cols_ptr[current_thread+1] = cols_ptr[current_thread];
  }

  return cols_ptr;
}

void SparseCuts::Build(DMatrix* dmat, uint32_t const max_num_bins) {
  monitor_.Start(__FUNCTION__);
  // Use group index for weights?
  auto use_group = UseGroup(dmat);
  uint32_t nthreads = omp_get_max_threads();
  CHECK_GT(nthreads, 0);
  std::vector<HistogramCuts> cuts_containers(nthreads);
  std::vector<std::unique_ptr<SparseCuts>> sparse_cuts(nthreads);
  for (size_t i = 0; i < nthreads; ++i) {
    sparse_cuts[i].reset(new SparseCuts(&cuts_containers[i]));
  }

  for (auto const& page : dmat->GetBatches<CSCPage>()) {
    CHECK_LE(page.Size(), dmat->Info().num_col_);
    monitor_.Start("Load balance");
    std::vector<size_t> col_ptr = LoadBalance(page, nthreads);
    monitor_.Stop("Load balance");
    // We here decouples the logic between build and parallelization
    // to simplify things a bit.
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (omp_ulong i = 0; i < nthreads; ++i) {
      common::Monitor t_monitor;
      t_monitor.Init("SingleThreadBuild: " + std::to_string(i));
      t_monitor.Start(std::to_string(i));
      sparse_cuts[i]->SingleThreadBuild(page, dmat->Info(), max_num_bins, use_group,
                                        col_ptr[i], col_ptr[i+1], i);
      t_monitor.Stop(std::to_string(i));
    }

    this->Concat(sparse_cuts, dmat->Info().num_col_);
  }

  monitor_.Stop(__FUNCTION__);
}

void SparseCuts::Concat(
    std::vector<std::unique_ptr<SparseCuts>> const& cuts, uint32_t n_cols) {
  monitor_.Start(__FUNCTION__);
  uint32_t nthreads = omp_get_max_threads();
  auto &local_min_vals = p_cuts_->min_vals_.HostVector();
  auto &local_cuts = p_cuts_->cut_values_.HostVector();
  auto &local_ptrs = p_cuts_->cut_ptrs_.HostVector();
  local_min_vals.resize(n_cols, std::numeric_limits<float>::max());
  size_t min_vals_tail = 0;

  for (uint32_t t = 0; t < nthreads; ++t) {
    auto& thread_min_vals = cuts[t]->p_cuts_->min_vals_.HostVector();
    auto& thread_cuts = cuts[t]->p_cuts_->cut_values_.HostVector();
    auto& thread_ptrs = cuts[t]->p_cuts_->cut_ptrs_.HostVector();

    // concat csc pointers.
    size_t const old_ptr_size = local_ptrs.size();
    local_ptrs.resize(
        thread_ptrs.size() + local_ptrs.size() - 1);
    size_t const new_icp_size = local_ptrs.size();
    auto tail = local_ptrs[old_ptr_size-1];
    for (size_t j = old_ptr_size; j < new_icp_size; ++j) {
      local_ptrs[j] = tail + thread_ptrs[j-old_ptr_size+1];
    }
    // concat csc values
    size_t const old_iv_size = local_cuts.size();
    local_cuts.resize(
        thread_cuts.size() + local_cuts.size());
    size_t const new_iv_size = local_cuts.size();
    for (size_t j = old_iv_size; j < new_iv_size; ++j) {
      local_cuts[j] = thread_cuts[j-old_iv_size];
    }
    // merge min values
    for (size_t j = 0; j < thread_min_vals.size(); ++j) {
       local_min_vals.at(min_vals_tail + j) =
          std::min(local_min_vals.at(min_vals_tail + j), thread_min_vals.at(j));
    }
    min_vals_tail += thread_min_vals.size();
  }
  monitor_.Stop(__FUNCTION__);
}

void DenseCuts::Build(DMatrix* p_fmat, uint32_t max_num_bins) {
  monitor_.Start(__FUNCTION__);
  const MetaInfo& info = p_fmat->Info();

  // safe factor for better accuracy
  constexpr int kFactor = 8;
  std::vector<WQSketch> sketchs;

  const int nthread = omp_get_max_threads();

  unsigned const nstep =
      static_cast<unsigned>((info.num_col_ + nthread - 1) / nthread);
  unsigned const ncol = static_cast<unsigned>(info.num_col_);
  sketchs.resize(info.num_col_);
  for (auto& s : sketchs) {
    s.Init(info.num_row_, 1.0 / (max_num_bins * kFactor));
  }

  // Data groups, used in ranking.
  std::vector<bst_uint> const& group_ptr = info.group_ptr_;
  size_t const num_groups = group_ptr.size() == 0 ? 0 : group_ptr.size() - 1;
  // Use group index for weights?
  bool const use_group = UseGroup(p_fmat);
  const bool isDense = p_fmat->IsDense();
  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    size_t group_ind = 0;
    if (use_group) {
      group_ind = this->SearchGroupIndFromRow(group_ptr, batch.base_rowid);
    }
#pragma omp parallel num_threads(nthread) firstprivate(group_ind, use_group)
    {
      CHECK_EQ(nthread, omp_get_num_threads());
      auto tid = static_cast<unsigned>(omp_get_thread_num());
      unsigned begin = std::min(nstep * tid, ncol);
      unsigned end = std::min(nstep * (tid + 1), ncol);

      // do not iterate if no columns are assigned to the thread
      if (begin < end && end <= ncol) {
        for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
          size_t const ridx = batch.base_rowid + i;
          SparsePage::Inst const inst = batch[i];
          if (use_group &&
              group_ptr[group_ind] == ridx &&
              // maximum equals to weights.size() - 1
              group_ind < num_groups - 1) {
            // move to next group
            group_ind++;
          }
          size_t w_idx = use_group ? group_ind : ridx;
          auto w = info.GetWeight(w_idx);
          if (isDense) {
            auto data = inst.data();
            for (size_t ii = begin; ii < end; ii++) {
              sketchs[ii].Push(data[ii].fvalue, w);
            }
          } else {
            for (auto const& entry : inst) {
              if (entry.index >= begin && entry.index < end) {
                sketchs[entry.index].Push(entry.fvalue, w);
              }
            }
          }
        }
      }
    }
  }

  Init(&sketchs, max_num_bins, info.num_row_);
  monitor_.Stop(__FUNCTION__);
}

/**
 * \param [in,out]  in_sketchs
 * \param           max_num_bins  The maximum number bins.
 * \param           max_rows      Number of rows in this DMatrix.
 */
void DenseCuts::Init
(std::vector<WQSketch>* in_sketchs, uint32_t max_num_bins, size_t max_rows) {
  monitor_.Start(__func__);
  std::vector<WQSketch>& sketchs = *in_sketchs;

  // Compute how many cuts samples we need at each node
  // Do not require more than the number of total rows  in training data
  // This allows efficient training on wide data
  size_t global_max_rows = max_rows;
  rabit::Allreduce<rabit::op::Sum>(&global_max_rows, 1);
  constexpr int kFactor = 8;
  size_t intermediate_num_cuts =
      std::min(global_max_rows, static_cast<size_t>(max_num_bins * kFactor));
  // gather the histogram data
  rabit::SerializeReducer<WQSketch::SummaryContainer> sreducer;
  std::vector<WQSketch::SummaryContainer> summary_array;
  summary_array.resize(sketchs.size());
  for (size_t i = 0; i < sketchs.size(); ++i) {
    WQSketch::SummaryContainer out;
    sketchs[i].GetSummary(&out);
    summary_array[i].Reserve(intermediate_num_cuts);
    summary_array[i].SetPrune(out, intermediate_num_cuts);
  }
  CHECK_EQ(summary_array.size(), in_sketchs->size());
  size_t nbytes = WQSketch::SummaryContainer::CalcMemCost(intermediate_num_cuts);
  // TODO(chenqin): rabit failure recovery assumes no boostrap onetime call after loadcheckpoint
  // we need to move this allreduce before loadcheckpoint call in future
  sreducer.Allreduce(dmlc::BeginPtr(summary_array), nbytes, summary_array.size());
  p_cuts_->min_vals_.HostVector().resize(sketchs.size());

  for (size_t fid = 0; fid < summary_array.size(); ++fid) {
    WQSketch::SummaryContainer a;
    a.Reserve(max_num_bins + 1);
    a.SetPrune(summary_array[fid], max_num_bins + 1);
    const bst_float mval = a.data[0].value;
    p_cuts_->min_vals_.HostVector()[fid] = mval - (fabs(mval) + 1e-5);
    AddCutPoint(a, max_num_bins);
    // push a value that is greater than anything
    const bst_float cpt
      = (a.size > 0) ? a.data[a.size - 1].value : p_cuts_->min_vals_.HostVector()[fid];
    // this must be bigger than last value in a scale
    const bst_float last = cpt + (fabs(cpt) + 1e-5);
    p_cuts_->cut_values_.HostVector().push_back(last);

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(p_cuts_->cut_values_.HostVector().size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(p_cuts_->cut_values_.HostVector().size());
    CHECK_GT(cut_size, p_cuts_->cut_ptrs_.HostVector().back());
    p_cuts_->cut_ptrs_.HostVector().push_back(cut_size);
  }
  monitor_.Stop(__func__);
}

void GHistIndexMatrix::Init(DMatrix* p_fmat, int max_bins) {
  cut.Build(p_fmat, max_bins);
  max_num_bins = max_bins;
  const int32_t nthread = omp_get_max_threads();
  const uint32_t nbins = cut.Ptrs().back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(nthread * nbins, 0);

  this->p_fmat = p_fmat;
  size_t new_size = 1;
  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    new_size += batch.Size();
  }

  row_ptr.resize(new_size);
  row_ptr[0] = 0;

  size_t rbegin = 0;
  size_t prev_sum = 0;
  const bool isDense = p_fmat->IsDense();
  this->isDense_ = isDense;

  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    // The number of threads is pegged to the batch size. If the OMP
    // block is parallelized on anything other than the batch/block size,
    // it should be reassigned
    const size_t batch_threads = std::max(
        size_t(1),
        std::min(batch.Size(), static_cast<size_t>(omp_get_max_threads())));
    MemStackAllocator<size_t, 128> partial_sums(batch_threads);
    size_t* p_part = partial_sums.Get();

    size_t block_size =  batch.Size() / batch_threads;

    #pragma omp parallel num_threads(batch_threads)
    {
      #pragma omp for
      for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads-1) ? batch.Size() : (block_size * (tid+1)));

        size_t sum = 0;
        for (size_t i = ibegin; i < iend; ++i) {
          sum += batch[i].size();
          row_ptr[rbegin + 1 + i] = sum;
        }
      }

      #pragma omp single
      {
        p_part[0] = prev_sum;
        for (size_t i = 1; i < batch_threads; ++i) {
          p_part[i] = p_part[i - 1] + row_ptr[rbegin + i*block_size];
        }
      }

      #pragma omp for
      for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads-1) ? batch.Size() : (block_size * (tid+1)));

        for (size_t i = ibegin; i < iend; ++i) {
          row_ptr[rbegin + 1 + i] += p_part[tid];
        }
      }
    }

    const size_t n_offsets = cut.Ptrs().size() - 1;
    const size_t n_index = row_ptr[rbegin + batch.Size()];
    ResizeIndex(rbegin, batch, n_offsets, n_index, isDense);

    CHECK_GT(cut.Values().size(), 0U);

    uint32_t* offsets = nullptr;
    if (isDense) {
      index.ResizeOffset(n_offsets);
      offsets = index.Offset();
      for (size_t i = 0; i < n_offsets; ++i) {
        offsets[i] = cut.Ptrs()[i];
      }
    }

    if (isDense) {
      BinTypeSize curent_bin_size = index.GetBinTypeSize();
      common::Span<const uint32_t> offsets_span = {offsets, n_offsets};
      if (curent_bin_size == kUint8BinsTypeSize) {
          common::Span<uint8_t> index_data_span = {index.data<uint8_t>(), n_index};
          SetIndexDataForDense(index_data_span, batch_threads, batch, rbegin, offsets_span, nbins);
      } else if (curent_bin_size == kUint16BinsTypeSize) {
          common::Span<uint16_t> index_data_span = {index.data<uint16_t>(), n_index};
          SetIndexDataForDense(index_data_span, batch_threads, batch, rbegin, offsets_span, nbins);
      } else {
          CHECK_EQ(curent_bin_size, kUint32BinsTypeSize);
          common::Span<uint32_t> index_data_span = {index.data<uint32_t>(), n_index};
          SetIndexDataForDense(index_data_span, batch_threads, batch, rbegin, offsets_span, nbins);
      }

    /* For sparse DMatrix we have to store index of feature for each bin
       in index field to chose right offset. So offset is nullptr and index is not reduced */
    } else {
      common::Span<uint32_t> index_data_span = {index.data<uint32_t>(), n_index};
      SetIndexDataForSparse(index_data_span, batch_threads, batch, rbegin, nbins);
    }

    #pragma omp parallel for num_threads(nthread) schedule(static)
    for (bst_omp_uint idx = 0; idx < bst_omp_uint(nbins); ++idx) {
      for (int32_t tid = 0; tid < nthread; ++tid) {
        hit_count[idx] += hit_count_tloc_[tid * nbins + idx];
        hit_count_tloc_[tid * nbins + idx] = 0;  // reset for next batch
      }
    }

    prev_sum = row_ptr[rbegin + batch.Size()];
    rbegin += batch.Size();
  }
}

template <typename BinIdxType>
static size_t GetConflictCount(const std::vector<bool>& mark,
                               const Column<BinIdxType>& column_input,
                               size_t max_cnt) {
  size_t ret = 0;
  if (column_input.GetType() == xgboost::common::kDenseColumn) {
    const DenseColumn<BinIdxType>& column
      = static_cast<const DenseColumn<BinIdxType>& >(column_input);
    for (size_t i = 0; i < column.Size(); ++i) {
      if ((!column.IsMissing(i)) && mark[i]) {
        ++ret;
        if (ret > max_cnt) {
          return max_cnt + 1;
        }
      }
    }
  } else {
    const SparseColumn<BinIdxType>& column
      = static_cast<const SparseColumn<BinIdxType>& >(column_input);
    for (size_t i = 0; i < column.Size(); ++i) {
      if (mark[column.GetRowIdx(i)]) {
        ++ret;
        if (ret > max_cnt) {
          return max_cnt + 1;
        }
      }
    }
  }
  return ret;
}

template <typename BinIdxType>
inline void
MarkUsed(std::vector<bool>* p_mark, const Column<BinIdxType>& column_input) {
  std::vector<bool>& mark = *p_mark;
  if (column_input.GetType() == xgboost::common::kDenseColumn) {
    const DenseColumn<BinIdxType>& column
      = static_cast<const DenseColumn<BinIdxType>& >(column_input);
    for (size_t i = 0; i < column.Size(); ++i) {
      if (!column.IsMissing(i)) {
        mark[i] = true;
      }
    }
  } else {
    const SparseColumn<BinIdxType>& column
      = static_cast<const SparseColumn<BinIdxType>& >(column_input);
    for (size_t i = 0; i < column.Size(); ++i) {
      mark[column.GetRowIdx(i)] = true;
    }
  }
}

template <typename BinIdxType>
inline void SetGroup(const unsigned fid, const Column<BinIdxType>& column,
  const size_t max_conflict_cnt, const std::vector<size_t>& search_groups,
  std::vector<size_t>* p_group_conflict_cnt,
  std::vector<std::vector<bool>>* p_conflict_marks,
  std::vector<std::vector<unsigned>>* p_groups,
  std::vector<size_t>* p_group_nnz, const size_t cur_fid_nnz, const size_t nrow) {
  bool need_new_group = true;
  std::vector<size_t>& group_conflict_cnt = *p_group_conflict_cnt;
  std::vector<std::vector<bool>>& conflict_marks = *p_conflict_marks;
  std::vector<std::vector<unsigned>>& groups = *p_groups;
  std::vector<size_t>& group_nnz = *p_group_nnz;

  // examine each candidate group: is it okay to insert fid?
  for (auto gid : search_groups) {
    const size_t rest_max_cnt = max_conflict_cnt - group_conflict_cnt[gid];
    const size_t cnt = GetConflictCount(conflict_marks[gid], column, rest_max_cnt);
    if (cnt <= rest_max_cnt) {
      need_new_group = false;
      groups[gid].push_back(fid);
      group_conflict_cnt[gid] += cnt;
      group_nnz[gid] += cur_fid_nnz - cnt;
      MarkUsed(&conflict_marks[gid], column);
      break;
    }
  }
  // create new group if necessary
  if (need_new_group) {
    groups.emplace_back();
    groups.back().push_back(fid);
    group_conflict_cnt.push_back(0);
    conflict_marks.emplace_back(nrow, false);
    MarkUsed(&conflict_marks.back(), column);
    group_nnz.emplace_back(cur_fid_nnz);
  }
}

inline std::vector<std::vector<unsigned>>
FindGroups(const std::vector<unsigned>& feature_list,
           const std::vector<size_t>& feature_nnz,
           const ColumnMatrix& colmat,
           size_t nrow,
           const tree::TrainParam& param) {
  /* Goal: Bundle features together that has little or no "overlap", i.e.
           only a few data points should have nonzero values for
           member features.
           Note that one-hot encoded features will be grouped together. */

  std::vector<std::vector<unsigned>> groups;
  std::vector<std::vector<bool>> conflict_marks;
  std::vector<size_t> group_nnz;
  std::vector<size_t> group_conflict_cnt;
  const auto max_conflict_cnt
    = static_cast<size_t>(param.max_conflict_rate * nrow);

  for (auto fid : feature_list) {
    const size_t cur_fid_nnz = feature_nnz[fid];

    // randomly choose some of existing groups as candidates
    std::vector<size_t> search_groups;
    for (size_t gid = 0; gid < groups.size(); ++gid) {
      if (group_nnz[gid] + cur_fid_nnz <= nrow + max_conflict_cnt) {
        search_groups.push_back(gid);
      }
    }
    std::shuffle(search_groups.begin(), search_groups.end(), common::GlobalRandom());
    if (param.max_search_group > 0 && search_groups.size() > param.max_search_group) {
      search_groups.resize(param.max_search_group);
    }

    BinTypeSize bins_type_size = colmat.GetTypeSize();
    if (bins_type_size == kUint8BinsTypeSize) {
        const auto column = colmat.GetColumn<uint8_t>(fid);
        SetGroup(fid, *(column.get()), max_conflict_cnt, search_groups,
                 &group_conflict_cnt, &conflict_marks, &groups, &group_nnz, cur_fid_nnz, nrow);
    } else if (bins_type_size == kUint16BinsTypeSize) {
        const auto column = colmat.GetColumn<uint16_t>(fid);
        SetGroup(fid, *(column.get()), max_conflict_cnt, search_groups,
                 &group_conflict_cnt, &conflict_marks, &groups, &group_nnz, cur_fid_nnz, nrow);
    } else {
        CHECK_EQ(bins_type_size, kUint32BinsTypeSize);
        const auto column = colmat.GetColumn<uint32_t>(fid);
        SetGroup(fid, *(column.get()), max_conflict_cnt, search_groups,
                 &group_conflict_cnt, &conflict_marks, &groups, &group_nnz, cur_fid_nnz, nrow);
    }
  }
  return groups;
}

inline std::vector<std::vector<unsigned>>
FastFeatureGrouping(const GHistIndexMatrix& gmat,
                    const ColumnMatrix& colmat,
                    const tree::TrainParam& param) {
  const size_t nrow = gmat.row_ptr.size() - 1;
  const size_t nfeature = gmat.cut.Ptrs().size() - 1;

  std::vector<unsigned> feature_list(nfeature);
  std::iota(feature_list.begin(), feature_list.end(), 0);

  // sort features by nonzero counts, descending order
  std::vector<size_t> feature_nnz(nfeature);
  std::vector<unsigned> features_by_nnz(feature_list);
  gmat.GetFeatureCounts(&feature_nnz[0]);
  std::sort(features_by_nnz.begin(), features_by_nnz.end(),
            [&feature_nnz](unsigned a, unsigned b) {
    return feature_nnz[a] > feature_nnz[b];
  });

  auto groups_alt1 = FindGroups(feature_list, feature_nnz, colmat, nrow, param);
  auto groups_alt2 = FindGroups(features_by_nnz, feature_nnz, colmat, nrow, param);
  auto& groups = (groups_alt1.size() > groups_alt2.size()) ? groups_alt2 : groups_alt1;

  // take apart small, sparse groups, as it won't help speed
  {
    std::vector<std::vector<unsigned>> ret;
    for (const auto& group : groups) {
      if (group.size() <= 1 || group.size() >= 5) {
        ret.push_back(group);  // keep singleton groups and large (5+) groups
      } else {
        size_t nnz = 0;
        for (auto fid : group) {
          nnz += feature_nnz[fid];
        }
        double nnz_rate = static_cast<double>(nnz) / nrow;
        // take apart small sparse group, due it will not gain on speed
        if (nnz_rate <= param.sparse_threshold) {
          for (auto fid : group) {
            ret.emplace_back();
            ret.back().push_back(fid);
          }
        } else {
          ret.push_back(group);
        }
      }
    }
    groups = std::move(ret);
  }

  // shuffle groups
  std::shuffle(groups.begin(), groups.end(), common::GlobalRandom());

  return groups;
}

void GHistIndexBlockMatrix::Init(const GHistIndexMatrix& gmat,
                                 const ColumnMatrix& colmat,
                                 const tree::TrainParam& param) {
  cut_ = &gmat.cut;

  const size_t nrow = gmat.row_ptr.size() - 1;
  const uint32_t nbins = gmat.cut.Ptrs().back();

  /* step 1: form feature groups */
  auto groups = FastFeatureGrouping(gmat, colmat, param);
  const auto nblock = static_cast<uint32_t>(groups.size());

  /* step 2: build a new CSR matrix for each feature group */
  std::vector<uint32_t> bin2block(nbins);  // lookup table [bin id] => [block id]
  for (uint32_t group_id = 0; group_id < nblock; ++group_id) {
    for (auto& fid : groups[group_id]) {
      const uint32_t bin_begin = gmat.cut.Ptrs()[fid];
      const uint32_t bin_end = gmat.cut.Ptrs()[fid + 1];
      for (uint32_t bin_id = bin_begin; bin_id < bin_end; ++bin_id) {
        bin2block[bin_id] = group_id;
      }
    }
  }

  std::vector<std::vector<uint32_t>> index_temp(nblock);
  std::vector<std::vector<size_t>> row_ptr_temp(nblock);
  for (uint32_t block_id = 0; block_id < nblock; ++block_id) {
    row_ptr_temp[block_id].push_back(0);
  }
  for (size_t rid = 0; rid < nrow; ++rid) {
    const size_t ibegin = gmat.row_ptr[rid];
    const size_t iend = gmat.row_ptr[rid + 1];
    for (size_t j = ibegin; j < iend; ++j) {
      const uint32_t bin_id = gmat.index[j];
      const uint32_t block_id = bin2block[bin_id];
      index_temp[block_id].push_back(bin_id);
    }
    for (uint32_t block_id = 0; block_id < nblock; ++block_id) {
      row_ptr_temp[block_id].push_back(index_temp[block_id].size());
    }
  }

  /* step 3: concatenate CSR matrices into one (index, row_ptr) pair */
  std::vector<size_t> index_blk_ptr;
  std::vector<size_t> row_ptr_blk_ptr;
  index_blk_ptr.push_back(0);
  row_ptr_blk_ptr.push_back(0);
  for (uint32_t block_id = 0; block_id < nblock; ++block_id) {
    index_.insert(index_.end(), index_temp[block_id].begin(), index_temp[block_id].end());
    row_ptr_.insert(row_ptr_.end(), row_ptr_temp[block_id].begin(), row_ptr_temp[block_id].end());
    index_blk_ptr.push_back(index_.size());
    row_ptr_blk_ptr.push_back(row_ptr_.size());
  }

  // save shortcut for each block
  for (uint32_t block_id = 0; block_id < nblock; ++block_id) {
    Block blk;
    blk.index_begin = &index_[index_blk_ptr[block_id]];
    blk.row_ptr_begin = &row_ptr_[row_ptr_blk_ptr[block_id]];
    blk.index_end = &index_[index_blk_ptr[block_id + 1]];
    blk.row_ptr_end = &row_ptr_[row_ptr_blk_ptr[block_id + 1]];
    blocks_.push_back(blk);
  }
}

/*!
 * \brief fill a histogram by zeros in range [begin, end)
 */
template<typename GradientSumT>
void InitilizeHistByZeroes(GHistRow<GradientSumT> hist, size_t begin, size_t end) {
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  std::fill(hist.begin() + begin, hist.begin() + end,
            xgboost::detail::GradientPairInternal<GradientSumT>());
#else  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  memset(hist.data() + begin, '\0', (end-begin)*
         sizeof(xgboost::detail::GradientPairInternal<GradientSumT>));
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
}
template void InitilizeHistByZeroes(GHistRow<float> hist, size_t begin,
                                    size_t end);
template void InitilizeHistByZeroes(GHistRow<double> hist, size_t begin,
                                    size_t end);

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
template<typename GradientSumT>
void IncrementHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> add,
                   size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* padd = reinterpret_cast<const GradientSumT*>(add.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] += padd[i];
  }
}
template void IncrementHist(GHistRow<float> dst, const GHistRow<float> add,
                            size_t begin, size_t end);
template void IncrementHist(GHistRow<double> dst, const GHistRow<double> add,
                            size_t begin, size_t end);

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
template<typename GradientSumT>
void CopyHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> src,
              size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* psrc = reinterpret_cast<const GradientSumT*>(src.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc[i];
  }
}
template void CopyHist(GHistRow<float> dst, const GHistRow<float> src,
                       size_t begin, size_t end);
template void CopyHist(GHistRow<double> dst, const GHistRow<double> src,
                       size_t begin, size_t end);

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
template<typename GradientSumT>
void SubtractionHist(GHistRow<GradientSumT> dst, const GHistRow<GradientSumT> src1,
                     const GHistRow<GradientSumT> src2,
                     size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.data());
  const GradientSumT* psrc1 = reinterpret_cast<const GradientSumT*>(src1.data());
  const GradientSumT* psrc2 = reinterpret_cast<const GradientSumT*>(src2.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc1[i] - psrc2[i];
  }
}
template void SubtractionHist(GHistRow<float> dst, const GHistRow<float> src1,
                              const GHistRow<float> src2,
                              size_t begin, size_t end);
template void SubtractionHist(GHistRow<double> dst, const GHistRow<double> src1,
                              const GHistRow<double> src2,
                              size_t begin, size_t end);

struct Prefetch {
 public:
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kPrefetchOffset = 10;

 private:
  static constexpr size_t kNoPrefetchSize =
      kPrefetchOffset + kCacheLineSize /
      sizeof(decltype(GHistIndexMatrix::row_ptr)::value_type);

 public:
  static size_t NoPrefetchSize(size_t rows) {
    return std::min(rows, kNoPrefetchSize);
  }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return Prefetch::kCacheLineSize / sizeof(T);
  }
};

constexpr size_t Prefetch::kNoPrefetchSize;


template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistDenseKernel(const std::vector<GradientPair>& gpair,
                          const RowSetCollection::Elem row_indices,
                          const GHistIndexMatrix& gmat,
                          const size_t n_features,
                          GHistRow<FPType> hist) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  FPType* hist_data = reinterpret_cast<FPType*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start = rid[i] * n_features;
    const size_t idx_gh = two * rid[i];

    if (do_prefetch) {
      const size_t icol_start_prefetch = rid[i + Prefetch::kPrefetchOffset] * n_features;

      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features;
           j += Prefetch::GetPrefetchStep<BinIdxType>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    for (size_t j = 0; j < n_features; ++j) {
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
                                      offsets[j]);

      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
}

template<typename FPType, bool do_prefetch>
void BuildHistSparseKernel(const std::vector<GradientPair>& gpair,
                           const RowSetCollection::Elem row_indices,
                           const GHistIndexMatrix& gmat,
                           GHistRow<FPType> hist) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const uint32_t* gradient_index = gmat.index.data<uint32_t>();
  const size_t* row_ptr =  gmat.row_ptr.data();
  FPType* hist_data = reinterpret_cast<FPType*>(hist.data());
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array

  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start = row_ptr[rid[i]];
    const size_t icol_end = row_ptr[rid[i]+1];
    const size_t idx_gh = two * rid[i];

    if (do_prefetch) {
      const size_t icol_start_prftch = row_ptr[rid[i+Prefetch::kPrefetchOffset]];
      const size_t icol_end_prefect = row_ptr[rid[i+Prefetch::kPrefetchOffset]+1];

      PREFETCH_READ_T0(pgh + two * rid[i + Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prftch; j < icol_end_prefect;
        j+=Prefetch::GetPrefetchStep<uint32_t>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    }
    for (size_t j = icol_start; j < icol_end; ++j) {
      const uint32_t idx_bin = two * gradient_index[j];
      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
}


template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistDispatchKernel(const std::vector<GradientPair>& gpair,
                     const RowSetCollection::Elem row_indices,
                     const GHistIndexMatrix& gmat, GHistRow<FPType> hist, bool isDense) {
  if (isDense) {
    const size_t* row_ptr =  gmat.row_ptr.data();
    const size_t n_features = row_ptr[row_indices.begin[0]+1] - row_ptr[row_indices.begin[0]];
    BuildHistDenseKernel<FPType, do_prefetch, BinIdxType>(gpair, row_indices,
                                                       gmat, n_features, hist);
  } else {
    BuildHistSparseKernel<FPType, do_prefetch>(gpair, row_indices,
                                                        gmat, hist);
  }
}

template<typename FPType, bool do_prefetch>
void BuildHistKernel(const std::vector<GradientPair>& gpair,
                     const RowSetCollection::Elem row_indices,
                     const GHistIndexMatrix& gmat, const bool isDense, GHistRow<FPType> hist) {
  const bool is_dense = row_indices.Size() && isDense;
  switch (gmat.index.GetBinTypeSize()) {
    case kUint8BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint8_t>(gpair, row_indices,
                                                            gmat, hist, is_dense);
      break;
    case kUint16BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint16_t>(gpair, row_indices,
                                                             gmat, hist, is_dense);
      break;
    case kUint32BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint32_t>(gpair, row_indices,
                                                             gmat, hist, is_dense);
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

template<typename GradientSumT>
void GHistBuilder<GradientSumT>::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRowT hist,
                             bool isDense) {
  const size_t nrows = row_indices.Size();
  const size_t no_prefetch_size = Prefetch::NoPrefetchSize(nrows);

  // if need to work with all rows from bin-matrix (e.g. root node)
  const bool contiguousBlock = (row_indices.begin[nrows - 1] - row_indices.begin[0]) == (nrows - 1);

  if (contiguousBlock) {
    // contiguous memory access, built-in HW prefetching is enough
    BuildHistKernel<GradientSumT, false>(gpair, row_indices, gmat, isDense, hist);
  } else {
    const RowSetCollection::Elem span1(row_indices.begin, row_indices.end - no_prefetch_size);
    const RowSetCollection::Elem span2(row_indices.end - no_prefetch_size, row_indices.end);

    BuildHistKernel<GradientSumT, true>(gpair, span1, gmat, isDense, hist);
    // no prefetching to avoid loading extra memory
    BuildHistKernel<GradientSumT, false>(gpair, span2, gmat, isDense, hist);
  }
}
template
void GHistBuilder<float>::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRow<float> hist,
                             bool isDense);
template
void GHistBuilder<double>::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRow<double> hist,
                             bool isDense);

template<typename GradientSumT>
void GHistBuilder<GradientSumT>::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const RowSetCollection::Elem row_indices,
                                  const GHistIndexBlockMatrix& gmatb,
                                  GHistRowT hist) {
  constexpr int kUnroll = 8;  // loop unrolling factor
  const size_t nblock = gmatb.GetNumBlock();
  const size_t nrows = row_indices.end - row_indices.begin;
  const size_t rest = nrows % kUnroll;
#if defined(_OPENMP)
  const auto nthread = static_cast<bst_omp_uint>(this->nthread_);  // NOLINT
#endif  // defined(_OPENMP)
  xgboost::detail::GradientPairInternal<GradientSumT>* p_hist = hist.data();

#pragma omp parallel for num_threads(nthread) schedule(guided)
  for (bst_omp_uint bid = 0; bid < nblock; ++bid) {
    auto gmat = gmatb[bid];

    for (size_t i = 0; i < nrows - rest; i += kUnroll) {
      size_t rid[kUnroll];
      size_t ibegin[kUnroll];
      size_t iend[kUnroll];
      GradientPair stat[kUnroll];

      for (int k = 0; k < kUnroll; ++k) {
        rid[k] = row_indices.begin[i + k];
        ibegin[k] = gmat.row_ptr[rid[k]];
        iend[k] = gmat.row_ptr[rid[k] + 1];
        stat[k] = gpair[rid[k]];
      }
      for (int k = 0; k < kUnroll; ++k) {
        for (size_t j = ibegin[k]; j < iend[k]; ++j) {
          const uint32_t bin = gmat.index[j];
          p_hist[bin].Add(stat[k].GetGrad(), stat[k].GetHess());
        }
      }
    }
    for (size_t i = nrows - rest; i < nrows; ++i) {
      const size_t rid = row_indices.begin[i];
      const size_t ibegin = gmat.row_ptr[rid];
      const size_t iend = gmat.row_ptr[rid + 1];
      const GradientPair stat = gpair[rid];
      for (size_t j = ibegin; j < iend; ++j) {
        const uint32_t bin = gmat.index[j];
        p_hist[bin].Add(stat.GetGrad(), stat.GetHess());
      }
    }
  }
}
template
void GHistBuilder<float>::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const RowSetCollection::Elem row_indices,
                                  const GHistIndexBlockMatrix& gmatb,
                                  GHistRow<float> hist);
template
void GHistBuilder<double>::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const RowSetCollection::Elem row_indices,
                                  const GHistIndexBlockMatrix& gmatb,
                                  GHistRow<double> hist);


template<typename GradientSumT>
void GHistBuilder<GradientSumT>::SubtractionTrick(GHistRowT self,
                                                  GHistRowT sibling,
                                                  GHistRowT parent) {
  const size_t size = self.size();
  CHECK_EQ(sibling.size(), size);
  CHECK_EQ(parent.size(), size);

  const size_t block_size = 1024;  // aproximatly 1024 values per block
  size_t n_blocks = size/block_size + !!(size%block_size);

#pragma omp parallel for
  for (omp_ulong iblock = 0; iblock < n_blocks; ++iblock) {
    const size_t ibegin = iblock*block_size;
    const size_t iend = (((iblock+1)*block_size > size) ? size : ibegin + block_size);
    SubtractionHist(self, parent, sibling, ibegin, iend);
  }
}
template
void GHistBuilder<float>::SubtractionTrick(GHistRow<float> self,
                                           GHistRow<float> sibling,
                                           GHistRow<float> parent);
template
void GHistBuilder<double>::SubtractionTrick(GHistRow<double> self,
                                            GHistRow<double> sibling,
                                            GHistRow<double> parent);

}  // namespace common
}  // namespace xgboost
