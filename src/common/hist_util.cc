/*!
 * Copyright 2017-2019 by Contributors
 * \file hist_util.cc
 */
#include <dmlc/timer.h>
#include <dmlc/omp.h>

#include <rabit/rabit.h>
#include <numeric>
#include <vector>

#include "xgboost/base.h"
#include "../common/common.h"
#include "./hist_util.h"
#include "./random.h"
#include "./column_matrix.h"
#include "./quantile.h"
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

HistogramCuts::HistogramCuts() {
  monitor_.Init(__FUNCTION__);
  cut_ptrs_.emplace_back(0);
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
  LOG(INFO) << "Total number of hist bins: " << cut_ptrs_.back();
}

bool CutsBuilder::UseGroup(DMatrix* dmat) {
  auto& info = dmat->Info();
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
  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;
  CHECK_GE(end_col, beg_col);
  constexpr float kFactor = 8;

  // Data groups, used in ranking.
  std::vector<bst_uint> const& group_ptr = info.group_ptr_;
  p_cuts_->min_vals_.resize(end_col - beg_col, 0);

  for (uint32_t col_id = beg_col; col_id < page.Size() && col_id < end_col; ++col_id) {
    // Using a local variable makes things easier, but at the cost of memory trashing.
    WXQSketch sketch;
    common::Span<xgboost::Entry const> const column = page[col_id];
    uint32_t const n_bins = std::min(static_cast<uint32_t>(column.size()),
                                     max_num_bins);
    if (n_bins == 0) {
      // cut_ptrs_ is initialized with a zero, so there's always an element at the back
      p_cuts_->cut_ptrs_.emplace_back(p_cuts_->cut_ptrs_.back());
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

    WXQSketch::SummaryContainer out_summary;
    sketch.GetSummary(&out_summary);
    WXQSketch::SummaryContainer summary;
    summary.Reserve(n_bins);
    summary.SetPrune(out_summary, n_bins);

    // Can be use data[1] as the min values so that we don't need to
    // store another array?
    float mval = summary.data[0].value;
    p_cuts_->min_vals_[col_id - beg_col]  = mval - (fabs(mval) + 1e-5);

    this->AddCutPoint(summary);

    bst_float cpt = (summary.size > 0) ?
                    summary.data[summary.size - 1].value :
                    p_cuts_->min_vals_[col_id - beg_col];
    cpt += fabs(cpt) + 1e-5;
    p_cuts_->cut_values_.emplace_back(cpt);

    p_cuts_->cut_ptrs_.emplace_back(p_cuts_->cut_values_.size());
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
  p_cuts_->min_vals_.resize(n_cols, std::numeric_limits<float>::max());
  size_t min_vals_tail = 0;

  for (uint32_t t = 0; t < nthreads; ++t) {
    // concat csc pointers.
    size_t const old_ptr_size = p_cuts_->cut_ptrs_.size();
    p_cuts_->cut_ptrs_.resize(
        cuts[t]->p_cuts_->cut_ptrs_.size() + p_cuts_->cut_ptrs_.size() - 1);
    size_t const new_icp_size = p_cuts_->cut_ptrs_.size();
    auto tail = p_cuts_->cut_ptrs_[old_ptr_size-1];
    for (size_t j = old_ptr_size; j < new_icp_size; ++j) {
      p_cuts_->cut_ptrs_[j] = tail + cuts[t]->p_cuts_->cut_ptrs_[j-old_ptr_size+1];
    }
    // concat csc values
    size_t const old_iv_size = p_cuts_->cut_values_.size();
    p_cuts_->cut_values_.resize(
        cuts[t]->p_cuts_->cut_values_.size() + p_cuts_->cut_values_.size());
    size_t const new_iv_size = p_cuts_->cut_values_.size();
    for (size_t j = old_iv_size; j < new_iv_size; ++j) {
      p_cuts_->cut_values_[j] = cuts[t]->p_cuts_->cut_values_[j-old_iv_size];
    }
    // merge min values
    for (size_t j = 0; j < cuts[t]->p_cuts_->min_vals_.size(); ++j) {
      p_cuts_->min_vals_.at(min_vals_tail + j) =
          std::min(p_cuts_->min_vals_.at(min_vals_tail + j), cuts.at(t)->p_cuts_->min_vals_.at(j));
    }
    min_vals_tail += cuts[t]->p_cuts_->min_vals_.size();
  }
  monitor_.Stop(__FUNCTION__);
}

void DenseCuts::Build(DMatrix* p_fmat, uint32_t max_num_bins) {
  monitor_.Start(__FUNCTION__);
  const MetaInfo& info = p_fmat->Info();

  // safe factor for better accuracy
  constexpr int kFactor = 8;
  std::vector<WXQSketch> sketchs;

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
          for (auto const& entry : inst) {
            if (entry.index >= begin && entry.index < end) {
              size_t w_idx = use_group ? group_ind : ridx;
              sketchs[entry.index].Push(entry.fvalue, info.GetWeight(w_idx));
            }
          }
        }
      }
    }
  }

  Init(&sketchs, max_num_bins);
  monitor_.Stop(__FUNCTION__);
}

void DenseCuts::Init
(std::vector<WXQSketch>* in_sketchs, uint32_t max_num_bins) {
  monitor_.Start(__func__);
  std::vector<WXQSketch>& sketchs = *in_sketchs;
  constexpr int kFactor = 8;
  // gather the histogram data
  rabit::SerializeReducer<WXQSketch::SummaryContainer> sreducer;
  std::vector<WXQSketch::SummaryContainer> summary_array;
  summary_array.resize(sketchs.size());
  for (size_t i = 0; i < sketchs.size(); ++i) {
    WXQSketch::SummaryContainer out;
    sketchs[i].GetSummary(&out);
    summary_array[i].Reserve(max_num_bins * kFactor);
    summary_array[i].SetPrune(out, max_num_bins * kFactor);
  }
  CHECK_EQ(summary_array.size(), in_sketchs->size());
  size_t nbytes = WXQSketch::SummaryContainer::CalcMemCost(max_num_bins * kFactor);
  // TODO(chenqin): rabit failure recovery assumes no boostrap onetime call after loadcheckpoint
  // we need to move this allreduce before loadcheckpoint call in future
  sreducer.Allreduce(dmlc::BeginPtr(summary_array), nbytes, summary_array.size());
  p_cuts_->min_vals_.resize(sketchs.size());

  for (size_t fid = 0; fid < summary_array.size(); ++fid) {
    WXQSketch::SummaryContainer a;
    a.Reserve(max_num_bins);
    a.SetPrune(summary_array[fid], max_num_bins);
    const bst_float mval = a.data[0].value;
    p_cuts_->min_vals_[fid] = mval - (fabs(mval) + 1e-5);
    AddCutPoint(a);
    // push a value that is greater than anything
    const bst_float cpt
      = (a.size > 0) ? a.data[a.size - 1].value : p_cuts_->min_vals_[fid];
    // this must be bigger than last value in a scale
    const bst_float last = cpt + (fabs(cpt) + 1e-5);
    p_cuts_->cut_values_.push_back(last);

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(p_cuts_->cut_values_.size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(p_cuts_->cut_values_.size());
    CHECK_GT(cut_size, p_cuts_->cut_ptrs_.back());
    p_cuts_->cut_ptrs_.push_back(cut_size);
  }
  monitor_.Stop(__func__);
}

void GHistIndexMatrix::Init(DMatrix* p_fmat, int max_num_bins) {
  cut.Build(p_fmat, max_num_bins);
  const int32_t nthread = omp_get_max_threads();
  const uint32_t nbins = cut.Ptrs().back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(nthread * nbins, 0);


  size_t new_size = 1;
  for (const auto &batch : p_fmat->GetBatches<SparsePage>()) {
    new_size += batch.Size();
  }

  row_ptr.resize(new_size);
  row_ptr[0] = 0;

  size_t rbegin = 0;
  size_t prev_sum = 0;

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

    index.resize(row_ptr[rbegin + batch.Size()]);

    CHECK_GT(cut.Values().size(), 0U);

    #pragma omp parallel for num_threads(batch_threads) schedule(static)
    for (omp_ulong i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      const int tid = omp_get_thread_num();
      size_t ibegin = row_ptr[rbegin + i];
      size_t iend = row_ptr[rbegin + i + 1];
      SparsePage::Inst inst = batch[i];

      CHECK_EQ(ibegin + inst.size(), iend);
      for (bst_uint j = 0; j < inst.size(); ++j) {
        uint32_t idx = cut.SearchBin(inst[j]);

        index[ibegin + j] = idx;
        ++hit_count_tloc_[tid * nbins + idx];
      }
      std::sort(index.begin() + ibegin, index.begin() + iend);
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

static size_t GetConflictCount(const std::vector<bool>& mark,
                               const Column& column,
                               size_t max_cnt) {
  size_t ret = 0;
  if (column.GetType() == xgboost::common::kDenseColumn) {
    for (size_t i = 0; i < column.Size(); ++i) {
      if (column.GetFeatureBinIdx(i) != std::numeric_limits<uint32_t>::max() && mark[i]) {
        ++ret;
        if (ret > max_cnt) {
          return max_cnt + 1;
        }
      }
    }
  } else {
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

inline void
MarkUsed(std::vector<bool>* p_mark, const Column& column) {
  std::vector<bool>& mark = *p_mark;
  if (column.GetType() == xgboost::common::kDenseColumn) {
    for (size_t i = 0; i < column.Size(); ++i) {
      if (column.GetFeatureBinIdx(i) != std::numeric_limits<uint32_t>::max()) {
        mark[i] = true;
      }
    }
  } else {
    for (size_t i = 0; i < column.Size(); ++i) {
      mark[column.GetRowIdx(i)] = true;
    }
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
    const Column& column = colmat.GetColumn(fid);

    const size_t cur_fid_nnz = feature_nnz[fid];
    bool need_new_group = true;

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
 * \brief fill a histogram by zeroes
 */
void InitilizeHistByZeroes(GHistRow hist, size_t begin, size_t end) {
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  std::fill(hist.begin() + begin, hist.begin() + end, tree::GradStats());
#else  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  memset(hist.data() + begin, '\0', (end-begin)*sizeof(tree::GradStats));
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
}

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
void IncrementHist(GHistRow dst, const GHistRow add, size_t begin, size_t end) {
  using FPType = decltype(tree::GradStats::sum_grad);
  FPType* pdst = reinterpret_cast<FPType*>(dst.data());
  const FPType* padd = reinterpret_cast<const FPType*>(add.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] += padd[i];
  }
}

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
void CopyHist(GHistRow dst, const GHistRow src, size_t begin, size_t end) {
  using FPType = decltype(tree::GradStats::sum_grad);
  FPType* pdst = reinterpret_cast<FPType*>(dst.data());
  const FPType* psrc = reinterpret_cast<const FPType*>(src.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc[i];
  }
}

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
void SubtractionHist(GHistRow dst, const GHistRow src1, const GHistRow src2,
                     size_t begin, size_t end) {
  using FPType = decltype(tree::GradStats::sum_grad);
  FPType* pdst = reinterpret_cast<FPType*>(dst.data());
  const FPType* psrc1 = reinterpret_cast<const FPType*>(src1.data());
  const FPType* psrc2 = reinterpret_cast<const FPType*>(src2.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc1[i] - psrc2[i];
  }
}


void GHistBuilder::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRow hist) {
  const size_t* rid =  row_indices.begin;
  const size_t nrows = row_indices.Size();
  const uint32_t* index = gmat.index.data();
  const size_t* row_ptr =  gmat.row_ptr.data();
  const float* pgh = reinterpret_cast<const float*>(gpair.data());

  double* hist_data = reinterpret_cast<double*>(hist.data());

  const size_t cache_line_size = 64;
  const size_t prefetch_offset = 10;
  size_t no_prefetch_size = prefetch_offset + cache_line_size/sizeof(*rid);
  no_prefetch_size = no_prefetch_size > nrows ? nrows : no_prefetch_size;

  for (size_t i = 0; i < nrows; ++i) {
    const size_t icol_start = row_ptr[rid[i]];
    const size_t icol_end = row_ptr[rid[i]+1];

    if (i < nrows - no_prefetch_size) {
      PREFETCH_READ_T0(row_ptr + rid[i + prefetch_offset]);
      PREFETCH_READ_T0(pgh + 2*rid[i + prefetch_offset]);
    }

    for (size_t j = icol_start; j < icol_end; ++j) {
      const uint32_t idx_bin = 2*index[j];
      const size_t idx_gh = 2*rid[i];

      hist_data[idx_bin] += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
}

void GHistBuilder::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const RowSetCollection::Elem row_indices,
                                  const GHistIndexBlockMatrix& gmatb,
                                  GHistRow hist) {
  constexpr int kUnroll = 8;  // loop unrolling factor
  const size_t nblock = gmatb.GetNumBlock();
  const size_t nrows = row_indices.end - row_indices.begin;
  const size_t rest = nrows % kUnroll;

#if defined(_OPENMP)
  const auto nthread = static_cast<bst_omp_uint>(this->nthread_);  // NOLINT
#endif  // defined(_OPENMP)
  tree::GradStats* p_hist = hist.data();

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
          p_hist[bin].Add(stat[k]);
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
        p_hist[bin].Add(stat);
      }
    }
  }
}

void GHistBuilder::SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent) {
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

}  // namespace common
}  // namespace xgboost
