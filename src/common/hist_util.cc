/*!
 * Copyright 2017-2019 by Contributors
  * \file hist_util.cc
 */
#include "./hist_util.h"
#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <dmlc/omp.h>
#include <numeric>
#include <vector>
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

HistCutMatrix::HistCutMatrix() {
  monitor_.Init("HistCutMatrix");
}

size_t HistCutMatrix::SearchGroupIndFromBaseRow(
    std::vector<bst_uint> const& group_ptr, size_t const base_rowid) const {
  using KIt = std::vector<bst_uint>::const_iterator;
  KIt res = std::lower_bound(group_ptr.cbegin(), group_ptr.cend() - 1, base_rowid);
  // Cannot use CHECK_NE because it will try to print the iterator.
  bool const found = res != group_ptr.cend() - 1;
  if (!found) {
    LOG(FATAL) << "Row " << base_rowid << " does not lie in any group!\n";
  }
  size_t group_ind = std::distance(group_ptr.cbegin(), res);
  return group_ind;
}

void HistCutMatrix::Init(DMatrix* p_fmat, uint32_t max_num_bins) {
  monitor_.Start("Init");
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

  const auto& weights = info.weights_.HostVector();

  // Data groups, used in ranking.
  std::vector<bst_uint> const& group_ptr = info.group_ptr_;
  size_t const num_groups = group_ptr.size() == 0 ? 0 : group_ptr.size() - 1;
  // Use group index for weights?
  bool const use_group_ind = num_groups != 0 && weights.size() != info.num_row_;

  for (const auto &batch : p_fmat->GetRowBatches()) {
    size_t group_ind = 0;
    if (use_group_ind) {
      group_ind = this->SearchGroupIndFromBaseRow(group_ptr, batch.base_rowid);
    }
#pragma omp parallel num_threads(nthread) firstprivate(group_ind, use_group_ind)
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
          if (use_group_ind &&
              group_ptr[group_ind] == ridx &&
              // maximum equals to weights.size() - 1
              group_ind < num_groups - 1) {
            // move to next group
            group_ind++;
          }
          for (auto const& entry : inst) {
            if (entry.index >= begin && entry.index < end) {
              size_t w_idx = use_group_ind ? group_ind : ridx;
              sketchs[entry.index].Push(entry.fvalue, info.GetWeight(w_idx));
            }
          }
        }
      }
    }
  }

  Init(&sketchs, max_num_bins);
  monitor_.Stop("Init");
}

void HistCutMatrix::Init
(std::vector<WXQSketch>* in_sketchs, uint32_t max_num_bins) {
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
  sreducer.Allreduce(dmlc::BeginPtr(summary_array), nbytes, summary_array.size());
  this->min_val.resize(sketchs.size());
  row_ptr.push_back(0);
  for (size_t fid = 0; fid < summary_array.size(); ++fid) {
    WXQSketch::SummaryContainer a;
    a.Reserve(max_num_bins);
    a.SetPrune(summary_array[fid], max_num_bins);
    const bst_float mval = a.data[0].value;
    this->min_val[fid] = mval - (fabs(mval) + 1e-5);
    if (a.size > 1 && a.size <= 16) {
      /* specialized code categorial / ordinal data -- use midpoints */
      for (size_t i = 1; i < a.size; ++i) {
        bst_float cpt = (a.data[i].value + a.data[i - 1].value) / 2.0f;
        if (i == 1 || cpt > cut.back()) {
          cut.push_back(cpt);
        }
      }
    } else {
      for (size_t i = 2; i < a.size; ++i) {
        bst_float cpt = a.data[i - 1].value;
        if (i == 2 || cpt > cut.back()) {
          cut.push_back(cpt);
        }
      }
    }
    // push a value that is greater than anything
    const bst_float cpt
      = (a.size > 0) ? a.data[a.size - 1].value : this->min_val[fid];
    // this must be bigger than last value in a scale
    const bst_float last = cpt + (fabs(cpt) + 1e-5);
    cut.push_back(last);

    // Ensure that every feature gets at least one quantile point
    CHECK_LE(cut.size(), std::numeric_limits<uint32_t>::max());
    auto cut_size = static_cast<uint32_t>(cut.size());
    CHECK_GT(cut_size, row_ptr.back());
    row_ptr.push_back(cut_size);
  }
}

uint32_t HistCutMatrix::GetBinIdx(const Entry& e) {
  unsigned fid = e.index;
  auto cbegin = cut.begin() + row_ptr[fid];
  auto cend = cut.begin() + row_ptr[fid + 1];
  CHECK(cbegin != cend);
  auto it = std::upper_bound(cbegin, cend, e.fvalue);
  if (it == cend) {
    it = cend - 1;
  }
  uint32_t idx = static_cast<uint32_t>(it - cut.begin());
  return idx;
}

void GHistIndexMatrix::Init(DMatrix* p_fmat, int max_num_bins) {
  cut.Init(p_fmat, max_num_bins);
  const int32_t nthread = omp_get_max_threads();
  const uint32_t nbins = cut.row_ptr.back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(nthread * nbins, 0);


  size_t new_size = 1;
  for (const auto &batch : p_fmat->GetRowBatches()) {
    new_size += batch.Size();
  }

  row_ptr.resize(new_size);
  row_ptr[0] = 0;

  size_t rbegin = 0;
  size_t prev_sum = 0;

  for (const auto &batch : p_fmat->GetRowBatches()) {
    // The number of threads is pegged to the batch size. If the OMP
    // block is parallelized on anything other than the batch/block size,
    // it should be reassigned
    const size_t batch_threads = std::min(batch.Size(), static_cast<size_t>(omp_get_max_threads()));
    MemStackAllocator<size_t, 128> partial_sums(batch_threads);
    size_t* p_part = partial_sums.Get();

    size_t block_size =  batch.Size() / batch_threads;

    #pragma omp parallel num_threads(batch_threads)
    {
      #pragma omp for
      for (int32_t tid = 0; tid < batch_threads; ++tid) {
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
        for (int32_t i = 1; i < batch_threads; ++i) {
          p_part[i] = p_part[i - 1] + row_ptr[rbegin + i*block_size];
        }
      }

      #pragma omp for
      for (int32_t tid = 0; tid < batch_threads; ++tid) {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads-1) ? batch.Size() : (block_size * (tid+1)));

        for (size_t i = ibegin; i < iend; ++i) {
          row_ptr[rbegin + 1 + i] += p_part[tid];
        }
      }
    }

    index.resize(row_ptr[rbegin + batch.Size()]);

    CHECK_GT(cut.cut.size(), 0U);

    #pragma omp parallel for num_threads(batch_threads) schedule(static)
    for (omp_ulong i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      const int tid = omp_get_thread_num();
      size_t ibegin = row_ptr[rbegin + i];
      size_t iend = row_ptr[rbegin + i + 1];
      SparsePage::Inst inst = batch[i];

      CHECK_EQ(ibegin + inst.size(), iend);
      for (bst_uint j = 0; j < inst.size(); ++j) {
        uint32_t idx = cut.GetBinIdx(inst[j]);

        index[ibegin + j] = idx;
        ++hit_count_tloc_[tid * nbins + idx];
      }
      std::sort(index.begin() + ibegin, index.begin() + iend);
    }

    #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int32_t idx = 0; idx < int32_t(nbins); ++idx) {
      for (int32_t tid = 0; tid < nthread; ++tid) {
        hit_count[idx] += hit_count_tloc_[tid * nbins + idx];
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
  const size_t nfeature = gmat.cut.row_ptr.size() - 1;

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
        float nnz_rate = static_cast<float>(nnz) / nrow;
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
  const uint32_t nbins = gmat.cut.row_ptr.back();

  /* step 1: form feature groups */
  auto groups = FastFeatureGrouping(gmat, colmat, param);
  const auto nblock = static_cast<uint32_t>(groups.size());

  /* step 2: build a new CSR matrix for each feature group */
  std::vector<uint32_t> bin2block(nbins);  // lookup table [bin id] => [block id]
  for (uint32_t group_id = 0; group_id < nblock; ++group_id) {
    for (auto& fid : groups[group_id]) {
      const uint32_t bin_begin = gmat.cut.row_ptr[fid];
      const uint32_t bin_end = gmat.cut.row_ptr[fid + 1];
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

// used when data layout is kDenseDataZeroBased or kDenseDataOneBased
// it means that "row_ptr" is not needed for hist computations
void BuildHistLocalDense(size_t istart, size_t iend, size_t nrows, const size_t* rid,
    const uint32_t* index, const GradientPair::ValueT* pgh, const size_t* row_ptr,
    GradStatHist::GradType* data_local_hist, GradStatHist* grad_stat_global) {
  GradStatHist grad_stat;  // make local var to prevent false sharing

  const size_t n_features = row_ptr[rid[istart]+1] - row_ptr[rid[istart]];
  const size_t cache_line_size = 64;
  const size_t prefetch_step = cache_line_size / sizeof(*index);
  const size_t prefetch_offset = 10;

  size_t no_prefetch_size = prefetch_offset + cache_line_size/sizeof(*rid);
  no_prefetch_size = no_prefetch_size > nrows ? nrows : no_prefetch_size;

  // if read each row in some block of bin-matrix - it's dense block
  // and we dont need SW prefetch in this case
  const bool denseBlock = (rid[iend-1] - rid[istart]) == (iend - istart - 1);

  if (iend < nrows - no_prefetch_size && !denseBlock) {
    for (size_t i = istart; i < iend; ++i) {
      const size_t icol_start = rid[i] * n_features;
      const size_t icol_start_prefetch = rid[i+prefetch_offset] * n_features;
      const size_t idx_gh = 2*rid[i];

      PREFETCH_READ_T0(pgh + 2*rid[i + prefetch_offset]);

      for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features;
          j += prefetch_step) {
        PREFETCH_READ_T0(index + j);
      }

      grad_stat.sum_grad += pgh[idx_gh];
      grad_stat.sum_hess += pgh[idx_gh+1];

      for (size_t j = icol_start; j < icol_start + n_features; ++j) {
        const uint32_t idx_bin = 2*index[j];
        data_local_hist[idx_bin] += pgh[idx_gh];
        data_local_hist[idx_bin+1] += pgh[idx_gh+1];
      }
    }
  } else {
    for (size_t i = istart; i < iend; ++i) {
      const size_t icol_start = rid[i] * n_features;
      const size_t idx_gh = 2*rid[i];
      grad_stat.sum_grad += pgh[idx_gh];
      grad_stat.sum_hess += pgh[idx_gh+1];

      for (size_t j = icol_start; j < icol_start + n_features; ++j) {
        const uint32_t idx_bin      = 2*index[j];
        data_local_hist[idx_bin]   += pgh[idx_gh];
        data_local_hist[idx_bin+1] += pgh[idx_gh+1];
      }
    }
  }
  grad_stat_global->Add(grad_stat);
}

// used when data layout is kSparseData
// it means that "row_ptr" is needed for hist computations
void BuildHistLocalSparse(size_t istart, size_t iend, size_t nrows, const size_t* rid,
    const uint32_t* index, const GradientPair::ValueT* pgh, const size_t* row_ptr,
    GradStatHist::GradType* data_local_hist, GradStatHist* grad_stat_global) {
  GradStatHist grad_stat;  // make local var to prevent false sharing

  const size_t cache_line_size = 64;
  const size_t prefetch_step = cache_line_size / sizeof(index[0]);
  const size_t prefetch_offset = 10;

  size_t no_prefetch_size = prefetch_offset + cache_line_size/sizeof(*rid);
  no_prefetch_size = no_prefetch_size > nrows ? nrows : no_prefetch_size;

  // if read each row in some block of bin-matrix - it's dense block
  // and we dont need SW prefetch in this case
  const bool denseBlock = (rid[iend-1] - rid[istart]) == (iend - istart);

  if (iend < nrows - no_prefetch_size && !denseBlock) {
    for (size_t i = istart; i < iend; ++i) {
      const size_t icol_start = row_ptr[rid[i]];
      const size_t icol_end = row_ptr[rid[i]+1];
      const size_t idx_gh = 2*rid[i];

      const size_t icol_start10 = row_ptr[rid[i+prefetch_offset]];
      const size_t icol_end10 = row_ptr[rid[i+prefetch_offset]+1];

      PREFETCH_READ_T0(pgh + 2*rid[i + prefetch_offset]);

      for (size_t j = icol_start10; j < icol_end10; j+=prefetch_step) {
        PREFETCH_READ_T0(index + j);
      }

      grad_stat.sum_grad += pgh[idx_gh];
      grad_stat.sum_hess += pgh[idx_gh+1];

      for (size_t j = icol_start; j < icol_end; ++j) {
        const uint32_t idx_bin      = 2*index[j];
        data_local_hist[idx_bin]   += pgh[idx_gh];
        data_local_hist[idx_bin+1] += pgh[idx_gh+1];
      }
    }
  } else {
    for (size_t i = istart; i < iend; ++i) {
      const size_t icol_start = row_ptr[rid[i]];
      const size_t icol_end = row_ptr[rid[i]+1];
      const size_t idx_gh = 2*rid[i];

      grad_stat.sum_grad += pgh[idx_gh];
      grad_stat.sum_hess += pgh[idx_gh+1];

      for (size_t j = icol_start; j < icol_end; ++j) {
        const uint32_t idx_bin      = 2*index[j];
        data_local_hist[idx_bin]   += pgh[idx_gh];
        data_local_hist[idx_bin+1] += pgh[idx_gh+1];
      }
    }
  }
  grad_stat_global->Add(grad_stat);
}

void SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent) {
  GradStatHist* p_self = self.data();
  GradStatHist* p_sibling = sibling.data();
  GradStatHist* p_parent = parent.data();

  const size_t size = self.size();
  CHECK_EQ(sibling.size(), size);
  CHECK_EQ(parent.size(), size);

  const size_t block_size = 1024;  // aproximatly 1024 values per block
  size_t n_blocks = size/block_size + !!(size%block_size);

  #pragma omp parallel for
  for (int iblock = 0; iblock < n_blocks; ++iblock) {
    const size_t ibegin = iblock*block_size;
    const size_t iend = (((iblock+1)*block_size > size) ? size : ibegin + block_size);
    for (bst_omp_uint bin_id = ibegin; bin_id < iend; bin_id++) {
      p_self[bin_id].SetSubstract(p_parent[bin_id], p_sibling[bin_id]);
    }
  }
}

}  // namespace common
}  // namespace xgboost
