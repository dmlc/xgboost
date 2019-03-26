/*!
 * Copyright 2017-2019 by Contributors
 * \file hist_util.h
 */
#include <rabit/rabit.h>
#include <dmlc/omp.h>
#include <numeric>
#include <vector>
#include <dmlc/timer.h>
#include "./random.h"
#include "./column_matrix.h"
#include "./hist_util.h"
#include "./quantile.h"
#include "./../tree/updater_quantile_hist.h"
#include <xmmintrin.h>

#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
  #define PREFETCH_READ_T1(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T1)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
  #define PREFETCH_READ_T1(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 2)
  #define PREFETCH_READ_NTA(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 0)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

namespace xgboost {
namespace common {
template class GHistBuilder<tree::QuantileHistMaker::Builder::HistTLS>;

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

  auto t1 = dmlc::GetTime();

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

  printf("use_group_ind = %d\n", use_group_ind);

  if (use_group_ind)
  {
    for (const auto &batch : p_fmat->GetRowBatches()) {
      size_t group_ind = this->SearchGroupIndFromBaseRow(group_ptr, batch.base_rowid);
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
            if (group_ptr[group_ind] == ridx &&
                // maximum equals to weights.size() - 1
                group_ind < num_groups - 1) {
              // move to next group
              group_ind++;
            }
            for (auto const& entry : inst) {
              if (entry.index >= begin && entry.index < end) {
                size_t w_idx = group_ind;
                sketchs[entry.index].Push(entry.fvalue, info.GetWeight(w_idx));
              }
            }
          }
        }
      }
    }
  } else {
    for (const auto &batch : p_fmat->GetRowBatches()) {
      size_t group_ind = 0;

      const size_t size = batch.Size();
      const size_t block_size = 1024;
      const size_t block_size_iter = block_size * nthread;
      const size_t n_blocks = size / block_size_iter + !!(size % block_size_iter);

      std::vector<std::vector<std::pair<float, float>>> buff(nthread);
      for(size_t tid = 0; tid < nthread; ++tid) {
        buff[tid].resize(block_size * ncol);
      }

      std::vector<size_t> sizes(nthread * ncol);

      for (size_t iblock = 0; iblock < n_blocks; ++iblock) {
        #pragma omp prallel num_threads(nthread)
        {
          int tid = omp_get_num_threads();

          const size_t ibegin = iblock * block_size_iter + tid * block_size;
          const size_t iend = std::min(ibegin + block_size, size);

          auto* p_sizes = sizes.data() + ncol * tid;
          auto* p_buff = buff[tid].data();

          for(size_t i = ibegin; i < iend; ++i) {
            size_t const ridx = batch.base_rowid + i;
            bst_float w = info.GetWeight(ridx);
            SparsePage::Inst const inst = batch[i];

            for (auto const& entry : inst) {
              const size_t idx = entry.index;
              p_buff[idx * block_size + p_sizes[idx]] = { entry.fvalue, w };
              p_sizes[idx]++;
            }
          }
          #pragma omp barrier
          #pragma omp for schedule(static)
          for(size_t icol = 0; icol < ncol; ++icol) {
            for(size_t tid = 0; tid < nthread; ++tid) {
              auto* p_sizes = sizes.data() + ncol * tid;
              auto* p_buff = buff[tid].data() + icol * block_size;

              for(size_t i = 0; i < p_sizes[icol]; ++i) {
                sketchs[icol].Push(p_buff[i].first,  p_buff[i].second);
              }

              p_sizes[icol] = 0;
            }
          }
        }
      }
    }
  }

  auto t2 = dmlc::GetTime();


  Init(&sketchs, max_num_bins);
  monitor_.Stop("Init");

  auto t3 = dmlc::GetTime();
  printf("HistCutMatrix::Init1 = %f\n", (t2-t1)*1000);
  printf("HistCutMatrix::Init2 = %f\n", (t3-t2)*1000);
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
  auto t1 = dmlc::GetTime();
  cut.Init(p_fmat, max_num_bins);
  const int nthread = omp_get_max_threads();
  // const int nthread = 1;
  const uint32_t nbins = cut.row_ptr.back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(nthread * nbins, 0);


  printf("GHistIndexMatrix::p_fmat->GetRowBatches() = %zu\n", p_fmat->GetRowBatches());

  size_t new_size = 1;
  for (const auto &batch : p_fmat->GetRowBatches()) {
    new_size += batch.Size();
  }

  row_ptr.resize(new_size);
  row_ptr[0] = 0;

  size_t rbegin = 0;
  size_t prev_sum = 0;

  for (const auto &batch : p_fmat->GetRowBatches()) {

    auto tt1 = dmlc::GetTime();

    MemStackAllocator<size_t, 128> partial_sums(nthread);
    size_t* p_part = partial_sums.Get();

    size_t block_size =  batch.Size() / nthread;

    #pragma omp parallel num_threads(nthread)
    {
      #pragma omp for
      for(size_t tid = 0; tid < nthread; ++tid)
      {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (nthread-1) ? batch.Size() : (block_size * (tid+1)));

        size_t sum = 0;
        for (size_t i = ibegin; i < iend; ++i) {
          sum += batch[i].size();
          row_ptr[rbegin + 1 + i] = sum;
        }
      }

      #pragma omp single
      {
        p_part[0] = prev_sum;
        for (size_t i = 1; i < nthread; ++i)
          p_part[i] = p_part[i - 1] + row_ptr[rbegin + i*block_size];
      }

      #pragma omp for
      for(size_t tid = 0; tid < nthread; ++tid) {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (nthread-1) ? batch.Size() : (block_size * (tid+1)));

        for (size_t i = ibegin; i < iend; ++i)
          row_ptr[rbegin + 1 + i] += p_part[tid];
      }
    }

    auto tt1_1 = dmlc::GetTime();

    index.resize(row_ptr.back());
    auto tt2 = dmlc::GetTime();

    printf("GHistIndexMatrix::PUSH_BACK = %f\n", (tt2-tt1)*1000);
    printf("GHistIndexMatrix::PUSH_BACK_ONLY = %f\n", (tt1_1-tt1)*1000);

    CHECK_GT(cut.cut.size(), 0U);

    auto bsize = static_cast<omp_ulong>(batch.Size());
    #pragma omp parallel for num_threads(nthread) schedule(static)
    for (omp_ulong i = 0; i < bsize; ++i) { // NOLINT(*)
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

    auto tt3 = dmlc::GetTime();
    printf("GHistIndexMatrix::PARALLEL_LOOP = %f\n", (tt3-tt2)*1000);
    #pragma omp parallel for num_threads(nthread) schedule(static)
    for (bst_omp_uint idx = 0; idx < bst_omp_uint(nbins); ++idx) {
      for (int tid = 0; tid < nthread; ++tid) {
        hit_count[idx] += hit_count_tloc_[tid * nbins + idx];
      }
    }
    auto tt4 = dmlc::GetTime();
    printf("GHistIndexMatrix::REDUCTION = %f\n", (tt4-tt3)*1000);

    prev_sum = row_ptr[rbegin + batch.Size()];
    rbegin += batch.Size();
  }
  auto t2 = dmlc::GetTime();
  printf("GHistIndexMatrix::GHistIndexMatrix = %f\n", (t2-t1)*1000);

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

template<typename TlsType>
void GHistBuilder<TlsType>::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             GHistRow hist,
                             TlsType& hist_tls,
                             common::ColumnSampler& column_sampler,
                             tree::RegTreeThreadSafe& tree,
                             int32_t parent_nid,
                             const tree::TrainParam& param,
                             GHistRow sibling,
                             GHistRow parent,
                             int32_t this_nid,
                             int32_t another_nid,
                             const bool is_dense_layout,
                             const bool sync_hist) {
  static float prep = 0, histcomp = 0, reduce = 0;

  auto t1  = dmlc::GetTime();

  const size_t nthread = static_cast<size_t>(this->nthread_);

  const size_t* rid =  row_indices.begin;
  const size_t nrows = row_indices.Size();
  const uint32_t* index = gmat.index.data();
  const size_t* row_ptr =  gmat.row_ptr.data();
  const float* pgh = reinterpret_cast<const float*>(gpair.data());

  float* hist_data = reinterpret_cast<float*>(hist.data());

  constexpr size_t block_size = 2048;
  constexpr size_t n_stack_size = 128;
  constexpr size_t cache_line_size = 64;
  constexpr size_t prefetch_offset = 10;

  size_t n_blocks = nrows/block_size;
  n_blocks += !!(nrows - n_blocks*block_size);

  const size_t nthread_to_process = std::min(nthread,  n_blocks);

  MemStackAllocator<uint32_t, n_stack_size> thread_init_alloc(nthread);
  MemStackAllocator<std::pair<tree::GradStats*, size_t>, n_stack_size> hist_local_alloc(nthread);
  MemStackAllocator<float, n_stack_size> thread_gh_sum_alloc(nthread*2);
  auto* p_thread_init = thread_init_alloc.Get();
  auto* p_hist_local = hist_local_alloc.Get();
  auto* p_thread_gh_sum = thread_gh_sum_alloc.Get();

  memset(p_thread_init, '\0', nthread*sizeof(p_thread_init[0]));
  memset(p_hist_local, '\0', nthread*sizeof(p_hist_local[0]));
  memset(p_thread_gh_sum, '\0', 2*nthread*sizeof(p_thread_gh_sum[0]));

  size_t no_prefetch_size = prefetch_offset + cache_line_size/sizeof(*rid);
  no_prefetch_size = no_prefetch_size > nrows ? nrows : no_prefetch_size;

  auto t2  = dmlc::GetTime();

  if (is_dense_layout) {
    ParallelFor(n_blocks, [&](size_t iblock) {
      dmlc::omp_uint tid = omp_get_thread_num();

      bool prev = p_thread_init[tid];
      if (!p_thread_init[tid]) {
        p_thread_init[tid] = true;
        p_hist_local[tid] = hist_tls.get(tid);
      }
      float* data_local_hist = ((nthread_to_process == 1) ? hist_data :
              reinterpret_cast<float*>(p_hist_local[tid].first));

      if (!prev)
        memset(data_local_hist, '\0', 2*nbins_*sizeof(float));

      const size_t istart = iblock*block_size;
      const size_t iend = (((iblock+1)*block_size > nrows) ? nrows : istart + block_size);

      const size_t n_features = row_ptr[rid[istart]+1] - row_ptr[rid[istart]];

      float gh_sum[2] = {0, 0};

      __m128 adds;
      float* addsPtr = (float*) (&adds);
      addsPtr[2] = 0;
      addsPtr[3] = 0;

      if (iend < nrows - no_prefetch_size) {
        for (size_t i = istart; i < iend; ++i) {
          const size_t icol_start = rid[i] * n_features;
          const size_t icol_start_prefetch = rid[i+prefetch_offset] * n_features;
          const size_t idx_gh = 2*rid[i];

          PREFETCH_READ_T0(pgh + 2*rid[i + prefetch_offset]);

          for (size_t j = icol_start_prefetch; j < icol_start_prefetch + n_features; j+=16)
            PREFETCH_READ_T0(index + j);

          gh_sum[0] += pgh[idx_gh];
          gh_sum[1] += pgh[idx_gh+1];


          addsPtr[0] = pgh[idx_gh];
          addsPtr[1] = pgh[idx_gh+1];

          for (size_t j = icol_start; j < icol_start + n_features; ++j) {
            const uint32_t idx_bin = 2*index[j];
            // data_local_hist[idx_bin] += pgh[idx_gh];
            // data_local_hist[idx_bin+1] += pgh[idx_gh+1];

            __m128 hist1    = _mm_loadu_ps(data_local_hist + idx_bin);
            __m128 newHist1 = _mm_add_ps(adds, hist1);
            _mm_storeu_ps(data_local_hist + idx_bin, newHist1);

          }
        }
      }
      else {
        for (size_t i = istart; i < iend; ++i) {
          const size_t icol_start = rid[i] * n_features;
          const size_t idx_gh = 2*rid[i];

          gh_sum[0] += pgh[idx_gh];
          gh_sum[1] += pgh[idx_gh+1];

          for (size_t j = icol_start; j < icol_start + n_features; ++j) {
            const uint32_t idx_bin      = 2*index[j];
            data_local_hist[idx_bin]   += pgh[idx_gh];
            data_local_hist[idx_bin+1] += pgh[idx_gh+1];
          }
        }
      }
      p_thread_gh_sum[tid*2] += gh_sum[0];
      p_thread_gh_sum[tid*2+1] += gh_sum[1];
    });
  } else { // Sparse case
    ParallelFor(n_blocks, [&](size_t iblock) {
      dmlc::omp_uint tid = omp_get_thread_num();

      bool prev = p_thread_init[tid];
      if (!p_thread_init[tid]) {
        p_thread_init[tid] = true;
        p_hist_local[tid] = hist_tls.get(tid);
      }
      float* data_local_hist = ((nthread_to_process == 1) ? hist_data :
              reinterpret_cast<float*>(p_hist_local[tid].first));

      if (!prev) {
        memset(data_local_hist, '\0', 2*nbins_*sizeof(float));
      }

      const size_t istart = iblock*block_size;
      const size_t iend = (((iblock+1)*block_size > nrows) ? nrows : istart + block_size);

      float gh_sum[2] = {0, 0};

      __m128 adds;
      float* addsPtr = (float*) (&adds);
      addsPtr[2] = 0;
      addsPtr[3] = 0;

      if (iend < nrows - no_prefetch_size) {
        for (size_t i = istart; i < iend; ++i) {
          const size_t icol_start = row_ptr[rid[i]];
          const size_t icol_end = row_ptr[rid[i]+1];
          const size_t idx_gh = 2*rid[i];

          // const size_t icol_start10 = row_ptr[rid[i+prefetch_offset]];
          // const size_t icol_end10 = row_ptr[rid[i+prefetch_offset]+1];

          // const size_t icol_start15 = row_ptr[rid[i+prefetch_offset+5]];
          // const size_t icol_end15 = row_ptr[rid[i+prefetch_offset+5]+1];

          // const size_t icol_start10 = row_ptr[rid[i+2*prefetch_offset]];
          // const size_t icol_end10 = row_ptr[rid[i+2*prefetch_offset]+1];

          // PREFETCH_READ_T1(row_ptr + rid[i + 2*prefetch_offset]);
          // PREFETCH_READ_T1(pgh + 2*rid[i + 2*prefetch_offset]);

          PREFETCH_READ_T0(row_ptr + rid[i + prefetch_offset]);
          PREFETCH_READ_T0(pgh + 2*rid[i + prefetch_offset]);

          // PREFETCH_READ_NTA(row_ptr + rid[i + prefetch_offset]);
          // PREFETCH_READ_NTA(pgh + 2*rid[i + prefetch_offset]);

          // for (size_t j = icol_start10; j < icol_end10; j+=16){
          //   PREFETCH_READ_T0(index + j);
          // }
          // for (size_t j = icol_start15; j < icol_end15; j+=16){
          //   PREFETCH_READ_T1(index + j);
          // }

          gh_sum[0] += pgh[idx_gh];
          gh_sum[1] += pgh[idx_gh+1];

          addsPtr[0] = pgh[idx_gh];
          addsPtr[1] = pgh[idx_gh+1];

          for (size_t j = icol_start; j < icol_end; ++j) {
            const uint32_t idx_bin = 2*index[j];
            // data_local_hist[idx_bin] += pgh[idx_gh];
            // data_local_hist[idx_bin+1] += pgh[idx_gh+1];

            __m128 hist1    = _mm_loadu_ps(data_local_hist + idx_bin);
            __m128 newHist1 = _mm_add_ps(adds, hist1);
            _mm_storeu_ps(data_local_hist + idx_bin, newHist1);

          }
        }
      }
      else {
        for (size_t i = istart; i < iend; ++i) {
          const size_t icol_start = row_ptr[rid[i]];
          const size_t icol_end = row_ptr[rid[i]+1];
          const size_t idx_gh = 2*rid[i];

          gh_sum[0] += pgh[idx_gh];
          gh_sum[1] += pgh[idx_gh+1];

          for (size_t j = icol_start; j < icol_end; ++j) {
            const uint32_t idx_bin      = 2*index[j];
            data_local_hist[idx_bin]   += pgh[idx_gh];
            data_local_hist[idx_bin+1] += pgh[idx_gh+1];
          }
        }
      }
      p_thread_gh_sum[tid*2] += gh_sum[0];
      p_thread_gh_sum[tid*2+1] += gh_sum[1];
    });
  }

  auto t3  = dmlc::GetTime();


  {
    size_t n_worked_bins = 0;

    if (nthread_to_process == 0) {
      memset(hist_data, '\0', 2*nbins_*sizeof(float));
    } else if (nthread_to_process > 1) {
      for (size_t i = 0; i < nthread; ++i) {
        if (p_thread_init[i]) {
          std::swap(p_hist_local[n_worked_bins], p_hist_local[i]);
          n_worked_bins++;
        }
      }
    }

    if(!sync_hist) {
      float gh_sum[2] = {0, 0};
      for (size_t i = 0; i < nthread; ++i) {
        gh_sum[0] += p_thread_gh_sum[2 * i];
        gh_sum[1] += p_thread_gh_sum[2 * i + 1];
      }

      tree::GradStats grad_st(gh_sum[0], gh_sum[1]);
      tree.Snode(this_nid).stats = grad_st;

      if (another_nid > -1) {
        auto& st = tree.Snode(parent_nid).stats;
        tree.Snode(another_nid).stats.SetSubstract(st, grad_st);
      }
    }

    const size_t size = (2*nbins_);
    const size_t block_size = 1024; // aproximatly 1024 values per block
    size_t n_blocks = size/block_size + !!(size%block_size);

    ParallelFor(n_blocks, [&](size_t iblock) {
      const size_t ibegin = iblock*block_size;
      const size_t iend = (((iblock+1)*block_size > size) ? size : ibegin + block_size);

      if (nthread_to_process > 1) {
        memcpy(hist_data + ibegin, (((float*)p_hist_local[0].first) + ibegin),
            sizeof(float)*(iend - ibegin));
        for (size_t i_bin_part = 1; i_bin_part < n_worked_bins; ++i_bin_part) {
          float* ptr = reinterpret_cast<float*>(p_hist_local[i_bin_part].first);
          for (int32_t i = ibegin; i < iend; i++) {
            hist_data[i] += ptr[i];
          }
        }
      }

      int32_t n_local_bins = (iend - ibegin)/2;

      const size_t tid = omp_get_thread_num();

      if (another_nid > -1) {
        float* other = (float*)sibling.data();
        float* par = (float*)parent.data();

        for (int32_t i = ibegin; i < iend; i++) {
          other[i] = par[i] - hist_data[i];
        }
      }
    });
  }

  for (uint32_t i = 0; i < nthread; i++) {
    if (p_hist_local[i].first) {
      hist_tls.release(p_hist_local[i]);
    }
  }


  auto t4  = dmlc::GetTime();

  prep += t2-t1;
  histcomp +=t3-t2;
  reduce += t4-t3;

  float total = prep + histcomp + reduce;

  if (this_nid == 0)
    printf("%f %f %f | %f %f %f\n", prep*1000, histcomp*1000, reduce*1000, prep/total*100, histcomp/total*100, reduce/total*100);
}

template<typename TlsType>
void GHistBuilder<TlsType>::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const RowSetCollection::Elem row_indices,
                                  const GHistIndexBlockMatrix& gmatb,
                                  GHistRow hist) {
  constexpr int kUnroll = 8;  // loop unrolling factor
  const size_t nblock = gmatb.GetNumBlock();
  const size_t nrows = row_indices.end - row_indices.begin;
  const size_t rest = nrows % kUnroll;

  ParallelFor(nblock, [&](size_t bid) {
    auto gmat = gmatb[bid];

    for (size_t i = 0; i < nrows - rest; i += kUnroll) {
      size_t rid[kUnroll];
      size_t ibegin[kUnroll];
      size_t iend[kUnroll];
      GradientPair stat[kUnroll];
      for (int k = 0; k < kUnroll; ++k) {
        rid[k] = row_indices.begin[i + k];
      }
      for (int k = 0; k < kUnroll; ++k) {
        ibegin[k] = gmat.row_ptr[rid[k]];
        iend[k] = gmat.row_ptr[rid[k] + 1];
      }
      for (int k = 0; k < kUnroll; ++k) {
        stat[k] = gpair[rid[k]];
      }
      for (int k = 0; k < kUnroll; ++k) {
        for (size_t j = ibegin[k]; j < iend[k]; ++j) {
          const uint32_t bin = gmat.index[j];
          hist[bin].Add(stat[k]);
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
        hist[bin].Add(stat);
      }
    }
  });
}

template<typename TlsType>
void GHistBuilder<TlsType>::SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent) {
  const uint32_t nbins = static_cast<bst_omp_uint>(nbins_);

  tree::GradStats* p_self = self.data();
  tree::GradStats* p_sibling = sibling.data();
  tree::GradStats* p_parent = parent.data();

  const size_t size = (2*nbins_);
  const size_t block_size = 1024; // aproximatly 1024 values per block
  size_t n_blocks = size/block_size + !!(size%block_size);

  ParallelFor(n_blocks, [&](size_t iblock) {
    const size_t ibegin = iblock*block_size;
    const size_t iend = (((iblock+1)*block_size > size) ? size : ibegin + block_size);
    for (bst_omp_uint bin_id = ibegin; bin_id < iend; bin_id++) {
      p_self[bin_id].SetSubstract(p_parent[bin_id], p_sibling[bin_id]);
    }
  });
}

}  // namespace common
}  // namespace xgboost
