/*!
 * Copyright 2017 by Contributors
 * \file hist_util.h
 * \brief Utilities to store histograms
 * \author Philip Cho, Tianqi Chen
 */
#include <dmlc/omp.h>
#include <numeric>
#include <vector>
#include "./sync.h"
#include "./random.h"
#include "./column_matrix.h"
#include "./hist_util.h"
#include "./quantile.h"

namespace xgboost {
namespace common {

void HistCutMatrix::Init(DMatrix* p_fmat, uint32_t max_num_bins) {
  using WXQSketch = common::WXQuantileSketch<bst_float, bst_float>;
  const MetaInfo& info = p_fmat->Info();

  // safe factor for better accuracy
  constexpr int kFactor = 8;
  std::vector<WXQSketch> sketchs;

  const int nthread = omp_get_max_threads();

  auto nstep = static_cast<unsigned>((info.num_col_ + nthread - 1) / nthread);
  auto ncol = static_cast<unsigned>(info.num_col_);
  sketchs.resize(info.num_col_);
  for (auto& s : sketchs) {
    s.Init(info.num_row_, 1.0 / (max_num_bins * kFactor));
  }

  auto iter = p_fmat->RowIterator();
  iter->BeforeFirst();
  while (iter->Next()) {
     auto &batch = iter->Value();
    #pragma omp parallel num_threads(nthread)
    {
      CHECK_EQ(nthread, omp_get_num_threads());
      auto tid = static_cast<unsigned>(omp_get_thread_num());
      unsigned begin = std::min(nstep * tid, ncol);
      unsigned end = std::min(nstep * (tid + 1), ncol);
      // do not iterate if no columns are assigned to the thread
      if (begin < end && end <= ncol) {
        for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
          size_t ridx = batch.base_rowid + i;
          SparsePage::Inst inst = batch[i];
          for (bst_uint j = 0; j < inst.length; ++j) {
            if (inst[j].index >= begin && inst[j].index < end) {
              sketchs[inst[j].index].Push(inst[j].fvalue, info.GetWeight(ridx));
            }
          }
        }
      }
    }
  }

  Init(&sketchs, max_num_bins);
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
    if (a.size != 0) {
      bst_float cpt = a.data[a.size - 1].value;
      // this must be bigger than last value in a scale
      bst_float last = cpt + (fabs(cpt) + 1e-5);
      cut.push_back(last);
    }

    row_ptr.push_back(static_cast<bst_uint>(cut.size()));
  }
}

uint32_t HistCutMatrix::GetBinIdx(const Entry& e) {
  unsigned fid = e.index;
  auto cbegin = cut.begin() + row_ptr[fid];
  auto cend = cut.begin() + row_ptr[fid + 1];
  CHECK(cbegin != cend);
  auto it = std::upper_bound(cbegin, cend, e.fvalue);
  if (it == cend) it = cend - 1;
  uint32_t idx = static_cast<uint32_t>(it - cut.begin());
  return idx;
}

void GHistIndexMatrix::Init(DMatrix* p_fmat, int max_num_bins) {
  cut.Init(p_fmat, max_num_bins);
  auto iter = p_fmat->RowIterator();

  const int nthread = omp_get_max_threads();
  const uint32_t nbins = cut.row_ptr.back();
  hit_count.resize(nbins, 0);
  hit_count_tloc_.resize(nthread * nbins, 0);

  iter->BeforeFirst();
  row_ptr.push_back(0);
  while (iter->Next()) {
     auto &batch = iter->Value();
    const size_t rbegin = row_ptr.size() - 1;
    for (size_t i = 0; i < batch.Size(); ++i) {
      row_ptr.push_back(batch[i].length + row_ptr.back());
    }
    index.resize(row_ptr.back());

    CHECK_GT(cut.cut.size(), 0U);
    CHECK_EQ(cut.row_ptr.back(), cut.cut.size());

    auto bsize = static_cast<omp_ulong>(batch.Size());
    #pragma omp parallel for num_threads(nthread) schedule(static)
    for (omp_ulong i = 0; i < bsize; ++i) { // NOLINT(*)
      const int tid = omp_get_thread_num();
      size_t ibegin = row_ptr[rbegin + i];
      size_t iend = row_ptr[rbegin + i + 1];
      SparsePage::Inst inst = batch[i];
      CHECK_EQ(ibegin + inst.length, iend);
      for (bst_uint j = 0; j < inst.length; ++j) {
        uint32_t idx = cut.GetBinIdx(inst[j]);
        index[ibegin + j] = idx;
        ++hit_count_tloc_[tid * nbins + idx];
      }
      std::sort(index.begin() + ibegin, index.begin() + iend);
    }

    #pragma omp parallel for num_threads(nthread) schedule(static)
    for (bst_omp_uint idx = 0; idx < bst_omp_uint(nbins); ++idx) {
      for (int tid = 0; tid < nthread; ++tid) {
        hit_count[idx] += hit_count_tloc_[tid * nbins + idx];
      }
    }
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
           const FastHistParam& param) {
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
                    const FastHistParam& param) {
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
                                 const FastHistParam& param) {
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

void GHistBuilder::BuildHist(const std::vector<GradientPair>& gpair,
                             const RowSetCollection::Elem row_indices,
                             const GHistIndexMatrix& gmat,
                             const std::vector<bst_uint>& feat_set,
                             GHistRow hist) {
  data_.resize(nbins_ * nthread_, GHistEntry());
  std::fill(data_.begin(), data_.end(), GHistEntry());

  constexpr int kUnroll = 8;  // loop unrolling factor
  const auto nthread = static_cast<bst_omp_uint>(this->nthread_);
  const size_t nrows = row_indices.end - row_indices.begin;
  const size_t rest = nrows % kUnroll;

  #pragma omp parallel for num_threads(nthread) schedule(guided)
  for (bst_omp_uint i = 0; i < nrows - rest; i += kUnroll) {
    const bst_omp_uint tid = omp_get_thread_num();
    const size_t off = tid * nbins_;
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
        data_[off + bin].Add(stat[k]);
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
      data_[bin].Add(stat);
    }
  }

  /* reduction */
  const uint32_t nbins = nbins_;
  #pragma omp parallel for num_threads(nthread) schedule(static)
  for (bst_omp_uint bin_id = 0; bin_id < bst_omp_uint(nbins); ++bin_id) {
    for (bst_omp_uint tid = 0; tid < nthread; ++tid) {
      hist.begin[bin_id].Add(data_[tid * nbins_ + bin_id]);
    }
  }
}

void GHistBuilder::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const RowSetCollection::Elem row_indices,
                                  const GHistIndexBlockMatrix& gmatb,
                                  const std::vector<bst_uint>& feat_set,
                                  GHistRow hist) {
  constexpr int kUnroll = 8;  // loop unrolling factor
  const size_t nblock = gmatb.GetNumBlock();
  const size_t nrows = row_indices.end - row_indices.begin;
  const size_t rest = nrows % kUnroll;

#if defined(_OPENMP)
  const auto nthread = static_cast<bst_omp_uint>(this->nthread_);
#endif

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
          hist.begin[bin].Add(stat[k]);
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
        hist.begin[bin].Add(stat);
      }
    }
  }
}

void GHistBuilder::SubtractionTrick(GHistRow self, GHistRow sibling, GHistRow parent) {
  const uint32_t nbins = static_cast<bst_omp_uint>(nbins_);
  constexpr int kUnroll = 8;  // loop unrolling factor
  const uint32_t rest = nbins % kUnroll;

#if defined(_OPENMP)
  const auto nthread = static_cast<bst_omp_uint>(this->nthread_);
#endif

  #pragma omp parallel for num_threads(nthread) schedule(static)
  for (bst_omp_uint bin_id = 0;
       bin_id < static_cast<bst_omp_uint>(nbins - rest); bin_id += kUnroll) {
    GHistEntry pb[kUnroll];
    GHistEntry sb[kUnroll];
    for (int k = 0; k < kUnroll; ++k) {
      pb[k] = parent.begin[bin_id + k];
    }
    for (int k = 0; k < kUnroll; ++k) {
      sb[k] = sibling.begin[bin_id + k];
    }
    for (int k = 0; k < kUnroll; ++k) {
      self.begin[bin_id + k].SetSubtract(pb[k], sb[k]);
    }
  }
  for (uint32_t bin_id = nbins - rest; bin_id < nbins; ++bin_id) {
    self.begin[bin_id].SetSubtract(parent.begin[bin_id], sibling.begin[bin_id]);
  }
}

}  // namespace common
}  // namespace xgboost
