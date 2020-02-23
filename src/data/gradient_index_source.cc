/*!
 * Copyright 2017-2020 by XGBoost Contributors
 */
#include <cstddef>
#include <ios>
#include <limits>

#include "gradient_index_source.h"
#include "xgboost/base.h"

namespace xgboost {

namespace common {
void ColumnMatrix::Init(const GradientIndexPage &gmat, double sparse_threshold) {
  const int32_t nfeature = static_cast<int32_t>(gmat.cut.Ptrs().size() - 1);
  const size_t nrow = gmat.row_ptr.size() - 1;

  // identify type of each column
  feature_counts_.resize(nfeature);
  type_.resize(nfeature);
  std::fill(feature_counts_.begin(), feature_counts_.end(), 0);

  uint32_t max_val = std::numeric_limits<uint32_t>::max();
  for (int32_t fid = 0; fid < nfeature; ++fid) {
    CHECK_LE(gmat.cut.Ptrs()[fid + 1] - gmat.cut.Ptrs()[fid], max_val);
  }

  gmat.GetFeatureCounts(&feature_counts_[0]);
  // classify features
  for (int32_t fid = 0; fid < nfeature; ++fid) {
    if (static_cast<double>(feature_counts_[fid]) < sparse_threshold * nrow) {
      type_[fid] = kSparseColumn;
    } else {
      type_[fid] = kDenseColumn;
    }
  }

  // want to compute storage boundary for each feature
  // using variants of prefix sum scan
  boundary_.resize(nfeature);
  size_t accum_index_ = 0;
  size_t accum_row_ind_ = 0;
  for (int32_t fid = 0; fid < nfeature; ++fid) {
    boundary_[fid].index_begin = accum_index_;
    boundary_[fid].row_ind_begin = accum_row_ind_;
    if (type_[fid] == kDenseColumn) {
      accum_index_ += static_cast<size_t>(nrow);
      accum_row_ind_ += static_cast<size_t>(nrow);
    } else {
      accum_index_ += feature_counts_[fid];
      accum_row_ind_ += feature_counts_[fid];
    }
    boundary_[fid].index_end = accum_index_;
    boundary_[fid].row_ind_end = accum_row_ind_;
  }

  index_.resize(boundary_[nfeature - 1].index_end);
  row_ind_.resize(boundary_[nfeature - 1].row_ind_end);

  // store least bin id for each feature
  index_base_.resize(nfeature);
  for (int32_t fid = 0; fid < nfeature; ++fid) {
    index_base_[fid] = gmat.cut.Ptrs()[fid];
  }

  // pre-fill index_ for dense columns

#pragma omp parallel for
  for (int32_t fid = 0; fid < nfeature; ++fid) {
    if (type_[fid] == kDenseColumn) {
      const size_t ibegin = boundary_[fid].index_begin;
      uint32_t *begin = &index_[ibegin];
      uint32_t *end = begin + nrow;
      std::fill(begin, end, std::numeric_limits<uint32_t>::max());
      // max() indicates missing values
    }
  }

  // loop over all rows and fill column entries
  // num_nonzeros[fid] = how many nonzeros have this feature accumulated so far?
  std::vector<size_t> num_nonzeros;
  num_nonzeros.resize(nfeature);
  std::fill(num_nonzeros.begin(), num_nonzeros.end(), 0);
  for (size_t rid = 0; rid < nrow; ++rid) {
    const size_t ibegin = gmat.row_ptr[rid];
    const size_t iend = gmat.row_ptr[rid + 1];
    size_t fid = 0;
    for (size_t i = ibegin; i < iend; ++i) {
      const uint32_t bin_id = gmat.index[i];
      auto iter = std::upper_bound(gmat.cut.Ptrs().cbegin() + fid,
                                   gmat.cut.Ptrs().cend(), bin_id);
      fid = std::distance(gmat.cut.Ptrs().cbegin(), iter) - 1;
      if (type_[fid] == kDenseColumn) {
        uint32_t *begin = &index_[boundary_[fid].index_begin];
        begin[rid] = bin_id - index_base_[fid];
      } else {
        uint32_t *begin = &index_[boundary_[fid].index_begin];
        begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
        row_ind_[boundary_[fid].row_ind_begin + num_nonzeros[fid]] = rid;
        ++num_nonzeros[fid];
      }
    }
  }
}
}  // namespace common

void GradientIndexPage::Init(DMatrix *p_fmat, int max_num_bins) {
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
    std::vector<size_t> partial_sums(batch_threads);
    size_t *p_part = partial_sums.data();

    size_t block_size = batch.Size() / batch_threads;

#pragma omp parallel num_threads(batch_threads)
    {
#pragma omp for
      for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads - 1) ? batch.Size()
                                                  : (block_size * (tid + 1)));

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
          p_part[i] = p_part[i - 1] + row_ptr[rbegin + i * block_size];
        }
      }

#pragma omp for
      for (omp_ulong tid = 0; tid < batch_threads; ++tid) {
        size_t ibegin = block_size * tid;
        size_t iend = (tid == (batch_threads - 1) ? batch.Size()
                                                  : (block_size * (tid + 1)));

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

template <typename Iter, typename Value>
Iter BinarySearchRange(Iter input_1, Iter input_2, Value const &r_1,
                       Value const &r_2) {
  static_assert(std::is_same<typename std::iterator_traits<Iter>::value_type,
                             Value>::value,
                "");
  auto beg = input_1, end = input_2;
  auto previous_middle = input_2;
  while (end != beg) {
    Iter mid = beg + (end - beg) / 2;
    if (mid == previous_middle) {
      break;
    }
    previous_middle = mid;
    if (r_1 <= *mid && *mid < r_2) {
      return mid;
    }
    if (*mid < r_1) {
      beg = mid;
    } else {
      end = mid;
    }
    CHECK(beg <= end);
  }

  return input_2;
}

float GradientIndexPage::GetFvalue(bst_row_t ridx, bst_feature_t fid) const {
  auto lower = cut.Ptrs()[fid];
  CHECK_NE(column_matrix.GetNumFeature(), 0);
  auto column = column_matrix.GetColumn(fid);
  auto row_id = column.GetRowIdx(ridx);
  if (column.GetType() == common::kDenseColumn) {
    CHECK_EQ(row_id, ridx);
  }
  auto bin = column.GetGlobalBinIdx(row_id);
  if (column.IsMissing(row_id)) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  common::Span<float const> cuts{cut.Values()};
  CHECK_GE(bin, lower);
  if (XGBOOST_EXPECT(bin == lower, false)) {
    CHECK_EQ(column.GetFeatureBinIdx(row_id), 0);
    return cut.MinValues().at(fid);
  } else {
    return cuts[bin-1];
  }
}

GradientIndexSource::GradientIndexSource(DMatrix* m, const BatchParam& param) {
  CHECK_GE(param.max_bin, 2);
  gradient_index_.Init(m, param.max_bin);
}
}  // namespace xgboost
