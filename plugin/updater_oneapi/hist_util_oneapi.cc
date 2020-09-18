/*!
 * Copyright 2017-2020 by Contributors
 * \file hist_util_oneapi.cc
 */
#include <dmlc/timer.h>
#include <dmlc/omp.h>

#include <rabit/rabit.h>
#include <numeric>
#include <vector>

#include "xgboost/base.h"
#include "../../src/common/common.h"
#include "../../src/common/random.h"
#include "../../src/common/quantile.h"
#include "./../../src/tree/updater_quantile_hist.h"
#include "column_matrix_oneapi.h"
#include "hist_util_oneapi.h"

#include "CL/sycl.hpp"

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

uint32_t SearchBin(const bst_float* cut_values, const uint32_t* cut_ptrs, float value, uint32_t column_id) {
  auto beg = cut_ptrs[column_id];
  auto end = cut_ptrs[column_id + 1];
  const auto &values = cut_values;
  auto it = std::upper_bound(cut_values + beg, cut_values + end, value);
  uint32_t idx = it - cut_values;
  if (idx == end) {
    idx -= 1;
  }
  return idx;
}

uint32_t SearchBin(const bst_float* cut_values, const uint32_t* cut_ptrs, EntryOneAPI const& e) {
  return SearchBin(cut_values, cut_ptrs, e.fvalue, e.index);
}

template <typename BinIdxType>
void GHistIndexMatrixOneAPI::SetIndexData(cl::sycl::queue qu, common::Span<BinIdxType> index_data_span,
                  const DeviceMatrixOneAPI &dmat_device,
                  size_t nbins, uint32_t* offsets) {
  const xgboost::EntryOneAPI *data_ptr = dmat_device.data.DataConst();
  const bst_row_t *offset_vec = dmat_device.row_ptr.DataConst();
  const size_t num_rows = dmat_device.row_ptr.Size() - 1;
  BinIdxType* index_data = index_data_span.data();
  const bst_float* cut_values = cut_device.Values().DataConst();
  const uint32_t* cut_ptrs = cut_device.Ptrs().DataConst();
//  int threads = qu.get_device().get_info<cl::sycl::info::device::max_compute_units>();
  LOG(INFO) << "SetIndexData, BinIdxType = " << sizeof(BinIdxType);
  LOG(INFO) << "SetIndexData, data_ptr = " << data_ptr << ", offset_vec = " << offset_vec << ", index_data = " << (uint32_t*)index_data << ", cut_values = " << cut_values << ", cut_ptrs = " << cut_ptrs;
/*  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<1>(num_rows), [=](cl::sycl::item<1> pid) {
      size_t i = pid.get_id(0);*/
  for (size_t i = 0; i < num_rows; i++) {
      size_t ibegin = offset_vec[i];
      size_t iend = offset_vec[i + 1];
      const size_t size = offset_vec[i + 1] - offset_vec[i];
      if (i < 5) {
        LOG(INFO) << "SetIndexData, nbins = " << nbins << ", i = " << i << ", range = " << ibegin << " " << iend << " " << size;
      }
      for (bst_uint j = 0; j < size; ++j) {
        uint32_t idx = SearchBin(cut_values, cut_ptrs, data_ptr[offset_vec[i] + j]);
        index_data[ibegin + j] = offsets ? idx - offsets[j] : idx;
      }
    }/*);
  }).wait();*/

  for (size_t i = 0; i < num_rows; i++) {
      size_t ibegin = offset_vec[i];
      size_t iend = offset_vec[i + 1];
      const size_t size = offset_vec[i + 1] - offset_vec[i];
      for (bst_uint j = 0; j < size; ++j) {
        uint32_t idx = offsets ? (uint32_t)index_data[ibegin + j] + offsets[j] : index_data[ibegin + j];
        ++hit_count[idx];
      }
  }
}

void GHistIndexMatrixOneAPI::ResizeIndex(const size_t n_offsets, const size_t n_index,
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

void GHistIndexMatrixOneAPI::Init(cl::sycl::queue qu, DMatrix* p_fmat, int max_bins) {
  LOG(INFO) << "GHistIndexMatrixOneAPI Init started";
  DeviceMatrixOneAPI p_fmat_device(qu, p_fmat);
  
  LOG(INFO) << "GHistIndexMatrixOneAPI Init 1st";

  cut = SketchOnDMatrix(p_fmat, max_bins);
  cut_device.Init(qu, cut);

  LOG(INFO) << "GHistIndexMatrixOneAPI Init 2st";
  max_num_bins = max_bins;
  const int32_t nthread = omp_get_max_threads();
  const uint32_t nbins = cut.Ptrs().back();
  this->nbins = nbins;
  hit_count.resize(nbins, 0);

  this->p_fmat = p_fmat;
  const bool isDense = p_fmat->IsDense();
  this->isDense_ = isDense;

  LOG(INFO) << "GHistIndexMatrixOneAPI Init 3rd, isDense = " << isDense;
  row_ptr = std::vector<size_t>(p_fmat_device.row_ptr.Begin(), p_fmat_device.row_ptr.End());
  row_ptr_device = p_fmat_device.row_ptr;

  index.setQueue(qu);

  const size_t n_offsets = cut.Ptrs().size() - 1;
  const size_t n_index = p_fmat_device.total_offset;
  ResizeIndex(n_offsets, n_index, isDense);

  CHECK_GT(cut.Values().size(), 0U);

  uint32_t* offsets = nullptr;
  if (isDense) {
    index.ResizeOffset(n_offsets);
    offsets = index.Offset();
    LOG(INFO) << "GHistIndexMatrixOneAPI Init, n_offsets = " << n_offsets << ", offsets = " << offsets;
    for (size_t i = 0; i < n_offsets; ++i) {
      offsets[i] = cut.Ptrs()[i];
    }
  }

  if (isDense) {
    BinTypeSize curent_bin_size = index.GetBinTypeSize();
    if (curent_bin_size == kUint8BinsTypeSize) {
      common::Span<uint8_t> index_data_span = {index.data<uint8_t>(),
                                               n_index};
      SetIndexData(qu, index_data_span, p_fmat_device, nbins, offsets);

    } else if (curent_bin_size == kUint16BinsTypeSize) {
      common::Span<uint16_t> index_data_span = {index.data<uint16_t>(),
                                                n_index};
      SetIndexData(qu, index_data_span, p_fmat_device, nbins, offsets);
    } else {
      CHECK_EQ(curent_bin_size, kUint32BinsTypeSize);
      common::Span<uint32_t> index_data_span = {index.data<uint32_t>(),
                                                n_index};
      SetIndexData(qu, index_data_span, p_fmat_device, nbins, offsets);
    }
  /* For sparse DMatrix we have to store index of feature for each bin
     in index field to chose right offset. So offset is nullptr and index is not reduced */
  } else {
    common::Span<uint32_t> index_data_span = {index.data<uint32_t>(), n_index};
    SetIndexData(qu, index_data_span, p_fmat_device, nbins, offsets);
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
           const ColumnMatrixOneAPI& colmat,
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
FastFeatureGrouping(const GHistIndexMatrixOneAPI& gmat,
                    const ColumnMatrixOneAPI& colmat,
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

void GHistIndexBlockMatrixOneAPI::Init(const GHistIndexMatrixOneAPI& gmat,
                                 const ColumnMatrixOneAPI& colmat,
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
void InitializeHistByZeroes(GHistRowOneAPI<GradientSumT>& hist, size_t begin, size_t end) {
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  std::fill(hist.Begin() + begin, hist.Begin() + end,
            xgboost::detail::GradientPairInternal<GradientSumT>());
#else  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  memset(hist.Data() + begin, '\0', (end-begin)*
         sizeof(xgboost::detail::GradientPairInternal<GradientSumT>));
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
}
template void InitializeHistByZeroes(GHistRowOneAPI<float>& hist, size_t begin,
                                    size_t end);
template void InitializeHistByZeroes(GHistRowOneAPI<double>& hist, size_t begin,
                                    size_t end);

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
template<typename GradientSumT>
void IncrementHist(GHistRowOneAPI<GradientSumT>& dst, const GHistRowOneAPI<GradientSumT>& add,
                   size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.Data());
  const GradientSumT* padd = reinterpret_cast<const GradientSumT*>(add.DataConst());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] += padd[i];
  }
}
template void IncrementHist(GHistRowOneAPI<float>& dst, const GHistRowOneAPI<float>& add,
                            size_t begin, size_t end);
template void IncrementHist(GHistRowOneAPI<double>& dst, const GHistRowOneAPI<double>& add,
                            size_t begin, size_t end);

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
template<typename GradientSumT>
void CopyHist(GHistRowOneAPI<GradientSumT>& dst, const GHistRowOneAPI<GradientSumT>& src,
              size_t begin, size_t end) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.Data());
  const GradientSumT* psrc = reinterpret_cast<const GradientSumT*>(src.DataConst());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc[i];
  }
}
template void CopyHist(GHistRowOneAPI<float>& dst, const GHistRowOneAPI<float>& src,
                       size_t begin, size_t end);
template void CopyHist(GHistRowOneAPI<double>& dst, const GHistRowOneAPI<double>& src,
                       size_t begin, size_t end);

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
template<typename GradientSumT>
void SubtractionHist(cl::sycl::queue qu,
                     GHistRowOneAPI<GradientSumT>& dst, const GHistRowOneAPI<GradientSumT>& src1,
                     const GHistRowOneAPI<GradientSumT>& src2,
                     size_t size) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst.Data());
  const GradientSumT* psrc1 = reinterpret_cast<const GradientSumT*>(src1.DataConst());
  const GradientSumT* psrc2 = reinterpret_cast<const GradientSumT*>(src2.DataConst());

  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<1>(2 * size), [=](cl::sycl::item<1> pid) {
      size_t i = pid.get_id(0);
      pdst[i] = psrc1[i] - psrc2[i];
    });
  }).wait();
}
template void SubtractionHist(cl::sycl::queue qu,
                              GHistRowOneAPI<float>& dst, const GHistRowOneAPI<float>& src1,
                              const GHistRowOneAPI<float>& src2,
                              size_t size);
template void SubtractionHist(cl::sycl::queue qu,
                              GHistRowOneAPI<double>& dst, const GHistRowOneAPI<double>& src1,
                              const GHistRowOneAPI<double>& src2,
                              size_t size);

struct PrefetchOneAPI {
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
    return PrefetchOneAPI::kCacheLineSize / sizeof(T);
  }
};

constexpr size_t PrefetchOneAPI::kNoPrefetchSize;

template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistDenseKernel(cl::sycl::queue qu,
                          const std::vector<GradientPair>& gpair,
                          const USMVector<GradientPair>& gpair_device,
                          const RowSetCollectionOneAPI::Elem row_indices,
                          const GHistIndexMatrixOneAPI& gmat,
                          const size_t n_features,
                          GHistRowOneAPI<FPType>& hist,
                          GHistRowOneAPI<FPType>& hist_buffer) {
  LOG(INFO) << "BuildHistDense, start";
  const size_t size = row_indices.Size();
  LOG(INFO) << "BuildHist, start 1, size = " << size;
  const size_t* rid = row_indices.begin;
  LOG(INFO) << "BuildHist, start 2, rid = " << rid;
  const float* pgh = reinterpret_cast<const float*>(gpair_device.DataConst());
  LOG(INFO) << "BuildHist, start 3, pgh = " << pgh;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  LOG(INFO) << "BuildHist, start 4, gradient_index = " << gradient_index;
  LOG(INFO) << "BuildHist, start 5, offsets = " << offsets;
  FPType* hist_data = reinterpret_cast<FPType*>(hist.Data());
  LOG(INFO) << "BuildHist, start 6, hist_data = " << hist_data;
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  LOG(INFO) << "BuildHist, start 7";
  const size_t nbins = gmat.nbins;

  LOG(INFO) << "BuildHist, start 8, nbins = " << nbins;
  const size_t max_nblocks = hist_buffer.Size() / (nbins * two);
  LOG(INFO) << "BuildHist, start 9";
  const size_t min_block_size = 128;
  LOG(INFO) << "BuildHist, start 10";
  const size_t nblocks = std::min(max_nblocks, size / min_block_size + !!(size % min_block_size));
  LOG(INFO) << "BuildHist, start 11";
  const size_t block_size = size / nblocks + !!(size % nblocks);
  LOG(INFO) << "BuildHist, start 12, nblocks = " << nblocks << ", block_size = " << block_size;

  FPType* hist_buffer_data = reinterpret_cast<FPType*>(hist_buffer.Data());
//  FPType* hist_buffer_data = cl::sycl::malloc_shared<FPType>(nblocks * nbins * 2, qu);//reinterpret_cast<FPType*>(hist_buffer.Data());

  LOG(INFO) << "BuildHist, size = " << size;

/*
  for (size_t i = 0; i < size; ++i) {
    const size_t icol_start = rid[i] * n_features;
    const size_t idx_gh = two * rid[i];

    const BinIdxType* gr_index_local = gradient_index + icol_start;
    for (size_t j = 0; j < n_features; ++j) {
      const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
                                      offsets[j]);

      hist_data[idx_bin]   += pgh[idx_gh];
      hist_data[idx_bin+1] += pgh[idx_gh+1];
    }
  }
*/
  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<2>(nblocks, nbins), [=](cl::sycl::item<2> pid) {
      size_t i = pid.get_id(0);
      size_t j = pid.get_id(1);
      hist_buffer_data[two * (i * nbins + j)] = 0.0f;
      hist_buffer_data[two * (i * nbins + j) + 1] = 0.0f;
    });
  }).wait();

  LOG(INFO) << "BuildHist, size = " << size << " kernel 1";

  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<1>(nblocks), [=](cl::sycl::item<1> pid) {
      size_t block = pid.get_id(0);

      size_t start = block * block_size;
      size_t end = (block + 1) * block_size;
      if (end > size) {
        end = size;
      }

      FPType* hist_local = hist_buffer_data + block * nbins * two;

      for (size_t i = start; i < end; ++i) {
        const size_t icol_start = rid[i] * n_features;
        const size_t idx_gh = two * rid[i];

        const BinIdxType* gr_index_local = gradient_index + icol_start;
      
        for (size_t j = 0; j < n_features; ++j) {
          const uint32_t idx_bin = two * (static_cast<uint32_t>(gr_index_local[j]) +
                                      offsets[j]);
          hist_local[idx_bin]   += pgh[idx_gh];
          hist_local[idx_bin+1] += pgh[idx_gh+1];
        }
      }
    });
  }).wait();

  LOG(INFO) << "BuildHist, size = " << size << " kernel 2";

/*  
  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<2>(nblocks, 32), [=](cl::sycl::item<2> pid) {
      size_t block = pid.get_id(0);
      size_t feat = pid.get_id(1);

      size_t start = block * block_size;
      size_t end = (block + 1) * block_size;
      if (end > size) {
        end = size;
      }

      FPType* hist_local = hist_buffer_data + block * nbins * two;

      for (size_t i = start; i < end; ++i) {
        const size_t icol_start = row_ptr[rid[i]];
        const size_t icol_end = row_ptr[rid[i]+1];
        const size_t idx_gh = two * rid[i];
      
        for (size_t j = icol_start + feat; j < icol_end; j += 32) {
          const uint32_t idx_bin = two * gradient_index[j];
          hist_local[idx_bin]   += pgh[idx_gh];
          hist_local[idx_bin+1] += pgh[idx_gh+1];
        }
      }
    });
  }).wait();
*/
  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<1>(nbins), [=](cl::sycl::item<1> pid) {
      size_t i = pid.get_id(0);

      const size_t idx_bin = two * i;

      FPType gsum = 0.0f;
      FPType hsum = 0.0f;

      for (size_t j = 0; j < nblocks; ++j) {
        gsum += hist_buffer_data[j * nbins * two + idx_bin];
        hsum += hist_buffer_data[j * nbins * two + idx_bin + 1];
      }

      hist_data[idx_bin] = gsum;
      hist_data[idx_bin + 1] = hsum;
    });
  }).wait();
  LOG(INFO) << "BuildHist, size = " << size << " kernel 3";
}

template<typename FPType, bool do_prefetch>
void BuildHistSparseKernel(cl::sycl::queue qu,
                           const std::vector<GradientPair>& gpair,
                           const USMVector<GradientPair>& gpair_device,
                           const RowSetCollectionOneAPI::Elem row_indices,
                           const GHistIndexMatrixOneAPI& gmat,
                           GHistRowOneAPI<FPType>& hist,
                           GHistRowOneAPI<FPType>& hist_buffer) {
  LOG(INFO) << "BuildHist, start";
  const size_t size = row_indices.Size();
  LOG(INFO) << "BuildHist, start 1";
  const size_t* rid = row_indices.begin;
  LOG(INFO) << "BuildHist, start 2";
  const float* pgh = reinterpret_cast<const float*>(gpair_device.DataConst());
  LOG(INFO) << "BuildHist, start 3";
  const uint32_t* gradient_index = gmat.index.data<uint32_t>();
  LOG(INFO) << "BuildHist, start 4";
  const size_t* row_ptr =  gmat.row_ptr_device.DataConst();
  LOG(INFO) << "BuildHist, start 5";
  FPType* hist_data = reinterpret_cast<FPType*>(hist.Data());
  LOG(INFO) << "BuildHist, start 6";
  const uint32_t two {2};  // Each element from 'gpair' and 'hist' contains
                           // 2 FP values: gradient and hessian.
                           // So we need to multiply each row-index/bin-index by 2
                           // to work with gradient pairs as a singe row FP array
  LOG(INFO) << "BuildHist, start 7";
  const size_t nbins = gmat.nbins;

  LOG(INFO) << "BuildHist, start 8";
  const size_t max_nblocks = hist_buffer.Size() / (nbins * two);
  LOG(INFO) << "BuildHist, start 9";
  const size_t min_block_size = 128;
  LOG(INFO) << "BuildHist, start 10";
  const size_t nblocks = std::min(max_nblocks, size / min_block_size + !!(size % min_block_size));
  LOG(INFO) << "BuildHist, start 11";
  const size_t block_size = size / nblocks + !!(size % nblocks);
  LOG(INFO) << "BuildHist, start 12";

  FPType* hist_buffer_data = reinterpret_cast<FPType*>(hist_buffer.Data());

  LOG(INFO) << "BuildHist, size = " << size;

  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<2>(nblocks, nbins), [=](cl::sycl::item<2> pid) {
      size_t i = pid.get_id(0);
      size_t j = pid.get_id(1);
      hist_buffer_data[two * (i * nbins + j)] = 0.0f;
      hist_buffer_data[two * (i * nbins + j) + 1] = 0.0f;
    });
  }).wait();

  LOG(INFO) << "BuildHist, size = " << size << " kernel 1";

  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<1>(nblocks), [=](cl::sycl::item<1> pid) {
      size_t block = pid.get_id(0);

      size_t start = block * block_size;
      size_t end = (block + 1) * block_size;
      if (end > size) {
        end = size;
      }

      FPType* hist_local = hist_buffer_data + block * nbins * two;

      for (size_t i = start; i < end; ++i) {
        const size_t icol_start = row_ptr[rid[i]];
        const size_t icol_end = row_ptr[rid[i]+1];
        const size_t idx_gh = two * rid[i];
      
        for (size_t j = icol_start; j < icol_end; ++j) {
          const uint32_t idx_bin = two * gradient_index[j];
          hist_local[idx_bin]   += pgh[idx_gh];
          hist_local[idx_bin+1] += pgh[idx_gh+1];
        }
      }
    });
  }).wait();

  LOG(INFO) << "BuildHist, size = " << size << " kernel 2";

/*  
  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<2>(nblocks, 32), [=](cl::sycl::item<2> pid) {
      size_t block = pid.get_id(0);
      size_t feat = pid.get_id(1);

      size_t start = block * block_size;
      size_t end = (block + 1) * block_size;
      if (end > size) {
        end = size;
      }

      FPType* hist_local = hist_buffer_data + block * nbins * two;

      for (size_t i = start; i < end; ++i) {
        const size_t icol_start = row_ptr[rid[i]];
        const size_t icol_end = row_ptr[rid[i]+1];
        const size_t idx_gh = two * rid[i];
      
        for (size_t j = icol_start + feat; j < icol_end; j += 32) {
          const uint32_t idx_bin = two * gradient_index[j];
          hist_local[idx_bin]   += pgh[idx_gh];
          hist_local[idx_bin+1] += pgh[idx_gh+1];
        }
      }
    });
  }).wait();
*/
  qu.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<>(cl::sycl::range<1>(nbins), [=](cl::sycl::item<1> pid) {
      size_t i = pid.get_id(0);

      const size_t idx_bin = two * i;

      FPType gsum = 0.0f;
      FPType hsum = 0.0f;

      for (size_t j = 0; j < nblocks; ++j) {
        gsum += hist_buffer_data[j * nbins * two + idx_bin];
        hsum += hist_buffer_data[j * nbins * two + idx_bin + 1];
      }

      hist_data[idx_bin] = gsum;
      hist_data[idx_bin + 1] = hsum;
    });
  }).wait();
  LOG(INFO) << "BuildHist, size = " << size << " kernel 3";
}


template<typename FPType, bool do_prefetch, typename BinIdxType>
void BuildHistDispatchKernel(cl::sycl::queue qu,
                             const std::vector<GradientPair>& gpair,
                             const USMVector<GradientPair>& gpair_device,
                             const RowSetCollectionOneAPI::Elem row_indices,
                             const GHistIndexMatrixOneAPI& gmat, GHistRowOneAPI<FPType>& hist, bool isDense,
                             GHistRowOneAPI<FPType>& hist_buffer) {
  LOG(INFO) << "BuildHistDispatchKernel start, isDense = " << isDense;
  if (isDense) {
    const size_t* row_ptr =  gmat.row_ptr.data();
    const size_t n_features = row_ptr[row_indices.begin[0]+1] - row_ptr[row_indices.begin[0]];
    BuildHistDenseKernel<FPType, do_prefetch, BinIdxType>(qu, gpair, gpair_device, row_indices,
                                                       gmat, n_features, hist, hist_buffer);
  } else {
    BuildHistSparseKernel<FPType, do_prefetch>(qu, gpair, gpair_device, row_indices,
                                                        gmat, hist, hist_buffer);
  }
}

template<typename FPType, bool do_prefetch>
void BuildHistKernel(cl::sycl::queue qu,
                     const std::vector<GradientPair>& gpair,
                     const USMVector<GradientPair>& gpair_device,
                     const RowSetCollectionOneAPI::Elem row_indices,
                     const GHistIndexMatrixOneAPI& gmat, const bool isDense, GHistRowOneAPI<FPType>& hist,
                     GHistRowOneAPI<FPType>& hist_buffer) {
  LOG(INFO) << "BuildHistKernel start, isDense = " << isDense;
  const bool is_dense = row_indices.Size() && isDense;
  switch (gmat.index.GetBinTypeSize()) {
    case kUint8BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint8_t>(qu, gpair, gpair_device, row_indices,
                                                            gmat, hist, is_dense, hist_buffer);
      break;
    case kUint16BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint16_t>(qu, gpair, gpair_device, row_indices,
                                                             gmat, hist, is_dense, hist_buffer);
      break;
    case kUint32BinsTypeSize:
      BuildHistDispatchKernel<FPType, do_prefetch, uint32_t>(qu, gpair, gpair_device, row_indices,
                                                             gmat, hist, is_dense, hist_buffer);
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

template <typename GradientSumT>
void GHistBuilderOneAPI<GradientSumT>::BuildHist(
    const std::vector<GradientPair> &gpair,
    const USMVector<GradientPair>& gpair_device,
    const RowSetCollectionOneAPI::Elem row_indices, const GHistIndexMatrixOneAPI &gmat,
    GHistRowT& hist, bool isDense, GHistRowT& hist_buffer) {
  LOG(INFO) << "BuildHist start, isDense = " << isDense;
  const size_t nrows = row_indices.Size();
  const size_t no_prefetch_size = PrefetchOneAPI::NoPrefetchSize(nrows);

  // if need to work with all rows from bin-matrix (e.g. root node)
  const bool contiguousBlock = (row_indices.begin[nrows - 1] - row_indices.begin[0]) == (nrows - 1);

//  if (contiguousBlock) {
    // contiguous memory access, built-in HW prefetching is enough
    BuildHistKernel<GradientSumT, false>(qu_, gpair, gpair_device, row_indices, gmat, isDense, hist, hist_buffer);
/*  } else {
    const RowSetCollectionOneAPI::Elem span1(row_indices.begin, row_indices.end - no_prefetch_size);
    const RowSetCollectionOneAPI::Elem span2(row_indices.end - no_prefetch_size, row_indices.end);

    BuildHistKernel<GradientSumT, true>(gpair, gpair_device, span1, gmat, isDense, hist);
    // no prefetching to avoid loading extra memory
    BuildHistKernel<GradientSumT, false>(gpair, gpair_device, span2, gmat, isDense, hist);
  }*/
}
template
void GHistBuilderOneAPI<float>::BuildHist(
                             const std::vector<GradientPair>& gpair,
                             const USMVector<GradientPair>& gpair_device,
                             const RowSetCollectionOneAPI::Elem row_indices,
                             const GHistIndexMatrixOneAPI& gmat,
                             GHistRowOneAPI<float>& hist,
                             bool isDense,
                             GHistRowOneAPI<float>& hist_buffer);
template
void GHistBuilderOneAPI<double>::BuildHist(
                             const std::vector<GradientPair>& gpair,
                             const USMVector<GradientPair>& gpair_device,
                             const RowSetCollectionOneAPI::Elem row_indices,
                             const GHistIndexMatrixOneAPI& gmat,
                             GHistRowOneAPI<double>& hist,
                             bool isDense,
                             GHistRowOneAPI<double>& hist_buffer);

template<typename GradientSumT>
void GHistBuilderOneAPI<GradientSumT>::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const USMVector<GradientPair>& gpair_device,
                                  const RowSetCollectionOneAPI::Elem row_indices,
                                  const GHistIndexBlockMatrixOneAPI& gmatb,
                                  GHistRowT& hist) {
  constexpr int kUnroll = 8;  // loop unrolling factor
  const size_t nblock = gmatb.GetNumBlock();
  const size_t nrows = row_indices.end - row_indices.begin;
  const size_t rest = nrows % kUnroll;
#if defined(_OPENMP)
  const auto nthread = static_cast<bst_omp_uint>(this->nthread_);  // NOLINT
#endif  // defined(_OPENMP)
  xgboost::detail::GradientPairInternal<GradientSumT>* p_hist = hist.Data();

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
void GHistBuilderOneAPI<float>::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const USMVector<GradientPair>& gpair_device,
                                  const RowSetCollectionOneAPI::Elem row_indices,
                                  const GHistIndexBlockMatrixOneAPI& gmatb,
                                  GHistRowOneAPI<float>& hist);
template
void GHistBuilderOneAPI<double>::BuildBlockHist(const std::vector<GradientPair>& gpair,
                                  const USMVector<GradientPair>& gpair_device,
                                  const RowSetCollectionOneAPI::Elem row_indices,
                                  const GHistIndexBlockMatrixOneAPI& gmatb,
                                  GHistRowOneAPI<double>& hist);


template<typename GradientSumT>
void GHistBuilderOneAPI<GradientSumT>::SubtractionTrick(GHistRowT& self,
                                                        GHistRowT& sibling,
                                                        GHistRowT& parent) {
  const size_t size = self.Size();
  CHECK_EQ(sibling.Size(), size);
  CHECK_EQ(parent.Size(), size);

  SubtractionHist(qu_, self, parent, sibling, size);
}
template
void GHistBuilderOneAPI<float>::SubtractionTrick(GHistRowOneAPI<float>& self,
                                           GHistRowOneAPI<float>& sibling,
                                           GHistRowOneAPI<float>& parent);
template
void GHistBuilderOneAPI<double>::SubtractionTrick(GHistRowOneAPI<double>& self,
                                            GHistRowOneAPI<double>& sibling,
                                            GHistRowOneAPI<double>& parent);

}  // namespace common
}  // namespace xgboost
