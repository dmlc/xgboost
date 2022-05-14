/*!
 * Copyright 2017-2022 by Contributors
 * \file column_matrix.h
 * \brief Utility for fast column-wise access
 * \author Philip Cho
 */

#ifndef XGBOOST_COMMON_COLUMN_MATRIX_H_
#define XGBOOST_COMMON_COLUMN_MATRIX_H_

#include <dmlc/endian.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>  // std::move
#include <vector>

#include "../data/gradient_index.h"
#include "hist_util.h"

namespace xgboost {
namespace common {

class ColumnMatrix;
/*! \brief column type */
enum ColumnType : uint8_t { kDenseColumn, kSparseColumn };

/*! \brief a column storage, to be used with ApplySplit. Note that each
    bin id is stored as index[i] + index_base.
    Different types of column index for each column allow
    to reduce the memory usage. */
template <typename BinIdxType>
class Column {
 public:
  static constexpr bst_bin_t kMissingId = -1;

  Column(common::Span<const BinIdxType> index, bst_bin_t least_bin_idx)
      : index_(index), index_base_(least_bin_idx) {}
  virtual ~Column() = default;

  bst_bin_t GetGlobalBinIdx(size_t idx) const {
    return index_base_ + static_cast<bst_bin_t>(index_[idx]);
  }

  /* returns number of elements in column */
  size_t Size() const { return index_.size(); }

 private:
  /* bin indexes in range [0, max_bins - 1] */
  common::Span<const BinIdxType> index_;
  /* bin index offset for specific feature */
  bst_bin_t const index_base_;
};

template <typename BinIdxT>
class SparseColumnIter : public Column<BinIdxT> {
 private:
  using Base = Column<BinIdxT>;
  /* indexes of rows */
  common::Span<const size_t> row_ind_;
  size_t idx_;

  size_t const* RowIndices() const { return row_ind_.data(); }

 public:
  SparseColumnIter(common::Span<const BinIdxT> index, bst_bin_t least_bin_idx,
                   common::Span<const size_t> row_ind, bst_row_t first_row_idx)
      : Base{index, least_bin_idx}, row_ind_(row_ind) {
    // first_row_id is the first row in the leaf partition
    const size_t* row_data = RowIndices();
    const size_t column_size = this->Size();
    // search first nonzero row with index >= rid_span.front()
    // note that the input row partition is always sorted.
    const size_t* p = std::lower_bound(row_data, row_data + column_size, first_row_idx);
    // column_size if all missing
    idx_ = p - row_data;
  }
  SparseColumnIter(SparseColumnIter const&) = delete;
  SparseColumnIter(SparseColumnIter&&) = default;

  size_t GetRowIdx(size_t idx) const { return RowIndices()[idx]; }
  bst_bin_t operator[](size_t rid) {
    const size_t column_size = this->Size();
    if (!((idx_) < column_size)) {
      return this->kMissingId;
    }
    // find next non-missing row
    while ((idx_) < column_size && GetRowIdx(idx_) < rid) {
      ++(idx_);
    }
    if (((idx_) < column_size) && GetRowIdx(idx_) == rid) {
      // non-missing row found
      return this->GetGlobalBinIdx(idx_);
    } else {
      // at the end of column
      return this->kMissingId;
    }
  }
};

template <typename BinIdxT, bool any_missing>
class DenseColumnIter : public Column<BinIdxT> {
 private:
  using Base = Column<BinIdxT>;
  /* flags for missing values in dense columns */
  std::vector<bool> const& missing_flags_;
  size_t feature_offset_;

 public:
  explicit DenseColumnIter(common::Span<const BinIdxT> index, bst_bin_t index_base,
                           std::vector<bool> const& missing_flags, size_t feature_offset)
      : Base{index, index_base}, missing_flags_{missing_flags}, feature_offset_{feature_offset} {}
  DenseColumnIter(DenseColumnIter const&) = delete;
  DenseColumnIter(DenseColumnIter&&) = default;

  bool IsMissing(size_t ridx) const { return missing_flags_[feature_offset_ + ridx]; }

  bst_bin_t operator[](size_t ridx) const {
    if (any_missing) {
      return IsMissing(ridx) ? this->kMissingId : this->GetGlobalBinIdx(ridx);
    } else {
      return this->GetGlobalBinIdx(ridx);
    }
  }
};

/**
 * \brief Column major matrix for gradient index. This matrix contains both dense column
 * and sparse column, the type of the column is controlled by sparse threshold. When the
 * number of missing values in a column is below the threshold it classified as dense
 * column.
 */
class ColumnMatrix {
 public:
  // get number of features
  bst_feature_t GetNumFeature() const { return static_cast<bst_feature_t>(type_.size()); }

  // construct column matrix from GHistIndexMatrix
  void Init(SparsePage const& page, const GHistIndexMatrix& gmat, double sparse_threshold,
            int32_t n_threads) {
    auto const nfeature = static_cast<bst_feature_t>(gmat.cut.Ptrs().size() - 1);
    const size_t nrow = gmat.row_ptr.size() - 1;
    // identify type of each column
    feature_counts_.resize(nfeature);
    type_.resize(nfeature);
    std::fill(feature_counts_.begin(), feature_counts_.end(), 0);
    uint32_t max_val = std::numeric_limits<uint32_t>::max();
    for (bst_feature_t fid = 0; fid < nfeature; ++fid) {
      CHECK_LE(gmat.cut.Ptrs()[fid + 1] - gmat.cut.Ptrs()[fid], max_val);
    }

    bool all_dense_column = true;
    gmat.GetFeatureCounts(&feature_counts_[0]);
    // classify features
    for (bst_feature_t fid = 0; fid < nfeature; ++fid) {
      if (static_cast<double>(feature_counts_[fid]) < sparse_threshold * nrow) {
        type_[fid] = kSparseColumn;
        all_dense_column = false;
      } else {
        type_[fid] = kDenseColumn;
      }
    }

    // want to compute storage boundary for each feature
    // using variants of prefix sum scan
    feature_offsets_.resize(nfeature + 1);
    size_t accum_index = 0;
    feature_offsets_[0] = accum_index;
    for (bst_feature_t fid = 1; fid < nfeature + 1; ++fid) {
      if (type_[fid - 1] == kDenseColumn) {
        accum_index += static_cast<size_t>(nrow);
      } else {
        accum_index += feature_counts_[fid - 1];
      }
      feature_offsets_[fid] = accum_index;
    }

    SetTypeSize(gmat.max_num_bins);
    auto storage_size =
        feature_offsets_.back() * static_cast<std::underlying_type_t<BinTypeSize>>(bins_type_size_);
    index_.resize(storage_size, 0);
    if (!all_dense_column) {
      row_ind_.resize(feature_offsets_[nfeature]);
    }

    // store least bin id for each feature
    index_base_ = const_cast<uint32_t*>(gmat.cut.Ptrs().data());

    any_missing_ = !gmat.IsDense();

    missing_flags_.clear();
    // pre-fill index_ for dense columns
    BinTypeSize gmat_bin_size = gmat.index.GetBinTypeSize();
    if (!any_missing_) {
      missing_flags_.resize(feature_offsets_[nfeature], false);
      // row index is compressed, we need to dispatch it.
      DispatchBinType(gmat_bin_size, [&](auto t) {
        using RowBinIdxT = decltype(t);
        SetIndexNoMissing(page, gmat.index.data<RowBinIdxT>(), nrow, nfeature, n_threads);
      });
    } else {
      missing_flags_.resize(feature_offsets_[nfeature], true);
      SetIndexMixedColumns(page, gmat.index.data<uint32_t>(), gmat, nfeature);
    }
  }

  /* Set the number of bytes based on numeric limit of maximum number of bins provided by user */
  void SetTypeSize(size_t max_bin_per_feat) {
    if ((max_bin_per_feat - 1) <= static_cast<int>(std::numeric_limits<uint8_t>::max())) {
      bins_type_size_ = kUint8BinsTypeSize;
    } else if ((max_bin_per_feat - 1) <= static_cast<int>(std::numeric_limits<uint16_t>::max())) {
      bins_type_size_ = kUint16BinsTypeSize;
    } else {
      bins_type_size_ = kUint32BinsTypeSize;
    }
  }

  template <typename BinIdxType>
  auto SparseColumn(bst_feature_t fidx, bst_row_t first_row_idx) const {
    const size_t feature_offset = feature_offsets_[fidx];  // to get right place for certain feature
    const size_t column_size = feature_offsets_[fidx + 1] - feature_offset;
    common::Span<const BinIdxType> bin_index = {
        reinterpret_cast<const BinIdxType*>(&index_[feature_offset * bins_type_size_]),
        column_size};
    return SparseColumnIter<BinIdxType>(bin_index, index_base_[fidx],
                                        {&row_ind_[feature_offset], column_size}, first_row_idx);
  }

  template <typename BinIdxType, bool any_missing>
  auto DenseColumn(bst_feature_t fidx) const {
    const size_t feature_offset = feature_offsets_[fidx];  // to get right place for certain feature
    const size_t column_size = feature_offsets_[fidx + 1] - feature_offset;
    common::Span<const BinIdxType> bin_index = {
        reinterpret_cast<const BinIdxType*>(&index_[feature_offset * bins_type_size_]),
        column_size};
    return std::move(DenseColumnIter<BinIdxType, any_missing>{
        bin_index, static_cast<bst_bin_t>(index_base_[fidx]), missing_flags_, feature_offset});
  }

  // all columns are dense column and has no missing value
  // FIXME(jiamingy): We don't need a column matrix if there's no missing value.
  template <typename RowBinIdxT>
  void SetIndexNoMissing(SparsePage const& page, RowBinIdxT const* row_index,
                         const size_t n_samples, const size_t n_features, int32_t n_threads) {
    DispatchBinType(bins_type_size_, [&](auto t) {
      using ColumnBinT = decltype(t);
      auto column_index = Span<ColumnBinT>{reinterpret_cast<ColumnBinT*>(index_.data()),
                                           index_.size() / sizeof(ColumnBinT)};
      ParallelFor(n_samples, n_threads, [&](auto rid) {
        const size_t ibegin = rid * n_features;
        const size_t iend = (rid + 1) * n_features;
        size_t j = 0;
        for (size_t i = ibegin; i < iend; ++i, ++j) {
          const size_t idx = feature_offsets_[j];
          // No need to add offset, as row index is compressed and stores the local index
          column_index[idx + rid] = row_index[i];
        }
      });
    });
  }

  /**
   * \brief Set column index for both dense and sparse columns
   */
  void SetIndexMixedColumns(SparsePage const& page, uint32_t const* row_index,
                            const GHistIndexMatrix& gmat, size_t n_features) {
    std::vector<size_t> num_nonzeros;
    num_nonzeros.resize(n_features, 0);

    DispatchBinType(bins_type_size_, [&](auto t) {
      using ColumnBinT = decltype(t);
      ColumnBinT* local_index = reinterpret_cast<ColumnBinT*>(index_.data());

      auto get_bin_idx = [&](auto bin_id, auto rid, bst_feature_t fid) {
        if (type_[fid] == kDenseColumn) {
          ColumnBinT* begin = &local_index[feature_offsets_[fid]];
          begin[rid] = bin_id - index_base_[fid];
          // not thread-safe with bool vector.
          missing_flags_[feature_offsets_[fid] + rid] = false;
        } else {
          ColumnBinT* begin = &local_index[feature_offsets_[fid]];
          begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
          row_ind_[feature_offsets_[fid] + num_nonzeros[fid]] = rid;
          ++num_nonzeros[fid];
        }
      };

      const xgboost::Entry* data_ptr = page.data.HostVector().data();
      const std::vector<bst_row_t>& offset_vec = page.offset.HostVector();
      const size_t batch_size = gmat.Size();
      CHECK_LT(batch_size, offset_vec.size());
      for (size_t rid = 0; rid < batch_size; ++rid) {
        const size_t ibegin = gmat.row_ptr[rid];
        const size_t iend = gmat.row_ptr[rid + 1];
        const size_t size = offset_vec[rid + 1] - offset_vec[rid];
        SparsePage::Inst inst = {data_ptr + offset_vec[rid], size};

        CHECK_EQ(ibegin + inst.size(), iend);
        size_t j = 0;
        for (size_t i = ibegin; i < iend; ++i, ++j) {
          const uint32_t bin_id = row_index[i];
          auto fid = inst[j].index;
          get_bin_idx(bin_id, rid, fid);
        }
      }
    });
  }

  BinTypeSize GetTypeSize() const { return bins_type_size_; }
  auto GetColumnType(bst_feature_t fidx) const { return type_[fidx]; }

  // And this returns part of state
  bool AnyMissing() const { return any_missing_; }

  // IO procedures for external memory.
  bool Read(dmlc::SeekStream* fi, uint32_t const* index_base) {
    fi->Read(&index_);
    fi->Read(&feature_counts_);
#if !DMLC_LITTLE_ENDIAN
    // s390x
    std::vector<std::underlying_type<ColumnType>::type> int_types;
    fi->Read(&int_types);
    type_.resize(int_types.size());
    std::transform(
        int_types.begin(), int_types.end(), type_.begin(),
        [](std::underlying_type<ColumnType>::type i) { return static_cast<ColumnType>(i); });
#else
    fi->Read(&type_);
#endif  // !DMLC_LITTLE_ENDIAN

    fi->Read(&row_ind_);
    fi->Read(&feature_offsets_);
    index_base_ = index_base;
#if !DMLC_LITTLE_ENDIAN
    std::underlying_type<BinTypeSize>::type v;
    fi->Read(&v);
    bins_type_size_ = static_cast<BinTypeSize>(v);
#else
    fi->Read(&bins_type_size_);
#endif

    fi->Read(&any_missing_);
    return true;
  }

  size_t Write(dmlc::Stream* fo) const {
    size_t bytes{0};

    auto write_vec = [&](auto const& vec) {
      fo->Write(vec);
      bytes += vec.size() * sizeof(typename std::remove_reference_t<decltype(vec)>::value_type) +
               sizeof(uint64_t);
    };
    write_vec(index_);
    write_vec(feature_counts_);
#if !DMLC_LITTLE_ENDIAN
    // s390x
    std::vector<std::underlying_type<ColumnType>::type> int_types(type_.size());
    std::transform(type_.begin(), type_.end(), int_types.begin(), [](ColumnType t) {
      return static_cast<std::underlying_type<ColumnType>::type>(t);
    });
    write_vec(int_types);
#else
    write_vec(type_);
#endif  // !DMLC_LITTLE_ENDIAN
    write_vec(row_ind_);
    write_vec(feature_offsets_);

#if !DMLC_LITTLE_ENDIAN
    auto v = static_cast<std::underlying_type<BinTypeSize>::type>(bins_type_size_);
    fo->Write(v);
#else
    fo->Write(bins_type_size_);
#endif  // DMLC_LITTLE_ENDIAN
    bytes += sizeof(bins_type_size_);
    fo->Write(any_missing_);
    bytes += sizeof(any_missing_);

    return bytes;
  }

 private:
  std::vector<uint8_t> index_;

  std::vector<size_t> feature_counts_;
  std::vector<ColumnType> type_;
  std::vector<size_t> row_ind_;
  /* indicate where each column's index and row_ind is stored. */
  std::vector<size_t> feature_offsets_;

  // index_base_[fid]: least bin id for feature fid
  uint32_t const* index_base_;
  std::vector<bool> missing_flags_;
  BinTypeSize bins_type_size_;
  bool any_missing_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
