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

#include "../data/adapter.h"
#include "../data/gradient_index.h"
#include "algorithm.h"
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
 public:
  using ByteType = bool;

 private:
  using Base = Column<BinIdxT>;
  /* flags for missing values in dense columns */
  std::vector<ByteType> const& missing_flags_;
  size_t feature_offset_;

 public:
  explicit DenseColumnIter(common::Span<const BinIdxT> index, bst_bin_t index_base,
                           std::vector<ByteType> const& missing_flags, size_t feature_offset)
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
 * number of missing values in a column is below the threshold it's classified as dense
 * column.
 */
class ColumnMatrix {
  void InitStorage(GHistIndexMatrix const& gmat, double sparse_threshold);

  template <typename ColumnBinT, typename BinT, typename RIdx>
  void SetBinSparse(BinT bin_id, RIdx rid, bst_feature_t fid, ColumnBinT* local_index) {
    if (type_[fid] == kDenseColumn) {
      ColumnBinT* begin = &local_index[feature_offsets_[fid]];
      begin[rid] = bin_id - index_base_[fid];
      // not thread-safe with bool vector.  FIXME(jiamingy): We can directly assign
      // kMissingId to the index to avoid missing flags.
      missing_flags_[feature_offsets_[fid] + rid] = false;
    } else {
      ColumnBinT* begin = &local_index[feature_offsets_[fid]];
      begin[num_nonzeros_[fid]] = bin_id - index_base_[fid];
      row_ind_[feature_offsets_[fid] + num_nonzeros_[fid]] = rid;
      ++num_nonzeros_[fid];
    }
  }

 public:
  using ByteType = bool;
  // get number of features
  bst_feature_t GetNumFeature() const { return static_cast<bst_feature_t>(type_.size()); }

  ColumnMatrix() = default;
  ColumnMatrix(GHistIndexMatrix const& gmat, double sparse_threshold) {
    this->InitStorage(gmat, sparse_threshold);
  }

  /**
   * \brief Initialize ColumnMatrix from GHistIndexMatrix with reference to the original
   *        SparsePage.
   */
  void InitFromSparse(SparsePage const& page, const GHistIndexMatrix& gmat, double sparse_threshold,
                      int32_t n_threads) {
    auto batch = data::SparsePageAdapterBatch{page.GetView()};
    this->InitStorage(gmat, sparse_threshold);
    // ignore base row id here as we always has one column matrix for each sparse page.
    this->PushBatch(n_threads, batch, std::numeric_limits<float>::quiet_NaN(), gmat, 0);
  }

  /**
   * \brief Initialize ColumnMatrix from GHistIndexMatrix without reference to actual
   * data.
   *
   *    This function requires a binary search for each bin to get back the feature index
   *    for those bins.
   */
  void InitFromGHist(Context const* ctx, GHistIndexMatrix const& gmat) {
    auto n_threads = ctx->Threads();
    if (!any_missing_) {
      // row index is compressed, we need to dispatch it.
      DispatchBinType(gmat.index.GetBinTypeSize(), [&, size = gmat.Size(), n_threads = n_threads,
                                                    n_features = gmat.Features()](auto t) {
        using RowBinIdxT = decltype(t);
        SetIndexNoMissing(gmat.base_rowid, gmat.index.data<RowBinIdxT>(), size, n_features,
                          n_threads);
      });
    } else {
      SetIndexMixedColumns(gmat);
    }
  }

  bool IsInitialized() const { return !type_.empty(); }

  /**
   * \brief Push batch of data for Quantile DMatrix support.
   *
   * \param batch      Input data wrapped inside a adapter batch.
   * \param gmat       The row-major histogram index that contains index for ALL data.
   * \param base_rowid The beginning row index for current batch.
   */
  template <typename Batch>
  void PushBatch(int32_t n_threads, Batch const& batch, float missing, GHistIndexMatrix const& gmat,
                 size_t base_rowid) {
    // pre-fill index_ for dense columns
    if (!any_missing_) {
      // row index is compressed, we need to dispatch it.

      // use base_rowid from input parameter as gmat is a single matrix that contains all
      // the histogram index instead of being only a batch.
      DispatchBinType(gmat.index.GetBinTypeSize(), [&, size = batch.Size(), n_threads = n_threads,
                                                    n_features = gmat.Features()](auto t) {
        using RowBinIdxT = decltype(t);
        SetIndexNoMissing(base_rowid, gmat.index.data<RowBinIdxT>(), size, n_features, n_threads);
      });
    } else {
      SetIndexMixedColumns(base_rowid, batch, gmat, missing);
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
  void SetIndexNoMissing(bst_row_t base_rowid, RowBinIdxT const* row_index, const size_t n_samples,
                         const size_t n_features, int32_t n_threads) {
    missing_flags_.resize(feature_offsets_[n_features], false);
    DispatchBinType(bins_type_size_, [&](auto t) {
      using ColumnBinT = decltype(t);
      auto column_index = Span<ColumnBinT>{reinterpret_cast<ColumnBinT*>(index_.data()),
                                           index_.size() / sizeof(ColumnBinT)};
      ParallelFor(n_samples, n_threads, [&](auto rid) {
        rid += base_rowid;
        const size_t ibegin = rid * n_features;
        const size_t iend = (rid + 1) * n_features;
        for (size_t i = ibegin, j = 0; i < iend; ++i, ++j) {
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
  template <typename Batch>
  void SetIndexMixedColumns(size_t base_rowid, Batch const& batch, const GHistIndexMatrix& gmat,
                            float missing) {
    auto n_features = gmat.Features();
    missing_flags_.resize(feature_offsets_[n_features], true);
    auto const* row_index = gmat.index.data<uint32_t>() + gmat.row_ptr[base_rowid];
    num_nonzeros_.resize(n_features, 0);
    auto is_valid = data::IsValidFunctor{missing};

    DispatchBinType(bins_type_size_, [&](auto t) {
      using ColumnBinT = decltype(t);
      ColumnBinT* local_index = reinterpret_cast<ColumnBinT*>(index_.data());
      size_t const batch_size = batch.Size();
      size_t k{0};
      for (size_t rid = 0; rid < batch_size; ++rid) {
        auto line = batch.GetLine(rid);
        for (size_t i = 0; i < line.Size(); ++i) {
          auto coo = line.GetElement(i);
          if (is_valid(coo)) {
            auto fid = coo.column_idx;
            const uint32_t bin_id = row_index[k];
            SetBinSparse(bin_id, rid + base_rowid, fid, local_index);
            ++k;
          }
        }
      }
    });
  }

  /**
   * \brief Set column index for both dense and sparse columns, but with only GHistMatrix
   *        available and requires a search for each bin.
   */
  void SetIndexMixedColumns(const GHistIndexMatrix& gmat) {
    auto n_features = gmat.Features();
    missing_flags_.resize(feature_offsets_[n_features], true);
    num_nonzeros_.resize(n_features, 0);

    DispatchBinType(bins_type_size_, [&](auto t) {
      using ColumnBinT = decltype(t);
      ColumnBinT* local_index = reinterpret_cast<ColumnBinT*>(index_.data());
      CHECK(this->any_missing_);
      AssignColumnBinIndex(gmat,
                           [&](auto bin_idx, std::size_t, std::size_t ridx, bst_feature_t fidx) {
                             SetBinSparse(bin_idx, ridx, fidx, local_index);
                           });
    });
  }

  BinTypeSize GetTypeSize() const { return bins_type_size_; }
  auto GetColumnType(bst_feature_t fidx) const { return type_[fidx]; }

  // And this returns part of state
  bool AnyMissing() const { return any_missing_; }

  // IO procedures for external memory.
  bool Read(dmlc::SeekStream* fi, uint32_t const* index_base) {
    fi->Read(&index_);
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

    std::vector<std::uint8_t> missing;
    fi->Read(&missing);
    missing_flags_.resize(missing.size());
    std::transform(missing.cbegin(), missing.cend(), missing_flags_.begin(),
                   [](std::uint8_t flag) { return !!flag; });

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
    // dmlc can not handle bool vector
    std::vector<std::uint8_t> missing(missing_flags_.size());
    std::transform(missing_flags_.cbegin(), missing_flags_.cend(), missing.begin(),
                   [](bool flag) { return static_cast<std::uint8_t>(flag); });
    write_vec(missing);

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

  std::vector<ColumnType> type_;
  /* indptr of a CSC matrix. */
  std::vector<size_t> row_ind_;
  /* indicate where each column's index and row_ind is stored. */
  std::vector<size_t> feature_offsets_;
  /* The number of nnz of each column. */
  std::vector<size_t> num_nonzeros_;

  // index_base_[fid]: least bin id for feature fid
  uint32_t const* index_base_;
  std::vector<ByteType> missing_flags_;
  BinTypeSize bins_type_size_;
  bool any_missing_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
