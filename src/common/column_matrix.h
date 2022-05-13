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
class Column {
 public:
  static constexpr bst_bin_t MissingIdx() { return -1; }

  Column(ColumnType type, common::Span<bst_bin_t const> index, const bst_bin_t index_base)
      : type_(type), index_(index), index_base_{index_base} {}

  virtual ~Column() = default;

  bst_bin_t GetGlobalBinIdx(size_t idx) const { return index_base_ + index_[idx]; }
  bst_bin_t GetFeatureBinIdx(size_t idx) const { return index_[idx]; }
  bst_bin_t GetBaseIdx() const { return index_base_; }
  common::Span<bst_bin_t const> GetFeatureBinIdxPtr() const { return index_; }

  ColumnType GetType() const { return type_; }

  /* returns number of elements in column */
  size_t Size() const { return index_.size(); }

 private:
  /* type of column */
  ColumnType type_;
  /* bin index offset for specific feature */
  bst_bin_t const index_base_;

 protected:
  /* bin indexes in range [0, max_bins - 1] */
  common::Span<bst_bin_t const> index_;
};

class SparseColumn : public Column {
 public:
  SparseColumn(ColumnType type, common::Span<bst_bin_t const> index, bst_bin_t index_base,
               common::Span<const size_t> row_ind)
      : Column(type, index, index_base), row_ind_(row_ind) {}

  const size_t* GetRowData() const { return row_ind_.data(); }

  bst_bin_t GetBinIdx(size_t rid, size_t* state) const {
    const size_t column_size = this->Size();
    if (!((*state) < column_size)) {
      return MissingIdx();
    }
    while ((*state) < column_size && GetRowIdx(*state) < rid) {
      ++(*state);
    }
    if (((*state) < column_size) && GetRowIdx(*state) == rid) {
      return this->GetGlobalBinIdx(*state);
    } else {
      return MissingIdx();
    }
  }

  size_t GetInitialState(const size_t first_row_id) const {
    const size_t* row_data = GetRowData();
    const size_t column_size = this->Size();
    // search first nonzero row with index >= rid_span.front()
    const size_t* p = std::lower_bound(row_data, row_data + column_size, first_row_id);
    // column_size if all messing
    return p - row_data;
  }

  size_t GetRowIdx(size_t idx) const { return row_ind_.data()[idx]; }

 private:
  /* indexes of rows */
  common::Span<const size_t> row_ind_;
};

template <bool any_missing>
class DenseColumn : public Column {
 public:
  DenseColumn(ColumnType type, common::Span<int32_t const> index, uint32_t index_base)
      : Column(type, index, index_base) {}
  bool IsMissing(size_t idx) const {
    return index_[idx] == Column::MissingIdx();
  }

  bst_bin_t GetBinIdx(size_t idx, size_t* state) const {
    if (any_missing) {
      return IsMissing(idx) ? Column::MissingIdx() : this->GetGlobalBinIdx(idx);
    } else {
      this->GetGlobalBinIdx(idx);
    }
    return this->GetGlobalBinIdx(idx);
  }
  size_t GetInitialState(const size_t first_row_id) const { return 0; }
};

/*! \brief a collection of columns, with support for construction from
    GHistIndexMatrix. */
class ColumnMatrix {
 public:
  // get number of features
  bst_feature_t GetNumFeature() const { return static_cast<bst_feature_t>(type_.size()); }

  // construct column matrix from GHistIndexMatrix
  inline void Init(SparsePage const& page, const GHistIndexMatrix& gmat, double sparse_threshold,
                   int32_t n_threads) {
    auto const nfeature = static_cast<bst_feature_t>(gmat.cut.Ptrs().size() - 1);
    const size_t nrow = gmat.row_ptr.size() - 1;
    // identify type of each column
    std::vector<size_t> feature_counts;
    feature_counts.resize(nfeature);
    type_.resize(nfeature);
    std::fill(feature_counts.begin(), feature_counts.end(), 0);
    uint32_t max_val = std::numeric_limits<uint32_t>::max();
    for (bst_feature_t fid = 0; fid < nfeature; ++fid) {
      CHECK_LE(gmat.cut.Ptrs()[fid + 1] - gmat.cut.Ptrs()[fid], max_val);
    }
    bool all_dense = gmat.IsDense();
    gmat.GetFeatureCounts(&feature_counts[0]);
    // classify features
    for (bst_feature_t fid = 0; fid < nfeature; ++fid) {
      if (static_cast<double>(feature_counts[fid]) < sparse_threshold * nrow) {
        type_[fid] = kSparseColumn;
        all_dense = false;
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
        accum_index += feature_counts[fid - 1];
      }
      feature_offsets_[fid] = accum_index;
    }

    index_.resize(feature_offsets_[nfeature], Column::MissingIdx());
    if (!all_dense) {
      row_ind_.resize(feature_offsets_[nfeature]);
    }

    // store least bin id for each feature
    index_base_ = const_cast<uint32_t*>(gmat.cut.Ptrs().data());

    const bool noMissingValues = NoMissingValues(gmat.row_ptr[nrow], nrow, nfeature);
    any_missing_ = !noMissingValues;

    // pre-fill index_ for dense columns
    if (all_dense) {
      BinTypeSize gmat_bin_size = gmat.index.GetBinTypeSize();
      if (gmat_bin_size == kUint8BinsTypeSize) {
        SetIndexAllDense(page, gmat.index.data<uint8_t>(), gmat, nrow, nfeature, noMissingValues,
                         n_threads);
      } else if (gmat_bin_size == kUint16BinsTypeSize) {
        SetIndexAllDense(page, gmat.index.data<uint16_t>(), gmat, nrow, nfeature, noMissingValues,
                         n_threads);
      } else {
        CHECK_EQ(gmat_bin_size, kUint32BinsTypeSize);
        SetIndexAllDense(page, gmat.index.data<uint32_t>(), gmat, nrow, nfeature, noMissingValues,
                         n_threads);
      }
      /* For sparse DMatrix gmat.index.getBinTypeSize() returns always kUint32BinsTypeSize
         but for ColumnMatrix we still have a chance to reduce the memory consumption */
    } else {
      SetIndex(page, gmat.index.data<uint32_t>(), gmat, nfeature);
    }
  }

  /* Fetch an individual column. This code should be used with type swith
     to determine type of bin id's */
  template <bool any_missing>
  std::unique_ptr<const Column> GetColumn(unsigned fid) const {
    const size_t feature_offset = feature_offsets_[fid];  // to get right place for certain feature
    const size_t column_size = feature_offsets_[fid + 1] - feature_offset;
    auto bin_index = Span<const int32_t>{index_}.subspan(feature_offset, column_size);
    std::unique_ptr<const Column> res;
    if (type_[fid] == ColumnType::kDenseColumn) {
      CHECK_EQ(any_missing, any_missing_);
      res.reset(new DenseColumn<any_missing>(type_[fid], bin_index, index_base_[fid]));
    } else {
      res.reset(new SparseColumn(type_[fid], bin_index, index_base_[fid],
                                 {&row_ind_[feature_offset], column_size}));
    }
    return res;
  }

  template <typename T>
  inline void SetIndexAllDense(SparsePage const& page, T const* row_index,
                               const GHistIndexMatrix& gmat, const size_t nrow,
                               const size_t nfeature, const bool noMissingValues,
                               int32_t n_threads) {
    bst_bin_t* local_index = index_.data();

    /* missing values make sense only for column with type kDenseColumn,
       and if no missing values were observed it could be handled much faster. */
    if (noMissingValues) {
      ParallelFor(nrow, n_threads, [&](auto rid) {
        const size_t ibegin = rid * nfeature;
        const size_t iend = (rid + 1) * nfeature;
        size_t j = 0;
        for (size_t i = ibegin; i < iend; ++i, ++j) {
          const size_t idx = feature_offsets_[j];
          local_index[idx + rid] = static_cast<bst_bin_t>(row_index[i]);
        }
      });
    } else {
      /* to handle rows in all batches, sum of all batch sizes equal to gmat.row_ptr.size() - 1 */
      auto get_bin_idx = [&](auto bin_id, auto rid, bst_feature_t fid) {
        const size_t idx = feature_offsets_[fid];
        /* rbegin allows to store indexes from specific SparsePage batch */
        local_index[idx + rid] = static_cast<bst_bin_t>(bin_id);
      };
      this->SetIndexSparse(page, row_index, gmat, nfeature, get_bin_idx);
    }
  }

  // FIXME(jiamingy): In the future we might want to simply use binary search to simplify
  // this and remove the dependency on SparsePage.  This way we can have quantilized
  // matrix for host similar to `DeviceQuantileDMatrix`.
  template <typename T, typename BinFn>
  void SetIndexSparse(SparsePage const& batch, T* row_index, const GHistIndexMatrix& gmat,
                      const size_t nfeature, BinFn&& assign_bin) {
    std::vector<size_t> num_nonzeros(nfeature, 0ul);
    const xgboost::Entry* data_ptr = batch.data.HostVector().data();
    const std::vector<bst_row_t>& offset_vec = batch.offset.HostVector();
    auto rbegin = 0;
    const size_t batch_size = gmat.Size();
    CHECK_LT(batch_size, offset_vec.size());

    for (size_t rid = 0; rid < batch_size; ++rid) {
      const size_t ibegin = gmat.row_ptr[rbegin + rid];
      const size_t iend = gmat.row_ptr[rbegin + rid + 1];
      const size_t size = offset_vec[rid + 1] - offset_vec[rid];
      SparsePage::Inst inst = {data_ptr + offset_vec[rid], size};

      CHECK_EQ(ibegin + inst.size(), iend);
      size_t j = 0;
      for (size_t i = ibegin; i < iend; ++i, ++j) {
        const uint32_t bin_id = row_index[i];
        auto fid = inst[j].index;
        assign_bin(bin_id, rid, fid);
      }
    }
  }

  void SetIndex(SparsePage const& page, uint32_t const* row_index, const GHistIndexMatrix& gmat,
                const size_t nfeature) {
    bst_bin_t* local_index = index_.data();
    std::vector<size_t> num_nonzeros;
    num_nonzeros.resize(nfeature);
    std::fill(num_nonzeros.begin(), num_nonzeros.end(), 0);

    auto get_bin_idx = [&](auto bin_id, auto rid, bst_feature_t fid) {
      if (type_[fid] == kDenseColumn) {
        auto* begin = &local_index[feature_offsets_[fid]];
        begin[rid] = bin_id - index_base_[fid];
      } else {
        auto* begin = &local_index[feature_offsets_[fid]];
        begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
        row_ind_[feature_offsets_[fid] + num_nonzeros[fid]] = rid;
        ++num_nonzeros[fid];
      }
    };
    this->SetIndexSparse(page, row_index, gmat, nfeature, get_bin_idx);
  }

  // This is just an utility function
  bool NoMissingValues(const size_t n_elements, const size_t n_row, const size_t n_features) {
    return n_elements == n_features * n_row;
  }

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
    index_base_ = index_base;

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

    fo->Write(any_missing_);
    bytes += sizeof(any_missing_);

    return bytes;
  }

 private:
  std::vector<bst_bin_t> index_;

  std::vector<ColumnType> type_;
  std::vector<size_t> row_ind_;
  /* indicate where each column's index and row_ind is stored. */
  std::vector<size_t> feature_offsets_;

  // index_base_[fid]: least bin id for feature fid
  uint32_t const* index_base_;
  bool any_missing_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
