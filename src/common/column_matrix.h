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
#include <utility>

#include "../data/gradient_index.h"
#include "hist_util.h"

namespace xgboost {
namespace common {

/*! \brief column type */
enum ColumnType : uint8_t { kDenseColumn, kSparseColumn };

/*! \brief a column storage, to be used with ApplySplit. Note that each
    bin id is stored as index[i] + index_base.
    Different types of column index for each column allow
    to reduce the memory usage. */
class Column {
 public:
  using ByteType = uint8_t;
  using BinCmpType = int32_t;
  static constexpr BinCmpType kMissingId = -1;

  Column(ColumnType type, BinTypeSize bin_type_size,
         common::Span<const ByteType> index, const uint32_t index_base)
      : type_(type),
        bin_type_size_(bin_type_size),
        index_(index),
        index_base_(index_base) {}

  virtual ~Column() = default;

  uint32_t GetGlobalBinIdx(size_t idx) const {
    uint32_t res = index_base_;
    if (GetBinTypeSize() == kUint8BinsTypeSize) {
      res += GetFeatureBinIdx<BinTypeMap<kUint8BinsTypeSize>::Type>(idx);
    } else if (GetBinTypeSize() == kUint16BinsTypeSize) {
      res += GetFeatureBinIdx<BinTypeMap<kUint16BinsTypeSize>::Type>(idx);
    } else {
      res += GetFeatureBinIdx<BinTypeMap<kUint32BinsTypeSize>::Type>(idx);
    }
    return res;
  }

  template <typename BinIdxType>
  BinIdxType GetFeatureBinIdx(size_t idx) const {
    const BinIdxType * ptr = reinterpret_cast<const BinIdxType *>(index_.data());
    return ptr[idx];
  }

  uint32_t GetBaseIdx() const { return index_base_; }

  template <typename BinIdxType>
  common::Span<const BinIdxType> GetFeatureBinIdxPtr() const {
    return { reinterpret_cast<BinIdxType>(index_), index_.size() / sizeof(BinIdxType)};
  }

  ColumnType GetType() const { return type_; }

  BinTypeSize GetBinTypeSize() const { return bin_type_size_; }

  /* returns number of elements in column */
  size_t Size() const { return index_.size() / bin_type_size_; }

 private:
  /* type of column */
  ColumnType type_;
  /* size of bin type idx*/
  BinTypeSize bin_type_size_;
  /* bin indexes in range [0, max_bins - 1] */
  common::Span<const ByteType> index_;
  /* bin index offset for specific feature */
  bst_bin_t const index_base_;
};

class SparseColumn: public Column {
 public:
  SparseColumn(BinTypeSize bin_type_size, common::Span<const ByteType> index,
              uint32_t index_base, common::Span<const size_t> row_ind)
      : Column(ColumnType::kSparseColumn, bin_type_size, index, index_base),
        row_ind_(row_ind) {}

  const size_t* GetRowData() const { return row_ind_.data(); }

  template <typename BinIdxType, typename CastType = Column::BinCmpType>
  CastType GetBinIdx(size_t rid, size_t* state) const {
    const size_t column_size = this->Size();
    if (!((*state) < column_size)) {
      return static_cast<CastType>(this->kMissingId);
    }
    while ((*state) < column_size && GetRowIdx(*state) < rid) {
      ++(*state);
    }
    if (((*state) < column_size) && GetRowIdx(*state) == rid) {
      return static_cast<CastType>(this->GetFeatureBinIdx<BinIdxType>(*state));
    } else {
      return static_cast<CastType>(this->kMissingId);
    }
  }

  /*
   * Additional method for maintaining the backward compatibility 
   * with the non-optimized partition builder
   * TODO: Delete after the optimizied partition builder will be implemented
   */
  template <typename CastType = Column::BinCmpType>
  CastType GetBinId(size_t rid, size_t* state) const {
    const size_t column_size = this->Size();
    if (!((*state) < column_size)) {
      return static_cast<CastType>(this->kMissingId);
    }
    while ((*state) < column_size && GetRowIdx(*state) < rid) {
      ++(*state);
    }
    if (((*state) < column_size) && GetRowIdx(*state) == rid) {
      return static_cast<CastType>(this->GetGlobalBinIdx(*state));
    } else {
      return static_cast<CastType>(this->kMissingId);
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

  size_t GetRowIdx(size_t idx) const {
    return row_ind_.data()[idx];
  }

 private:
  /* indexes of rows */
  common::Span<const size_t> row_ind_;
};

class DenseColumn: public Column {
 public:
  DenseColumn(BinTypeSize bin_type_size, common::Span<const ByteType> index,
              uint32_t index_base, const bool any_missing,
              const std::vector<ByteType>& missing_flags,
              size_t feature_offset)
      : Column(ColumnType::kDenseColumn, bin_type_size, index, index_base),
        any_missing_(any_missing),
        missing_flags_(missing_flags),
        feature_offset_(feature_offset) {}
  bool IsMissing(size_t idx) const {
  const bool res = missing_flags_[feature_offset_ + idx];
    return res;}

  template <typename BinIdxType, typename CastType = Column::BinCmpType>
  CastType GetBinIdx(size_t idx, size_t* state) const {
    return static_cast<CastType>(
              (any_missing_ && IsMissing(idx))
              ? this->kMissingId
              : this->GetFeatureBinIdx<BinIdxType>(idx));
  }

  /*
   * Additional method for maintaining the backward compatibility 
   * with the non-optimized partition builder
   * TODO: Delete after the optimizied partition builder will be implemented
   */
  template <typename CastType = Column::BinCmpType>
  CastType GetBinId(size_t idx, size_t* state) const {
    return static_cast<CastType>(
              (any_missing_ && IsMissing(idx))
              ? this->kMissingId
              : this->GetGlobalBinIdx(idx));
  }

  size_t GetInitialState(const size_t first_row_id) const { return 0; }

 private:
  const bool any_missing_;
  /* flags for missing values in dense columns */
  const std::vector<ByteType>& missing_flags_;
  size_t feature_offset_;
};

class ColumnView final {
 public:
  ColumnView() = delete;
  explicit ColumnView(const SparseColumn * sparse_clmn_ptr) :
    sparse_clmn_ptr_(sparse_clmn_ptr),
    dense_clmn_ptr_(nullptr) { }
  explicit ColumnView(const DenseColumn * dense_clmn_ptr) :
    sparse_clmn_ptr_(nullptr),
    dense_clmn_ptr_(dense_clmn_ptr) { }

  uint32_t GetGlobalBinIdx(size_t idx) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetGlobalBinIdx(idx)
                            : dense_clmn_ptr_->GetGlobalBinIdx(idx);
  }

  bool IsMissing(size_t idx) const {
    return sparse_clmn_ptr_ ? false
                            : dense_clmn_ptr_->IsMissing(idx);
  }

  template <typename BinIdxType>
  BinIdxType GetFeatureBinIdx(size_t idx) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetFeatureBinIdx<BinIdxType>(idx)
                            : dense_clmn_ptr_->GetFeatureBinIdx<BinIdxType>(idx);
  }

  uint32_t GetBaseIdx() const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetBaseIdx()
                            : dense_clmn_ptr_->GetBaseIdx();
  }

  template <typename BinIdxType>
  common::Span<const BinIdxType> GetFeatureBinIdxPtr() const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetFeatureBinIdxPtr<BinIdxType>()
                            : dense_clmn_ptr_->GetFeatureBinIdxPtr<BinIdxType>();
  }

  ColumnType GetType() const {
    return sparse_clmn_ptr_ ? ColumnType::kSparseColumn : ColumnType::kDenseColumn;
  }

  size_t Size() const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->Size() : dense_clmn_ptr_->Size();
  }

  const size_t* GetRowData() const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetRowData() : nullptr;
  }

  template <typename BinIdxType, typename CastType = Column::BinCmpType>
  CastType GetBinIdx(size_t rid, size_t* state) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetBinIdx<BinIdxType, CastType>(rid, state)
                            : dense_clmn_ptr_->GetBinIdx<BinIdxType, CastType>(rid, state);
  }

  /*
   * Additional method for maintaining the backward compatibility 
   * with the non-optimized partition builder
   * TODO: Delete after the optimizied partition builder will be implemented
   */
  template <typename CastType = Column::BinCmpType>
  CastType GetBinId(size_t rid, size_t* state) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetBinId<CastType>(rid, state)
                            : dense_clmn_ptr_->GetBinId<CastType>(rid, state);
  }

  size_t GetInitialState(const size_t first_row_id) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetInitialState(first_row_id)
                            : dense_clmn_ptr_->GetInitialState(first_row_id);
  }

  size_t GetRowIdx(size_t idx) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetRowIdx(idx) : 0;
  }

 private:
  const SparseColumn * sparse_clmn_ptr_;
  const DenseColumn* dense_clmn_ptr_;
};

/*! \brief a collection of columns, with support for construction from
    GHistIndexMatrix. */
class ColumnMatrix {
 public:
  using ByteType = uint8_t;
  using ColumnListType = std::vector<std::shared_ptr<const Column>>;
  using ColumnViewListType = std::vector<std::shared_ptr<const ColumnView>>;
  // get number of features
  bst_feature_t GetNumFeature() const { return static_cast<bst_feature_t>(type_.size()); }
  // get index data ptr
  template <typename Data>
  const Data* GetIndexData() const {
    return reinterpret_cast<const Data*>(index_.data());
  }

  // get index data ptr
  const ByteType* GetIndexData() const {
    return index_.data();
  }

  // construct column matrix from GHistIndexMatrix
  inline void Init(SparsePage const& page, const GHistIndexMatrix& gmat, double sparse_threshold,
                   int32_t n_thread) {
    auto const n_feature = static_cast<bst_feature_t>(gmat.cut.Ptrs().size() - 1);
    const size_t n_row = gmat.row_ptr.size() - 1;
    // identify type of each column
    feature_counts_.resize(n_feature);
    type_.resize(n_feature);
    std::fill(feature_counts_.begin(), feature_counts_.end(), 0);
    uint32_t max_val = std::numeric_limits<uint32_t>::max();
    for (bst_feature_t fid = 0; fid < n_feature; ++fid) {
      CHECK_LE(gmat.cut.Ptrs()[fid + 1] - gmat.cut.Ptrs()[fid], max_val);
    }
    bool all_dense = gmat.IsDense();
    gmat.GetFeatureCounts(&feature_counts_[0]);
    // classify features
    for (bst_feature_t fid = 0; fid < n_feature; ++fid) {
      if (static_cast<double>(feature_counts_[fid]) < sparse_threshold * n_row) {
        type_[fid] = kSparseColumn;
        all_dense = false;
      } else {
        type_[fid] = kDenseColumn;
      }
    }

    // want to compute storage boundary for each feature
    // using variants of prefix sum scan
    feature_offsets_.resize(n_feature + 1);
    size_t accum_index_ = 0;
    feature_offsets_[0] = accum_index_;
    for (bst_feature_t fid = 1; fid < n_feature + 1; ++fid) {
      if (type_[fid - 1] == kDenseColumn) {
        accum_index_ += static_cast<size_t>(n_row);
      } else {
        accum_index_ += feature_counts_[fid - 1];
      }
      feature_offsets_[fid] = accum_index_;
    }

    SetTypeSize(gmat.max_num_bins);
    index_.resize(feature_offsets_[n_feature] * bin_type_size_, 0);
    if (!all_dense) {
      row_ind_.resize(feature_offsets_[n_feature]);
    }

    // store least bin id for each feature
    cut_ = gmat.cut;
    index_base_ = const_cast<uint32_t*>(cut_.Ptrs().data());

    const bool no_missing_values = NoMissingValues(gmat.row_ptr[n_row], n_row, n_feature);
    any_missing_ = !no_missing_values;

    missing_flags_.clear();
    if (no_missing_values) {
      missing_flags_.resize(feature_offsets_[n_feature], false);
    } else {
      missing_flags_.resize(feature_offsets_[n_feature], true);
    }

    // pre-fill index_ for dense columns
    if (all_dense) {
      BinTypeSize gmat_bin_size = gmat.index.GetBinTypeSize();
      if (gmat_bin_size == kUint8BinsTypeSize) {
        SetIndexAllDense(page, gmat.index.data<uint8_t>(), gmat, n_row, n_feature,
                         no_missing_values, n_thread);
      } else if (gmat_bin_size == kUint16BinsTypeSize) {
        SetIndexAllDense(page, gmat.index.data<uint16_t>(), gmat, n_row, n_feature,
                         no_missing_values, n_thread);
      } else {
        CHECK_EQ(gmat_bin_size, kUint32BinsTypeSize);
        SetIndexAllDense(page, gmat.index.data<uint32_t>(), gmat, n_row, n_feature,
                         no_missing_values, n_thread);
      }
      /* For sparse DMatrix gmat.index.getBinTypeSize() returns always kUint32BinsTypeSize
         but for ColumnMatrix we still have a chance to reduce the memory consumption */
    } else {
      if (bin_type_size_ == kUint8BinsTypeSize) {
        SetIndex<uint8_t>(page, gmat.index.data<uint32_t>(), gmat, n_feature);
      } else if (bin_type_size_ == kUint16BinsTypeSize) {
        SetIndex<uint16_t>(page, gmat.index.data<uint32_t>(), gmat, n_feature);
      } else {
        CHECK_EQ(bin_type_size_, kUint32BinsTypeSize);
        SetIndex<uint32_t>(page, gmat.index.data<uint32_t>(), gmat, n_feature);
      }
    }
    FillColumnViewList(n_feature);
  }

  const ColumnListType& GetColumnList() const { return column_list_; }
  const ColumnViewListType& GetColumnViewList() const { return column_view_list_; }

  /* Set the number of bytes based on numeric limit of maximum number of bins provided by user */
  void SetTypeSize(size_t max_num_bins) {
    if ((max_num_bins - 1) <= static_cast<int>(std::numeric_limits<uint8_t>::max())) {
      bin_type_size_ = kUint8BinsTypeSize;
    } else if ((max_num_bins - 1) <= static_cast<int>(std::numeric_limits<uint16_t>::max())) {
      bin_type_size_ = kUint16BinsTypeSize;
    } else {
      bin_type_size_ = kUint32BinsTypeSize;
    }
  }

  template <typename T>
  inline void SetIndexAllDense(SparsePage const& page, T const* index,
                               const GHistIndexMatrix& gmat,
                               const size_t n_row, const size_t n_feature,
                               const bool no_missing_values,
                               int32_t n_thread) {
    T* local_index = reinterpret_cast<T*>(&index_[0]);

    /* missing values make sense only for column with type kDenseColumn,
       and if no missing values were observed it could be handled much faster. */
    if (no_missing_values) {
      ParallelFor(n_row, n_thread, [&](auto rid) {
        const size_t ibegin = rid * n_feature;
        const size_t iend = (rid + 1) * n_feature;
        size_t j = 0;
        for (size_t i = ibegin; i < iend; ++i, ++j) {
          const size_t idx = feature_offsets_[j];
          local_index[idx + rid] = index[i];
        }
      });
    } else {
      /* to handle rows in all batches, sum of all batch sizes equal to gmat.row_ptr.size() - 1 */
      auto get_bin_idx = [&](auto bin_id, auto rid, bst_feature_t fid) {
        const size_t idx = feature_offsets_[fid];
        /* rbegin allows to store indexes from specific SparsePage batch */
        local_index[idx + rid] = bin_id;

        missing_flags_[idx + rid] = false;
      };
      this->SetIndexSparse(page, index, gmat, n_feature, get_bin_idx);
    }
  }

  // FIXME(jiamingy): In the future we might want to simply use binary search to simplify
  // this and remove the dependency on SparsePage.  This way we can have quantilized
  // matrix for host similar to `DeviceQuantileDMatrix`.
  template <typename T, typename BinFn>
  void SetIndexSparse(SparsePage const& batch, T* index, const GHistIndexMatrix& gmat,
                      const size_t n_feature, BinFn&& assign_bin) {
    std::vector<size_t> num_nonzeros(n_feature, 0ul);
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
        const uint32_t bin_id = index[i];
        auto fid = inst[j].index;
        assign_bin(bin_id, rid, fid);
      }
    }
  }

  template <typename T>
  inline void SetIndex(SparsePage const& page, uint32_t const* index, const GHistIndexMatrix& gmat,
                       const size_t n_feature) {
    T* local_index = reinterpret_cast<T*>(&index_[0]);
    std::vector<size_t> num_nonzeros;
    num_nonzeros.resize(n_feature);
    std::fill(num_nonzeros.begin(), num_nonzeros.end(), 0);

    auto get_bin_idx = [&](auto bin_id, auto rid, bst_feature_t fid) {
      if (type_[fid] == kDenseColumn) {
        T* begin = &local_index[feature_offsets_[fid]];
        begin[rid] = bin_id - index_base_[fid];
        missing_flags_[feature_offsets_[fid] + rid] = false;
      } else {
        T* begin = &local_index[feature_offsets_[fid]];
        begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
        row_ind_[feature_offsets_[fid] + num_nonzeros[fid]] = rid;
        ++num_nonzeros[fid];
      }
    };
    this->SetIndexSparse(page, index, gmat, n_feature, get_bin_idx);
  }

  BinTypeSize GetTypeSize() const {
    return bin_type_size_;
  }

  size_t GetSizeMissing() const {
    return missing_flags_.size();
  }

  const std::vector<ByteType>* GetMissing() const {
    return &missing_flags_;
  }

  // This is just an utility function
  bool NoMissingValues(const size_t n_element, const size_t n_row, const size_t n_feature) {
    return n_element == n_feature * n_row;
  }

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
    fi->Read(&missing_flags_);
    index_base_ = index_base;
#if !DMLC_LITTLE_ENDIAN
    std::underlying_type<BinTypeSize>::type v;
    fi->Read(&v);
    bin_type_size_ = static_cast<BinTypeSize>(v);
#else
    fi->Read(&bin_type_size_);
#endif
    fi->Read(&any_missing_);

    FillColumnViewList(type_.size());

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
    write_vec(missing_flags_);
#if !DMLC_LITTLE_ENDIAN
    auto v = static_cast<std::underlying_type<BinTypeSize>::type>(bin_type_size_);
    fo->Write(v);
#else
    fo->Write(bin_type_size_);
#endif  // DMLC_LITTLE_ENDIAN
    bytes += sizeof(bin_type_size_);
    fo->Write(any_missing_);
    bytes += sizeof(any_missing_);

    return bytes;
  }

  const size_t* GetRowId() const {
    return row_ind_.data();
  }

 private:
  template <typename ColumnType, typename ... Args>
  void AddColumnToList(size_t fid, Args&& ... args) {
        auto clmn = std::make_shared<const ColumnType>(std::forward<Args>(args) ...);
        column_list_[fid] = clmn;
        column_view_list_[fid] = std::make_shared<const ColumnView>(clmn.get());
  }

  /* Filling list of helpers for operating with columns */
  void FillColumnViewList(const size_t n_feature) {
    column_list_.resize(n_feature);
    column_view_list_.resize(n_feature);
    for (auto fid = 0; fid < n_feature; ++fid) {
      // to get right place for certain feature
      const size_t feature_offset = feature_offsets_[fid];
      const size_t column_size = feature_offsets_[fid + 1] - feature_offset;
      common::Span<const ByteType> bin_index = { &index_[feature_offset * bin_type_size_],
                                                   column_size * bin_type_size_ };

      if (type_[fid] == ColumnType::kDenseColumn) {
        AddColumnToList<DenseColumn>(fid, GetTypeSize(), bin_index,
                              index_base_[fid],
                              any_missing_, missing_flags_, feature_offset);
      } else {
        AddColumnToList<SparseColumn>(fid, GetTypeSize(), bin_index,
                              index_base_[fid],
                              common::Span<const size_t>(&row_ind_[feature_offset], column_size));
      }
    }
  }

 private:
  std::vector<ByteType> index_;

  std::vector<size_t> feature_counts_;
  std::vector<ColumnType> type_;
  std::vector<size_t> row_ind_;
  /* indicate where each column's index and row_ind is stored. */
  std::vector<size_t> feature_offsets_;

  // index_base_[fid]: least bin id for feature fid
  uint32_t const* index_base_ = nullptr;
  std::vector<ByteType> missing_flags_;
  BinTypeSize bin_type_size_ = static_cast<BinTypeSize>(0);
  bool any_missing_;
  common::HistogramCuts cut_;

  ColumnListType column_list_;
  ColumnViewListType column_view_list_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_"
