/**
 * Copyright 2017-2025, XGBoost Contributors
 * \file column_matrix.h
 * \brief Utility for fast column-wise access
 * \author Philip Cho
 */

#ifndef XGBOOST_COMMON_COLUMN_MATRIX_H_
#define XGBOOST_COMMON_COLUMN_MATRIX_H_

#include <algorithm>
#include <cstddef>  // for size_t, byte
#include <cstdint>  // for uint8_t
#include <limits>
#include <memory>
#include <type_traits>  // for enable_if_t, is_same_v, is_signed_v

#include "../data/adapter.h"
#include "../data/gradient_index.h"
#include "bitfield.h"  // for RBitField8
#include "hist_util.h"
#include "ref_resource_view.h"  // for RefResourceView
#include "xgboost/base.h"       // for bst_bin_t
#include "xgboost/span.h"       // for Span

namespace xgboost::common {
class ColumnMatrix;
class AlignedFileWriteStream;
class AlignedResourceReadStream;

/*! \brief column type */
enum ColumnType : std::uint8_t { kDenseColumn, kSparseColumn };

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

  [[nodiscard]] bst_bin_t GetGlobalBinIdx(size_t idx) const {
    return index_base_ + static_cast<bst_bin_t>(index_.data()[idx]);
  }

  /* returns number of elements in column */
  [[nodiscard]] size_t Size() const { return index_.size(); }

 private:
  /* bin indexes in range [0, max_bins - 1] */
  common::Span<BinIdxType const> index_;
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

  [[nodiscard]] size_t const* RowIndices() const { return row_ind_.data(); }

 public:
  SparseColumnIter(common::Span<const BinIdxT> index, bst_bin_t least_bin_idx,
                   common::Span<const size_t> row_ind, bst_idx_t first_row_idx)
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

  [[nodiscard]] size_t GetRowIdx(size_t idx) const { return RowIndices()[idx]; }
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

/**
 * @brief Column stored as a dense vector. It might still contain missing values as
 *        indicated by the missing flags.
 */
template <typename BinIdxT, bool any_missing>
class DenseColumnIter : public Column<BinIdxT> {
 private:
  using Base = Column<BinIdxT>;
  /* flags for missing values in dense columns */
  LBitField32 missing_flags_;
  size_t feature_offset_;

 public:
  explicit DenseColumnIter(common::Span<const BinIdxT> index, bst_bin_t index_base,
                           LBitField32 missing_flags, size_t feature_offset)
      : Base{index, index_base}, missing_flags_{missing_flags}, feature_offset_{feature_offset} {}
  DenseColumnIter(DenseColumnIter const&) = delete;
  DenseColumnIter(DenseColumnIter&&) = default;

  [[nodiscard]] bool IsMissing(size_t ridx) const {
    return missing_flags_.Check(feature_offset_ + ridx);
  }

  bst_bin_t operator[](size_t ridx) const {
    if (any_missing) {
      return IsMissing(ridx) ? this->kMissingId : this->GetGlobalBinIdx(ridx);
    } else {
      return this->GetGlobalBinIdx(ridx);
    }
  }
};

/**
 * @brief Column major matrix for gradient index on CPU.
 *
 *    This matrix contains both dense columns and sparse columns, the type of the column
 *    is controlled by the sparse threshold parameter. When the number of missing values
 *    in a column is below the threshold it's classified as dense column.
 */
class ColumnMatrix {
  /**
   * @brief A bit set for indicating whether an element in a dense column is missing.
   */
  struct MissingIndicator {
    using BitFieldT = LBitField32;
    using T = typename BitFieldT::value_type;

    BitFieldT missing;
    RefResourceView<T> storage;
    static_assert(std::is_same_v<T, std::uint32_t>);

    template <typename U>
    [[nodiscard]] std::enable_if_t<!std::is_signed_v<U>, U> static InitValue(bool init) {
      return init ? ~U{0} : U{0};
    }

    MissingIndicator() = default;
    /**
     * @param n_elements Size of the bit set
     * @param init       Initialize the indicator to true or false.
     */
    MissingIndicator(std::size_t n_elements, bool init) {
      auto m_size = missing.ComputeStorageSize(n_elements);
      storage = common::MakeFixedVecWithMalloc(m_size, InitValue<T>(init));
      this->InitView();
    }
    /** @brief Set the i^th element to be a valid element (instead of missing). */
    void SetValid(typename LBitField32::index_type i) { missing.Clear(i); }
    /** @brief assign the storage to the view. */
    void InitView() {
      missing = LBitField32{Span{storage.data(), static_cast<size_t>(storage.size())}};
    }

    void GrowTo(std::size_t n_elements, bool init) {
      CHECK(storage.Resource()->Type() == ResourceHandler::kMalloc)
          << "[Internal Error]: Cannot grow the vector when external memory is used.";
      auto m_size = missing.ComputeStorageSize(n_elements);
      CHECK_GE(m_size, storage.size());
      if (m_size == storage.size()) {
        return;
      }
      // grow the storage
      auto resource = std::dynamic_pointer_cast<common::MallocResource>(storage.Resource());
      CHECK(resource);
      resource->Resize(m_size * sizeof(T), InitValue<std::byte>(init));
      storage = RefResourceView<T>{resource->DataAs<T>(), m_size, resource};

      this->InitView();
    }
  };

  void InitStorage(GHistIndexMatrix const& gmat, double sparse_threshold);

  template <typename ColumnBinT, typename BinT, typename RIdx>
  void SetBinSparse(BinT bin_id, RIdx rid, bst_feature_t fid, ColumnBinT* local_index) {
    if (type_[fid] == kDenseColumn) {
      ColumnBinT* begin = &local_index[feature_offsets_[fid]];
      begin[rid] = bin_id - index_base_[fid];
      // not thread-safe with bit field.
      // FIXME(jiamingy): We can directly assign kMissingId to the index to avoid missing
      // flags.
      missing_.SetValid(feature_offsets_[fid] + rid);
    } else {
      ColumnBinT* begin = &local_index[feature_offsets_[fid]];
      begin[num_nonzeros_[fid]] = bin_id - index_base_[fid];
      row_ind_[feature_offsets_[fid] + num_nonzeros_[fid]] = rid;
      ++num_nonzeros_[fid];
    }
  }

 public:
  // get number of features
  [[nodiscard]] bst_feature_t GetNumFeature() const {
    return static_cast<bst_feature_t>(type_.size());
  }

  ColumnMatrix() = default;
  ColumnMatrix(GHistIndexMatrix const& gmat, double sparse_threshold) {
    this->InitStorage(gmat, sparse_threshold);
  }

  /**
   * @brief Initialize ColumnMatrix from GHistIndexMatrix with reference to the original
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
   * @brief Initialize ColumnMatrix from GHistIndexMatrix without reference to actual
   *        data.
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

  [[nodiscard]] bool IsInitialized() const { return !type_.empty(); }

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
  auto SparseColumn(bst_feature_t fidx, bst_idx_t first_row_idx) const {
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
    return DenseColumnIter<BinIdxType, any_missing>{
        bin_index, static_cast<bst_bin_t>(index_base_[fidx]), missing_.missing, feature_offset};
  }

  // all columns are dense column and has no missing value
  // FIXME(jiamingy): We don't need a column matrix if there's no missing value.
  template <typename RowBinIdxT>
  void SetIndexNoMissing(bst_idx_t base_rowid, RowBinIdxT const* row_index, const size_t n_samples,
                         const size_t n_features, int32_t n_threads) {
    missing_.GrowTo(feature_offsets_[n_features], false);

    DispatchBinType(bins_type_size_, [&](auto t) {
      using ColumnBinT = decltype(t);
      auto column_index = Span<ColumnBinT>{reinterpret_cast<ColumnBinT*>(index_.data()),
                                           static_cast<size_t>(index_.size() / sizeof(ColumnBinT))};
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

    missing_.GrowTo(feature_offsets_[n_features], true);
    auto const* row_index = gmat.index.data<std::uint32_t>() + gmat.row_ptr[base_rowid];
    if (num_nonzeros_.empty()) {
      num_nonzeros_ = common::MakeFixedVecWithMalloc(n_features, std::size_t{0});
    } else {
      CHECK_EQ(num_nonzeros_.size(), n_features);
    }

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

    missing_ = MissingIndicator{feature_offsets_[n_features], true};
    num_nonzeros_ = common::MakeFixedVecWithMalloc(n_features, std::size_t{0});

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

  [[nodiscard]] BinTypeSize GetTypeSize() const { return bins_type_size_; }
  [[nodiscard]] auto GetColumnType(bst_feature_t fidx) const { return type_[fidx]; }

  // And this returns part of state
  [[nodiscard]] bool AnyMissing() const { return any_missing_; }

  // IO procedures for external memory.
  [[nodiscard]] bool Read(AlignedResourceReadStream* fi, uint32_t const* index_base);
  [[nodiscard]] std::size_t Write(AlignedFileWriteStream* fo) const;
  [[nodiscard]] MissingIndicator const& Missing() const { return missing_; }

 private:
  RefResourceView<std::uint8_t> index_;

  RefResourceView<ColumnType> type_;
  /** @brief indptr of a CSC matrix. */
  RefResourceView<std::size_t> row_ind_;
  /** @brief indicate where each column's index and row_ind is stored. */
  RefResourceView<std::size_t> feature_offsets_;
  /** @brief The number of nnz of each column. */
  RefResourceView<std::size_t> num_nonzeros_;

  // index_base_[fid]: least bin id for feature fid
  std::uint32_t const* index_base_;

  MissingIndicator missing_;

  BinTypeSize bins_type_size_;
  bool any_missing_;
};
}  // namespace xgboost::common
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
