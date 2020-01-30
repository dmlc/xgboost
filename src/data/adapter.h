/*!
 *  Copyright (c) 2019 by Contributors
 * \file adapter.h
 */
#ifndef XGBOOST_DATA_ADAPTER_H_
#define XGBOOST_DATA_ADAPTER_H_
#include <dmlc/data.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/data.h"

namespace xgboost {
namespace data {

/**  External data formats should implement an adapter as below. The
 * adapter provides a uniform access to data outside xgboost, allowing
 * construction of DMatrix objects from a range of sources without duplicating
 * code.
 *
 * The adapter object is an iterator that returns batches of data. Each batch
 * contains a number of "lines". A line represents a set of elements from a
 * sparse input matrix, normally a row in the case of a CSR matrix or a column
 * for a CSC matrix. Typically in sparse matrix formats we can efficiently
 * access subsets of elements at a time, but cannot efficiently lookups elements
 * by random access, hence the "line" abstraction, allowing the sparse matrix to
 * return subsets of elements efficiently. Individual elements are described by
 * a COO tuple (row index, column index, value).
 *
 * This abstraction allows us to read through different sparse matrix formats
 * using the same interface. In particular we can write a DMatrix constructor
 * that uses the same code to construct itself from a CSR matrix, CSC matrix,
 * dense matrix, csv, libsvm file, or potentially other formats. To see why this
 * is necessary, imagine we have 5 external matrix formats and 5 internal
 * DMatrix types where each DMatrix needs a custom constructor for each possible
 * input. The number of constructors is 5*5=25. Using an abstraction over the
 * input data types the number of constructors is reduced to 5, as each DMatrix
 * is oblivious to the external data format. Adding a new input source is simply
 * a case of implementing an adapter.
 *
 * Most of the below adapters do not need more than one batch as the data
 * originates from an in memory source. The file adapter does require batches to
 * avoid loading the entire file in memory.
 *
 * An important detail is empty row/column handling. Files loaded from disk do
 * not provide meta information about the number of rows/columns to expect, this
 * needs to be inferred during construction. Other sparse formats may specify a
 * number of rows/columns, but we can encounter entirely sparse rows or columns,
 * leading to disagreement between the inferred number and the meta-info
 * provided. To resolve this, adapters have methods specifying the number of
 * rows/columns expected, these methods may return zero where these values must
 * be inferred from data. A constructed DMatrix should agree with the input
 * source on numbers of rows/columns, appending empty rows if necessary.
 *  */

/** \brief An adapter can return this value for number of rows or columns
 * indicating that this value is currently unknown and should be inferred while
 * passing over the data. */
constexpr size_t kAdapterUnknownSize = std::numeric_limits<size_t >::max();

struct COOTuple {
  XGBOOST_DEVICE COOTuple(size_t row_idx, size_t column_idx, float value)
      : row_idx(row_idx), column_idx(column_idx), value(value) {}

  size_t row_idx{0};
  size_t column_idx{0};
  float value{0};
};

namespace detail {

/**
 * \brief Simplifies the use of DataIter when there is only one batch.
 */
template <typename DType>
class SingleBatchDataIter : dmlc::DataIter<DType> {
 public:
  void BeforeFirst() override { counter_ = 0; }
  bool Next() override {
    if (counter_ == 0) {
      counter_++;
      return true;
    }
    return false;
  }

 private:
  int counter_{0};
};

/** \brief Indicates this data source cannot contain meta-info such as labels,
 * weights or qid. */
class NoMetaInfo {
 public:
  const float* Labels() const { return nullptr; }
  const float* Weights() const { return nullptr; }
  const uint64_t* Qid() const { return nullptr; }
  const float* BaseMargin() const { return nullptr; }
};

};  // namespace detail

class CSRAdapterBatch : public detail::NoMetaInfo {
 public:
  class Line {
   public:
    Line(size_t row_idx, size_t size, const unsigned* feature_idx,
         const float* values)
        : row_idx_(row_idx),
          size_(size),
          feature_idx_(feature_idx),
          values_(values) {}

    size_t Size() const { return size_; }
    COOTuple GetElement(size_t idx) const {
      return COOTuple{row_idx_, feature_idx_[idx], values_[idx]};
    }

   private:
    size_t row_idx_;
    size_t size_;
    const unsigned* feature_idx_;
    const float* values_;
  };
  CSRAdapterBatch(const size_t* row_ptr, const unsigned* feature_idx,
                  const float* values, size_t num_rows, size_t num_elements,
                  size_t num_features)
      : row_ptr_(row_ptr),
        feature_idx_(feature_idx),
        values_(values),
        num_rows_(num_rows) {}
  const Line GetLine(size_t idx) const {
    size_t begin_offset = row_ptr_[idx];
    size_t end_offset = row_ptr_[idx + 1];
    return Line(idx, end_offset - begin_offset, &feature_idx_[begin_offset],
                &values_[begin_offset]);
  }
  size_t Size() const { return num_rows_; }

 private:
  const size_t* row_ptr_;
  const unsigned* feature_idx_;
  const float* values_;
  size_t num_rows_;
};

class CSRAdapter : public detail::SingleBatchDataIter<CSRAdapterBatch> {
 public:
  CSRAdapter(const size_t* row_ptr, const unsigned* feature_idx,
             const float* values, size_t num_rows, size_t num_elements,
             size_t num_features)
      : batch_(row_ptr, feature_idx, values, num_rows, num_elements,
               num_features),
        num_rows_(num_rows),
        num_columns_(num_features) {}
  const CSRAdapterBatch& Value() const override { return batch_; }
  size_t NumRows() const { return num_rows_; }
  size_t NumColumns() const { return num_columns_; }

 private:
  CSRAdapterBatch batch_;
  size_t num_rows_;
  size_t num_columns_;
};

class DenseAdapterBatch : public detail::NoMetaInfo {
 public:
  DenseAdapterBatch(const float* values, size_t num_rows, size_t num_features)
      : values_(values),
        num_rows_(num_rows),
        num_features_(num_features) {}

 private:
  class Line {
   public:
    Line(const float* values, size_t size, size_t row_idx)
        : row_idx_(row_idx), size_(size), values_(values) {}

    size_t Size() const { return size_; }
    COOTuple GetElement(size_t idx) const {
      return COOTuple{row_idx_, idx, values_[idx]};
    }

   private:
    size_t row_idx_;
    size_t size_;
    const float* values_;
  };

 public:
  size_t Size() const { return num_rows_; }
  const Line GetLine(size_t idx) const {
    return Line(values_ + idx * num_features_, num_features_, idx);
  }

 private:
  const float* values_;
  size_t num_rows_;
  size_t num_features_;
};

class DenseAdapter : public detail::SingleBatchDataIter<DenseAdapterBatch> {
 public:
  DenseAdapter(const float* values, size_t num_rows, size_t num_features)
      : batch_(values, num_rows, num_features),
        num_rows_(num_rows),
        num_columns_(num_features) {}
  const DenseAdapterBatch& Value() const override { return batch_; }

  size_t NumRows() const { return num_rows_; }
  size_t NumColumns() const { return num_columns_; }

 private:
  DenseAdapterBatch batch_;
  size_t num_rows_;
  size_t num_columns_;
};

class CSCAdapterBatch : public detail::NoMetaInfo {
 public:
  CSCAdapterBatch(const size_t* col_ptr, const unsigned* row_idx,
                  const float* values, size_t num_features)
      : col_ptr_(col_ptr),
        row_idx_(row_idx),
        values_(values),
        num_features_(num_features) {}

 private:
  class Line {
   public:
    Line(size_t col_idx, size_t size, const unsigned* row_idx,
         const float* values)
        : col_idx_(col_idx), size_(size), row_idx_(row_idx), values_(values) {}

    size_t Size() const { return size_; }
    COOTuple GetElement(size_t idx) const {
      return COOTuple{row_idx_[idx], col_idx_, values_[idx]};
    }

   private:
    size_t col_idx_;
    size_t size_;
    const unsigned* row_idx_;
    const float* values_;
  };

 public:
  size_t Size() const { return num_features_; }
  const Line GetLine(size_t idx) const {
    size_t begin_offset = col_ptr_[idx];
    size_t end_offset = col_ptr_[idx + 1];
    return Line(idx, end_offset - begin_offset, &row_idx_[begin_offset],
                &values_[begin_offset]);
  }

 private:
  const size_t* col_ptr_;
  const unsigned* row_idx_;
  const float* values_;
  size_t num_features_;
};

class CSCAdapter : public detail::SingleBatchDataIter<CSCAdapterBatch> {
 public:
  CSCAdapter(const size_t* col_ptr, const unsigned* row_idx,
             const float* values, size_t num_features, size_t num_rows)
      : batch_(col_ptr, row_idx, values, num_features),
        num_rows_(num_rows),
        num_columns_(num_features) {}
  const CSCAdapterBatch& Value() const override { return batch_; }

  // JVM package sends 0 as unknown
  size_t NumRows() const {
    return num_rows_ == 0 ? kAdapterUnknownSize : num_rows_;
  }
  size_t NumColumns() const { return num_columns_; }

 private:
  CSCAdapterBatch batch_;
  size_t num_rows_;
  size_t num_columns_;
};

class DataTableAdapterBatch : public detail::NoMetaInfo {
 public:
  DataTableAdapterBatch(void** data, const char** feature_stypes,
                        size_t num_rows, size_t num_features)
      : data_(data),
        feature_stypes_(feature_stypes),
        num_features_(num_features),
        num_rows_(num_rows) {}

 private:
  enum class DTType : uint8_t {
    kFloat32 = 0,
    kFloat64 = 1,
    kBool8 = 2,
    kInt32 = 3,
    kInt8 = 4,
    kInt16 = 5,
    kInt64 = 6,
    kUnknown = 7
  };

  DTType DTGetType(std::string type_string) const {
    if (type_string == "float32") {
      return DTType::kFloat32;
    } else if (type_string == "float64") {
      return DTType::kFloat64;
    } else if (type_string == "bool8") {
      return DTType::kBool8;
    } else if (type_string == "int32") {
      return DTType::kInt32;
    } else if (type_string == "int8") {
      return DTType::kInt8;
    } else if (type_string == "int16") {
      return DTType::kInt16;
    } else if (type_string == "int64") {
      return DTType::kInt64;
    } else {
      LOG(FATAL) << "Unknown data table type.";
      return DTType::kUnknown;
    }
  }

  class Line {
    float DTGetValue(const void* column, DTType dt_type, size_t ridx) const {
      float missing = std::numeric_limits<float>::quiet_NaN();
      switch (dt_type) {
        case DTType::kFloat32: {
          float val = reinterpret_cast<const float*>(column)[ridx];
          return std::isfinite(val) ? val : missing;
        }
        case DTType::kFloat64: {
          double val = reinterpret_cast<const double*>(column)[ridx];
          return std::isfinite(val) ? static_cast<float>(val) : missing;
        }
        case DTType::kBool8: {
          bool val = reinterpret_cast<const bool*>(column)[ridx];
          return static_cast<float>(val);
        }
        case DTType::kInt32: {
          int32_t val = reinterpret_cast<const int32_t*>(column)[ridx];
          return val != (-2147483647 - 1) ? static_cast<float>(val) : missing;
        }
        case DTType::kInt8: {
          int8_t val = reinterpret_cast<const int8_t*>(column)[ridx];
          return val != -128 ? static_cast<float>(val) : missing;
        }
        case DTType::kInt16: {
          int16_t val = reinterpret_cast<const int16_t*>(column)[ridx];
          return val != -32768 ? static_cast<float>(val) : missing;
        }
        case DTType::kInt64: {
          int64_t val = reinterpret_cast<const int64_t*>(column)[ridx];
          return val != -9223372036854775807 - 1 ? static_cast<float>(val)
                                                 : missing;
        }
        default: {
          LOG(FATAL) << "Unknown data table type.";
          return 0.0f;
        }
      }
    }

   public:
    Line(DTType type, size_t size, size_t column_idx, const void* column)
        : type_(type), size_(size), column_idx_(column_idx), column_(column) {}

    size_t Size() const { return size_; }
    COOTuple GetElement(size_t idx) const {
      return COOTuple{idx, column_idx_, DTGetValue(column_, type_, idx)};
    }

   private:
    DTType type_;
    size_t size_;
    size_t column_idx_;
    const void* column_;
  };

 public:
  size_t Size() const { return num_features_; }
  const Line GetLine(size_t idx) const {
    return Line(DTGetType(feature_stypes_[idx]), num_rows_, idx, data_[idx]);
  }

 private:
  void** data_;
  const char** feature_stypes_;
  size_t num_features_;
  size_t num_rows_;
};

class DataTableAdapter
    : public detail::SingleBatchDataIter<DataTableAdapterBatch> {
 public:
  DataTableAdapter(void** data, const char** feature_stypes, size_t num_rows,
                   size_t num_features)
      : batch_(data, feature_stypes, num_rows, num_features),
        num_rows_(num_rows),
        num_columns_(num_features) {}
  const DataTableAdapterBatch& Value() const override { return batch_; }
  size_t NumRows() const { return num_rows_; }
  size_t NumColumns() const { return num_columns_; }

 private:
  DataTableAdapterBatch batch_;
  size_t num_rows_;
  size_t num_columns_;
};

class FileAdapterBatch {
 public:
  class Line {
   public:
    Line(size_t row_idx, const uint32_t* feature_idx, const float* value,
         size_t size)
        : row_idx_(row_idx),
          feature_idx_(feature_idx),
          value_(value),
          size_(size) {}

    size_t Size() { return size_; }
    COOTuple GetElement(size_t idx) {
      float fvalue = value_ == nullptr ? 1.0f : value_[idx];
      return COOTuple{row_idx_, feature_idx_[idx], fvalue};
    }

   private:
    size_t row_idx_;
    const uint32_t* feature_idx_;
    const float* value_;
    size_t size_;
  };
  FileAdapterBatch(const dmlc::RowBlock<uint32_t>* block, size_t row_offset)
      : block_(block), row_offset_(row_offset) {}
  Line GetLine(size_t idx) const {
    auto begin = block_->offset[idx];
    auto end = block_->offset[idx + 1];
    return Line{idx + row_offset_, &block_->index[begin], &block_->value[begin],
                end - begin};
  }
  const float* Labels() const { return block_->label; }
  const float* Weights() const { return block_->weight; }
  const uint64_t* Qid() const { return block_->qid; }
  const float* BaseMargin() const { return nullptr; }

  size_t Size() const { return block_->size; }

 private:
  const dmlc::RowBlock<uint32_t>* block_;
  size_t row_offset_;
};

/** \brief FileAdapter wraps dmlc::parser to read files and provide access in a
 * common interface. */
class FileAdapter : dmlc::DataIter<FileAdapterBatch> {
 public:
  explicit FileAdapter(dmlc::Parser<uint32_t>* parser) : parser_(parser) {}

  const FileAdapterBatch& Value() const override { return *batch_.get(); }
  void BeforeFirst() override {
    batch_.reset();
    parser_->BeforeFirst();
    row_offset_ = 0;
  }
  bool Next() override {
    bool next = parser_->Next();
    batch_.reset(new FileAdapterBatch(&parser_->Value(), row_offset_));
    row_offset_ += parser_->Value().size;
    return next;
  }
  // Indicates a number of rows/columns must be inferred
  size_t NumRows() const { return kAdapterUnknownSize; }
  size_t NumColumns() const { return kAdapterUnknownSize; }

 private:
  size_t row_offset_{0};
  std::unique_ptr<FileAdapterBatch> batch_;
  dmlc::Parser<uint32_t>* parser_;
};

class DMatrixSliceAdapterBatch {
 public:
  // Fetch metainfo values according to sliced rows
  template <typename T>
  std::vector<T> Gather(const std::vector<T>& in) {
    if (in.empty()) return {};

    std::vector<T> out(this->Size());
    for (auto i = 0ull; i < this->Size(); i++) {
      out[i] = in[ridx_set[i]];
    }
    return out;
  }
  DMatrixSliceAdapterBatch(const SparsePage& batch, DMatrix* dmat,
                           common::Span<const int> ridx_set)
      : batch(batch), ridx_set(ridx_set) {
    batch_labels = this->Gather(dmat->Info().labels_.HostVector());
    batch_weights = this->Gather(dmat->Info().weights_.HostVector());
    batch_base_margin = this->Gather(dmat->Info().base_margin_.HostVector());
  }

  class Line {
   public:
    Line(const SparsePage::Inst& inst, size_t row_idx)
        : inst_(inst), row_idx_(row_idx) {}

    size_t Size() { return inst_.size(); }
    COOTuple GetElement(size_t idx) {
      return COOTuple{row_idx_, inst_[idx].index, inst_[idx].fvalue};
    }

   private:
    SparsePage::Inst inst_;
    size_t row_idx_;
  };
  Line GetLine(size_t idx) const { return Line(batch[ridx_set[idx]], idx); }
  const float* Labels() const {
    if (batch_labels.empty()) {
      return nullptr;
    }
    return batch_labels.data();
  }
  const float* Weights() const {
    if (batch_weights.empty()) {
      return nullptr;
    }
    return batch_weights.data();
  }
  const uint64_t* Qid() const { return nullptr; }
  const float* BaseMargin() const {
    if (batch_base_margin.empty()) {
      return nullptr;
    }
    return batch_base_margin.data();
  }

  size_t Size() const { return ridx_set.size(); }
  const SparsePage& batch;
  common::Span<const int> ridx_set;
  std::vector<float> batch_labels;
  std::vector<float> batch_weights;
  std::vector<float> batch_base_margin;
};

// Group pointer is not exposed
// This is because external bindings currently manipulate the group values
// manually when slicing This could potentially be moved to internal C++ code if
// needed

class DMatrixSliceAdapter
    : public detail::SingleBatchDataIter<DMatrixSliceAdapterBatch> {
 public:
  DMatrixSliceAdapter(DMatrix* dmat, common::Span<const int> ridx_set)
      : dmat_(dmat),
        ridx_set_(ridx_set),
        batch_(*dmat_->GetBatches<SparsePage>().begin(), dmat_, ridx_set) {}
  const DMatrixSliceAdapterBatch& Value() const override { return batch_; }
  // Indicates a number of rows/columns must be inferred
  size_t NumRows() const { return ridx_set_.size(); }
  size_t NumColumns() const { return dmat_->Info().num_col_; }

 private:
  DMatrix* dmat_;
  common::Span<const int> ridx_set_;
  DMatrixSliceAdapterBatch batch_;
};
};  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_ADAPTER_H_
