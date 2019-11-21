/*!
 *  Copyright (c) 2019 by Contributors
 * \file adapter.h
 */
#ifndef XGBOOST_C_API_ADAPTER_H_
#define XGBOOST_C_API_ADAPTER_H_
#include <string>
#include <limits>
namespace xgboost {

/** \brief External data formats should implement an adapter as below. The
 * adapter provides a uniform access to data outside xgboost, allowing
 * construction of DMatrix objects from a range of sources without duplicating
 * code. The adapter should translate external data into batches of COO tuples
 * containing the floating point value, row index and column index.
 * 
 * Why return batches? Sparse matrix formats such as CSR and CSC do not provide
 * efficient random access to matrix elements, allowing these formats to provide
 * data by rows or columns allows us to efficiently read data.
 *  */
class ExternalDataAdapter {
 public:
  ExternalDataAdapter(size_t num_features, size_t num_rows, size_t num_elements)
      : num_features(num_features),
        num_rows(num_rows),
        num_elements(num_elements) {}
  struct COOTuple {
    COOTuple(size_t row_idx, size_t column_idx, float value)
        : row_idx(row_idx), column_idx(column_idx), value(value) {}

    size_t row_idx{0};
    size_t column_idx{0};
    float value{0};
  };
  size_t GetNumFeatures() const { return num_features; }
  size_t GetNumRows() const { return num_rows; }
  size_t GetNumElements() const { return num_elements; }

 protected:
  size_t num_features;
  size_t num_rows;
  size_t num_elements;
};

class CSRAdapter : public ExternalDataAdapter {
 public:
  CSRAdapter(const size_t* row_ptr, const unsigned* feature_idx,
             const float* values, size_t num_rows, size_t num_elements,
             size_t num_features)
      : ExternalDataAdapter(num_features, num_rows, num_elements),
        row_ptr(row_ptr),
        feature_idx(feature_idx),
        values(values) {}

 private:
  class CSRAdapterBatch {
   public:
    CSRAdapterBatch(size_t row_idx, size_t size, const unsigned* feature_idx,
                    const float* values)
        : row_idx(row_idx),
          size(size),
          feature_idx(feature_idx),
          values(values) {}

    size_t Size() const { return size; }
    COOTuple GetElement(size_t idx) const {
      return COOTuple(row_idx, feature_idx[idx], values[idx]);
    }

   private:
    size_t row_idx;
    size_t size;
    const unsigned* feature_idx;
    const float* values;
  };

 public:
  size_t Size() const { return num_rows; }
  const CSRAdapterBatch operator[](size_t idx) const {
    size_t begin_offset = row_ptr[idx];
    size_t end_offset = row_ptr[idx + 1];
    return CSRAdapterBatch(idx, end_offset - begin_offset,
                           &feature_idx[begin_offset], &values[begin_offset]);
  }

 private:
  const size_t* row_ptr;
  const unsigned* feature_idx;
  const float* values;
};

class DenseAdapter : public ExternalDataAdapter {
 public:
  DenseAdapter(const float* values, size_t num_rows, size_t num_elements,
               size_t num_features)
      : ExternalDataAdapter(num_features, num_rows, num_elements),
        values(values) {}

 private:
  class DenseAdapterBatch {
   public:
    DenseAdapterBatch(const float* values, size_t size, size_t row_idx)
        : row_idx(row_idx), size(size), values(values) {}

    size_t Size() const { return size; }
    COOTuple GetElement(size_t idx) const {
      return COOTuple(row_idx, idx, values[idx]);
    }

   private:
    size_t row_idx;
    size_t size;
    const float* values;
  };

 public:
  size_t Size() const { return num_rows; }
  const DenseAdapterBatch operator[](size_t idx) const {
    return DenseAdapterBatch(values + idx * num_features, num_features, idx);
  }

 private:
  const float* values;
};

class CSCAdapter : public ExternalDataAdapter {
 public:
  CSCAdapter(const size_t* col_ptr, const unsigned* row_idx,
             const float* values, size_t num_rows, size_t num_elements,
             size_t num_features)
      : ExternalDataAdapter(num_features, num_rows, num_elements),
        col_ptr(col_ptr),
        row_idx(row_idx),
        values(values) {}

 private:
  class CSCAdapterBatch {
   public:
    CSCAdapterBatch(size_t col_idx, size_t size, const unsigned* row_idx,
                    const float* values)
        : col_idx(col_idx), size(size), row_idx(row_idx), values(values) {}

    size_t Size() const { return size; }
    COOTuple GetElement(size_t idx) const {
      return COOTuple(row_idx[idx], col_idx, values[idx]);
    }

   private:
    size_t col_idx;
    size_t size;
    const unsigned* row_idx;
    const float* values;
  };

 public:
  size_t Size() const { return num_features; }
  const CSCAdapterBatch operator[](size_t idx) const {
    size_t begin_offset = col_ptr[idx];
    size_t end_offset = col_ptr[idx + 1];
    return CSCAdapterBatch(idx, end_offset - begin_offset,
                           &row_idx[begin_offset], &values[begin_offset]);
  }

 private:
  const size_t* col_ptr;
  const unsigned* row_idx;
  const float* values;
};

class DataTableAdapter : public ExternalDataAdapter {
 public:
  DataTableAdapter(void** data, const char** feature_stypes, size_t num_rows,
                   size_t num_elements, size_t num_features)
      : ExternalDataAdapter(num_features, num_rows, num_elements),
        data(data),
        feature_stypes(feature_stypes) {}

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

  class DataTableAdapterBatch {
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
    DataTableAdapterBatch(DTType type, size_t size, size_t column_idx,
                          const void* column)
        : type(type), size(size), column_idx(column_idx), column(column) {}

    size_t Size() const { return size; }
    COOTuple GetElement(size_t idx) const {
      return COOTuple(idx, column_idx, DTGetValue(column, type, idx));
    }

   private:
    DTType type;
    size_t size;
    size_t column_idx;
    const void* column;
  };

 public:
  size_t Size() const { return num_features; }
  const DataTableAdapterBatch operator[](size_t idx) const {
    return DataTableAdapterBatch(DTGetType(feature_stypes[idx]), num_rows, idx,
                                 data[idx]);
  }

 private:
  void** data;
  const char** feature_stypes;
};
}  // namespace xgboost
#endif  // XGBOOST_C_API_ADAPTER_H_
