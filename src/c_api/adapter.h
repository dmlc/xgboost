/*!
 *  Copyright (c) 2019 by Contributors
 * \file adapter.h
 */
#ifndef XGBOOST_ADAPTER_H_
#define XGBOOST_ADAPTER_H_
#include <xgboost/data.h>
namespace xgboost {
class CSRAdapter : public ExternalDataAdapter {
 public:
  CSRAdapter(size_t* row_ptr, unsigned* feature_idx, float* values,
             size_t num_rows, size_t num_elements, size_t num_features)
      : ExternalDataAdapter(num_features, num_rows, num_elements),
        row_ptr(row_ptr),
        feature_idx(feature_idx),
        values(values) {}

 private:
  class CSRAdapterBatch : public Batch {
   public:
    CSRAdapterBatch(size_t row_idx, size_t size, unsigned* feature_idx,
                    float* values)
        : row_idx(row_idx),
          size(size),
          feature_idx(feature_idx),
          values(values) {}

    size_t Size() const override { return size; }
    COOTuple GetElement(size_t idx) const override {
      return {row_idx, feature_idx[idx], values[idx]};
    }

   private:
    size_t row_idx;
    size_t size;
    unsigned* feature_idx;
    float* values;
  };

 public:
  size_t Size() const { return num_rows; }
  std::unique_ptr<const Batch> operator[](size_t idx) const {
    size_t begin_offset = row_ptr[idx];
    size_t end_offset = row_ptr[idx + 1];
    return std::unique_ptr<const Batch>(
        new CSRAdapterBatch(idx, end_offset - begin_offset,
                            &feature_idx[begin_offset], &values[begin_offset]));
  }

 private:
  size_t* row_ptr;
  unsigned* feature_idx;
  float* values;
};

class DenseAdapter : public ExternalDataAdapter {
 public:
  DenseAdapter(const float* values, size_t num_rows, size_t num_elements,
               size_t num_features)
      : ExternalDataAdapter(num_features, num_rows, num_elements),
        values(values) {}

 private:
  class DenseAdapterBatch : public Batch {
   public:
    DenseAdapterBatch(const float* values, size_t size, size_t row_idx)
        : row_idx(row_idx), size(size), values(values) {}

    size_t Size() const override { return size; }
    COOTuple GetElement(size_t idx) const override {
      return {row_idx, idx, values[idx]};
    }

   private:
    size_t row_idx;
    size_t size;
    const float* values;
  };

 public:
  size_t Size() const { return num_rows; }
  std::unique_ptr<const Batch> operator[](size_t idx) const {
    return std::unique_ptr<const Batch>(
        new DenseAdapterBatch(values + idx * num_features, num_features, idx));
  }

 private:
  const float* values;
};

class CSCAdapter : public ExternalDataAdapter {
 public:
  CSCAdapter(size_t* col_ptr, unsigned* row_idx, float* values, size_t num_rows,
             size_t num_elements, size_t num_features)
      : ExternalDataAdapter(num_features, num_rows, num_elements),
        col_ptr(col_ptr),
        row_idx(row_idx),
        values(values) {}

 private:
  class CSCAdapterBatch : public Batch {
   public:
    CSCAdapterBatch(size_t col_idx, size_t size, unsigned* row_idx,
                    float* values)
        : col_idx(col_idx), size(size), row_idx(row_idx), values(values) {}

    size_t Size() const override { return size; }
    COOTuple GetElement(size_t idx) const override {
      return {row_idx[idx], col_idx, values[idx]};
    }

   private:
    size_t col_idx;
    size_t size;
    unsigned* row_idx;
    float* values;
  };

 public:
  size_t Size() const { return num_features; }
  std::unique_ptr<const Batch> operator[](size_t idx) const {
    size_t begin_offset = col_ptr[idx];
    size_t end_offset = col_ptr[idx + 1];
    return std::unique_ptr<const Batch>(
        new CSCAdapterBatch(idx, end_offset - begin_offset,
                            &row_idx[begin_offset], &values[begin_offset]));
  }

 private:
  size_t* col_ptr;
  unsigned* row_idx;
  float* values;
};
}  // namespace xgboost
#endif  // XGBOOST_ADAPTER_H_
