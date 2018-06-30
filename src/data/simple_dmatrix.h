/*!
 * Copyright 2015 by Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SIMPLE_DMATRIX_H_
#define XGBOOST_DATA_SIMPLE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <vector>
#include <algorithm>
#include <cstring>

namespace xgboost {
namespace data {

class SimpleDMatrix : public DMatrix {
 public:
  explicit SimpleDMatrix(std::unique_ptr<DataSource>&& source)
      : source_(std::move(source)) {}

  MetaInfo& Info() override {
    return source_->info;
  }

  const MetaInfo& Info() const override {
    return source_->info;
  }

  dmlc::DataIter<SparsePage>* RowIterator() override {
    auto iter = source_.get();
    iter->BeforeFirst();
    return iter;
  }

  bool HaveColAccess(bool sorted) const override {
    return col_iter_.sorted_ == sorted && col_iter_.column_page_!= nullptr;
  }

  const RowSet& BufferedRowset() const override {
    return buffered_rowset_;
  }

  size_t GetColSize(size_t cidx) const override {
    auto& batch = *col_iter_.column_page_;
    return batch[cidx].length;
  }

  float GetColDensity(size_t cidx) const override {
    size_t nmiss = buffered_rowset_.Size() - GetColSize(cidx);
    return 1.0f - (static_cast<float>(nmiss)) / buffered_rowset_.Size();
  }

  dmlc::DataIter<SparsePage>* ColIterator() override;

  void InitColAccess(
    size_t max_row_perbatch, bool sorted) override;

  bool SingleColBlock() const override;

 private:
  // in-memory column batch iterator.
  struct ColBatchIter: dmlc::DataIter<SparsePage> {
   public:
    ColBatchIter()  = default;
    void BeforeFirst() override {
      data_ = 0;
    }
    const SparsePage &Value() const override {
      return *column_page_;
    }
    bool Next() override;

   private:
    // allow SimpleDMatrix to access it.
    friend class SimpleDMatrix;
    // column sparse page
    std::unique_ptr<SparsePage> column_page_;
    // data pointer
    size_t data_{0};
    // Is column sorted?
    bool sorted_{false};
  };

  // source data pointer.
  std::unique_ptr<DataSource> source_;
  // column iterator
  ColBatchIter col_iter_;
  // list of row index that are buffered.
  RowSet buffered_rowset_;

  // internal function to make one batch from row iter.
  void MakeOneBatch(
    SparsePage *pcol, bool sorted);
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
