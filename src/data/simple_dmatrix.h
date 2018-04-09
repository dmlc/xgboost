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
#include "./sparse_batch_page.h"

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

  dmlc::DataIter<RowBatch>* RowIterator() override {
    dmlc::DataIter<RowBatch>* iter = source_.get();
    iter->BeforeFirst();
    return iter;
  }

  bool HaveColAccess(bool sorted) const override {
    return col_size_.size() != 0 && col_iter_.sorted_ == sorted;
  }

  const RowSet& BufferedRowset() const override {
    return buffered_rowset_;
  }

  size_t GetColSize(size_t cidx) const override {
    return col_size_[cidx];
  }

  float GetColDensity(size_t cidx) const override {
    size_t nmiss = buffered_rowset_.Size() - col_size_[cidx];
    return 1.0f - (static_cast<float>(nmiss)) / buffered_rowset_.Size();
  }

  dmlc::DataIter<ColBatch>* ColIterator() override;

  dmlc::DataIter<ColBatch>* ColIterator(const std::vector<bst_uint>& fset) override;

  void InitColAccess(const std::vector<bool>& enabled,
                     float subsample,
                     size_t max_row_perbatch, bool sorted) override;

  bool SingleColBlock() const override;

 private:
  // in-memory column batch iterator.
  struct ColBatchIter: dmlc::DataIter<ColBatch> {
   public:
    ColBatchIter()  = default;
    void BeforeFirst() override {
      data_ptr_ = 0;
    }
    const ColBatch &Value() const override {
      return batch_;
    }
    bool Next() override;

   private:
    // allow SimpleDMatrix to access it.
    friend class SimpleDMatrix;
    // data content
    std::vector<bst_uint> col_index_;
    // column content
    std::vector<ColBatch::Inst> col_data_;
    // column sparse pages
    std::vector<std::unique_ptr<SparsePage> > cpages_;
    // data pointer
    size_t data_ptr_{0};
    // temporal space for batch
    ColBatch batch_;
    // Is column sorted?
    bool sorted_{false};
  };

  // source data pointer.
  std::unique_ptr<DataSource> source_;
  // column iterator
  ColBatchIter col_iter_;
  // list of row index that are buffered.
  RowSet buffered_rowset_;
  /*! \brief sizeof column data */
  std::vector<size_t> col_size_;

  // internal function to make one batch from row iter.
  void MakeOneBatch(const std::vector<bool>& enabled,
                    float pkeep,
                    SparsePage *pcol, bool sorted);

  void MakeManyBatch(const std::vector<bool>& enabled,
                     float pkeep,
                     size_t max_row_perbatch, bool sorted);

  void MakeColPage(const RowBatch& batch,
                   size_t buffer_begin,
                   const std::vector<bool>& enabled,
                   SparsePage* pcol, bool sorted);
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
