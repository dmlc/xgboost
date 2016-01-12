/*!
 * Copyright 2015 by Contributors
 * \file simple_dmatrix.h
 * \brief In-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
#define XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <dmlc/threadediter.h>
#include <vector>
#include <algorithm>
#include <string>
#include "./sparse_batch_page.h"

namespace xgboost {
namespace data {

class SparsePageDMatrix : public DMatrix {
 public:
  explicit SparsePageDMatrix(std::unique_ptr<DataSource>&& source,
                             const std::string& cache_prefix)
      : source_(std::move(source)),
        cache_prefix_(cache_prefix) {}

  MetaInfo& info() override {
    return source_->info;
  }

  const MetaInfo& info() const override {
    return source_->info;
  }

  dmlc::DataIter<RowBatch>* RowIterator() override {
    dmlc::DataIter<RowBatch>* iter = source_.get();
    iter->BeforeFirst();
    return iter;
  }

  bool HaveColAccess() const override {
    return col_iter_.get() != nullptr;
  }

  const std::vector<bst_uint>& buffered_rowset() const override {
    return buffered_rowset_;
  }

  size_t GetColSize(size_t cidx) const {
    return col_size_[cidx];
  }

  float GetColDensity(size_t cidx) const override {
    size_t nmiss = buffered_rowset_.size() - col_size_[cidx];
    return 1.0f - (static_cast<float>(nmiss)) / buffered_rowset_.size();
  }

  bool SingleColBlock() const override {
    return false;
  }

  dmlc::DataIter<ColBatch>* ColIterator() override;

  dmlc::DataIter<ColBatch>* ColIterator(const std::vector<bst_uint>& fset) override;

  void InitColAccess(const std::vector<bool>& enabled,
                     float subsample,
                     size_t max_row_perbatch) override;

  /*! \brief page size 256 MB */
  static const size_t kPageSize = 256UL << 20UL;
  /*! \brief Maximum number of rows per batch. */
  static const size_t kMaxRowPerBatch = 64UL << 10UL;

 private:
  // declare the column batch iter.
  class ColPageIter : public dmlc::DataIter<ColBatch> {
   public:
    explicit ColPageIter(std::unique_ptr<dmlc::SeekStream>&& fi);
    virtual ~ColPageIter();
    void BeforeFirst() override {
      prefetcher_.BeforeFirst();
    }
    const ColBatch &Value() const override {
      return out_;
    }
    bool Next() override;
    // initialize the column iterator with the specified index set.
    void Init(const std::vector<bst_uint>& index_set, bool load_all);

   private:
    // data file pointer.
    std::unique_ptr<dmlc::SeekStream> fi_;
    // the temp page.
    SparsePage* page_;
    // page format.
    std::unique_ptr<SparsePage::Format> format_;
    // The index set to be loaded.
    std::vector<bst_uint> index_set_;
    // The index set by the outsiders
    std::vector<bst_uint> set_index_set_;
    // whether to load data dataset.
    bool set_load_all_, load_all_;
    // data prefetcher.
    dmlc::ThreadedIter<SparsePage> prefetcher_;
    // temporal space for batch
    ColBatch out_;
    // the pointer data.
    std::vector<SparseBatch::Inst> col_data_;
  };
  /*!
   * \brief Try to intitialize column data.
   * \return true if data already exists, false if they do not.
   */
  bool TryInitColData();
  // source data pointer.
  std::unique_ptr<DataSource> source_;
  // the cache prefix
  std::string cache_prefix_;
  /*! \brief list of row index that are buffered */
  std::vector<bst_uint> buffered_rowset_;
  // count for column data
  std::vector<size_t> col_size_;
  // internal column iter.
  std::unique_ptr<ColPageIter> col_iter_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
