/*!
 * Copyright 2015 by Contributors
 * \file sparse_page_dmatrix.h
 * \brief External-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
#define XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>
#include <dmlc/threadediter.h>
#include <utility>
#include <vector>
#include <algorithm>
#include <string>
#include "../common/common.h"
#include "./sparse_page_writer.h"

namespace xgboost {
namespace data {

class SparsePageDMatrix : public DMatrix {
 public:
  explicit SparsePageDMatrix(std::unique_ptr<DataSource>&& source,
                             std::string  cache_info)
      : source_(std::move(source)), cache_info_(std::move(cache_info)) {
  }

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
    return col_iter_ != nullptr && col_iter_->sorted == sorted;
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

  bool SingleColBlock() const override {
    return false;
  }

  dmlc::DataIter<SparsePage>* ColIterator() override;

  void InitColAccess(
    size_t max_row_perbatch, bool sorted) override;

  /*! \brief page size 256 MB */
  static const size_t kPageSize = 256UL << 20UL;
  /*! \brief Maximum number of rows per batch. */
  static const size_t kMaxRowPerBatch = 64UL << 10UL;

 private:
  // declare the column batch iter.
  class ColPageIter : public dmlc::DataIter<SparsePage> {
   public:
    explicit ColPageIter(std::vector<std::unique_ptr<dmlc::SeekStream> >&& files);
    ~ColPageIter() override;
    void BeforeFirst() override;
    const SparsePage &Value() const override {
      return *page_;
    }
    bool Next() override;
    // initialize the column iterator with the specified index set.
    void Init(const std::vector<bst_uint>& index_set);
    // If the column features are sorted
    bool sorted;

   private:
    // the temp page.
    SparsePage* page_;
    // internal clock ptr.
    size_t clock_ptr_;
    // data file pointer.
    std::vector<std::unique_ptr<dmlc::SeekStream> > files_;
    // page format.
    std::vector<std::unique_ptr<SparsePageFormat> > formats_;
    /*! \brief internal prefetcher. */
    std::vector<std::unique_ptr<dmlc::ThreadedIter<SparsePage> > > prefetchers_;
    // The index set to be loaded.
    std::vector<bst_uint> index_set_;
    // The index set by the outsiders
    std::vector<bst_uint> set_index_set_;
    // whether to load data dataset.
    bool set_load_all_, load_all_;
  };
  /*!
   * \brief Try to initialize column data.
   * \return true if data already exists, false if they do not.
   */
  bool TryInitColData(bool sorted);
  // source data pointer.
  std::unique_ptr<DataSource> source_;
  // the cache prefix
  std::string cache_info_;
  /*! \brief list of row index that are buffered */
  RowSet buffered_rowset_;
  // count for column data
  std::vector<size_t> col_size_;
  // internal column iter.
  std::unique_ptr<ColPageIter> col_iter_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
