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
#include <vector>
#include <algorithm>
#include <string>
#include "./sparse_batch_page.h"
#include "../common/common.h"
#include "../common/random.h"

namespace xgboost {
namespace data {

class SparsePageDMatrix : public DMatrix {
 public:
  explicit SparsePageDMatrix(std::unique_ptr<DataSource>&& source,
                             const std::string& cache_info)
      : source_(std::move(source)), cache_info_(cache_info),
        shuffled_indices_(initialise_shuffled_indices_()) {
  }

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

  bool HaveColAccess(bool sorted) const override {
    return col_iter_.get() != nullptr && col_iter_->sorted == sorted;
  }

  const RowSet& buffered_rowset() const override {
    return buffered_rowset_;
  }

  size_t GetColSize(size_t cidx) const override {
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
                     size_t max_row_perbatch, bool sorted) override;

  /*! \brief page size 256 MB */
  static const size_t kPageSize = 256UL << 20UL;
  /*! \brief Maximum number of rows per batch. */
  static const size_t kMaxRowPerBatch = 64UL << 10UL;

  std::vector<uint64_t> get_subsampled_indices(
      const float subsample,
      const std::vector<xgboost::bst_gpair>& gpair
  ) override {
      auto &random_number_generator = common::GlobalRandom();
      const size_t indices_length = shuffled_indices_.size();
      const size_t subsample_size = static_cast<size_t>(indices_length * subsample);
      size_t number_of_subsamples = 0;
      std::vector<uint64_t> subsampled_indices;

      for (
          size_t index = 0; 
          number_of_subsamples != subsample_size && index != indices_length; 
          ++index
      ) {
      
          std::uniform_int_distribution<size_t> index_to_swap(index, indices_length - 1);
          const size_t swap_index = index_to_swap(random_number_generator);

          if (index != swap_index) {
              std::swap(shuffled_indices_.at(index), shuffled_indices_.at(swap_index));
          }

          if (gpair[shuffled_indices_.at(index)].GetHess() >= 0.0f) {
              subsampled_indices.push_back(shuffled_indices_.at(index));
              ++number_of_subsamples;
          }
      }

      return subsampled_indices;
  }

 private:
  // declare the column batch iter.
  class ColPageIter : public dmlc::DataIter<ColBatch> {
   public:
    explicit ColPageIter(std::vector<std::unique_ptr<dmlc::SeekStream> >&& files);
    virtual ~ColPageIter();
    void BeforeFirst() override;
    const ColBatch &Value() const override {
      return out_;
    }
    bool Next() override;
    // initialize the column iterator with the specified index set.
    void Init(const std::vector<bst_uint>& index_set, bool load_all);
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
    std::vector<std::unique_ptr<SparsePage::Format> > formats_;
    /*! \brief internal prefetcher. */
    std::vector<std::unique_ptr<dmlc::ThreadedIter<SparsePage> > > prefetchers_;
    // The index set to be loaded.
    std::vector<bst_uint> index_set_;
    // The index set by the outsiders
    std::vector<bst_uint> set_index_set_;
    // whether to load data dataset.
    bool set_load_all_, load_all_;
    // temporal space for batch
    ColBatch out_;
    // the pointer data.
    std::vector<SparseBatch::Inst> col_data_;
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
  // indices to shuffle.
  std::vector<uint64_t> shuffled_indices_;

  std::vector<uint64_t> initialise_shuffled_indices_() {
      
    const MetaInfo& dataset_info = info();
    const uint64_t number_of_datapoints = dataset_info.num_row;
    std::vector<uint64_t> shuffled_indices_(number_of_datapoints);

        for (uint64_t i = 0; i != number_of_datapoints; ++i) {
            shuffled_indices_.push_back(i);
        }

    return shuffled_indices_;

  }

};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
