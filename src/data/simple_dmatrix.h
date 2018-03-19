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
#include "../common/random.h"

namespace xgboost {
namespace data {

class SimpleDMatrix : public DMatrix {
 public:
  explicit SimpleDMatrix(std::unique_ptr<DataSource>&& source)
      : source_(std::move(source)), shuffled_indices_(initialise_shuffled_indices_()) {}

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
    return col_size_.size() != 0 && col_iter_.sorted == sorted;
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

  dmlc::DataIter<ColBatch>* ColIterator() override;

  dmlc::DataIter<ColBatch>* ColIterator(const std::vector<bst_uint>& fset) override;

  void InitColAccess(const std::vector<bool>& enabled,
                     float subsample,
                     size_t max_row_perbatch, bool sorted) override;

  bool SingleColBlock() const override;

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
  // in-memory column batch iterator.
  struct ColBatchIter: dmlc::DataIter<ColBatch> {
   public:
    ColBatchIter() : data_ptr_(0), sorted(false) {}
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
    size_t data_ptr_;
    // temporal space for batch
    ColBatch batch_;
    // Is column sorted?
    bool sorted;
  };

  // source data pointer.
  std::unique_ptr<DataSource> source_;
  // column iterator
  ColBatchIter col_iter_;
  // list of row index that are buffered.
  RowSet buffered_rowset_;
  /*! \brief sizeof column data */
  std::vector<size_t> col_size_;
  
  // indices to shuffle.
  std::vector<uint64_t> shuffled_indices_;

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
#endif  // XGBOOST_DATA_SIMPLE_DMATRIX_H_
