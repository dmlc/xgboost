/*!
 * Copyright 2015 by Contributors
 * \file sparse_page_dmatrix.h
 * \brief External-memory version of DMatrix.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
#define XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_

#include <xgboost/data.h>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ellpack_page_source.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {
// Used for external memory.
class SparsePageDMatrix : public DMatrix {
 public:
  template <typename AdapterT>
  explicit SparsePageDMatrix(AdapterT* adapter, float missing, int nthread,
                             const std::string& cache_prefix,
                             size_t page_size = kPageSize)
      : cache_info_(std::move(cache_prefix)) {
    row_source_.reset(new data::SparsePageSource(adapter, missing, nthread,
                                                 cache_prefix, page_size));
  }
  ~SparsePageDMatrix() override = default;

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  bool SingleColBlock() const override { return false; }
  DMatrix *Slice(common::Span<int32_t const> ridxs) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for external memory.";
    return nullptr;
  }

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  // source data pointers.
  std::unique_ptr<SparsePageSource> row_source_;
  std::unique_ptr<CSCPageSource> column_source_;
  std::unique_ptr<SortedCSCPageSource> sorted_column_source_;
  std::unique_ptr<EllpackPageSource> ellpack_source_;
  // saved batch param
  BatchParam batch_param_;
  // the cache prefix
  std::string cache_info_;
  // Store column densities to avoid recalculating
  std::vector<float> col_density_;

  bool EllpackExists() const override {
    return static_cast<bool>(ellpack_source_);
  }
  bool SparsePageExists() const override {
    return static_cast<bool>(row_source_);
  }
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
