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
#include <string>
#include <utility>
#include <vector>
#include "sparse_page_source.h"

namespace xgboost {
namespace data {

class SparsePageDMatrix : public DMatrix {
 public:
  explicit SparsePageDMatrix(std::unique_ptr<DataSource>&& source,
                             std::string cache_info)
      : row_source_(std::move(source)), cache_info_(std::move(cache_info)) {}

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  BatchSet GetRowBatches() override;

  BatchSet GetSortedColumnBatches() override;

  BatchSet GetColumnBatches() override;

  float GetColDensity(size_t cidx) override;

  bool SingleColBlock() const override;

 private:
  /*! \brief page size 256 MB */
  static const size_t kPageSize = 256UL << 20UL;

  // source data pointers.
  std::unique_ptr<DataSource> row_source_;
  std::unique_ptr<SparsePageSource> column_source_;
  std::unique_ptr<SparsePageSource> sorted_column_source_;
  // the cache prefix
  std::string cache_info_;
  // Store column densities to avoid recalculating
  std::vector<float> col_density_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
