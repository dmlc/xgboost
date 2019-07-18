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

#include "sparse_page_source.h"

namespace xgboost {
namespace data {
// Used for external memory.
class SparsePageDMatrix {
 public:
  explicit SparsePageDMatrix(std::unique_ptr<DataSource>&& source,
                             std::string cache_info)
      : row_source_(std::move(source)), cache_info_(std::move(cache_info)) {}
  virtual ~SparsePageDMatrix() = default;

  MetaInfo& Info();

  const MetaInfo& Info() const;

  BatchSet GetRowBatches();

  BatchSet GetSortedColumnBatches(DMatrix* dmat);

  BatchSet GetColumnBatches(DMatrix* dmat);

  float GetColDensity(DMatrix* dmat, size_t cidx);

  bool SingleColBlock() const;

 private:
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
