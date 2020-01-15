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
class SparsePageDMatrix : public DMatrix {
 public:
  explicit SparsePageDMatrix(std::unique_ptr<DataSource<SparsePage>>&& source,
                             std::string cache_info)
      : row_source_(std::move(source)), cache_info_(std::move(cache_info)) {}
  ~SparsePageDMatrix() override = default;

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  float GetColDensity(size_t cidx) override;

  bool SingleColBlock() const override;

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches() override;

  // source data pointers.
  std::unique_ptr<DataSource<SparsePage>> row_source_;
  std::unique_ptr<SparsePageSource<CSCPage>> column_source_;
  std::unique_ptr<SparsePageSource<SortedCSCPage>> sorted_column_source_;
  std::unique_ptr<EllpackPage> ellpack_page_;
  // the cache prefix
  std::string cache_info_;
  // Store column densities to avoid recalculating
  std::vector<float> col_density_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
