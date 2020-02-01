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
  explicit SparsePageDMatrix(std::unique_ptr<DataSource<SparsePage>>&& source,
                             std::string cache_info)
      : row_source_(std::move(source)), cache_info_(std::move(cache_info)) {}

  template <typename AdapterT>
  explicit SparsePageDMatrix(AdapterT* adapter, float missing, int nthread,
                             const std::string& cache_prefix,
                             size_t page_size = kPageSize)
      : cache_info_(std::move(cache_prefix)) {
    if (!data::SparsePageSource<SparsePage>::CacheExist(cache_prefix,
                                                        ".row.page")) {
      data::SparsePageSource<SparsePage>::CreateRowPage(
          adapter, missing, nthread, cache_prefix, page_size);
    }
    row_source_.reset(
        new data::SparsePageSource<SparsePage>(cache_prefix, ".row.page"));
  }
    // Set number of threads but keep old value so we can reset it after
  ~SparsePageDMatrix() override = default;

  MetaInfo& Info() override;

  const MetaInfo& Info() const override;

  float GetColDensity(size_t cidx) override;

  bool SingleColBlock() const override;

 private:
  BatchSet<SparsePage> GetRowBatches() override;
  BatchSet<CSCPage> GetColumnBatches() override;
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override;
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  // source data pointers.
  std::unique_ptr<DataSource<SparsePage>> row_source_;
  std::unique_ptr<SparsePageSource<CSCPage>> column_source_;
  std::unique_ptr<SparsePageSource<SortedCSCPage>> sorted_column_source_;
  std::unique_ptr<EllpackPageSource> ellpack_source_;
  // saved batch param
  BatchParam batch_param_;
  // the cache prefix
  std::string cache_info_;
  // Store column densities to avoid recalculating
  std::vector<float> col_density_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_DMATRIX_H_
