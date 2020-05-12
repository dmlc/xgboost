/*!
 * Copyright 2019 by XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <xgboost/data.h>
#include <memory>
#include <string>

#include "../common/timer.h"
#include "../common/hist_util.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {

/*!
 * \brief External memory data source for ELLPACK format.
 *
 */
class EllpackPageSource {
 public:
  /*!
   * \brief Create source from cache files the cache_prefix.
   * \param cache_prefix The prefix of cache we want to solve.
   */
  explicit EllpackPageSource(DMatrix* dmat,
                             const std::string& cache_info,
                             const BatchParam& param) noexcept(false);

  BatchSet<EllpackPage> GetBatchSet() {
    auto begin_iter = BatchIterator<EllpackPage>(
        new SparseBatchIteratorImpl<ExternalMemoryPrefetcher<EllpackPage>,
                                    EllpackPage>(external_prefetcher_.get()));
    return BatchSet<EllpackPage>(begin_iter);
  }

  ~EllpackPageSource() {
    external_prefetcher_.reset();
    for (auto file : cache_info_.name_shards) {
      TryDeleteCacheFile(file);
    }
  }

 private:
  void WriteEllpackPages(int device, DMatrix* dmat,
                         const common::HistogramCuts& cuts,
                         const std::string& cache_info,
                         size_t row_stride) const;

  /*! \brief The page type string for ELLPACK. */
  const std::string kPageType_{".ellpack.page"};

  size_t page_size_{DMatrix::kPageSize};
  common::Monitor monitor_;
  std::unique_ptr<ExternalMemoryPrefetcher<EllpackPage>> external_prefetcher_;
  CacheInfo cache_info_;
};

}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
