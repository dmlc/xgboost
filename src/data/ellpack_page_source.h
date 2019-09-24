/*!
 * Copyright 2019 by XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <xgboost/data.h>
#include <string>

#include "sparse_page_source.h"

namespace xgboost {
namespace data {

/*!
 * \brief External memory data source for ELLPACK format.
 */
class EllpackPageSource : public DataSource<EllpackPage> {
 public:
  /*!
   * \brief Create source from cache files the cache_prefix.
   * \param cache_prefix The prefix of cache we want to solve.
   */
  explicit EllpackPageSource(DMatrix* src, const std::string& cache_info) noexcept(false);

  /*!
   * \brief Create ELLPACK source cache by copy content from DMatrix.
   * \param cache_info The cache_info of cache file location.
   */
  static void CreateEllpackPage(DMatrix* src, const std::string& cache_info);

  /*! \brief destructor */
  ~EllpackPageSource() override = default;

  void BeforeFirst() override {}
  bool Next() override {
    return false;
  }

  EllpackPage& Value() {
    return page_;
  }

  const EllpackPage& Value() const override {
    return page_;
  }

 private:
  EllpackPage page_;
};

}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
