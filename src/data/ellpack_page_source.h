/*!
 * Copyright 2019 by XGBoost Contributors
 */

#ifndef XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_

#include <xgboost/data.h>
#include <memory>
#include <string>

#include "../common/timer.h"

namespace xgboost {
namespace data {

class EllpackPageSourceImpl;

/*!
 * \brief External memory data source for ELLPACK format.
 *
 * This class uses the PImpl idiom (https://en.cppreference.com/w/cpp/language/pimpl) to avoid
 * including CUDA-specific implementation details in the header.
 */
class EllpackPageSource : public DataSource<EllpackPage> {
 public:
  /*!
   * \brief Create source from cache files the cache_prefix.
   * \param cache_prefix The prefix of cache we want to solve.
   */
  explicit EllpackPageSource(DMatrix* dmat,
                             const std::string& cache_info,
                             const BatchParam& param) noexcept(false);

  /*! \brief destructor */
  ~EllpackPageSource() override = default;

  void BeforeFirst() override;
  bool Next() override;
  EllpackPage& Value();
  const EllpackPage& Value() const override;

  const EllpackPageSourceImpl* Impl() const { return impl_.get(); }
  EllpackPageSourceImpl* Impl() { return impl_.get(); }

 private:
  std::shared_ptr<EllpackPageSourceImpl> impl_;
};

}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ELLPACK_PAGE_SOURCE_H_
