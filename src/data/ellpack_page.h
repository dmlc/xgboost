/**
 * Copyright 2017-2023 by XGBoost Contributors
 */
#ifndef XGBOOST_DATA_ELLPACK_PAGE_H_
#define XGBOOST_DATA_ELLPACK_PAGE_H_

#include <memory>  // for unique_ptr

#include "../common/hist_util.h"  // for HistogramCuts
#include "xgboost/context.h"      // for Context
#include "xgboost/data.h"         // for DMatrix, BatchParam

namespace xgboost {
class EllpackPageImpl;
/**
 * @brief A page stored in ELLPACK format.
 *
 * This class uses the PImpl idiom (https://en.cppreference.com/w/cpp/language/pimpl) to avoid
 * including CUDA-specific implementation details in the header.
 */
class EllpackPage {
 public:
  /**
   * @brief Default constructor.
   *
   * This is used in the external memory case. An empty ELLPACK page is constructed with its content
   * set later by the reader.
   */
  EllpackPage();
  /**
   * @brief Constructor from an existing DMatrix.
   *
   * This is used in the in-memory case. The ELLPACK page is constructed from an existing DMatrix
   * in CSR format.
   */
  explicit EllpackPage(Context const* ctx, DMatrix* dmat, const BatchParam& param);

  /*! \brief Destructor. */
  ~EllpackPage();

  EllpackPage(EllpackPage&& that);

  /*! \return Number of instances in the page. */
  [[nodiscard]] bst_idx_t Size() const;

  /*! \brief Set the base row id for this page. */
  void SetBaseRowId(std::size_t row_id);

  [[nodiscard]] const EllpackPageImpl* Impl() const { return impl_.get(); }
  EllpackPageImpl* Impl() { return impl_.get(); }

  [[nodiscard]] common::HistogramCuts const& Cuts() const;

 private:
  std::unique_ptr<EllpackPageImpl> impl_;
};
}  // namespace xgboost
#endif  // XGBOOST_DATA_ELLPACK_PAGE_H_
