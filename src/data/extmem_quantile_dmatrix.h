/**
 * Copyright 2024, XGBoost Contributors
 */
#pragma once

#include <map>      // for map
#include <memory>   // for shared_ptr
#include <string>   // for string
#include <variant>  // for variant

#include "ellpack_page_source.h"         // for EllpackPageSource, EllpackPageHostSource
#include "gradient_index_page_source.h"  // for GradientIndexPageSource
#include "quantile_dmatrix.h"            // for QuantileDMatrix, ExternalIter
#include "xgboost/base.h"                // for bst_bin_t
#include "xgboost/c_api.h"               // for DataIterHandle, DMatrixHandle
#include "xgboost/context.h"             // for Context
#include "xgboost/data.h"                // for MetaInfo, BatchParam

namespace xgboost::data {
/**
 * @brief A DMatrix class for building a `QuantileDMatrix` from external memory iterator.
 */
class ExtMemQuantileDMatrix : public QuantileDMatrix {
 public:
  ExtMemQuantileDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                        std::shared_ptr<DMatrix> ref, DataIterResetCallback *reset,
                        XGDMatrixCallbackNext *next, float missing, std::int32_t n_threads,
                        std::string cache, bst_bin_t max_bin);
  ~ExtMemQuantileDMatrix() override;

  [[nodiscard]] bool SingleColBlock() const override { return false; }

 private:
  void InitFromCPU(
      Context const *ctx,
      std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
      DMatrixHandle proxy, BatchParam const &p, float missing, std::shared_ptr<DMatrix> ref);
  void InitFromCUDA(
      Context const *ctx,
      std::shared_ptr<DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>> iter,
      DMatrixHandle proxy_handle, BatchParam const &p, float missing, std::shared_ptr<DMatrix> ref);

  BatchSet<GHistIndexMatrix> GetGradientIndexImpl();
  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const *ctx, BatchParam const &param) override;

  BatchSet<EllpackPage> GetEllpackBatches(Context const *ctx, const BatchParam &param) override;

  [[nodiscard]] bool EllpackExists() const override {
    return std::visit([](auto &&v) { return static_cast<bool>(v); }, ellpack_page_source_);
  }
  [[nodiscard]] bool GHistIndexExists() const override { return true; }

  [[nodiscard]] BatchSet<ExtSparsePage> GetExtBatches(Context const *ctx,
                                                      BatchParam const &param) override;

  std::map<std::string, std::shared_ptr<Cache>> cache_info_;
  std::string cache_prefix_;  // fixme
  BatchParam batch_;

  using EllpackDiskPtr = std::shared_ptr<EllpackPageSource>;
  using EllpackHostPtr = std::shared_ptr<EllpackPageHostSource>;
  std::variant<EllpackDiskPtr, EllpackHostPtr> ellpack_page_source_;
  std::shared_ptr<ExtGradientIndexPageSource> ghist_index_source_;
};
}  // namespace xgboost::data
