/**
 * Copyright 2020-2024, XGBoost Contributors
 * \file iterative_dmatrix.h
 *
 * \brief Implementation of the higher-level `QuantileDMatrix`.
 */
#ifndef XGBOOST_DATA_ITERATIVE_DMATRIX_H_
#define XGBOOST_DATA_ITERATIVE_DMATRIX_H_

#include <memory>  // for shared_ptr

#include "quantile_dmatrix.h"     // for QuantileDMatrix
#include "xgboost/base.h"         // for bst_bin_t
#include "xgboost/c_api.h"        // for DataIterHandle, DMatrixHandle
#include "xgboost/context.h"      // for Context
#include "xgboost/data.h"         // for BatchSet

namespace xgboost {
namespace common {
class HistogramCuts;
}

namespace data {
/**
 * @brief DMatrix type for `QuantileDMatrix`, the naming `IterativeDMatix` is due to its
 *        construction process.
 *
 * During initializaion, it walks through the data multiple times iteratively in order to
 * perform quantilization. This design helps us reduce memory usage significantly by
 * avoiding data concatenation along with removing the CSR matrix `SparsePage`.
 */
class IterativeDMatrix : public QuantileDMatrix {
  std::shared_ptr<EllpackPage> ellpack_;
  std::shared_ptr<GHistIndexMatrix> ghist_;
  BatchParam batch_;

  DMatrixHandle proxy_;
  DataIterResetCallback *reset_;
  XGDMatrixCallbackNext *next_;

  void InitFromCUDA(Context const *ctx, BatchParam const &p, std::int64_t max_quantile_blocks,
                    DataIterHandle iter_handle, float missing, std::shared_ptr<DMatrix> ref);
  void InitFromCPU(Context const *ctx, BatchParam const &p, DataIterHandle iter_handle,
                   float missing, std::shared_ptr<DMatrix> ref);

 public:
  explicit IterativeDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                            std::shared_ptr<DMatrix> ref, DataIterResetCallback *reset,
                            XGDMatrixCallbackNext *next, float missing, int nthread,
                            bst_bin_t max_bin, std::int64_t max_quantile_blocks);
  /**
   * @param Directly construct a QDM from an existing one.
   */
  IterativeDMatrix(std::shared_ptr<EllpackPage> ellpack, MetaInfo const &info, BatchParam batch);

  ~IterativeDMatrix() override = default;

  bool EllpackExists() const override { return static_cast<bool>(ellpack_); }
  bool GHistIndexExists() const override { return static_cast<bool>(ghist_); }

  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const *ctx, BatchParam const &param) override;

  BatchSet<EllpackPage> GetEllpackBatches(Context const *ctx, const BatchParam &param) override;
  BatchSet<ExtSparsePage> GetExtBatches(Context const *ctx, BatchParam const &param) override;
};
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ITERATIVE_DMATRIX_H_
