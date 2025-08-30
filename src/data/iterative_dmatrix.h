/**
 * Copyright 2020-2025, XGBoost Contributors
 *
 * @brief Implementation of the higher-level `QuantileDMatrix`.
 */
#ifndef XGBOOST_DATA_ITERATIVE_DMATRIX_H_
#define XGBOOST_DATA_ITERATIVE_DMATRIX_H_

#include <memory>   // for shared_ptr
#include <utility>  // for move

#include "quantile_dmatrix.h"     // for QuantileDMatrix
#include "xgboost/base.h"         // for bst_bin_t
#include "xgboost/c_api.h"        // for DataIterHandle, DMatrixHandle
#include "xgboost/context.h"      // for Context
#include "xgboost/data.h"         // for BatchSet

namespace xgboost {
namespace common {
class HistogramCuts;
class AlignedFileWriteStream;
class AlignedResourceReadStream;
}  // namespace common

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

  void InitFromCUDA(Context const *ctx, BatchParam const &p, std::int64_t max_quantile_blocks,
                    DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> &&iter,
                    float missing, std::shared_ptr<DMatrix> ref);
  void InitFromCPU(Context const *ctx, BatchParam const &p,
                   DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext> &&iter,
                   float missing, std::shared_ptr<DMatrix> ref);

  explicit IterativeDMatrix(std::shared_ptr<EllpackPage> ellpack) : ellpack_{std::move(ellpack)} {
    this->fmat_ctx_.UpdateAllowUnknown(Args{{"device", DeviceSym::CUDA()}});
  }

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

  [[nodiscard]] bool EllpackExists() const override { return static_cast<bool>(ellpack_); }
  [[nodiscard]] bool GHistIndexExists() const override { return static_cast<bool>(ghist_); }

  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const *ctx, BatchParam const &param) override;
  BatchSet<EllpackPage> GetEllpackBatches(Context const *ctx, const BatchParam &param) override;
  BatchSet<ExtSparsePage> GetExtBatches(Context const *ctx, BatchParam const &param) override;

  void Save(common::AlignedFileWriteStream *fo) const;
  [[nodiscard]] static IterativeDMatrix *Load(common::AlignedResourceReadStream *fi);
};
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ITERATIVE_DMATRIX_H_
