/**
 * Copyright 2020-2023 by XGBoost Contributors
 * \file iterative_dmatrix.h
 *
 * \brief Implementation of the higher-level `QuantileDMatrix`.
 */
#ifndef XGBOOST_DATA_ITERATIVE_DMATRIX_H_
#define XGBOOST_DATA_ITERATIVE_DMATRIX_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../common/error_msg.h"
#include "proxy_dmatrix.h"
#include "simple_batch_iterator.h"
#include "xgboost/base.h"
#include "xgboost/c_api.h"
#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"

namespace xgboost {
namespace common {
class HistogramCuts;
}

namespace data {
/**
 * \brief DMatrix type for `QuantileDMatrix`, the naming `IterativeDMatix` is due to its
 *        construction process.
 *
 * `QuantileDMatrix` is an intermediate storage for quantilization results including
 * quantile cuts and histogram index. Quantilization is designed to be performed on stream
 * of data (or batches of it). As a result, the `QuantileDMatrix` is also designed to work
 * with batches of data. During initializaion, it walks through the data multiple times
 * iteratively in order to perform quantilization. This design helps us reduce memory
 * usage significantly by avoiding data concatenation along with removing the CSR matrix
 * `SparsePage`. However, it has its limitation (can be fixed if needed):
 *
 * - It's only supported by hist tree method (both CPU and GPU) since approx requires a
 *   re-calculation of quantiles for each iteration. We can fix this by retaining a
 *   reference to the callback if there are feature requests.
 *
 * - The CPU format and the GPU format are different, the former uses a CSR + CSC for
 *   histogram index while the latter uses only Ellpack.
 */
class IterativeDMatrix : public DMatrix {
  MetaInfo info_;
  std::shared_ptr<EllpackPage> ellpack_;
  std::shared_ptr<GHistIndexMatrix> ghist_;
  BatchParam batch_;

  DMatrixHandle proxy_;
  DataIterResetCallback *reset_;
  XGDMatrixCallbackNext *next_;
  Context fmat_ctx_;

  void CheckParam(BatchParam const &param) {
    CHECK_EQ(param.max_bin, batch_.max_bin) << error::InconsistentMaxBin();
    CHECK(!param.regen && param.hess.empty())
        << "Only `hist` and `gpu_hist` tree method can use `QuantileDMatrix`.";
  }

  template <typename Page>
  static auto InvalidTreeMethod() {
    LOG(FATAL) << "Only `hist` and `gpu_hist` tree method can use `QuantileDMatrix`.";
    return BatchSet<Page>(BatchIterator<Page>(nullptr));
  }

  void InitFromCUDA(Context const *ctx, BatchParam const &p, DataIterHandle iter_handle,
                    float missing, std::shared_ptr<DMatrix> ref);
  void InitFromCPU(Context const *ctx, BatchParam const &p, DataIterHandle iter_handle,
                   float missing, std::shared_ptr<DMatrix> ref);

 public:
  explicit IterativeDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                            std::shared_ptr<DMatrix> ref, DataIterResetCallback *reset,
                            XGDMatrixCallbackNext *next, float missing, int nthread,
                            bst_bin_t max_bin);
  ~IterativeDMatrix() override = default;

  bool EllpackExists() const override { return static_cast<bool>(ellpack_); }
  bool GHistIndexExists() const override { return static_cast<bool>(ghist_); }
  bool SparsePageExists() const override { return false; }

  DMatrix *Slice(common::Span<int32_t const>) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for Quantile DMatrix.";
    return nullptr;
  }
  DMatrix *SliceCol(int, int) override {
    LOG(FATAL) << "Slicing DMatrix columns is not supported for Quantile DMatrix.";
    return nullptr;
  }
  BatchSet<SparsePage> GetRowBatches() override {
    LOG(FATAL) << "Not implemented.";
    return BatchSet<SparsePage>(BatchIterator<SparsePage>(nullptr));
  }
  BatchSet<CSCPage> GetColumnBatches(Context const *) override {
    return InvalidTreeMethod<CSCPage>();
  }
  BatchSet<SortedCSCPage> GetSortedColumnBatches(Context const *) override {
    return InvalidTreeMethod<SortedCSCPage>();
  }
  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const *ctx, BatchParam const &param) override;

  BatchSet<EllpackPage> GetEllpackBatches(Context const *ctx, const BatchParam &param) override;
  BatchSet<ExtSparsePage> GetExtBatches(Context const *ctx, BatchParam const &param) override;

  bool SingleColBlock() const override { return true; }

  MetaInfo &Info() override { return info_; }
  MetaInfo const &Info() const override { return info_; }

  Context const *Ctx() const override { return &fmat_ctx_; }
};

/**
 * \brief Get quantile cuts from reference (Quantile)DMatrix.
 *
 * \param ctx The context of the new DMatrix.
 * \param ref The reference DMatrix.
 * \param n_features Number of features, used for validation only.
 * \param p Batch parameter for the new DMatrix.
 * \param p_cuts Output quantile cuts.
 */
void GetCutsFromRef(Context const *ctx, std::shared_ptr<DMatrix> ref, bst_feature_t n_features,
                    BatchParam p, common::HistogramCuts *p_cuts);
/**
 * \brief Get quantile cuts from ellpack page.
 */
void GetCutsFromEllpack(EllpackPage const &page, common::HistogramCuts *cuts);
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ITERATIVE_DMATRIX_H_
