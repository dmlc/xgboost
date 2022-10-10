/*!
 * Copyright 2020-2022 by Contributors
 * \file iterative_dmatrix.h
 */
#ifndef XGBOOST_DATA_ITERATIVE_DMATRIX_H_
#define XGBOOST_DATA_ITERATIVE_DMATRIX_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "proxy_dmatrix.h"
#include "simple_batch_iterator.h"
#include "xgboost/base.h"
#include "xgboost/c_api.h"
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
  Context ctx_;
  BatchParam batch_param_;
  std::shared_ptr<EllpackPage> ellpack_;
  std::shared_ptr<GHistIndexMatrix> ghist_;

  DMatrixHandle proxy_;
  DataIterResetCallback *reset_;
  XGDMatrixCallbackNext *next_;

  void CheckParam(BatchParam const &param) {
    // FIXME(Jiamingy): https://github.com/dmlc/xgboost/issues/7976
    if (param.max_bin != batch_param_.max_bin && param.max_bin != 0) {
      LOG(WARNING) << "Inconsistent max_bin between Quantile DMatrix and Booster:" << param.max_bin
                   << " vs. " << batch_param_.max_bin;
    }
    CHECK(!param.regen && param.hess.empty())
        << "Only `hist` and `gpu_hist` tree method can use `QuantileDMatrix`.";
  }

  template <typename Page>
  static auto InvalidTreeMethod() {
    LOG(FATAL) << "Only `hist` and `gpu_hist` tree method can use `QuantileDMatrix`.";
    return BatchSet<Page>(BatchIterator<Page>(nullptr));
  }

  void InitFromCUDA(DataIterHandle iter, float missing, std::shared_ptr<DMatrix> ref);
  void InitFromCPU(DataIterHandle iter_handle, float missing, std::shared_ptr<DMatrix> ref);

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
  BatchSet<SparsePage> GetRowBatches() override {
    LOG(FATAL) << "Not implemented.";
    return BatchSet<SparsePage>(BatchIterator<SparsePage>(nullptr));
  }
  BatchSet<CSCPage> GetColumnBatches() override { return InvalidTreeMethod<CSCPage>(); }
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override {
    return InvalidTreeMethod<SortedCSCPage>();
  }
  BatchSet<GHistIndexMatrix> GetGradientIndex(BatchParam const &param) override;

  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam &param) override;
  BatchSet<ExtSparsePage> GetExtBatches(BatchParam const& param) override;

  bool SingleColBlock() const override { return true; }

  MetaInfo &Info() override { return info_; }
  MetaInfo const &Info() const override { return info_; }

  Context const *Ctx() const override { return &ctx_; }
};

/**
 * \brief Get quantile cuts from reference Quantile DMatrix.
 */
void GetCutsFromRef(std::shared_ptr<DMatrix> ref_, bst_feature_t n_features, BatchParam p,
                    common::HistogramCuts *p_cuts);
/**
 * \brief Get quantile cuts from ellpack page.
 */
void GetCutsFromEllpack(EllpackPage const &page, common::HistogramCuts *cuts);

#if !defined(XGBOOST_USE_CUDA)
inline void IterativeDMatrix::InitFromCUDA(DataIterHandle, float, std::shared_ptr<DMatrix>) {
  // silent the warning about unused variables.
  (void)(proxy_);
  (void)(reset_);
  (void)(next_);
  common::AssertGPUSupport();
}
inline BatchSet<EllpackPage> IterativeDMatrix::GetEllpackBatches(const BatchParam &) {
  common::AssertGPUSupport();
  auto begin_iter = BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(ellpack_));
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}

inline void GetCutsFromEllpack(EllpackPage const &, common::HistogramCuts *) {
  common::AssertGPUSupport();
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ITERATIVE_DMATRIX_H_
