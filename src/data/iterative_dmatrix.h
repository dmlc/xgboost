/*!
 * Copyright 2020-2022 by Contributors
 * \file iterative_dmatrix.h
 */
#ifndef XGBOOST_DATA_ITERATIVE_DMATRIX_H_
#define XGBOOST_DATA_ITERATIVE_DMATRIX_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/c_api.h"
#include "proxy_dmatrix.h"
#include "simple_batch_iterator.h"

namespace xgboost {
namespace data {

class IterativeDMatrix : public DMatrix {
  MetaInfo info_;
  Context ctx_;
  BatchParam batch_param_;
  std::shared_ptr<EllpackPage> page_;

  DMatrixHandle proxy_;
  DataIterResetCallback *reset_;
  XGDMatrixCallbackNext *next_;

 public:
  void InitFromCUDA(DataIterHandle iter, float missing);

 public:
  explicit IterativeDMatrix(DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback *reset,
                            XGDMatrixCallbackNext *next, float missing, int nthread, int max_bin)
      : proxy_{proxy}, reset_{reset}, next_{next} {
    batch_param_ = BatchParam{MakeProxy(proxy_)->DeviceIdx(), max_bin};
    ctx_.UpdateAllowUnknown(Args{{"nthread", std::to_string(nthread)}});
    this->InitFromCUDA(iter, missing);
  }
  ~IterativeDMatrix() override = default;

  bool EllpackExists() const override { return true; }
  bool SparsePageExists() const override { return false; }
  DMatrix *Slice(common::Span<int32_t const>) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for Quantile DMatrix.";
    return nullptr;
  }
  BatchSet<SparsePage> GetRowBatches() override {
    LOG(FATAL) << "Not implemented.";
    return BatchSet<SparsePage>(BatchIterator<SparsePage>(nullptr));
  }
  BatchSet<CSCPage> GetColumnBatches() override {
    LOG(FATAL) << "Not implemented.";
    return BatchSet<CSCPage>(BatchIterator<CSCPage>(nullptr));
  }
  BatchSet<SortedCSCPage> GetSortedColumnBatches() override {
    LOG(FATAL) << "Not implemented.";
    return BatchSet<SortedCSCPage>(BatchIterator<SortedCSCPage>(nullptr));
  }
  BatchSet<GHistIndexMatrix> GetGradientIndex(const BatchParam&) override {
    LOG(FATAL) << "Not implemented.";
    return BatchSet<GHistIndexMatrix>(BatchIterator<GHistIndexMatrix>(nullptr));
  }

  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  bool SingleColBlock() const override { return true; }

  MetaInfo &Info() override { return info_; }
  MetaInfo const &Info() const override { return info_; }

  Context const *Ctx() const override { return &ctx_; }
};

#if !defined(XGBOOST_USE_CUDA)
inline void IterativeDMatrix::InitFromCUDA(DataIterHandle iter, float missing) {
  // silent the warning about unused variables.
  (void)(proxy_);
  (void)(reset_);
  (void)(next_);
  common::AssertGPUSupport();
}
inline BatchSet<EllpackPage> IterativeDMatrix::GetEllpackBatches(const BatchParam& param) {
  common::AssertGPUSupport();
  auto begin_iter =
      BatchIterator<EllpackPage>(new SimpleBatchIteratorImpl<EllpackPage>(page_));
  return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(begin_iter));
}
#endif  // !defined(XGBOOST_USE_CUDA)
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ITERATIVE_DMATRIX_H_
