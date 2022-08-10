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
    CHECK(!param.regen) << "Only `hist` and `gpu_hist` tree method can use `QuantileDMatrix`.";
  }

  template <typename Page>
  static auto InvalidTreeMethod() {
    LOG(FATAL) << "Only `hist` and `gpu_hist` tree method can use `QuantileDMatrix`.";
    return BatchSet<Page>(BatchIterator<Page>(nullptr));
  }

 public:
  void InitFromCUDA(DataIterHandle iter, float missing, std::shared_ptr<DMatrix> ref);
  void InitFromCPU(DataIterHandle iter_handle, float missing, std::shared_ptr<DMatrix> ref);

 public:
  explicit IterativeDMatrix(DataIterHandle iter_handle, DMatrixHandle proxy,
                            std::shared_ptr<DMatrix> ref, DataIterResetCallback *reset,
                            XGDMatrixCallbackNext *next, float missing, int nthread,
                            bst_bin_t max_bin)
      : proxy_{proxy}, reset_{reset}, next_{next} {
    // fetch the first batch
    auto iter =
        DataIterProxy<DataIterResetCallback, XGDMatrixCallbackNext>{iter_handle, reset_, next_};
    iter.Reset();
    bool valid = iter.Next();
    CHECK(valid) << "Iterative DMatrix must have at least 1 batch.";

    auto d = MakeProxy(proxy_)->DeviceIdx();
    if (batch_param_.gpu_id != Context::kCpuId) {
      CHECK_EQ(d, batch_param_.gpu_id) << "All batch should be on the same device.";
    }
    batch_param_ = BatchParam{d, max_bin};
    batch_param_.sparse_thresh = 0.2;  // default from TrainParam

    ctx_.UpdateAllowUnknown(Args{{"nthread", std::to_string(nthread)}});
    if (d == Context::kCpuId) {
      this->InitFromCPU(iter_handle, missing, ref);
    } else {
      this->InitFromCUDA(iter_handle, missing, ref);
    }
  }
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
inline void IterativeDMatrix::InitFromCUDA(DataIterHandle iter, float missing,
                                           std::shared_ptr<DMatrix> ref) {
  // silent the warning about unused variables.
  (void)(proxy_);
  (void)(reset_);
  (void)(next_);
  common::AssertGPUSupport();
}
inline BatchSet<EllpackPage> IterativeDMatrix::GetEllpackBatches(const BatchParam &param) {
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
