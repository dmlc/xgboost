/*!
 * Copyright 2020 by Contributors
 * \file iterative_device_dmatrix.h
 */
#ifndef XGBOOST_DATA_ITERATIVE_DEVICE_DMATRIX_H_
#define XGBOOST_DATA_ITERATIVE_DEVICE_DMATRIX_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/c_api.h"
#include "proxy_dmatrix.h"

namespace xgboost {
namespace data {

class IterativeDeviceDMatrix : public DMatrix {
  MetaInfo info_;
  BatchParam batch_param_;
  std::shared_ptr<EllpackPage> page_;

  DMatrixHandle proxy_;
  DataIterResetCallback *reset_;
  XGDMatrixCallbackNext *next_;

 public:
  void Initialize(DataIterHandle iter, float missing, int nthread);

 public:
  explicit IterativeDeviceDMatrix(DataIterHandle iter, DMatrixHandle proxy,
                                  DataIterResetCallback *reset,
                                  XGDMatrixCallbackNext *next, float missing,
                                  int nthread, int max_bin)
      : proxy_{proxy}, reset_{reset}, next_{next} {
    batch_param_ = BatchParam{0, max_bin, 0};
    this->Initialize(iter, missing, nthread);
  }

  bool EllpackExists() const override { return true; }
  bool SparsePageExists() const override { return false; }
  DMatrix *Slice(common::Span<int32_t const> ridxs) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for Device DMatrix.";
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

  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override;

  bool SingleColBlock() const override { return false; }

  MetaInfo& Info() override {
    return info_;
  }
  MetaInfo const& Info() const override {
    return info_;
  }
};
}  // namespace data
}  // namespace xgboost

#endif  // XGBOOST_DATA_ITERATIVE_DEVICE_DMATRIX_H_
