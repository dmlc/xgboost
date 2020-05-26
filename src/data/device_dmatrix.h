/*!
 * Copyright 2020 by Contributors
 * \file device_dmatrix.h
 * \brief Device-memory version of DMatrix.
 */
#ifndef XGBOOST_DATA_DEVICE_DMATRIX_H_
#define XGBOOST_DATA_DEVICE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <memory>

#include "adapter.h"
#include "simple_batch_iterator.h"
#include "simple_dmatrix.h"

namespace xgboost {
namespace data {

class DeviceDMatrix : public DMatrix {
 public:
  template <typename AdapterT>
  explicit DeviceDMatrix(AdapterT* adapter, float missing, int nthread, int max_bin);

  MetaInfo& Info() override { return info_; }

  const MetaInfo& Info() const override { return info_; }

  bool SingleColBlock() const override { return true; }

  bool EllpackExists() const override { return true; }
  bool SparsePageExists() const override { return false; }
  DMatrix *Slice(common::Span<int32_t const> ridxs) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for Device DMatrix.";
    return nullptr;
  }

 private:
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
  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override {
    auto begin_iter = BatchIterator<EllpackPage>(
        new SimpleBatchIteratorImpl<EllpackPage>(ellpack_page_.get()));
    return BatchSet<EllpackPage>(begin_iter);
  }

  MetaInfo info_;
  // source data pointer.
  std::unique_ptr<EllpackPage> ellpack_page_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_DEVICE_DMATRIX_H_
