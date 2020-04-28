/*!
 * Copyright 2020 by Contributors
 * \file iterative_device_dmatrix.h
 */
#ifndef XGBOOST_DATA_ITERATIVE_DEVICE_DMATRIX_H_
#define XGBOOST_DATA_ITERATIVE_DEVICE_DMATRIX_H_

#include <xgboost/base.h>
#include <xgboost/data.h>

#include <vector>

#include "../common/hist_util.h"
#include "sparse_page_source.h"

namespace xgboost {
namespace data {

class IteraterSource : dmlc::DataIter<EllpackPage> {
  std::shared_ptr<std::vector<EllpackPage>> ptr_;
  size_t current_ {0};

 public:
  explicit IteraterSource(std::shared_ptr<std::vector<EllpackPage>> ptr)
      : ptr_{std::move(ptr)} {}
  ~IteraterSource() noexcept(false) override = default;

  void BeforeFirst() override { current_ = 0; }
  /*! \brief move to next item */
  bool Next() override {
    if (current_ == ptr_->size()) {
      return false;
    }
    current_++;
    return true;
  }
  /*! \brief get current data */
  EllpackPage const &Value() const override {
    return (*ptr_).at(current_);
  }
  EllpackPage &Value() {
    return (*ptr_).at(current_);
  }
};

class IterativeDeviceDMatrix : public DMatrix {
  MetaInfo info_;
  BatchParam p_;
  std::shared_ptr<std::vector<EllpackPage>> pages_;
  std::unique_ptr<IteraterSource> iter_;

 public:
  template <typename Adapter>
  explicit IterativeDeviceDMatrix(Adapter* adapter, float missing,
                                  int nthread, int max_bin);

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

  BatchSet<EllpackPage> GetEllpackBatches(const BatchParam& param) override {
    CHECK(pages_);
    if (!iter_) {
      iter_.reset(new IteraterSource(pages_));
    }
    auto begin_iter = BatchIterator<EllpackPage>(
        new SparseBatchIteratorImpl<IteraterSource, EllpackPage>(iter_.get()));
    return BatchSet<EllpackPage>(begin_iter);
  }

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
