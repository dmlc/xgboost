/*!
 * Copyright 2020 XGBoost contributors
 */
#ifndef XGBOOST_DATA_PROXY_DMATRIX_H_
#define XGBOOST_DATA_PROXY_DMATRIX_H_

#include <dmlc/any.h>

#include <memory>
#include <string>
#include <utility>

#include "xgboost/data.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/c_api.h"
#include "adapter.h"

namespace xgboost {
namespace data {
/*
 * \brief A proxy to external iterator.
 */
template <typename ResetFn, typename NextFn>
class DataIterProxy {
  DataIterHandle iter_;
  ResetFn* reset_;
  NextFn* next_;

 public:
  DataIterProxy(DataIterHandle iter, ResetFn* reset, NextFn* next) :
      iter_{iter},
      reset_{reset}, next_{next} {}

  bool Next() {
    return next_(iter_);
  }
  void Reset() {
    reset_(iter_);
  }
};

/*
 * \brief A proxy of DMatrix used by external iterator.
 */
class DMatrixProxy : public DMatrix {
  MetaInfo info_;
  dmlc::any batch_;
  int32_t device_ { xgboost::GenericParameter::kCpuId };

#if defined(XGBOOST_USE_CUDA)
  void FromCudaColumnar(std::string interface_str);
  void FromCudaArray(std::string interface_str);
#endif  // defined(XGBOOST_USE_CUDA)

 public:
  int DeviceIdx() const { return device_; }

  void SetData(char const* c_interface) {
    common::AssertGPUSupport();
#if defined(XGBOOST_USE_CUDA)
    std::string interface_str = c_interface;
    Json json_array_interface =
        Json::Load({interface_str.c_str(), interface_str.size()});
    if (IsA<Array>(json_array_interface)) {
      this->FromCudaColumnar(interface_str);
    } else {
      this->FromCudaArray(interface_str);
    }
    if (this->info_.num_row_ == 0) {
      this->device_ = GenericParameter::kCpuId;
    }
#endif  // defined(XGBOOST_USE_CUDA)
  }

  MetaInfo& Info() override { return info_; }
  MetaInfo const& Info() const override { return info_; }
  bool SingleColBlock() const override { return true; }
  bool EllpackExists() const override { return true; }
  bool SparsePageExists() const override { return false; }
  DMatrix *Slice(common::Span<int32_t const> ridxs) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for Proxy DMatrix.";
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
    LOG(FATAL) << "Not implemented.";
    return BatchSet<EllpackPage>(BatchIterator<EllpackPage>(nullptr));
  }

  dmlc::any Adapter() const {
    return batch_;
  }
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_PROXY_DMATRIX_H_
