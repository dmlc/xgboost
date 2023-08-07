/**
 * Copyright 2020-2023, XGBoost contributors
 */
#ifndef XGBOOST_DATA_PROXY_DMATRIX_H_
#define XGBOOST_DATA_PROXY_DMATRIX_H_

#include <any>  // for any, any_cast
#include <memory>
#include <string>
#include <utility>

#include "adapter.h"
#include "xgboost/c_api.h"
#include "xgboost/context.h"
#include "xgboost/data.h"

namespace xgboost::data {
/*
 * \brief A proxy to external iterator.
 */
template <typename ResetFn, typename NextFn>
class DataIterProxy {
  DataIterHandle iter_;
  ResetFn* reset_;
  NextFn* next_;

 public:
  DataIterProxy(DataIterHandle iter, ResetFn* reset, NextFn* next)
      : iter_{iter}, reset_{reset}, next_{next} {}

  bool Next() { return next_(iter_); }
  void Reset() { reset_(iter_); }
};

/*
 * \brief A proxy of DMatrix used by external iterator.
 */
class DMatrixProxy : public DMatrix {
  MetaInfo info_;
  std::any batch_;
  Context ctx_;

#if defined(XGBOOST_USE_CUDA)
  void FromCudaColumnar(StringView interface_str);
  void FromCudaArray(StringView interface_str);
#endif  // defined(XGBOOST_USE_CUDA)

 public:
  int DeviceIdx() const { return ctx_.gpu_id; }

  void SetCUDAArray(char const* c_interface) {
    common::AssertGPUSupport();
    CHECK(c_interface);
#if defined(XGBOOST_USE_CUDA)
    StringView interface_str{c_interface};
    Json json_array_interface = Json::Load(interface_str);
    if (IsA<Array>(json_array_interface)) {
      this->FromCudaColumnar(interface_str);
    } else {
      this->FromCudaArray(interface_str);
    }
#endif  // defined(XGBOOST_USE_CUDA)
  }

  void SetArrayData(char const* c_interface);
  void SetCSRData(char const* c_indptr, char const* c_indices, char const* c_values,
                  bst_feature_t n_features, bool on_host);

  MetaInfo& Info() override { return info_; }
  MetaInfo const& Info() const override { return info_; }
  Context const* Ctx() const override { return &ctx_; }

  bool SingleColBlock() const override { return false; }
  bool EllpackExists() const override { return false; }
  bool GHistIndexExists() const override { return false; }
  bool SparsePageExists() const override { return false; }

  template <typename Page>
  BatchSet<Page> NoBatch() {
    LOG(FATAL) << "Proxy DMatrix cannot return data batch.";
    return BatchSet<Page>(BatchIterator<Page>(nullptr));
  }

  DMatrix* Slice(common::Span<int32_t const> /*ridxs*/) override {
    LOG(FATAL) << "Slicing DMatrix is not supported for Proxy DMatrix.";
    return nullptr;
  }
  DMatrix* SliceCol(int, int) override {
    LOG(FATAL) << "Slicing DMatrix columns is not supported for Proxy DMatrix.";
    return nullptr;
  }
  BatchSet<SparsePage> GetRowBatches() override { return NoBatch<SparsePage>(); }
  BatchSet<CSCPage> GetColumnBatches(Context const*) override { return NoBatch<CSCPage>(); }
  BatchSet<SortedCSCPage> GetSortedColumnBatches(Context const*) override {
    return NoBatch<SortedCSCPage>();
  }
  BatchSet<EllpackPage> GetEllpackBatches(Context const*, BatchParam const&) override {
    return NoBatch<EllpackPage>();
  }
  BatchSet<GHistIndexMatrix> GetGradientIndex(Context const*, BatchParam const&) override {
    return NoBatch<GHistIndexMatrix>();
  }
  BatchSet<ExtSparsePage> GetExtBatches(Context const*, BatchParam const&) override {
    return NoBatch<ExtSparsePage>();
  }
  std::any Adapter() const { return batch_; }
};

inline DMatrixProxy* MakeProxy(DMatrixHandle proxy) {
  auto proxy_handle = static_cast<std::shared_ptr<DMatrix>*>(proxy);
  CHECK(proxy_handle) << "Invalid proxy handle.";
  DMatrixProxy* typed = static_cast<DMatrixProxy*>(proxy_handle->get());
  CHECK(typed) << "Invalid proxy handle.";
  return typed;
}

template <typename Fn>
decltype(auto) HostAdapterDispatch(DMatrixProxy const* proxy, Fn fn, bool* type_error = nullptr) {
  if (proxy->Adapter().type() == typeid(std::shared_ptr<CSRArrayAdapter>)) {
    auto value = std::any_cast<std::shared_ptr<CSRArrayAdapter>>(proxy->Adapter())->Value();
    if (type_error) {
      *type_error = false;
    }
    return fn(value);
  } else if (proxy->Adapter().type() == typeid(std::shared_ptr<ArrayAdapter>)) {
    auto value = std::any_cast<std::shared_ptr<ArrayAdapter>>(proxy->Adapter())->Value();
    if (type_error) {
      *type_error = false;
    }
    return fn(value);
  } else {
    if (type_error) {
      *type_error = true;
    } else {
      LOG(FATAL) << "Unknown type: " << proxy->Adapter().type().name();
    }
    return std::result_of_t<Fn(decltype(std::declval<std::shared_ptr<ArrayAdapter>>()->Value()))>();
  }
}
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_PROXY_DMATRIX_H_
