/**
 * Copyright 2021-2023 XGBoost contributors
 */
#include <any>  // for any, any_cast

#include "device_adapter.cuh"
#include "proxy_dmatrix.h"

namespace xgboost::data::cuda_impl {
template <bool get_value = true, typename Fn>
decltype(auto) Dispatch(DMatrixProxy const* proxy, Fn fn) {
  if (proxy->Adapter().type() == typeid(std::shared_ptr<CupyAdapter>)) {
    if constexpr (get_value) {
      auto value = std::any_cast<std::shared_ptr<CupyAdapter>>(proxy->Adapter())->Value();
      return fn(value);
    } else {
      auto value = std::any_cast<std::shared_ptr<CupyAdapter>>(proxy->Adapter());
      return fn(value);
    }
  } else if (proxy->Adapter().type() == typeid(std::shared_ptr<CudfAdapter>)) {
    if constexpr (get_value) {
      auto value = std::any_cast<std::shared_ptr<CudfAdapter>>(proxy->Adapter())->Value();
      return fn(value);
    } else {
      auto value = std::any_cast<std::shared_ptr<CudfAdapter>>(proxy->Adapter());
      return fn(value);
    }
  } else {
    LOG(FATAL) << "Unknown type: " << proxy->Adapter().type().name();
    if constexpr (get_value) {
      auto value = std::any_cast<std::shared_ptr<CudfAdapter>>(proxy->Adapter())->Value();
      return fn(value);
    } else {
      auto value = std::any_cast<std::shared_ptr<CudfAdapter>>(proxy->Adapter());
      return fn(value);
    }
  }
}
}  // namespace xgboost::data::cuda_impl
