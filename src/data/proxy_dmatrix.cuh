/**
 * Copyright 2021-2023 XGBoost contributors
 */
#include <any>  // for any, any_cast

#include "device_adapter.cuh"
#include "proxy_dmatrix.h"

namespace xgboost::data {
template <typename Fn>
decltype(auto) Dispatch(DMatrixProxy const* proxy, Fn fn) {
  if (proxy->Adapter().type() == typeid(std::shared_ptr<CupyAdapter>)) {
    auto value = std::any_cast<std::shared_ptr<CupyAdapter>>(proxy->Adapter())->Value();
    return fn(value);
  } else if (proxy->Adapter().type() == typeid(std::shared_ptr<CudfAdapter>)) {
    auto value = std::any_cast<std::shared_ptr<CudfAdapter>>(proxy->Adapter())->Value();
    return fn(value);
  } else {
    LOG(FATAL) << "Unknown type: " << proxy->Adapter().type().name();
    auto value = std::any_cast<std::shared_ptr<CudfAdapter>>(proxy->Adapter())->Value();
    return fn(value);
  }
}
}  // namespace xgboost::data
