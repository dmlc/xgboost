/*!
 * Copyright 2021 XGBoost contributors
 */
#include "device_adapter.cuh"
#include "proxy_dmatrix.h"

namespace xgboost {
namespace data {
template <typename Fn>
decltype(auto) Dispatch(DMatrixProxy const* proxy, Fn fn) {
  if (proxy->Adapter().type() == typeid(std::shared_ptr<CupyAdapter>)) {
    auto value = dmlc::get<std::shared_ptr<CupyAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  } else if (proxy->Adapter().type() == typeid(std::shared_ptr<CudfAdapter>)) {
    auto value = dmlc::get<std::shared_ptr<CudfAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  } else {
    LOG(FATAL) << "Unknown type: " << proxy->Adapter().type().name();
    auto value = dmlc::get<std::shared_ptr<CudfAdapter>>(
        proxy->Adapter())->Value();
    return fn(value);
  }
}
}  // namespace data
}  // namespace xgboost
