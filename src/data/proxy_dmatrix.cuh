/**
 * Copyright 2021-2025, XGBoost contributors
 */
#include <any>     // for any_cast
#include <memory>  // for shared_ptr

#include "device_adapter.cuh"  // for MakeEncColumnarBatch
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
    auto adapter = std::any_cast<std::shared_ptr<CudfAdapter>>(proxy->Adapter());
    if constexpr (get_value) {
      auto value = adapter->Value();
      if (adapter->HasRefCategorical()) {
        auto [batch, mapping] = MakeEncColumnarBatch(proxy->Ctx(), adapter.get());
        return fn(batch);
      }
      return fn(value);
    } else {
      return fn(adapter);
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
