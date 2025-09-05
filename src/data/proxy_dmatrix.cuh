/**
 * Copyright 2021-2025, XGBoost contributors
 */
#include <any>     // for any_cast
#include <memory>  // for shared_ptr

#include "device_adapter.cuh"  // for MakeEncColumnarBatch
#include "proxy_dmatrix.h"

namespace xgboost::data::cuda_impl {
// See the cpu impl for parameter documentation.
template <bool get_value = true, template <typename A> typename AddPtrT = std::shared_ptr,
          typename Fn>
decltype(auto) DispatchAny(Context const* ctx, std::any x, Fn&& fn, bool* type_error = nullptr) {
  auto has_type = [&] {
    if (type_error) {
      *type_error = false;
    }
  };
  if (x.type() == typeid(AddPtrT<CupyAdapter>)) {
    has_type();
    if constexpr (get_value) {
      auto value = std::any_cast<AddPtrT<CupyAdapter>>(x)->Value();
      return fn(value);
    } else {
      auto value = std::any_cast<AddPtrT<CupyAdapter>>(x);
      return fn(value);
    }
  } else if (x.type() == typeid(AddPtrT<CudfAdapter>)) {
    has_type();
    auto adapter = std::any_cast<AddPtrT<CudfAdapter>>(x);
    if constexpr (get_value) {
      auto value = adapter->Value();
      if (adapter->HasRefCategorical()) {
        auto [batch, mapping] = MakeEncColumnarBatch(ctx, adapter);
        return fn(batch);
      }
      return fn(value);
    } else {
      return fn(adapter);
    }
  } else {
    if (type_error) {
      *type_error = true;
    } else {
      LOG(FATAL) << "Unknown type: " << x.type().name();
    }
  }

  // Dummy return value
  if constexpr (get_value) {
    auto value = std::any_cast<AddPtrT<CudfAdapter>>(x)->Value();
    return fn(value);
  } else {
    auto value = std::any_cast<AddPtrT<CudfAdapter>>(x);
    return fn(value);
  }
}

template <bool get_value = true, typename Fn>
decltype(auto) DispatchAny(DMatrixProxy const* proxy, Fn&& fn, bool* type_error = nullptr) {
  return DispatchAny<get_value>(proxy->Ctx(), proxy->Adapter(), std::forward<Fn>(fn), type_error);
}
}  // namespace xgboost::data::cuda_impl
