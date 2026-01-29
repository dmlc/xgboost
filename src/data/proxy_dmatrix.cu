/**
 * Copyright 2020-2025, XGBoost contributors
 */
#include "../encoder/ordinal.h"  // for DeviceColumnsView
#include "device_adapter.cuh"
#include "proxy_dmatrix.cuh"
#include "../common/type.h"  // for GetValueT
#include "proxy_dmatrix.h"

namespace xgboost::data {
void DMatrixProxy::SetCudaColumnar(StringView data) {
  auto adapter{std::make_shared<CudfAdapter>(data)};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  if (!adapter->Device().IsCUDA()) {
    // empty data
    CHECK_EQ(this->Info().num_row_, 0);
    ctx_ = ctx_.MakeCUDA(dh::CurrentDevice());
    return;
  }
  ctx_ = ctx_.MakeCUDA(adapter->Device().ordinal);
}

void DMatrixProxy::SetCudaArray(StringView data) {
  auto adapter(std::make_shared<CupyAdapter>(StringView{data}));
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  if (!adapter->Device().IsCUDA()) {
    // empty data
    CHECK_EQ(this->Info().num_row_, 0);
    ctx_ = ctx_.MakeCUDA(dh::CurrentDevice());
    return;
  }
  ctx_ = ctx_.MakeCUDA(adapter->Device().ordinal);
}

namespace cuda_impl {
std::shared_ptr<DMatrix> CreateDMatrixFromProxy(Context const* ctx,
                                                std::shared_ptr<DMatrixProxy> proxy,
                                                float missing) {
  return DispatchAny<false>(proxy.get(), [&](auto const& adapter) {
    auto p_fmat = std::shared_ptr<DMatrix>{DMatrix::Create(adapter.get(), missing, ctx->Threads())};
    CHECK_EQ(p_fmat->Info().num_row_, adapter->NumRows());
    return p_fmat;
  });
}

[[nodiscard]] bst_idx_t BatchSamples(DMatrixProxy const* proxy) {
  return cuda_impl::DispatchAny(proxy, [](auto const& value) { return value.NumRows(); });
}

[[nodiscard]] bst_idx_t BatchColumns(DMatrixProxy const* proxy) {
  return cuda_impl::DispatchAny(proxy, [](auto const& value) { return value.NumCols(); });
}

[[nodiscard]] bool BatchCatsIsRef(DMatrixProxy const* proxy) {
  return DispatchAny<false>(proxy, [&](auto const& adapter) {
    using AdapterT = typename common::GetValueT<decltype(adapter)>::element_type;
    if constexpr (std::is_same_v<AdapterT, CudfAdapter>) {
      return adapter->HasRefCategorical();
    }
    return false;
  });
}

[[nodiscard]] enc::DeviceColumnsView BatchCats(DMatrixProxy const* proxy) {
  return DispatchAny<false>(proxy, [&](auto const& adapter) {
    using AdapterT = typename common::GetValueT<decltype(adapter)>::element_type;
    if constexpr (std::is_same_v<AdapterT, CudfAdapter>) {
      if (adapter->HasRefCategorical()) {
        return adapter->RefCats();
      }
      return adapter->Cats();
    }
    return enc::DeviceColumnsView{};
  });
}
}  // namespace cuda_impl
}  // namespace xgboost::data
