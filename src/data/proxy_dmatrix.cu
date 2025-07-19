/**
 * Copyright 2020-2025, XGBoost contributors
 */
#include "../encoder/ordinal.h"  // for DeviceColumnsView
#include "device_adapter.cuh"
#include "proxy_dmatrix.cuh"
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
  return Dispatch<false>(proxy.get(), [&](auto const& adapter) {
    auto p_fmat = std::shared_ptr<DMatrix>{DMatrix::Create(adapter.get(), missing, ctx->Threads())};
    return p_fmat;
  });
}

[[nodiscard]] bst_idx_t BatchSamples(DMatrixProxy const* proxy) {
  return cuda_impl::Dispatch(proxy, [](auto const& value) { return value.NumRows(); });
}

[[nodiscard]] bst_idx_t BatchColumns(DMatrixProxy const* proxy) {
  return cuda_impl::Dispatch(proxy, [](auto const& value) { return value.NumCols(); });
}

[[nodiscard]] enc::DeviceColumnsView BatchCats(DMatrixProxy const* proxy, bool ref_if_avail) {
  return Dispatch<false>(proxy, [&](auto const& adapter) {
    using AdapterT = typename std::remove_reference_t<decltype(adapter)>::element_type;
    if constexpr (std::is_same_v<AdapterT, CudfAdapter>) {
      if (ref_if_avail && adapter->HasRefCategorical()) {
        return adapter->RefCats();
      }
      return adapter->Cats();
    }
    return enc::DeviceColumnsView{};
  });
}
}  // namespace cuda_impl
}  // namespace xgboost::data
