/**
 * Copyright 2020-2023, XGBoost contributors
 */
#include "device_adapter.cuh"
#include "proxy_dmatrix.cuh"
#include "proxy_dmatrix.h"

namespace xgboost::data {
void DMatrixProxy::FromCudaColumnar(StringView interface_str) {
  auto adapter{std::make_shared<CudfAdapter>(interface_str)};
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

void DMatrixProxy::FromCudaArray(StringView interface_str) {
  auto adapter(std::make_shared<CupyAdapter>(StringView{interface_str}));
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
}  // namespace cuda_impl
}  // namespace xgboost::data
