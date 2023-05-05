/**
 * Copyright 2020-2023, XGBoost contributors
 */
#include "device_adapter.cuh"
#include "proxy_dmatrix.cuh"
#include "proxy_dmatrix.h"

namespace xgboost::data {
void DMatrixProxy::FromCudaColumnar(StringView interface_str) {
  std::shared_ptr<data::CudfAdapter> adapter{new CudfAdapter{interface_str}};
  auto const& value = adapter->Value();
  this->batch_ = adapter;
  ctx_.gpu_id = adapter->DeviceIdx();
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  if (ctx_.gpu_id < 0) {
    CHECK_EQ(this->Info().num_row_, 0);
    ctx_.gpu_id = dh::CurrentDevice();
  }
}

void DMatrixProxy::FromCudaArray(StringView interface_str) {
  std::shared_ptr<CupyAdapter> adapter(new CupyAdapter{StringView{interface_str}});
  this->batch_ = adapter;
  ctx_.gpu_id = adapter->DeviceIdx();
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  if (ctx_.gpu_id < 0) {
    CHECK_EQ(this->Info().num_row_, 0);
    ctx_.gpu_id = dh::CurrentDevice();
  }
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
