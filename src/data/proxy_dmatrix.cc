/**
 * Copyright 2021-2023, XGBoost Contributors
 * \file proxy_dmatrix.cc
 */

#include "proxy_dmatrix.h"

namespace xgboost::data {
void DMatrixProxy::SetArrayData(StringView interface_str) {
  std::shared_ptr<ArrayAdapter> adapter{new ArrayAdapter{interface_str}};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_.gpu_id = Context::kCpuId;
}

void DMatrixProxy::SetCSRData(char const *c_indptr, char const *c_indices,
                              char const *c_values, bst_feature_t n_features, bool on_host) {
  CHECK(on_host) << "Not implemented on device.";
  std::shared_ptr<CSRArrayAdapter> adapter{new CSRArrayAdapter(
      StringView{c_indptr}, StringView{c_indices}, StringView{c_values}, n_features)};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_.gpu_id = Context::kCpuId;
}

namespace cuda_impl {
std::shared_ptr<DMatrix> CreateDMatrixFromProxy(Context const *ctx,
                                                std::shared_ptr<DMatrixProxy> proxy, float missing);
#if !defined(XGBOOST_USE_CUDA)
std::shared_ptr<DMatrix> CreateDMatrixFromProxy(Context const *, std::shared_ptr<DMatrixProxy>,
                                                float) {
  return nullptr;
}
#endif  // XGBOOST_USE_CUDA
}  // namespace cuda_impl

std::shared_ptr<DMatrix> CreateDMatrixFromProxy(Context const *ctx,
                                                std::shared_ptr<DMatrixProxy> proxy,
                                                float missing) {
  bool type_error{false};
  std::shared_ptr<DMatrix> p_fmat{nullptr};
  if (proxy->Ctx()->IsCPU()) {
    p_fmat = data::HostAdapterDispatch<false>(
        proxy.get(),
        [&](auto const &adapter) {
          auto p_fmat =
              std::shared_ptr<DMatrix>(DMatrix::Create(adapter.get(), missing, ctx->Threads()));
          return p_fmat;
        },
        &type_error);
  } else {
    p_fmat = cuda_impl::CreateDMatrixFromProxy(ctx, proxy, missing);
  }

  CHECK(p_fmat) << "Failed to fallback.";
  p_fmat->Info() = proxy->Info().Copy();
  return p_fmat;
}
}  // namespace xgboost::data
