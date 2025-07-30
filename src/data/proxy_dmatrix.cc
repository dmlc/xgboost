/**
 * Copyright 2021-2025, XGBoost Contributors
 */

#include "proxy_dmatrix.h"

#include <memory>  // for shared_ptr

#include "xgboost/context.h"  // for Context
#include "xgboost/data.h"     // for DMatrix
#include "xgboost/logging.h"
#include "xgboost/string_view.h"  // for StringView

#if !defined(XGBOOST_USE_CUDA)
#include "../common/common.h"  // for AssertGPUSupport
#endif

namespace xgboost::data {
void DMatrixProxy::SetColumnar(StringView data) {
  std::shared_ptr<ColumnarAdapter> adapter{new ColumnarAdapter{data}};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_.Init(Args{{"device", "cpu"}});
}

void DMatrixProxy::SetArray(StringView data) {
  std::shared_ptr<ArrayAdapter> adapter{new ArrayAdapter{data}};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_.Init(Args{{"device", "cpu"}});
}

void DMatrixProxy::SetCsr(char const *c_indptr, char const *c_indices, char const *c_values,
                          bst_feature_t n_features, bool on_host) {
  CHECK(on_host) << "Not implemented on device.";
  std::shared_ptr<CSRArrayAdapter> adapter{new CSRArrayAdapter(
      StringView{c_indptr}, StringView{c_indices}, StringView{c_values}, n_features)};
  this->batch_ = adapter;
  this->Info().num_col_ = adapter->NumColumns();
  this->Info().num_row_ = adapter->NumRows();
  this->ctx_.Init(Args{{"device", "cpu"}});
}

#if !defined(XGBOOST_USE_CUDA)
void DMatrixProxy::SetCudaArray(StringView) { common::AssertGPUSupport(); }
void DMatrixProxy::SetCudaColumnar(StringView) { common::AssertGPUSupport(); }
#endif  // !defined(XGBOOST_USE_CUDA)

namespace cuda_impl {
#if !defined(XGBOOST_USE_CUDA)
[[nodiscard]] bst_idx_t BatchSamples(DMatrixProxy const *) {
  common::AssertGPUSupport();
  return 0;
}
[[nodiscard]] bst_idx_t BatchColumns(DMatrixProxy const *) {
  common::AssertGPUSupport();
  return 0;
}
#else
std::shared_ptr<DMatrix> CreateDMatrixFromProxy(Context const *ctx,
                                                std::shared_ptr<DMatrixProxy> proxy, float missing);
#endif  // XGBOOST_USE_CUDA
}  // namespace cuda_impl

std::shared_ptr<DMatrix> CreateDMatrixFromProxy(Context const *ctx,
                                                std::shared_ptr<DMatrixProxy> proxy,
                                                float missing) {
  bool type_error{false};
  std::shared_ptr<DMatrix> p_fmat{nullptr};

  if (proxy->Ctx()->IsCUDA()) {
#if defined(XGBOOST_USE_CUDA)
    p_fmat = cuda_impl::CreateDMatrixFromProxy(ctx, proxy, missing);
#else
    common::AssertGPUSupport();
#endif
  } else {
    p_fmat = data::HostAdapterDispatch<false>(
        proxy.get(),
        [&](auto const &adapter) {
          auto p_fmat =
              std::shared_ptr<DMatrix>(DMatrix::Create(adapter.get(), missing, ctx->Threads()));
          CHECK_EQ(p_fmat->Info().num_row_, adapter->NumRows());
          return p_fmat;
        },
        &type_error);
  }

  CHECK(p_fmat) << "Failed to fallback.";
  p_fmat->Info().Extend(proxy->Info(), /*accumulate_rows=*/false, true);
  return p_fmat;
}
}  // namespace xgboost::data
