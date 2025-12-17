/**
 * Copyright 2025, XGBoost Contributors
 */
#include "grad_c_api.h"

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr

#include "../data/proxy_dmatrix.h"  // for DMatrixProxy
#include "c_api_error.h"
#include "c_api_utils.h"
#include "xgboost/c_api.h"
#include "xgboost/string_view.h"  // for StringView

using namespace xgboost;  // NOLINT

XGB_DLL int XGGradientContainerCreate(BoosterHandle handle, DMatrixHandle dtrain,
                                      GradientContainerHandle *out) {
  API_BEGIN();
  CHECK_HANDLE();
  xgboost_CHECK_C_ARG_PTR(out);
  // auto jconfig = Json::Load(StringView{config});
  // auto n_samples = get<Integer const>(jconfig["n_samples"]);
  auto p_fmat = CastDMatrixHandle(dtrain);
  auto *learner = static_cast<Learner *>(handle);
  auto n_targets = learner->OutputLength();
  std::size_t shape[2]{static_cast<std::size_t>(p_fmat->Info().num_row_), n_targets};
  *out = new GradientContainerWithCtx{learner->Ctx(), common::Span<std::size_t const, 2>{shape}};
  API_END();
}

XGB_DLL int XGGradientContainerFree(GradientContainerHandle handle) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(handle);
  auto p_grads = CastGradientContainerHandle(handle);
  delete p_grads;
  API_END();
}

namespace xgboost {
namespace {
template <typename Fn>
int PushGradImpl(GradientContainerHandle handle, JArrayStr grad, JArrayStr hess, Fn &&fn) {
  API_BEGIN();

  xgboost_CHECK_C_ARG_PTR(grad);
  xgboost_CHECK_C_ARG_PTR(hess);
  auto p_grads = CastGradientContainerHandle(handle);
  fn(p_grads->ctx, StringView{grad}, StringView{hess}, &p_grads->gpairs);
  API_END();
}
}  // namespace
}  // namespace xgboost

XGB_DLL int XGGradientContainerPushGrad(GradientContainerHandle handle, JArrayStr grad,
                                        JArrayStr hess) {
  return PushGradImpl(handle, grad, hess,
                      [](Context const *ctx, StringView grad, StringView hess,
                         GradientContainer *gpair) { gpair->PushGrad(ctx, grad, hess); });
}

XGB_DLL int XGGradientContainerPushValueGrad(GradientContainerHandle handle, JArrayStr value_grad,
                                             JArrayStr value_hess) {
  return PushGradImpl(handle, value_grad, value_hess,
                      [](Context const *ctx, StringView grad, StringView hess,
                         GradientContainer *gpair) { gpair->PushValueGrad(ctx, grad, hess); });
}

struct InfoIter {
  std::int32_t batch_idx{0};
  std::shared_ptr<xgboost::DMatrix> p_fmat;
};

typedef void *InfoIterHandle;  // NOLINT

XGB_DLL int XGDMatrixGetInfoBatches(DMatrixHandle handle, InfoIterHandle *out) {
  API_BEGIN();
  CHECK_HANDLE();

  xgboost_CHECK_C_ARG_PTR(out);
  auto p_iter = new InfoIter{};
  auto p_fmat = xgboost::CastDMatrixHandle(handle);
  p_iter->p_fmat = p_fmat;
  *out = p_iter;

  API_END();
}

XGB_DLL int XGDMatrixInfoBatchNext(InfoIterHandle iter_handle, DMatrixHandle out) {
  API_BEGIN();

  xgboost_CHECK_C_ARG_PTR(iter_handle);
  xgboost_CHECK_C_ARG_PTR(out);
  auto proxy = xgboost::GetDMatrixProxy(out);

  auto iter = static_cast<InfoIter *>(iter_handle);
  CHECK(iter->p_fmat);

  auto ctx = iter->p_fmat->Ctx();

  linalg::Extent range;
  // We cannot safely split the data if there are groups. Although it doesn't reduce
  // accuracy too much for most cases, we split the data for distributed training as
  // well. But it's better to be a bit less efficient than be less accurate.
  if (iter->p_fmat->Info().IsRanking()) {
    range = linalg::Range<std::size_t>(0, iter->p_fmat->Info().num_row_);
  } else {
    auto begin = iter->p_fmat->BaseRowId(iter->batch_idx);
    auto size = iter->p_fmat->BatchSize(iter->batch_idx);
    range = linalg::Range<std::size_t>(begin, begin + size);
  }

  // TODO(jiamingy): Implement an actual cached iterator.
  iter->p_fmat->Info().Slice(ctx, range, &proxy->Info());

  iter->batch_idx++;
  if (iter->batch_idx == iter->p_fmat->NumBatches() || iter->p_fmat->Info().IsRanking()) {
    // Mark the iterator as invalid. For LTR, this is a single iteration.
    iter->p_fmat = nullptr;
  }

  API_END();
}

XGB_DLL int XGDMatrixInfoBatchIsValid(InfoIterHandle iter_handle, int32_t *is_valid) {
  API_BEGIN();

  xgboost_CHECK_C_ARG_PTR(iter_handle);

  auto iter = static_cast<InfoIter *>(iter_handle);
  CHECK(iter);
  *is_valid = static_cast<bool>(iter->p_fmat);
  API_END();
}

XGB_DLL int XGDMatrixInfoBatchEnd(InfoIterHandle iter_handle) {
  API_BEGIN();

  xgboost_CHECK_C_ARG_PTR(iter_handle);

  auto iter = static_cast<InfoIter *>(iter_handle);
  CHECK(iter);
  delete iter;

  API_END();
}
