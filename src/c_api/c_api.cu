/**
 * Copyright 2019-2025, XGBoost Contributors
 */
#include <thrust/transform.h>  // for transform

#include "../common/api_entry.h"       // for XGBAPIThreadLocalEntry
#include "../common/cuda_context.cuh"  // for CUDAContext
#include "../data/array_interface.h"  // for DispatchDType, ArrayInterface
#include "../data/device_adapter.cuh"
#include "../data/proxy_dmatrix.h"
#include "c_api_error.h"
#include "c_api_utils.h"
#include "xgboost/c_api.h"
#include "xgboost/data.h"
#include "xgboost/json.h"
#include "xgboost/learner.h"
#if defined(XGBOOST_USE_NCCL)
#include <nccl.h>
#endif
#if defined(XGBOOST_USE_NVCOMP)
#include <nvcomp/version.h>
#endif  // defined(XGBOOST_USE_NVCOMP)

namespace xgboost {
void XGBBuildInfoDevice(Json *p_info) {
  auto &info = *p_info;

  info["USE_CUDA"] = true;

  std::vector<Json> v{Json{Integer{THRUST_MAJOR_VERSION}}, Json{Integer{THRUST_MINOR_VERSION}},
                      Json{Integer{THRUST_SUBMINOR_VERSION}}};
  info["THRUST_VERSION"] = v;

  v = {Json{Integer{dh::CUDAVersion().first}}, Json{Integer{dh::CUDAVersion().second}}};
  info["CUDA_VERSION"] = v;

#if defined(XGBOOST_USE_NCCL)
  info["USE_NCCL"] = Boolean{true};
  v = {Json{Integer{NCCL_MAJOR}}, Json{Integer{NCCL_MINOR}}, Json{Integer{NCCL_PATCH}}};
  info["NCCL_VERSION"] = v;

#if defined(XGBOOST_USE_DLOPEN_NCCL)
  info["USE_DLOPEN_NCCL"] = Boolean{true};
#else
  info["USE_DLOPEN_NCCL"] = Boolean{false};
#endif  // defined(XGBOOST_USE_DLOPEN_NCCL)

#else
  info["USE_NCCL"] = Boolean{false};
  info["USE_DLOPEN_NCCL"] = Boolean{false};
#endif

#if defined(XGBOOST_USE_RMM)
  info["USE_RMM"] = Boolean{true};
  v = {Json{Integer{RMM_VERSION_MAJOR}}, Json{Integer{RMM_VERSION_MINOR}},
       Json{Integer{RMM_VERSION_PATCH}}};
  info["RMM_VERSION"] = v;
#else
  info["USE_RMM"] = Boolean{false};
#endif

#if defined(XGBOOST_USE_NVCOMP)
  info["USE_NVCOMP"] = Boolean{true};
  v = {Json{Integer{NVCOMP_VER_MAJOR}}, Json{Integer{NVCOMP_VER_MINOR}},
       Json{Integer{NVCOMP_VER_PATCH}}};
  info["NVCOMP_VERSION"] = v;
#else
  info["USE_NVCOMP"] = Boolean{false};
#endif
}

void XGBoostAPIGuard::SetGPUAttribute() {
  // Not calling `safe_cuda` to avoid unnecessary exception handling overhead.
  // If errors, do nothing, assuming running on CPU only machine.
  cudaGetDevice(&device_id_);
}

void XGBoostAPIGuard::RestoreGPUAttribute() {
  // Not calling `safe_cuda` to avoid unnecessary exception handling overhead.
  // If errors, do nothing, assuming running on CPU only machine.
  cudaSetDevice(device_id_);
}

void CopyGradientFromCUDAArrays(Context const *ctx, ArrayInterface<2, false> const &grad,
                                ArrayInterface<2, false> const &hess,
                                linalg::Matrix<GradientPair> *out_gpair) {
  auto grad_dev = dh::CudaGetPointerDevice(grad.data);
  auto hess_dev = dh::CudaGetPointerDevice(hess.data);
  CHECK_EQ(grad_dev, hess_dev) << "gradient and hessian should be on the same device.";
  auto &gpair = *out_gpair;
  gpair.SetDevice(DeviceOrd::CUDA(grad_dev));
  gpair.Reshape(grad.Shape<0>(), grad.Shape<1>());
  auto d_gpair = gpair.View(DeviceOrd::CUDA(grad_dev));
  auto cuctx = ctx->CUDACtx();

  DispatchDType(grad, DeviceOrd::CUDA(grad_dev), [&](auto &&t_grad) {
    DispatchDType(hess, DeviceOrd::CUDA(hess_dev), [&](auto &&t_hess) {
      CHECK_EQ(t_grad.Size(), t_hess.Size());
      thrust::for_each_n(cuctx->CTP(), thrust::make_counting_iterator(0ul), t_grad.Size(),
                         detail::CustomGradHessOp{t_grad, t_hess, d_gpair});
    });
  });
}
}                        // namespace xgboost

using namespace xgboost;  // NOLINT

XGB_DLL int XGDMatrixCreateFromCudaColumnar(char const *data,
                                            char const* c_json_config,
                                            DMatrixHandle *out) {
  API_BEGIN();

  xgboost_CHECK_C_ARG_PTR(c_json_config);
  xgboost_CHECK_C_ARG_PTR(data);

  std::string json_str{data};
  auto config = Json::Load(StringView{c_json_config});

  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, std::int64_t>(config, "nthread", 0);
  data::CudfAdapter adapter(json_str);
  *out =
      new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, n_threads));
  API_END();
}

XGB_DLL int XGDMatrixCreateFromCudaArrayInterface(char const *data,
                                                  char const* c_json_config,
                                                  DMatrixHandle *out) {
  API_BEGIN();
  std::string json_str{data};
  auto config = Json::Load(StringView{c_json_config});
  float missing = GetMissing(config);
  auto n_threads = OptionalArg<Integer, std::int64_t>(config, "nthread", 0);
  data::CupyAdapter adapter(json_str);
  *out =
      new std::shared_ptr<DMatrix>(DMatrix::Create(&adapter, missing, n_threads));
  API_END();
}

template <bool is_columnar>
int InplacePreidctCUDA(BoosterHandle handle, char const *data, char const *c_json_config,
                       std::shared_ptr<DMatrix> p_m, xgboost::bst_ulong const **out_shape,
                       xgboost::bst_ulong *out_dim, const float **out_result) {
  API_BEGIN();
  CHECK_HANDLE();
  if (!p_m) {
    p_m.reset(new data::DMatrixProxy);
  }
  auto proxy = dynamic_cast<data::DMatrixProxy *>(p_m.get());
  CHECK(proxy) << "Invalid input type for inplace predict.";
  xgboost_CHECK_C_ARG_PTR(data);

  if constexpr (is_columnar) {
    proxy->SetCudaColumnar(data);
  } else {
    proxy->SetCudaArray(data);
  }

  auto config = Json::Load(StringView{c_json_config});
  auto *learner = static_cast<Learner *>(handle);

  HostDeviceVector<float> *p_predt{nullptr};
  auto type = PredictionType(RequiredArg<Integer>(config, "type", __func__));
  float missing = GetMissing(config);

  learner->InplacePredict(p_m, type, missing, &p_predt,
                          RequiredArg<Integer>(config, "iteration_begin", __func__),
                          RequiredArg<Integer>(config, "iteration_end", __func__));
  CHECK(p_predt);
  if (learner->Ctx()->IsCUDA()) {
    CHECK(p_predt->DeviceCanRead() && !p_predt->HostCanRead());
  }
  p_predt->SetDevice(proxy->Device());

  auto &shape = learner->GetThreadLocal().prediction_shape;
  size_t n_samples = p_m->Info().num_row_;
  auto chunksize = n_samples == 0 ? 0 : p_predt->Size() / n_samples;
  bool strict_shape = RequiredArg<Boolean>(config, "strict_shape", __func__);

  xgboost_CHECK_C_ARG_PTR(out_result);
  xgboost_CHECK_C_ARG_PTR(out_shape);
  xgboost_CHECK_C_ARG_PTR(out_dim);

  CalcPredictShape(strict_shape, type, n_samples, p_m->Info().num_col_, chunksize,
                   learner->Groups(), learner->BoostedRounds(), &shape, out_dim);
  *out_shape = dmlc::BeginPtr(shape);
  *out_result = p_predt->ConstDevicePointer();
  API_END();
}

XGB_DLL int XGBoosterPredictFromCudaColumnar(BoosterHandle handle, char const *data,
                                             char const *c_json_config, DMatrixHandle m,
                                             xgboost::bst_ulong const **out_shape,
                                             xgboost::bst_ulong *out_dim,
                                             const float **out_result) {
  std::shared_ptr<DMatrix> p_m{nullptr};
  xgboost_CHECK_C_ARG_PTR(c_json_config);
  if (m) {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  return InplacePreidctCUDA<true>(handle, data, c_json_config, p_m, out_shape, out_dim, out_result);
}

XGB_DLL int XGBoosterPredictFromCudaArray(BoosterHandle handle, char const *data,
                                          char const *c_json_config, DMatrixHandle m,
                                          xgboost::bst_ulong const **out_shape,
                                          xgboost::bst_ulong *out_dim, const float **out_result) {
  std::shared_ptr<DMatrix> p_m{nullptr};
  if (m) {
    p_m = *static_cast<std::shared_ptr<DMatrix> *>(m);
  }
  xgboost_CHECK_C_ARG_PTR(out_result);
  return InplacePreidctCUDA<false>(handle, data, c_json_config, p_m, out_shape, out_dim,
                                   out_result);
}
