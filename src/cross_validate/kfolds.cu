/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, XGBoost Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

#include <cstddef>
#include <cstdint>
#include <cub/device/device_histogram.cuh>
#include <limits>

#include "../c_api/c_api_error.h"
#include "../common/cuda_context.cuh"
#include "../common/device_helpers.cuh"
#include "../data/array_interface.h"
#include "kfolds.h"

namespace xgboost::cv {
FoldAssignment::FoldAssignment(Context const* ctx, std::size_t k_folds,
                               common::Span<std::int64_t const> ids) {
  CHECK(ctx->IsCUDA()) << "Fused CV requires CUDA.";
  CHECK_GE(k_folds, 2);
  CHECK_LE(k_folds, ids.size());
  CHECK_LE(ids.size(), std::numeric_limits<FoldId>::max());
  using thrust::placeholders::_1;
  CHECK(thrust::all_of(ctx->CUDACtx()->CTP(), dh::tbegin(ids), dh::tend(ids),
                       (_1 >= 0) && (_1 < static_cast<std::int64_t>(k_folds))))
      << "Fold IDs must be in [0, k_folds).";
  ids_.SetDevice(ctx->Device());
  ids_.Resize(ids.size());
  thrust::copy(ctx->CUDACtx()->CTP(), dh::tbegin(ids), dh::tend(ids),
               dh::tbegin(ids_.DeviceSpan()));
  counts_.resize(k_folds);
  counts_ = this->CountValidation(ctx, {0, this->Size()});
}

MembershipView FoldAssignment::ReadRows(RowRange rows) const {
  CHECK_LE(rows.begin, rows.end);
  CHECK_LE(rows.end, this->Size());
  return {this->Ids().subspan(rows.begin, rows.Size()), rows.begin};
}

std::vector<bst_idx_t> FoldAssignment::CountValidation(Context const* ctx, RowRange rows) const {
  CHECK(ctx->Device() == this->Device());
  auto view = this->ReadRows(rows);
  // CUB's 64-bit atomic counters require unsigned long long.
  static_assert(sizeof(unsigned long long) == sizeof(bst_idx_t));  // NOLINT
  dh::DeviceUVector<unsigned long long> counts(this->KFolds());    // NOLINT
  auto out = dh::ToSpan(counts);
  std::size_t n_bytes{0};
  auto histogram = [&](void* temp) {
    dh::safe_cuda(cub::DeviceHistogram::HistogramEven(
        temp, n_bytes, view.ids.data(), out.data(), static_cast<int>(this->KFolds()) + 1,
        std::uint64_t{0}, static_cast<std::uint64_t>(this->KFolds()),
        static_cast<std::int64_t>(view.ids.size()), ctx->CUDACtx()->Stream()));
  };
  histogram(nullptr);
  dh::CachingDeviceUVector<std::byte> temp(n_bytes);
  histogram(temp.data());
  // The constructor counts the full range to establish the nonempty-fold invariant.
  if (rows.Size() == this->Size()) {
    CHECK(thrust::all_of(ctx->CUDACtx()->CTP(), counts.begin(), counts.end(),
                         [] XGBOOST_DEVICE(bst_idx_t n) { return n != 0; }))
        << "Every fold must hold out at least one row.";
  }
  std::vector<bst_idx_t> result(counts.size());
  // Only O(K) allocation sizes cross to the host. Membership stays on the GPU.
  dh::safe_cuda(cudaMemcpyAsync(result.data(), counts.data(), out.size_bytes(),
                                cudaMemcpyDeviceToHost, ctx->CUDACtx()->Stream()));
  ctx->CUDACtx()->Stream().Sync();
  return result;
}
}  // namespace xgboost::cv

using namespace xgboost;  // NOLINT

XGB_DLL int XGBCvFoldAssignmentCreate(char const* c_ids, bst_ulong k_folds,
                                      FoldAssignmentHandle* out) {
  API_BEGIN();
  xgboost_CHECK_C_ARG_PTR(c_ids);
  xgboost_CHECK_C_ARG_PTR(out);
  auto ctx = Context{}.MakeCUDA(curt::CurrentDevice());
  auto json = Json::Load(StringView{c_ids});
  CHECK(get<Object const>(json).count("stream")) << "Fold IDs require the CUDA array interface.";
  ArrayInterface<1> array{json};
  CHECK(array.is_contiguous && array.type == ArrayInterfaceHandler::Type::kI8);
  CHECK_EQ(array.valid.Capacity(), 0) << "Missing fold IDs are not supported.";
  CHECK_EQ(dh::CudaGetPointerDevice(array.data), ctx.Ordinal());
  auto ids = common::Span<std::int64_t const>{static_cast<std::int64_t const*>(array.data),
                                              array.shape[0]};
  *out = new cv::FoldAssignmentPtr{std::make_shared<cv::FoldAssignment>(&ctx, k_folds, ids)};
  API_END();
}

XGB_DLL int XGBCvFoldAssignmentFree(FoldAssignmentHandle hdl) {
  API_BEGIN();
  delete static_cast<cv::FoldAssignmentPtr*>(hdl);
  API_END();
}
