/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <cuda/std/variant>  // for visit

#include <cstdint>  // for uint64_t

#include "../common/cuda_context.cuh"    // for CUDAContext
#include "../common/device_helpers.cuh"  // for LaunchN, TemporaryArray
#include "../encoder/ordinal.h"          // for DeviceColumnsView
#include "cat_container_hash.h"          // for CatContentDigest, k* constants

namespace xgboost::data {
/**
 * @brief Device sibling of @ref HashCatHostContent.
 *
 * Single-thread kernel because the byte fold is sequential; only the 16-byte digest
 * crosses PCIe. Mixed CPU/GPU clusters compare digests directly only when host and device
 * column views expose byte-identical offset and value buffers per feature.
 *
 * @param ctx CUDA context whose stream owns the kernel + memcpy.
 * @param view Device columns view; offsets and values bytes are folded per column.
 * @return Dual-component digest seeded with `n_total_cats` and `columns.size()`.
 */
[[nodiscard]] inline CatContentDigest HashCatDeviceContent(
    Context const* ctx, enc::DeviceColumnsView const& view) {
  dh::TemporaryArray<CatContentDigest> d_digest(1);
  auto* p_digest = d_digest.data().get();
  auto columns = view.columns;
  auto n_total_cats = view.n_total_cats;
  dh::LaunchN(1, ctx->CUDACtx()->Stream(), [=] __device__(std::size_t) {
    std::uint64_t h1 = kHashSeedPrimary ^ static_cast<std::uint64_t>(n_total_cats);
    std::uint64_t h2 = kHashSeedSecondary ^ static_cast<std::uint64_t>(columns.size());
    auto fold = [&](void const* p, std::size_t n) {
      auto const* b = static_cast<std::uint8_t const*>(p);
      for (std::size_t i = 0; i < n; ++i) {
        h1 = (h1 ^ b[i]) * kHashPrimePrimary;
        h2 = (h2 ^ b[i]) * kHashPrimeSecondary;
      }
    };
    for (auto const& column : columns) {
      cuda::std::visit(
          enc::Overloaded{
              [&](enc::CatStrArrayView const& s) {
                fold(s.offsets.data(), s.offsets.size_bytes());
                fold(s.values.data(), s.values.size_bytes());
              },
              [&](auto const& idx) { fold(idx.data(), idx.size_bytes()); }},
          column);
    }
    *p_digest = CatContentDigest{h1, h2};
  });
  CatContentDigest result{};
  dh::safe_cuda(cudaMemcpyAsync(&result, p_digest, sizeof(result), cudaMemcpyDeviceToHost,
                                ctx->CUDACtx()->Stream()));
  ctx->CUDACtx()->Stream().Sync();
  return result;
}
}  // namespace xgboost::data
