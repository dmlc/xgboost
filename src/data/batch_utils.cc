/**
 * Copyright 2023-2025, XGBoost Contributors
 */
#include "batch_utils.h"

#include <algorithm>  // for max
#include <cstddef>    // for size_t
#include <cstdint>    // for int64_t
#include <utility>    // for pair

#include "../common/common.h"         // for AssertGPUSupport
#include "../common/cuda_rt_utils.h"  // for TotalMemory
#include "../common/error_msg.h"      // for InconsistentMaxBin

#if defined(XGBOOST_USE_CUDA)

#include "../common/cuda_dr_utils.h"  // for GetC2cLinkCountFromSmiGlobal

#endif  // defined(XGBOOST_USE_CUDA)

namespace xgboost::data::detail {
void CheckParam(BatchParam const& init, BatchParam const& param) {
  CHECK_EQ(param.max_bin, init.max_bin) << error::InconsistentMaxBin();
  CHECK(!param.regen && param.hess.empty())
      << "Only the `hist` tree method can use the `QuantileDMatrix`.";
}

[[nodiscard]] std::pair<double, std::int64_t> DftPageSizeHostRatio(
    std::size_t n_cache_bytes, bool is_validation, double cache_host_ratio,
    std::int64_t min_cache_page_bytes) {
  common::AssertGPUSupport();

  if (!HostRatioIsAuto(cache_host_ratio)) {
    // Use user config.
    CHECK_GE(cache_host_ratio, 0.0f) << error::CacheHostRatioInvalid();
    CHECK_LE(cache_host_ratio, 1.0f) << error::CacheHostRatioInvalid();
  }

#if defined(XGBOOST_USE_CUDA)
  auto n_d_bytes = curt::TotalMemory();

  using xgboost::cuda_impl::CachePageRatio;

  auto lc = cudr::GetC2cLinkCountFromSmiGlobal();
  if (lc >= 10) {
    // >= 10, life is easy.
    if (CachePageBytesIsAuto(min_cache_page_bytes)) {
      min_cache_page_bytes = n_d_bytes * CachePageRatio();
    }
    if (HostRatioIsAuto(cache_host_ratio)) {
      cache_host_ratio = 1.0;
    }
    return {cache_host_ratio, min_cache_page_bytes};
  }

  /**
   * Configure the min_cache_page_bytes
   */
  // -1 if PCIe device, or something went wrong when running nvidia-smi
  //
  // GH200 1 CPU + 1 GPU has 10. For 1 CPU + 2 GPU, it's 5.
  //
  // Either way, we configure the cache based on the ratio between cache sizes and the
  // available memory.
  // Use half of the device memory for cache.
  auto d_cache_nbytes = n_d_bytes / 2;

  // Since half of the device is used for the cache, we have to use smaller page size.
  if (CachePageBytesIsAuto(min_cache_page_bytes)) {
    min_cache_page_bytes = n_d_bytes * (CachePageRatio() / 2.0);
  }

  /**
   * Configure the ratio.
   */
  if (!HostRatioIsAuto(cache_host_ratio)) {
    // Do nothing if it's provided by the user
    return {cache_host_ratio, min_cache_page_bytes};
  } else if (is_validation) {
    // Use full host cache for the validation dataset.
    cache_host_ratio = 1.0;
  } else if (n_cache_bytes <= d_cache_nbytes) {
    // The total size of the cache is smaller than the available device cache.
    cache_host_ratio = 0.0;
  } else {
    // The number of bytes that must be in the host memory.
    auto h_cache_nbytes = n_cache_bytes - d_cache_nbytes * 0.85;
    cache_host_ratio = static_cast<double>(h_cache_nbytes) / static_cast<double>(n_cache_bytes);
    if (lc > 0) {
      // 0 < lc < 10, C2C is available, but with reduced link count.
      // No need to exceed half in practice.
      cache_host_ratio = std::max(cache_host_ratio, 0.5);
    }
  }
#else
  (void)n_cache_bytes;
  (void)is_validation;
#endif  // defined(XGBOOST_USE_CUDA)
  return {cache_host_ratio, min_cache_page_bytes};
}
}  // namespace xgboost::data::detail
