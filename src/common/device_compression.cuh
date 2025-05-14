/**
 * Copyright 2025, XGBoost contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <cstdint>  // for uint8_t

#include "compressed_iterator.h"    // for CompressedByteT
#include "cuda_dr_utils.h"          // for CUDA_HW_DECOM_AVAILABLE
#include "cuda_pinned_allocator.h"  // for HostPinnedMemPool
#include "device_compression.h"     // for CuMemParams
#include "device_vector.cuh"        // for DeviceUVector
#include "ref_resource_view.h"      // for RefResourceView
#include "xgboost/span.h"           // for Span

namespace xgboost::dc {
/**
 * @brief Use nvcomp to compress the data.
 *
 * @param ctx Context, provides the CUDA stream and execution policy.
 * @param in  Input buffer, data to be compressed
 * @param p_out Output buffer, storing comprssed data.
 * @param chunk_size The number of bytes for each chunk.
 */
[[nodiscard]] CuMemParams CompressSnappy(Context const* ctx,
                                         common::Span<common::CompressedByteT const> in,
                                         dh::DeviceUVector<std::uint8_t>* p_out,
                                         std::size_t chunk_size);
/**
 * @brief Run decompression with meta data cached in a mgr object.
 *
 * @param stream CUDA stream, it should be an asynchronous stream.
 * @param mgr Cache for decompression-related data.
 * @param out Pre-allocated output buffer based on the @ref CuMemParams returned from
 *   compression.
 * @param allow_fallback Allow fallback to nvcomp implementation if hardware accelerated
 *   implementation is not available. Used for testing.
 */
void DecompressSnappy(dh::CUDAStreamView stream, SnappyDecomprMgr const& mgr,
                      common::Span<common::CompressedByteT> out, bool allow_fallback);

/**
 * @brief Coalesce the compressed chunks into a contiguous host pinned buffer.
 *
 * @param stream CUDA stream.
 * @param in_params Params from @ref CompressSnappy, specifies the chunks.
 * @param in_buf The buffer storing compressed chunks.
 * @param p_out Re-newed parameters to keep track of the buffers.
 */
[[nodiscard]] common::RefResourceView<std::uint8_t> CoalesceCompressedBuffersToHost(
    dh::CUDAStreamView stream, CuMemParams const& in_params,
    dh::DeviceUVector<std::uint8_t> const& in_buf, CuMemParams* p_out);

// We store decompression parameters in struct of vectors. This is due to nvcomp works
// with this format. But the CUDA driver works with vector of structs. We can optimize
// toward the driver decompression function if the overhead is significant (too many
// chunks).
struct SnappyDecomprMgrImpl {
  // src of the CUmemDecompressParams
  dh::device_vector<void const*> d_in_chunk_ptrs;
  // srcNumBytes of the CUmemDecompressParams
  dh::device_vector<std::size_t> d_in_chunk_sizes;
  // dstNumBytes of the CUmemDecompressParams
  dh::device_vector<std::size_t> d_out_chunk_sizes;
  // dstActBytes of the CUmemDecompressParams
  dh::device_vector<std::size_t> act_nbytes;

#if defined(CUDA_HW_DECOM_AVAILABLE)
  using DeParams = common::RefResourceView<CUmemDecompressParams>;
  DeParams de_params;
  DeParams de_params_copy;
#endif  // defined(CUDA_HW_DECOM_AVAILABLE)

  [[nodiscard]] std::size_t Chunks() const {
#if defined(CUDA_HW_DECOM_AVAILABLE)
    return de_params.size();
#else
    LOG(FATAL) << "CUDA >= 12.8 is required.";
    return 0;
#endif  // defined(CUDA_HW_DECOM_AVAILABLE)
  }

  SnappyDecomprMgrImpl(dh::CUDAStreamView s,
                       std::shared_ptr<common::cuda_impl::HostPinnedMemPool> pool,
                       CuMemParams params, common::Span<std::uint8_t const> in_compressed_data);

#if defined(CUDA_HW_DECOM_AVAILABLE) && defined(XGBOOST_USE_NVCOMP)
  common::Span<CUmemDecompressParams> GetParams(common::Span<common::CompressedByteT> out);
#endif  // defined(CUDA_HW_DECOM_AVAILABLE)

  // big 5
  SnappyDecomprMgrImpl() = default;
  SnappyDecomprMgrImpl(SnappyDecomprMgrImpl const& that) = delete;
  SnappyDecomprMgrImpl(SnappyDecomprMgrImpl&& that) = default;
  SnappyDecomprMgrImpl& operator=(SnappyDecomprMgrImpl const&) = delete;
  SnappyDecomprMgrImpl& operator=(SnappyDecomprMgrImpl&&) = default;
};

inline auto MakeSnappyDecomprMgr(dh::CUDAStreamView s,
                                 std::shared_ptr<common::cuda_impl::HostPinnedMemPool> pool,
                                 CuMemParams params,
                                 common::Span<std::uint8_t const> in_compressed_data) {
  SnappyDecomprMgr mgr;
  *mgr.Impl() = SnappyDecomprMgrImpl{s, std::move(pool), std::move(params), in_compressed_data};
  return mgr;
}
}  // namespace xgboost::dc
