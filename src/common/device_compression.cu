/**
 * Copyright 2025, XGBoost contributors
 *
 * We use NVComp to perform compression and access the DE API directly for
 * decompression. Invoking the DE directly can help us avoid unnecessary kernal launches
 * and CUDA API calls and any potential blocking behaviours.
 */

#include <cstddef>  // for size_t
#include <cstdint>  // for uint8_t, uint32_t, int32_t
#include <memory>   // for shared_ptr

#include "device_compression.cuh"
#include "device_helpers.cuh"  // for CUDAStreamView, MemcpyBatchAsync
#include "xgboost/span.h"      // for Span

#if defined(XGBOOST_USE_NVCOMP)

#include <nvcomp/snappy.h>   // for nvcompBatchedSnappyDecompressAsync
#include <thrust/logical.h>  // for all_of
#include <thrust/reduce.h>   // for reduce

#include <algorithm>  // for transform, min
#include <cstring>    // for memset
#include <mutex>      // for once_flag, call_once
#include <vector>     // for vector

#include "common.h"               // for HumanMemUnit
#include "compressed_iterator.h"  // for CompressedByteT
#include "cuda_context.cuh"       // for CUDAContext
#include "cuda_dr_utils.h"        // for GetGlobalCuDriverApi
#include "cuda_rt_utils.h"        // for CurrentDevice
#include "device_compression.h"
#include "device_vector.cuh"      // for DeviceUVector
#include "nvtx_utils.h"           // for xgboost_NVTX_FN_RANGE
#include "ref_resource_view.cuh"  // for MakeFixedVecWithPinnedMemPool
#include "ref_resource_view.h"    // for RefResourceView

namespace xgboost::dc {
namespace {
// Parse snappy header
XGBOOST_DEVICE std::uint32_t GetUncompressedSize(std::uint8_t const* src, std::size_t src_bytes,
                                                 std::uint32_t* p_header_nbytes,
                                                 std::int32_t* p_status) {
  auto& n_bytes = *p_header_nbytes;
  n_bytes = 0;

  *p_status = 1;
  std::uint32_t uncompressed_size = src[n_bytes++];
  if (uncompressed_size > 0x7f) {
    std::uint32_t c = (n_bytes < src_bytes) ? src[n_bytes++] : 0;
    uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);
    if (uncompressed_size >= (0x80 << 7)) {
      c = (n_bytes < src_bytes) ? src[n_bytes++] : 0;
      uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
      if (uncompressed_size >= (0x80 << 14)) {
        c = (n_bytes < src_bytes) ? src[n_bytes++] : 0;
        uncompressed_size = (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
        if (uncompressed_size >= (0x80 << 21)) {
          c = (n_bytes < src_bytes) ? src[n_bytes++] : 0;
          if (c < 0x8) {
            uncompressed_size =
                (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) |
                (c << 28);
          } else {
            *p_status = 0;
          }
        }
      }
    }
  }

  return uncompressed_size;
}

void FillDecompParams(void const* const* d_in_chunk_ptrs, std::size_t const* d_in_chunk_nbytes,
                      common::Span<CUmemDecompressParams> de_params, size_t* d_act_nbytes,
                      std::size_t const* d_out_chunk_nbytes, std::int32_t* statuses,
                      dh::CUDAStreamView stream) {
  auto n_chunks = de_params.size();
  dh::LaunchN(n_chunks, stream,
              [d_in_chunk_ptrs, d_in_chunk_nbytes, d_out_chunk_nbytes, d_act_nbytes, de_params,
               statuses, n_chunks] XGBOOST_DEVICE(std::size_t ix_chunk) {
                std::size_t const dev_in_bytes = d_in_chunk_nbytes[ix_chunk];

                // Parse the input buffer to determine the number of bytes to skip
                // First byte with a 0 msb indicates no more bytes in the header
                auto cur = reinterpret_cast<std::uint8_t const*>(d_in_chunk_ptrs[ix_chunk]);
                std::uint32_t header_nbytes = 0;
                std::uint32_t uncompressed_size =
                    GetUncompressedSize(cur, dev_in_bytes, &header_nbytes, &statuses[ix_chunk]);
                if (statuses[ix_chunk] == 0) {
                  return;
                }

                de_params[ix_chunk].src = reinterpret_cast<const void*>(cur + header_nbytes);
                de_params[ix_chunk].dst = nullptr;  // not know yet
                de_params[ix_chunk].dstNumBytes = d_out_chunk_nbytes[ix_chunk];
                d_act_nbytes[ix_chunk] = 0;
                de_params[ix_chunk].dstActBytes =
                    reinterpret_cast<cuuint32_t*>(&d_act_nbytes[ix_chunk]);
                de_params[ix_chunk].srcNumBytes = dev_in_bytes - header_nbytes;
                de_params[ix_chunk].algo = CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY;
                statuses[ix_chunk] = 1;
              });
}

struct ChkOp {
  XGBOOST_DEVICE bool operator()(int s) { return s == 1; }
};

void CheckAlign(nvcompAlignmentRequirements_t alignment) {
  CHECK_EQ(alignment.input, 1);
  CHECK_EQ(alignment.output, 1);
  CHECK_EQ(alignment.temp, 1);
}

void SafeNvComp(nvcompStatus_t status) {
  if (status != nvcompSuccess) {
    LOG(FATAL) << "NVComp error:" << static_cast<std::int32_t>(status);
  }
}
}  // namespace

[[nodiscard]] DeStatus const& GetGlobalDeStatus() {
  std::once_flag static flag;
  DeStatus static de;
  std::call_once(flag, [&] {
    // First check driver, we don't need to worry about mismatched libcuda version and rm
    // version here. The first DE-enabled GPU requires >= 12.8 to work.
    std::int32_t driver_version = 0;
    dh::safe_cuda(cudaDriverGetVersion(&driver_version));
    if (driver_version < 12080) {
      return;
    }

    // Then check HW
    auto device = curt::CurrentDevice();
    std::int32_t mask = 0;
    safe_cu(cudr::GetGlobalCuDriverApi().cuDeviceGetAttribute(
        &mask, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK, device));
    de.avail = static_cast<bool>(mask);
    if (!de.avail) {
      return;
    }

    std::int32_t max_supported_size = 0;
    // this refers to the output length of the decomp
    safe_cu(cudr::GetGlobalCuDriverApi().cuDeviceGetAttribute(
        &max_supported_size, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH, device));
    de.max_output_size = static_cast<std::size_t>(max_supported_size);
    LOG(INFO) << "The maximum supported size of the DE:" << max_supported_size << std::endl;
  });

  return de;
}

SnappyDecomprMgrImpl::SnappyDecomprMgrImpl(dh::CUDAStreamView s,
                                           std::shared_ptr<HostPinnedMemPool> pool,
                                           CuMemParams params,
                                           common::Span<std::uint8_t const> in_compressed_data)
    : n_dst_bytes{params.TotalDstBytes()} {
  std::size_t n_chunks = params.size();
  if (n_chunks == 0) {
    return;
  }

  std::size_t last_in = 0, last_out = 0;

  std::vector<void const*> in_chunk_ptrs(n_chunks);
  std::vector<std::size_t> in_chunk_sizes(n_chunks);
  std::vector<std::size_t> out_chunk_sizes(n_chunks);

  dh::DeviceUVector<std::int32_t> status(n_chunks);
  for (std::size_t i = 0; i < n_chunks; ++i) {
    in_chunk_ptrs[i] = in_compressed_data.subspan(last_in, params[i].src_act_nbytes).data();
    in_chunk_sizes[i] = params[i].src_act_nbytes;
    out_chunk_sizes[i] = params[i].dst_nbytes;

    last_in += params[i].src_nbytes;
    last_out += params[i].dst_nbytes;
  }
  CHECK_EQ(this->n_dst_bytes, last_out);

  // copy to d
  dh::CopyTo(in_chunk_ptrs, &this->d_in_chunk_ptrs, s);
  dh::CopyTo(in_chunk_sizes, &this->d_in_chunk_sizes, s);
  dh::CopyTo(out_chunk_sizes, &this->d_out_chunk_sizes, s);
  this->act_nbytes.resize(n_chunks, 0);

  this->de_params = common::MakeFixedVecWithPinnedMemPool<decltype(this->de_params)::value_type>(
      pool, n_chunks, s);
  for (std::size_t i = 0; i < n_chunks; ++i) {
    std::memset(this->de_params.data() + i, 0, sizeof(CUmemDecompressParams));
  }

  FillDecompParams(d_in_chunk_ptrs.data().get(), d_in_chunk_sizes.data().get(), de_params.ToSpan(),
                   this->act_nbytes.data().get(), d_out_chunk_sizes.data().get(), status.data(), s);
  dh::XGBCachingDeviceAllocator<char> alloc;
  bool valid = thrust::all_of(thrust::cuda::par_nosync(alloc).on(s), status.cbegin(), status.cend(),
                              ChkOp{});
  CHECK(valid);

  auto max_supported_size = GetGlobalDeStatus().max_output_size;
  auto max_chunk_size = *std::max_element(out_chunk_sizes.cbegin(), out_chunk_sizes.cend());
  if (GetGlobalDeStatus().avail) {
    CHECK_GE(max_supported_size, max_chunk_size);
  }

  this->de_params_copy =
      common::MakeFixedVecWithPinnedMemPool<decltype(this->de_params)::value_type>(pool, n_chunks,
                                                                                   s);
}

common::Span<CUmemDecompressParams> SnappyDecomprMgrImpl::GetParams(
    common::Span<common::CompressedByteT> out) {
  xgboost_NVTX_FN_RANGE_C(3, 252, 198);
  if (this->de_params.empty()) {
    return {};
  }
  auto n_chunks = this->de_params.size();
  CHECK(!this->de_params_copy.empty());
  // Set the output buffers.
  std::size_t last_out = 0;
  for (std::size_t i = 0; i < n_chunks; ++i) {
    this->de_params_copy[i] = this->de_params[i];
    this->de_params_copy[i].dst = out.subspan(last_out, de_params[i].dstNumBytes).data();
    last_out += de_params[i].dstNumBytes;
  }

  return this->de_params_copy.ToSpan();
}

[[nodiscard]] bool SnappyDecomprMgrImpl::Empty() const {
#if defined(CUDA_HW_DECOM_AVAILABLE)
  return this->de_params.empty();
#else
  return true;
#endif
}

SnappyDecomprMgr::SnappyDecomprMgr() : pimpl_{std::make_unique<SnappyDecomprMgrImpl>()} {}
SnappyDecomprMgr::SnappyDecomprMgr(SnappyDecomprMgr&& that) = default;
SnappyDecomprMgr& SnappyDecomprMgr::operator=(SnappyDecomprMgr&& that) = default;

SnappyDecomprMgr::~SnappyDecomprMgr() = default;

[[nodiscard]] bool SnappyDecomprMgr::Empty() const { return this->Impl()->Empty(); }

[[nodiscard]] std::size_t SnappyDecomprMgr::DecompressedBytes() const {
  return this->Impl()->n_dst_bytes;
}

SnappyDecomprMgrImpl* SnappyDecomprMgr::Impl() const { return this->pimpl_.get(); }

void DecompressSnappy(dh::CUDAStreamView stream, SnappyDecomprMgr const& mgr,
                      common::Span<common::CompressedByteT> out, bool allow_fallback) {
  xgboost_NVTX_FN_RANGE();
  auto mgr_impl = mgr.Impl();
  auto params = mgr_impl->GetParams(out);
  if (params.empty()) {
    CHECK(out.empty());
    return;
  }
  if (GetGlobalDeStatus().avail &&
      cudr::GetGlobalCuDriverApi().cuMemBatchDecompressAsync != nullptr) {
    // Invoke the DE.
#if defined(CUDA_HW_DECOM_AVAILABLE)
    std::size_t error_index;
    safe_cu(cudr::GetGlobalCuDriverApi().cuMemBatchDecompressAsync(
        params.data(), params.size(), 0 /*unused*/, &error_index, stream));
#else
    static_assert(false, "`cuMemBatchDecompressAsync` requires CUDA >= 12.8.")
#endif  // defined(CUDA_HW_DECOM_AVAILABLE)
  } else {
    // Fallback to nvcomp. This is only used during tests where we don't have access to DE
    // but still want the test coverage.
    CHECK(allow_fallback);
    CheckAlign(nvcompBatchedSnappyDecompressRequiredAlignments);
    auto n_chunks = mgr_impl->Chunks();
    // Get sketch space
    std::size_t n_tmp_bytes = 0;
    SafeNvComp(nvcompBatchedSnappyDecompressGetTempSize(n_chunks, /*unused*/ 0, &n_tmp_bytes));
    dh::device_vector<char> tmp(n_tmp_bytes, 0);

    dh::device_vector<nvcompStatus_t> status(n_chunks, nvcompSuccess);

    // Build output vector
    std::vector<void*> h_out_ptrs(n_chunks);
    std::transform(params.cbegin(), params.cend(), h_out_ptrs.begin(),
                   [](auto const& p) { return p.dst; });
    dh::device_vector<void*> d_out_ptrs(n_chunks);
    dh::safe_cuda(cudaMemcpyAsync(d_out_ptrs.data().get(), h_out_ptrs.data(),
                                  dh::ToSpan(d_out_ptrs).size_bytes(), cudaMemcpyDefault, stream));
    // Run nvcomp
    SafeNvComp(nvcompBatchedSnappyDecompressAsync(
        mgr_impl->d_in_chunk_ptrs.data().get(), mgr_impl->d_in_chunk_sizes.data().get(),
        mgr_impl->d_out_chunk_sizes.data().get(), mgr_impl->act_nbytes.data().get(), n_chunks,
        tmp.data().get(), n_tmp_bytes, d_out_ptrs.data().get(), status.data().get(), stream));
  }
}

[[nodiscard]] CuMemParams CompressSnappy(Context const* ctx,
                                         common::Span<common::CompressedByteT const> in,
                                         dh::DeviceUVector<std::uint8_t>* p_out,
                                         std::size_t chunk_size) {
  CHECK_GT(chunk_size, 0);
  auto cuctx = ctx->CUDACtx();
  auto nvcomp_batched_snappy_opts = nvcompBatchedSnappyDefaultOpts;

  nvcompAlignmentRequirements_t compression_alignment_reqs;
  SafeNvComp(nvcompBatchedSnappyCompressGetRequiredAlignments(nvcomp_batched_snappy_opts,
                                                              &compression_alignment_reqs));
  CheckAlign(compression_alignment_reqs);

  /**
   * Inputs
   */
  std::size_t n_chunks = (in.size() + chunk_size - 1) / chunk_size;
  if (n_chunks == 0) {
    p_out->clear();
    return {};
  }
  std::size_t last = 0;

  std::vector<common::CompressedByteT const*> h_in_ptrs(n_chunks);
  std::vector<std::size_t> h_in_sizes(n_chunks);
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto n = std::min(chunk_size, in.size() - last);
    auto chunk = in.subspan(last, n);
    last += n;

    h_in_sizes[i] = chunk.size();
    h_in_ptrs[i] = chunk.data();
  }
  CHECK_EQ(last, in.size());

  dh::DeviceUVector<void const*> in_ptrs(h_in_ptrs.size());
  dh::safe_cuda(cudaMemcpyAsync(in_ptrs.data(), h_in_ptrs.data(),
                                common::Span{h_in_ptrs}.size_bytes(), cudaMemcpyDefault,
                                cuctx->Stream()));
  dh::DeviceUVector<std::size_t> in_sizes(h_in_sizes.size());
  dh::safe_cuda(cudaMemcpyAsync(in_sizes.data(), h_in_sizes.data(),
                                common::Span{h_in_sizes}.size_bytes(), cudaMemcpyDefault,
                                cuctx->Stream()));

  CHECK_EQ(n_chunks, in_sizes.size());
  std::size_t max_in_nbytes = *std::max_element(h_in_sizes.cbegin(), h_in_sizes.cend());

  /**
   * Outputs
   */
  std::size_t comp_temp_bytes;
  SafeNvComp(nvcompBatchedSnappyCompressGetTempSize(n_chunks, chunk_size,
                                                    nvcomp_batched_snappy_opts, &comp_temp_bytes));
  CHECK_EQ(comp_temp_bytes, 0);
  dh::DeviceUVector<char> comp_tmp(comp_temp_bytes);

  std::size_t max_out_nbytes = 0;
  SafeNvComp(nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
      std::min(max_in_nbytes, chunk_size), nvcomp_batched_snappy_opts, &max_out_nbytes));
  p_out->resize(max_out_nbytes * n_chunks);
  std::vector<void*> h_out_ptrs(n_chunks);
  std::vector<std::size_t> h_out_sizes(n_chunks);
  auto s_out = dh::ToSpan(*p_out);
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto chunk = s_out.subspan(max_out_nbytes * i, max_out_nbytes);
    h_out_ptrs[i] = chunk.data();
    h_out_sizes[i] = chunk.size();
  }
  dh::DeviceUVector<void*> out_ptrs(h_out_ptrs.size());
  dh::safe_cuda(cudaMemcpyAsync(out_ptrs.data(), h_out_ptrs.data(),
                                common::Span{h_out_ptrs}.size_bytes(), cudaMemcpyDefault));
  dh::DeviceUVector<std::size_t> out_sizes(h_out_sizes.size());
  dh::safe_cuda(cudaMemcpyAsync(out_sizes.data(), h_out_sizes.data(),
                                common::Span{h_out_sizes}.size_bytes(), cudaMemcpyDefault));

  /**
   * Compress
   */
  SafeNvComp(nvcompBatchedSnappyCompressAsync(
      in_ptrs.data(), in_sizes.data(), max_in_nbytes, n_chunks, comp_tmp.data(), comp_temp_bytes,
      out_ptrs.data(), out_sizes.data(), nvcomp_batched_snappy_opts, cuctx->Stream()));
  auto n_bytes = thrust::reduce(cuctx->CTP(), out_sizes.cbegin(), out_sizes.cend());
  auto n_total_bytes = p_out->size();
  auto ratio = static_cast<double>(n_total_bytes) / in.size_bytes();
  auto ratio_act = static_cast<double>(n_bytes) / in.size_bytes();
  LOG(DEBUG) << "[snappy] Input: " << common::HumanMemUnit(in.size_bytes())
             << ", need:" << common::HumanMemUnit(n_bytes)
             << ", allocated:" << common::HumanMemUnit(n_total_bytes) << ", ratio:" << ratio
             << ", actual ratio:" << ratio_act;

  /**
   * Meta
   */
  CuMemParams params(n_chunks);
  std::vector<std::size_t> h_act_nbytes(out_sizes.size());
  dh::safe_cuda(cudaMemcpyAsync(h_act_nbytes.data(), out_sizes.data(),
                                common::Span{h_out_sizes}.size_bytes(), cudaMemcpyDefault,
                                cuctx->Stream()));
  for (std::size_t i = 0; i < n_chunks; ++i) {
    auto& p = params[i];
    p.src_nbytes = h_out_sizes[i];
    p.src_act_nbytes = h_act_nbytes[i];
    p.dst_nbytes = h_in_sizes[i];
    p.algo = ComprParam::kSnappy;
  }
  return params;
}

[[nodiscard]] common::RefResourceView<std::uint8_t> CoalesceCompressedBuffersToHost(
    dh::CUDAStreamView stream, std::shared_ptr<HostPinnedMemPool> pool,
    CuMemParams const& in_params, dh::DeviceUVector<std::uint8_t> const& in_buf,
    CuMemParams* p_out) {
  std::size_t n_total_act_bytes = in_params.TotalSrcActBytes();
  std::size_t n_total_bytes = in_params.TotalSrcBytes();
  if (n_total_bytes == 0) {
    CHECK_EQ(n_total_act_bytes, 0);
    p_out->resize(0);
    return {};
  }
  // copy from device buffer to the host cache.
  CHECK_EQ(n_total_bytes, in_buf.size());
  CHECK(pool);
  auto c_page =
      common::MakeFixedVecWithPinnedMemPool<std::remove_reference_t<decltype(in_buf)>::value_type>(
          pool, n_total_act_bytes, stream);
  std::vector<std::uint8_t const*> srcs(in_params.size());
  std::vector<std::uint8_t*> dsts(in_params.size());
  std::vector<std::size_t> sizes(in_params.size());

  decltype(srcs)::value_type sptr = in_buf.data();
  decltype(dsts)::value_type dptr = c_page.data();

  for (std::size_t i = 0; i < in_params.size(); ++i) {
    CHECK_LE(in_params[i].src_act_nbytes, in_params[i].src_nbytes);
    sizes[i] = in_params[i].src_act_nbytes;

    srcs[i] = sptr;
    dsts[i] = dptr;

    sptr += in_params[i].src_nbytes;
    dptr += in_params[i].src_act_nbytes;
  }
  std::size_t fail_idx = 0;
  dh::safe_cuda(dh::MemcpyBatchAsync<cudaMemcpyDeviceToHost>(dsts.data(), srcs.data(), sizes.data(),
                                                             in_params.size(), &fail_idx, stream));

  auto& out_params = *p_out;
  out_params.resize(in_params.size());
  for (std::size_t i = 0; i < in_params.size(); ++i) {
    out_params[i].algo = in_params[i].algo;
    out_params[i].dst_nbytes = in_params[i].dst_nbytes;
    out_params[i].src_nbytes = in_params[i].src_act_nbytes;  // change to act
    out_params[i].src_act_nbytes = in_params[i].src_act_nbytes;
  }
  return c_page;
}
}  // namespace xgboost::dc

#else

namespace xgboost::dc {
// Impl
SnappyDecomprMgrImpl::SnappyDecomprMgrImpl(dh::CUDAStreamView,
                                           std::shared_ptr<common::cuda_impl::HostPinnedMemPool>,
                                           CuMemParams,
                                           common::Span<common::CompressedByteT const>) {}

// SnappyDecomprMgr
SnappyDecomprMgr::SnappyDecomprMgr() = default;
SnappyDecomprMgr::SnappyDecomprMgr(SnappyDecomprMgr&& that) = default;
SnappyDecomprMgr& SnappyDecomprMgr::operator=(SnappyDecomprMgr&& that) = default;
SnappyDecomprMgr::~SnappyDecomprMgr() = default;
SnappyDecomprMgrImpl* SnappyDecomprMgr::Impl() const { return nullptr; }

[[nodiscard]] bool SnappyDecomprMgr::Empty() const { return true; }
[[nodiscard]] std::size_t SnappyDecomprMgr::DecompressedBytes() const { return 0; }

// Round-trip compression
void DecompressSnappy(dh::CUDAStreamView, SnappyDecomprMgr const&,
                      common::Span<common::CompressedByteT>, bool) {
  common::AssertNvCompSupport();
}

[[nodiscard]] CuMemParams CompressSnappy(Context const*,
                                         common::Span<common::CompressedByteT const> in,
                                         dh::DeviceUVector<std::uint8_t>*, std::size_t) {
  if (in.empty()) {
    return {};
  }
  common::AssertNvCompSupport();
  return {};
}

[[nodiscard]] common::RefResourceView<std::uint8_t> CoalesceCompressedBuffersToHost(
    dh::CUDAStreamView, std::shared_ptr<HostPinnedMemPool>, CuMemParams const& in_params,
    dh::DeviceUVector<std::uint8_t> const&, CuMemParams*) {
  std::size_t n_total_bytes = in_params.TotalSrcBytes();
  if (n_total_bytes == 0) {
    return {};
  }
  common::AssertNvCompSupport();
  return {};
}

[[nodiscard]] DeStatus const& GetGlobalDeStatus() {
  static thread_local DeStatus de;
  return de;
}
}  // namespace xgboost::dc

#endif  // defined(XGBOOST_USE_NVCOMP)
