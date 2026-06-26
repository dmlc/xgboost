/**
 * Copyright 2019-2026, XGBoost contributors
 */
#include <dmlc/registry.h>

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../common/cuda_context.cuh"       // for CUDAContext
#include "../common/cuda_stream.h"          // for Event
#include "../common/io.h"                   // for AlignedResourceReadStream, AlignedFileWriteStream
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/ref_resource_view.h"    // for ReadVec, WriteVec
#include "ellpack_page.cuh"                 // for EllpackPage
#include "ellpack_page_raw_format.h"
#include "ellpack_page_source.h"

namespace xgboost::data {
DMLC_REGISTRY_FILE_TAG(ellpack_page_raw_format);

namespace {
// Function to support system without HMM or ATS
template <typename T>
[[nodiscard]] bool ReadDeviceVec(Context const* ctx, common::AlignedResourceReadStream* fi,
                                 common::RefResourceView<T>* vec) {
  xgboost_NVTX_FN_RANGE();

  std::uint64_t n{0};
  if (!fi->Read(&n)) {
    return false;
  }
  if (n == 0) {
    return true;
  }

  auto expected_bytes = sizeof(T) * n;

  auto [ptr, n_bytes] = fi->Consume(expected_bytes);
  if (n_bytes != expected_bytes) {
    return false;
  }

  *vec = common::MakeFixedVecWithCudaMalloc<T>(n);
  dh::safe_cuda(
      cudaMemcpyAsync(vec->data(), ptr, n_bytes, cudaMemcpyDefault, ctx->CUDACtx()->Stream()));
  return true;
}
}  // namespace

#define RET_IF_NOT(expr) \
  if (!(expr)) {         \
    return false;        \
  }

[[nodiscard]] bool EllpackPageRawFormat::Read(EllpackPage* page,
                                              common::AlignedResourceReadStream* fi) {
  xgboost_NVTX_FN_RANGE();
  auto* impl = page->Impl();

  RET_IF_NOT(fi->Read(&impl->n_rows));
  RET_IF_NOT(fi->Read(&impl->is_dense));
  RET_IF_NOT(fi->Read(&impl->info.row_stride));

  if (this->param_.prefetch_copy || !has_hmm_ats_) {
    RET_IF_NOT(ReadDeviceVec(ctx_, fi, &impl->gidx_buffer));
  } else {
    RET_IF_NOT(common::ReadVec(fi, &impl->gidx_buffer));
  }
  RET_IF_NOT(fi->Read(&impl->base_rowid));
  bst_idx_t n_symbols{0};
  RET_IF_NOT(fi->Read(&n_symbols));
  impl->SetNumSymbols(n_symbols);

  impl->SetCuts(this->cuts_);

  ctx_->CUDACtx()->Stream().Sync();
  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(EllpackPage const& page,
                                                      common::AlignedFileWriteStream* fo) {
  xgboost_NVTX_FN_RANGE();

  std::size_t bytes{0};
  auto* impl = page.Impl();
  bytes += fo->Write(impl->n_rows);
  bytes += fo->Write(impl->is_dense);
  bytes += fo->Write(impl->info.row_stride);
  std::vector<common::CompressedByteT> h_gidx_buffer;
  // write data into the h_gidx_buffer
  [[maybe_unused]] auto h_accessor = impl->GetHostEllpack(ctx_, &h_gidx_buffer);
  bytes += common::WriteVec(fo, h_gidx_buffer);
  bytes += fo->Write(impl->base_rowid);
  bytes += fo->Write(impl->NumSymbols());

  ctx_->CUDACtx()->Stream().Sync();
  return bytes;
}

[[nodiscard]] bool EllpackPageRawFormat::Read(EllpackPage* page, EllpackHostCacheStream* fi) const {
  xgboost_NVTX_FN_RANGE_C(252, 198, 3);

  auto* impl = page->Impl();
  CHECK(this->cuts_->cut_values_.DeviceCanRead());

  auto stream = ctx_->CUDACtx()->Stream();

  auto dispatch = [&] {
    fi->Read(ctx_, page, this->param_.prefetch_copy || !this->has_hmm_ats_);
    impl->SetCuts(this->cuts_);
  };

  if (ConsoleLogger::GlobalVerbosity() == ConsoleLogger::LogVerbosity::kDebug) {
    curt::Event start{false}, stop{false};
    float milliseconds = 0;
    start.Record(stream);

    dispatch();

    stop.Record(stream);
    stop.Sync();
    dh::safe_cuda(cudaEventElapsedTime(&milliseconds, start, stop));
    double n_bytes = page->Impl()->MemCostBytes();
    double tp = (n_bytes / static_cast<double>((1ul << 30))) * 1000.0 / milliseconds;
    LOG(DEBUG) << "Ellpack " << __func__ << " throughput:" << tp << "GB/s";
  } else {
    dispatch();
  }

  stream.Sync();

  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(EllpackPage const& page,
                                                      EllpackHostCacheStream* fo) const {
  xgboost_NVTX_FN_RANGE_C(3, 252, 198);

  bool new_page = fo->Write(ctx_, page);
  ctx_->CUDACtx()->Stream().Sync();

  if (new_page) {
    auto cache = fo->Share();
    return cache->SizeBytes(cache->Size() - 1);  // last page
  } else {
    return InvalidPageSize();
  }
}

#undef RET_IF_NOT
}  // namespace xgboost::data
