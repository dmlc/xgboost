/**
 * Copyright 2019-2025, XGBoost contributors
 */
#include <dmlc/registry.h>

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../common/cuda_rt_utils.h"
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
[[nodiscard]] bool ReadDeviceVec(common::AlignedResourceReadStream* fi,
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
      cudaMemcpyAsync(vec->data(), ptr, n_bytes, cudaMemcpyDefault, curt::DefaultStream()));
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
    RET_IF_NOT(ReadDeviceVec(fi, &impl->gidx_buffer));
  } else {
    RET_IF_NOT(common::ReadVec(fi, &impl->gidx_buffer));
  }
  RET_IF_NOT(fi->Read(&impl->base_rowid));
  bst_idx_t n_symbols{0};
  RET_IF_NOT(fi->Read(&n_symbols));
  impl->SetNumSymbols(n_symbols);

  impl->SetCuts(this->cuts_);

  curt::DefaultStream().Sync();
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
  Context ctx = Context{}.MakeCUDA(curt::CurrentDevice());
  // write data into the h_gidx_buffer
  [[maybe_unused]] auto h_accessor = impl->GetHostEllpack(&ctx, &h_gidx_buffer);
  bytes += common::WriteVec(fo, h_gidx_buffer);
  bytes += fo->Write(impl->base_rowid);
  bytes += fo->Write(impl->NumSymbols());

  curt::DefaultStream().Sync();
  return bytes;
}

[[nodiscard]] bool EllpackPageRawFormat::Read(EllpackPage* page, EllpackHostCacheStream* fi) const {
  xgboost_NVTX_FN_RANGE_C(252, 198, 3);

  auto* impl = page->Impl();
  CHECK(this->cuts_->cut_values_.DeviceCanRead());

  auto ctx = Context{}.MakeCUDA(curt::CurrentDevice());

  auto dispatch = [&] {
    fi->Read(&ctx, page, this->param_.prefetch_copy || !this->has_hmm_ats_);
    impl->SetCuts(this->cuts_);
  };

  if (ConsoleLogger::GlobalVerbosity() == ConsoleLogger::LogVerbosity::kDebug) {
    curt::Event start{false}, stop{false};
    float milliseconds = 0;
    start.Record(ctx.CUDACtx()->Stream());

    dispatch();

    stop.Record(ctx.CUDACtx()->Stream());
    stop.Sync();
    dh::safe_cuda(cudaEventElapsedTime(&milliseconds, start, stop));
    double n_bytes = page->Impl()->MemCostBytes();
    double tp = (n_bytes / static_cast<double>((1ul << 30))) * 1000.0 / milliseconds;
    LOG(DEBUG) << "Ellpack " << __func__ << " throughput:" << tp << "GB/s";
  } else {
    dispatch();
  }

  curt::DefaultStream().Sync();

  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(EllpackPage const& page,
                                                      EllpackHostCacheStream* fo) const {
  xgboost_NVTX_FN_RANGE_C(3, 252, 198);

  bool new_page = fo->Write(page);
  curt::DefaultStream().Sync();

  if (new_page) {
    auto cache = fo->Share();
    return cache->SizeBytes(cache->Size() - 1);  // last page
  } else {
    return InvalidPageSize();
  }
}

#undef RET_IF_NOT
}  // namespace xgboost::data
