/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <dmlc/registry.h>

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "../common/cuda_rt_utils.h"
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

  auto ctx = Context{}.MakeCUDA(common::CurrentDevice());
  *vec = common::MakeFixedVecWithCudaMalloc<T>(&ctx, n);
  dh::safe_cuda(cudaMemcpyAsync(vec->data(), ptr, n_bytes, cudaMemcpyDefault, dh::DefaultStream()));
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

  impl->SetCuts(this->cuts_);
  RET_IF_NOT(fi->Read(&impl->n_rows));
  RET_IF_NOT(fi->Read(&impl->is_dense));
  RET_IF_NOT(fi->Read(&impl->row_stride));

  if (has_hmm_ats_) {
    RET_IF_NOT(common::ReadVec(fi, &impl->gidx_buffer));
  } else {
    RET_IF_NOT(ReadDeviceVec(fi, &impl->gidx_buffer));
  }
  RET_IF_NOT(fi->Read(&impl->base_rowid));
  dh::DefaultStream().Sync();
  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(const EllpackPage& page,
                                                      common::AlignedFileWriteStream* fo) {
  xgboost_NVTX_FN_RANGE();

  std::size_t bytes{0};
  auto* impl = page.Impl();
  bytes += fo->Write(impl->n_rows);
  bytes += fo->Write(impl->is_dense);
  bytes += fo->Write(impl->row_stride);
  std::vector<common::CompressedByteT> h_gidx_buffer;
  Context ctx = Context{}.MakeCUDA(common::CurrentDevice());
  [[maybe_unused]] auto h_accessor = impl->GetHostAccessor(&ctx, &h_gidx_buffer);
  bytes += common::WriteVec(fo, h_gidx_buffer);
  bytes += fo->Write(impl->base_rowid);
  dh::DefaultStream().Sync();
  return bytes;
}

[[nodiscard]] bool EllpackPageRawFormat::Read(EllpackPage* page, EllpackHostCacheStream* fi) const {
  xgboost_NVTX_FN_RANGE();

  auto* impl = page->Impl();
  CHECK(this->cuts_->cut_values_.DeviceCanRead());
  impl->SetCuts(this->cuts_);

  // Read vector
  Context ctx = Context{}.MakeCUDA(common::CurrentDevice());
  auto read_vec = [&] {
    common::NvtxScopedRange range{common::NvtxEventAttr{"read-vec", common::NvtxRgb{127, 255, 0}}};
    bst_idx_t n{0};
    RET_IF_NOT(fi->Read(&n));
    if (n == 0) {
      return true;
    }
    impl->gidx_buffer = common::MakeFixedVecWithCudaMalloc<common::CompressedByteT>(&ctx, n);
    RET_IF_NOT(fi->Read(impl->gidx_buffer.data(), impl->gidx_buffer.size_bytes()));
    return true;
  };
  RET_IF_NOT(read_vec());

  RET_IF_NOT(fi->Read(&impl->n_rows));
  RET_IF_NOT(fi->Read(&impl->is_dense));
  RET_IF_NOT(fi->Read(&impl->row_stride));
  RET_IF_NOT(fi->Read(&impl->base_rowid));

  dh::DefaultStream().Sync();
  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(const EllpackPage& page,
                                                      EllpackHostCacheStream* fo) const {
  xgboost_NVTX_FN_RANGE();

  bst_idx_t bytes{0};
  auto* impl = page.Impl();

  // Write vector
  auto write_vec = [&] {
    common::NvtxScopedRange range{common::NvtxEventAttr{"write-vec", common::NvtxRgb{127, 255, 0}}};
    bst_idx_t n = impl->gidx_buffer.size();
    bytes += fo->Write(n);

    if (!impl->gidx_buffer.empty()) {
      bytes += fo->Write(impl->gidx_buffer.data(), impl->gidx_buffer.size_bytes());
    }
  };

  write_vec();

  bytes += fo->Write(impl->n_rows);
  bytes += fo->Write(impl->is_dense);
  bytes += fo->Write(impl->row_stride);
  bytes += fo->Write(impl->base_rowid);

  dh::DefaultStream().Sync();
  return bytes;
}

#undef RET_IF_NOT
}  // namespace xgboost::data
