/**
 * Copyright 2019-2024, XGBoost contributors
 */
#include <dmlc/registry.h>

#include <cstddef>  // for size_t
#include <cstdint>  // for uint64_t

#include "../common/io.h"                   // for AlignedResourceReadStream, AlignedFileWriteStream
#include "../common/ref_resource_view.cuh"  // for MakeFixedVecWithCudaMalloc
#include "../common/ref_resource_view.h"    // for ReadVec, WriteVec
#include "ellpack_page.cuh"                 // for EllpackPage
#include "ellpack_page_raw_format.h"
#include "ellpack_page_source.h"

namespace xgboost::data {
DMLC_REGISTRY_FILE_TAG(ellpack_page_raw_format);

namespace {
template <typename T>
[[nodiscard]] bool ReadDeviceVec(common::AlignedResourceReadStream* fi,
                                 common::RefResourceView<T>* vec) {
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

  Context ctx = Context{}.MakeCUDA(0);  // FIXME
  *vec = common::MakeFixedVecWithCudaMalloc(&ctx, n, static_cast<common::CompressedByteT>(0));
  auto d_vec = vec->data();
  dh::safe_cuda(cudaMemcpyAsync(d_vec, ptr, n_bytes, cudaMemcpyDefault, dh::DefaultStream()));
  return true;
}
}  // namespace

[[nodiscard]] bool EllpackPageRawFormat::Read(EllpackPage* page,
                                              common::AlignedResourceReadStream* fi) {
  auto* impl = page->Impl();
  impl->SetCuts(this->cuts_);
  if (!fi->Read(&impl->n_rows)) {
    return false;
  }
  if (!fi->Read(&impl->is_dense)) {
    return false;
  }
  if (!fi->Read(&impl->row_stride)) {
    return false;
  }
  if (!common::ReadVec(fi, &impl->gidx_buffer)) {
    return false;
  }
  if (!fi->Read(&impl->base_rowid)) {
    return false;
  }
  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(const EllpackPage& page,
                                                      common::AlignedFileWriteStream* fo) {
  std::size_t bytes{0};
  auto* impl = page.Impl();
  bytes += fo->Write(impl->n_rows);
  bytes += fo->Write(impl->is_dense);
  bytes += fo->Write(impl->row_stride);
  std::vector<common::CompressedByteT> h_gidx_buffer;
  Context ctx = Context{}.MakeCUDA(0);  // FIXME
  [[maybe_unused]] auto h_accessor = impl->GetHostAccessor(&ctx, &h_gidx_buffer);
  bytes += common::WriteVec(fo, h_gidx_buffer);
  bytes += fo->Write(impl->base_rowid);
  dh::DefaultStream().Sync();
  return bytes;
}

[[nodiscard]] bool EllpackPageRawFormat::Read(EllpackPage* page, EllpackHostCacheStream* fi) const {
  auto* impl = page->Impl();
  CHECK(this->cuts_->cut_values_.DeviceCanRead());
  impl->SetCuts(this->cuts_);
  if (!fi->Read(&impl->n_rows)) {
    return false;
  }
  if (!fi->Read(&impl->is_dense)) {
    return false;
  }
  if (!fi->Read(&impl->row_stride)) {
    return false;
  }

  // Read vec
  bst_idx_t n{0};
  if (!fi->Read(&n)) {
    return false;
  }
  Context ctx = Context{}.MakeCUDA(0);
  if (n != 0) {
    impl->gidx_buffer =
        common::MakeFixedVecWithCudaMalloc(&ctx, n, static_cast<common::CompressedByteT>(0));
    if (!fi->Read(impl->gidx_buffer.data(), impl->gidx_buffer.size_bytes())) {
      return false;
    }
  }

  if (!fi->Read(&impl->base_rowid)) {
    return false;
  }

  dh::DefaultStream().Sync();
  return true;
}

[[nodiscard]] std::size_t EllpackPageRawFormat::Write(const EllpackPage& page,
                                                      EllpackHostCacheStream* fo) const {
  bst_idx_t bytes{0};
  auto* impl = page.Impl();
  bytes += fo->Write(impl->n_rows);
  bytes += fo->Write(impl->is_dense);
  bytes += fo->Write(impl->row_stride);

  // Write vector
  bst_idx_t n = impl->gidx_buffer.size();
  bytes += fo->Write(n);

  if (!impl->gidx_buffer.empty()) {
    bytes += fo->Write(impl->gidx_buffer.data(), impl->gidx_buffer.size_bytes());
  }
  bytes += fo->Write(impl->base_rowid);

  dh::DefaultStream().Sync();
  return bytes;
}
}  // namespace xgboost::data
