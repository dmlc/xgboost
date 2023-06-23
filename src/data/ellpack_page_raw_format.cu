/**
 * Copyright 2019-2023, XGBoost contributors
 */
#include <dmlc/registry.h>

#include <cstddef>  // for size_t

#include "../common/io.h"                 // for AlignedResourceReadStream, AlignedFileWriteStream
#include "../common/ref_resource_view.h"  // for ReadVec, WriteVec
#include "ellpack_page.cuh"
#include "histogram_cut_format.h"  // for ReadHistogramCuts, WriteHistogramCuts
#include "sparse_page_writer.h"    // for SparsePageFormat

namespace xgboost::data {
DMLC_REGISTRY_FILE_TAG(ellpack_page_raw_format);

class EllpackPageRawFormat : public SparsePageFormat<EllpackPage> {
 public:
  bool Read(EllpackPage* page, common::AlignedResourceReadStream* fi) override {
    auto* impl = page->Impl();
    if (!ReadHistogramCuts(&impl->Cuts(), fi)) {
      return false;
    }
    if (!fi->Read(&impl->n_rows)) {
      return false;
    }
    if (!fi->Read(&impl->is_dense)) {
      return false;
    }
    if (!fi->Read(&impl->row_stride)) {
      return false;
    }
    if (!common::ReadVec(fi, &impl->gidx_buffer.HostVector())) {
      return false;
    }
    if (!fi->Read(&impl->base_rowid)) {
      return false;
    }
    return true;
  }

  size_t Write(const EllpackPage& page, common::AlignedFileWriteStream* fo) override {
    std::size_t bytes{0};
    auto* impl = page.Impl();
    bytes += WriteHistogramCuts(impl->Cuts(), fo);
    bytes += fo->Write(impl->n_rows);
    bytes += fo->Write(impl->is_dense);
    bytes += fo->Write(impl->row_stride);
    CHECK(!impl->gidx_buffer.ConstHostVector().empty());
    bytes += common::WriteVec(fo, impl->gidx_buffer.HostVector());
    bytes += fo->Write(impl->base_rowid);
    return bytes;
  }
};

XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(raw)
    .describe("Raw ELLPACK binary data format.")
    .set_body([]() { return new EllpackPageRawFormat(); });
}  // namespace xgboost::data
