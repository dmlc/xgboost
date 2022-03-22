/*!
 * Copyright 2019-2021 XGBoost contributors
 */
#include <xgboost/data.h>
#include <dmlc/registry.h>

#include "ellpack_page.cuh"
#include "sparse_page_writer.h"
#include "histogram_cut_format.h"

namespace xgboost {
namespace data {

DMLC_REGISTRY_FILE_TAG(ellpack_page_raw_format);


class EllpackPageRawFormat : public SparsePageFormat<EllpackPage> {
 public:
  bool Read(EllpackPage* page, dmlc::SeekStream* fi) override {
    auto* impl = page->Impl();
    if (!ReadHistogramCuts(&impl->Cuts(), fi)) {
      return false;
    }
    fi->Read(&impl->n_rows);
    fi->Read(&impl->is_dense);
    fi->Read(&impl->row_stride);
    fi->Read(&impl->gidx_buffer.HostVector());
    if (!fi->Read(&impl->base_rowid)) {
      return false;
    }
    return true;
  }

  size_t Write(const EllpackPage& page, dmlc::Stream* fo) override {
    size_t bytes = 0;
    auto* impl = page.Impl();
    bytes += WriteHistogramCuts(impl->Cuts(), fo);
    fo->Write(impl->n_rows);
    bytes += sizeof(impl->n_rows);
    fo->Write(impl->is_dense);
    bytes += sizeof(impl->is_dense);
    fo->Write(impl->row_stride);
    bytes += sizeof(impl->row_stride);
    CHECK(!impl->gidx_buffer.ConstHostVector().empty());
    fo->Write(impl->gidx_buffer.HostVector());
    bytes += impl->gidx_buffer.ConstHostSpan().size_bytes() + sizeof(uint64_t);
    fo->Write(impl->base_rowid);
    bytes += sizeof(impl->base_rowid);
    return bytes;
  }
};

XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(raw)
    .describe("Raw ELLPACK binary data format.")
    .set_body([]() {
      return new EllpackPageRawFormat();
    });

}  // namespace data
}  // namespace xgboost
