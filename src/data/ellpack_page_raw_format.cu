/*!
 * Copyright 2019 XGBoost contributors
 */

#include <xgboost/data.h>
#include <dmlc/registry.h>

#include "./ellpack_page.cuh"
#include "./sparse_page_writer.h"

namespace xgboost {
namespace data {

DMLC_REGISTRY_FILE_TAG(ellpack_page_raw_format);

class EllpackPageRawFormat : public SparsePageFormat<EllpackPage> {
 public:
  bool Read(EllpackPage* page, dmlc::SeekStream* fi) override {
    auto* impl = page->Impl();
    fi->Read(&impl->Cuts().cut_values_.HostVector());
    fi->Read(&impl->Cuts().cut_ptrs_.HostVector());
    fi->Read(&impl->Cuts().min_vals_.HostVector());
    fi->Read(&impl->n_rows);
    fi->Read(&impl->is_dense);
    fi->Read(&impl->row_stride);
    if (!fi->Read(&impl->gidx_buffer.HostVector())) {
      return false;
    }
    return true;
  }

  bool Read(EllpackPage* page,
            dmlc::SeekStream* fi,
            const std::vector<bst_uint>& sorted_index_set) override {
    LOG(FATAL) << "Not implemented";
    return false;
  }

  void Write(const EllpackPage& page, dmlc::Stream* fo) override {
    auto* impl = page.Impl();
    fo->Write(impl->Cuts().cut_values_.ConstHostVector());
    fo->Write(impl->Cuts().cut_ptrs_.ConstHostVector());
    fo->Write(impl->Cuts().min_vals_.ConstHostVector());
    fo->Write(impl->n_rows);
    fo->Write(impl->is_dense);
    fo->Write(impl->row_stride);
    CHECK(!impl->gidx_buffer.ConstHostVector().empty());
    fo->Write(impl->gidx_buffer.HostVector());
  }
};

XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(raw)
    .describe("Raw ELLPACK binary data format.")
    .set_body([]() {
      return new EllpackPageRawFormat();
    });

}  // namespace data
}  // namespace xgboost
