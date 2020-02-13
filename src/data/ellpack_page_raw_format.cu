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
    impl->Clear();
    if (!fi->Read(&impl->matrix.n_rows))  return false;
    return fi->Read(&impl->idx_buffer);
  }

  bool Read(EllpackPage* page,
            dmlc::SeekStream* fi,
            const std::vector<bst_uint>& sorted_index_set) override {
    auto* impl = page->Impl();
    impl->Clear();
    if (!fi->Read(&impl->matrix.n_rows))  return false;
    return fi->Read(&page->Impl()->idx_buffer);
  }

  void Write(const EllpackPage& page, dmlc::Stream* fo) override {
    auto* impl = page.Impl();
    fo->Write(impl->matrix.n_rows);
    auto buffer = impl->idx_buffer;
    CHECK(!buffer.empty());
    fo->Write(buffer);
  }
};

XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(raw)
    .describe("Raw ELLPACK binary data format.")
    .set_body([]() {
      return new EllpackPageRawFormat();
    });

}  // namespace data
}  // namespace xgboost
