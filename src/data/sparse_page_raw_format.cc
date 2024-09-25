/**
 * Copyright 2015-2023, XGBoost Contributors
 * \file sparse_page_raw_format.cc
 *  Raw binary format of sparse page.
 */
#include <dmlc/registry.h>

#include "../common/io.h"                 // for AlignedResourceReadStream, AlignedFileWriteStream
#include "../common/ref_resource_view.h"  // for WriteVec
#include "./sparse_page_writer.h"
#include "xgboost/data.h"
#include "xgboost/logging.h"

namespace xgboost::data {
DMLC_REGISTRY_FILE_TAG(sparse_page_raw_format);

template <typename T>
class SparsePageRawFormat : public SparsePageFormat<T> {
 public:
  bool Read(T* page, common::AlignedResourceReadStream* fi) override {
    auto& offset_vec = page->offset.HostVector();
    if (!common::ReadVec(fi, &offset_vec)) {
      return false;
    }
    auto& data_vec = page->data.HostVector();
    CHECK_NE(page->offset.Size(), 0U) << "Invalid SparsePage file";
    data_vec.resize(offset_vec.back());
    if (page->data.Size() != 0) {
      if (!common::ReadVec(fi, &data_vec)) {
        return false;
      }
    }
    if (!fi->Read(&page->base_rowid, sizeof(page->base_rowid))) {
      return false;
    }
    return true;
  }

  std::size_t Write(const T& page, common::AlignedFileWriteStream* fo) override {
    const auto& offset_vec = page.offset.HostVector();
    const auto& data_vec = page.data.HostVector();
    CHECK(page.offset.Size() != 0 && offset_vec[0] == 0);
    CHECK_EQ(offset_vec.back(), page.data.Size());

    std::size_t bytes{0};
    bytes += common::WriteVec(fo, offset_vec);
    if (page.data.Size() != 0) {
      bytes += common::WriteVec(fo, data_vec);
    }
    bytes += fo->Write(&page.base_rowid, sizeof(page.base_rowid));
    return bytes;
  }

 private:
};

#define SparsePageFmt SparsePageFormat<SparsePage>
DMLC_REGISTRY_REGISTER(SparsePageFormatReg<SparsePage>, SparsePageFmt, raw)
    .describe("Raw binary data format.")
    .set_body([]() { return new SparsePageRawFormat<SparsePage>(); });

#define CSCPageFmt SparsePageFormat<CSCPage>
DMLC_REGISTRY_REGISTER(SparsePageFormatReg<CSCPage>, CSCPageFmt, raw)
    .describe("Raw binary data format.")
    .set_body([]() { return new SparsePageRawFormat<CSCPage>(); });

#define SortedCSCPageFmt SparsePageFormat<SortedCSCPage>
DMLC_REGISTRY_REGISTER(SparsePageFormatReg<SortedCSCPage>, SortedCSCPageFmt, raw)
    .describe("Raw binary data format.")
    .set_body([]() { return new SparsePageRawFormat<SortedCSCPage>(); });
}  // namespace xgboost::data
