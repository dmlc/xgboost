/*!
 * Copyright (c) 2015-2021 by Contributors
 * \file sparse_page_raw_format.cc
 *  Raw binary format of sparse page.
 */
#include <xgboost/data.h>
#include <dmlc/registry.h>

#include "xgboost/logging.h"
#include "./sparse_page_writer.h"

namespace xgboost {
namespace data {

DMLC_REGISTRY_FILE_TAG(sparse_page_raw_format);

template<typename T>
class SparsePageRawFormat : public SparsePageFormat<T> {
 public:
  bool Read(T* page, dmlc::SeekStream* fi) override {
    auto& offset_vec = page->offset.HostVector();
    if (!fi->Read(&offset_vec)) {
      return false;
    }
    auto& data_vec = page->data.HostVector();
    CHECK_NE(page->offset.Size(), 0U) << "Invalid SparsePage file";
    data_vec.resize(offset_vec.back());
    if (page->data.Size() != 0) {
      size_t n_bytes = fi->Read(dmlc::BeginPtr(data_vec),
                                (page->data).Size() * sizeof(Entry));
      CHECK_EQ(n_bytes, (page->data).Size() * sizeof(Entry))
          << "Invalid SparsePage file";
    }
    fi->Read(&page->base_rowid, sizeof(page->base_rowid));
    return true;
  }

  size_t Write(const T& page, dmlc::Stream* fo) override {
    const auto& offset_vec = page.offset.HostVector();
    const auto& data_vec = page.data.HostVector();
    CHECK(page.offset.Size() != 0 && offset_vec[0] == 0);
    CHECK_EQ(offset_vec.back(), page.data.Size());
    fo->Write(offset_vec);
    auto bytes = page.MemCostBytes();
    bytes += sizeof(uint64_t);
    if (page.data.Size() != 0) {
      fo->Write(dmlc::BeginPtr(data_vec), page.data.Size() * sizeof(Entry));
    }
    fo->Write(&page.base_rowid, sizeof(page.base_rowid));
    bytes += sizeof(page.base_rowid);
    return bytes;
  }

 private:
  /*! \brief external memory column offset */
  std::vector<size_t> disk_offset_;
};

XGBOOST_REGISTER_SPARSE_PAGE_FORMAT(raw)
.describe("Raw binary data format.")
.set_body([]() {
    return new SparsePageRawFormat<SparsePage>();
  });

XGBOOST_REGISTER_CSC_PAGE_FORMAT(raw)
.describe("Raw binary data format.")
.set_body([]() {
    return new SparsePageRawFormat<CSCPage>();
  });

XGBOOST_REGISTER_SORTED_CSC_PAGE_FORMAT(raw)
.describe("Raw binary data format.")
.set_body([]() {
    return new SparsePageRawFormat<SortedCSCPage>();
  });

}  // namespace data
}  // namespace xgboost
