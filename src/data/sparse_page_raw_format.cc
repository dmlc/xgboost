/*!
 * Copyright (c) 2015 by Contributors
 * \file sparse_page_raw_format.cc
 *  Raw binary format of sparse page.
 */
#include <xgboost/data.h>
#include <dmlc/registry.h>
#include "./sparse_page_writer.h"

namespace xgboost {
namespace data {

DMLC_REGISTRY_FILE_TAG(sparse_page_raw_format);

class SparsePageRawFormat : public SparsePageFormat {
 public:
  bool Read(SparsePage* page, dmlc::SeekStream* fi) override {
    if (!fi->Read(&(page->offset))) return false;
    CHECK_NE(page->offset.size(), 0U) << "Invalid SparsePage file";
    page->data.resize(page->offset.back());
    if (page->data.size() != 0) {
      CHECK_EQ(fi->Read(dmlc::BeginPtr(page->data),
                        (page->data).size() * sizeof(Entry)),
               (page->data).size() * sizeof(Entry))
          << "Invalid SparsePage file";
    }
    return true;
  }

  bool Read(SparsePage* page,
            dmlc::SeekStream* fi,
            const std::vector<bst_uint>& sorted_index_set) override {
    if (!fi->Read(&disk_offset_)) return false;
    // setup the offset
    page->offset.clear();
    page->offset.push_back(0);
    for (unsigned int fid : sorted_index_set) {
      CHECK_LT(fid + 1, disk_offset_.size());
      size_t size = disk_offset_[fid + 1] - disk_offset_[fid];
      page->offset.push_back(page->offset.back() + size);
    }
    page->data.resize(page->offset.back());
    // read in the data
    size_t begin = fi->Tell();
    size_t curr_offset = 0;
    for (size_t i = 0; i < sorted_index_set.size();) {
      bst_uint fid = sorted_index_set[i];
      if (disk_offset_[fid] != curr_offset) {
        CHECK_GT(disk_offset_[fid], curr_offset);
        fi->Seek(begin + disk_offset_[fid] * sizeof(Entry));
        curr_offset = disk_offset_[fid];
      }
      size_t j, size_to_read = 0;
      for (j = i; j < sorted_index_set.size(); ++j) {
        if (disk_offset_[sorted_index_set[j]] == disk_offset_[fid] + size_to_read) {
          size_to_read += page->offset[j + 1] - page->offset[j];
        } else {
          break;
        }
      }

      if (size_to_read != 0) {
        CHECK_EQ(fi->Read(dmlc::BeginPtr(page->data) + page->offset[i],
                          size_to_read * sizeof(Entry)),
                 size_to_read * sizeof(Entry))
            << "Invalid SparsePage file";
        curr_offset += size_to_read;
      }
      i = j;
    }
    // seek to end of record
    if (curr_offset != disk_offset_.back()) {
      fi->Seek(begin + disk_offset_.back() * sizeof(Entry));
    }
    return true;
  }

  void Write(const SparsePage& page, dmlc::Stream* fo) override {
    CHECK(page.offset.size() != 0 && page.offset[0] == 0);
    CHECK_EQ(page.offset.back(), page.data.size());
    fo->Write(page.offset);
    if (page.data.size() != 0) {
      fo->Write(dmlc::BeginPtr(page.data), page.data.size() * sizeof(Entry));
    }
  }

 private:
  /*! \brief external memory column offset */
  std::vector<size_t> disk_offset_;
};

XGBOOST_REGISTER_SPARSE_PAGE_FORMAT(raw)
.describe("Raw binary data format.")
.set_body([]() {
    return new SparsePageRawFormat();
  });
}  // namespace data
}  // namespace xgboost
