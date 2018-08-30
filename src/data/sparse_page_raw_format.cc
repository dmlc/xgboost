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
    auto& offset_vec = page->offset.HostVector();
    if (!fi->Read(&offset_vec)) return false;
    auto& data_vec = page->data.HostVector();
    CHECK_NE(page->offset.Size(), 0U) << "Invalid SparsePage file";
    data_vec.resize(offset_vec.back());
    if (page->data.Size() != 0) {
      CHECK_EQ(fi->Read(dmlc::BeginPtr(data_vec),
                        (page->data).Size() * sizeof(Entry)),
               (page->data).Size() * sizeof(Entry))
          << "Invalid SparsePage file";
    }
    return true;
  }

  bool Read(SparsePage* page,
            dmlc::SeekStream* fi,
            const std::vector<bst_uint>& sorted_index_set) override {
    if (!fi->Read(&disk_offset_)) return false;
    auto& offset_vec = page->offset.HostVector();
    auto& data_vec = page->data.HostVector();
    // setup the offset
    offset_vec.clear();
    offset_vec.push_back(0);
    for (unsigned int fid : sorted_index_set) {
      CHECK_LT(fid + 1, disk_offset_.size());
      size_t size = disk_offset_[fid + 1] - disk_offset_[fid];
      offset_vec.push_back(offset_vec.back() + size);
    }
    data_vec.resize(offset_vec.back());
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
          size_to_read += offset_vec[j + 1] - offset_vec[j];
        } else {
          break;
        }
      }

      if (size_to_read != 0) {
        CHECK_EQ(fi->Read(dmlc::BeginPtr(data_vec) + offset_vec[i],
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
    const auto& offset_vec = page.offset.HostVector();
    const auto& data_vec = page.data.HostVector();
    CHECK(page.offset.Size() != 0 && offset_vec[0] == 0);
    CHECK_EQ(offset_vec.back(), page.data.Size());
    fo->Write(offset_vec);
    if (page.data.Size() != 0) {
      fo->Write(dmlc::BeginPtr(data_vec), page.data.Size() * sizeof(Entry));
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
