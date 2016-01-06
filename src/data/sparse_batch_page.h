/*!
 * Copyright (c) 2014 by Contributors
 * \file sparse_batch_page.h
 *   content holder of sparse batch that can be saved to disk
 *   the representation can be effectively
 *   use in external memory computation
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_BATCH_PAGE_H_
#define XGBOOST_DATA_SPARSE_BATCH_PAGE_H_

#include <xgboost/data.h>
#include <dmlc/io.h>
#include <vector>
#include <algorithm>

namespace xgboost {
namespace data {
/*!
 * \brief in-memory storage unit of sparse batch
 */
class SparsePage {
 public:
  /*! \brief offset of the segments */
  std::vector<size_t> offset;
  /*! \brief the data of the segments */
  std::vector<SparseBatch::Entry> data;

  /*! \brief constructor */
  SparsePage() {
    this->Clear();
  }
  /*! \return number of instance in the page */
  inline size_t Size() const {
    return offset.size() - 1;
  }
  /*!
   * \brief load only the segments we are interested in
   * \param fi the input stream of the file
   * \param sorted_index_set sorted index of segments we are interested in
   * \return true of the loading as successful, false if end of file was reached
   */
  inline bool Load(dmlc::SeekStream *fi,
                   const std::vector<bst_uint> &sorted_index_set) {
    if (!fi->Read(&disk_offset_)) return false;
    // setup the offset
    offset.clear(); offset.push_back(0);
    for (size_t i = 0; i < sorted_index_set.size(); ++i) {
      bst_uint fid = sorted_index_set[i];
      CHECK_LT(fid + 1, disk_offset_.size());
      size_t size = disk_offset_[fid + 1] - disk_offset_[fid];
      offset.push_back(offset.back() + size);
    }
    data.resize(offset.back());
    // read in the data
    size_t begin = fi->Tell();
    size_t curr_offset = 0;
    for (size_t i = 0; i < sorted_index_set.size();) {
      bst_uint fid = sorted_index_set[i];
      if (disk_offset_[fid] != curr_offset) {
        CHECK_GT(disk_offset_[fid], curr_offset);
        fi->Seek(begin + disk_offset_[fid] * sizeof(SparseBatch::Entry));
        curr_offset = disk_offset_[fid];
      }
      size_t j, size_to_read = 0;
      for (j = i; j < sorted_index_set.size(); ++j) {
        if (disk_offset_[sorted_index_set[j]] == disk_offset_[fid] + size_to_read) {
          size_to_read += offset[j + 1] - offset[j];
        } else {
          break;
        }
      }

      if (size_to_read != 0) {
        CHECK_EQ(fi->Read(dmlc::BeginPtr(data) + offset[i],
                          size_to_read * sizeof(SparseBatch::Entry)),
                 size_to_read * sizeof(SparseBatch::Entry))
            << "Invalid SparsePage file";
        curr_offset += size_to_read;
      }
      i = j;
    }
    // seek to end of record
    if (curr_offset != disk_offset_.back()) {
      fi->Seek(begin + disk_offset_.back() * sizeof(SparseBatch::Entry));
    }
    return true;
  }
  /*!
   * \brief load all the segments
   * \param fi the input stream of the file
   * \return true of the loading as successful, false if end of file was reached
   */
  inline bool Load(dmlc::Stream *fi) {
    if (!fi->Read(&offset)) return false;
    CHECK_NE(offset.size(), 0) << "Invalid SparsePage file";
    data.resize(offset.back());
    if (data.size() != 0) {
      CHECK_EQ(fi->Read(dmlc::BeginPtr(data), data.size() * sizeof(SparseBatch::Entry)),
               data.size() * sizeof(SparseBatch::Entry))
          << "Invalid SparsePage file";
    }
    return true;
  }
  /*!
   * \brief save the data to fo, when a page was written
   *    to disk it must contain all the elements in the
   * \param fo output stream
   */
  inline void Save(dmlc::Stream *fo) const {
    CHECK(offset.size() != 0 && offset[0] == 0);
    CHECK_EQ(offset.back(), data.size());
    fo->Write(offset);
    if (data.size() != 0) {
      fo->Write(dmlc::BeginPtr(data), data.size() * sizeof(SparseBatch::Entry));
    }
  }
  /*! \return estimation of memory cost of this page */
  inline size_t MemCostBytes(void) const {
    return offset.size() * sizeof(size_t) + data.size() * sizeof(SparseBatch::Entry);
  }
  /*! \brief clear the page */
  inline void Clear(void) {
    offset.clear();
    offset.push_back(0);
    data.clear();
  }
  /*!
   * \brief load all the segments and add it to existing batch
   * \param fi the input stream of the file
   * \return true of the loading as successful, false if end of file was reached
   */
  inline bool PushLoad(dmlc::Stream *fi) {
    if (!fi->Read(&disk_offset_)) return false;
    data.resize(offset.back() + disk_offset_.back());
    if (disk_offset_.back() != 0) {
      CHECK_EQ(fi->Read(dmlc::BeginPtr(data) + offset.back(),
                        disk_offset_.back() * sizeof(SparseBatch::Entry)),
               disk_offset_.back() * sizeof(SparseBatch::Entry))
          << "Invalid SparsePage file";
    }
    size_t top = offset.back();
    size_t begin = offset.size();
    offset.resize(offset.size() + disk_offset_.size());
    for (size_t i = 0; i < disk_offset_.size(); ++i) {
      offset[i + begin] = top + disk_offset_[i];
    }
    return true;
  }
  /*!
   * \brief Push row batch into the page
   * \param batch the row batch
   */
  inline void Push(const RowBatch &batch) {
    data.resize(offset.back() + batch.ind_ptr[batch.size]);
    std::memcpy(dmlc::BeginPtr(data) + offset.back(),
                batch.data_ptr + batch.ind_ptr[0],
                sizeof(SparseBatch::Entry) * batch.ind_ptr[batch.size]);
    size_t top = offset.back();
    size_t begin = offset.size();
    offset.resize(offset.size() + batch.size);
    for (size_t i = 0; i < batch.size; ++i) {
      offset[i + begin] = top + batch.ind_ptr[i + 1] - batch.ind_ptr[0];
    }
  }
  /*!
   * \brief Push a sparse page
   * \param batch the row page
   */
  inline void Push(const SparsePage &batch) {
    size_t top = offset.back();
    data.resize(top + batch.data.size());
    std::memcpy(dmlc::BeginPtr(data) + top,
                dmlc::BeginPtr(batch.data),
                sizeof(SparseBatch::Entry) * batch.data.size());
    size_t begin = offset.size();
    offset.resize(begin + batch.Size());
    for (size_t i = 0; i < batch.Size(); ++i) {
      offset[i + begin] = top + batch.offset[i + 1];
    }
  }
  /*!
   * \brief Push one instance into page
   *  \param row an instance row
   */
  inline void Push(const SparseBatch::Inst &inst) {
    offset.push_back(offset.back() + inst.length);
    size_t begin = data.size();
    data.resize(begin + inst.length);
    if (inst.length != 0) {
      std::memcpy(dmlc::BeginPtr(data) + begin, inst.data,
                  sizeof(SparseBatch::Entry) * inst.length);
    }
  }
  /*!
   * \param base_rowid base_rowid of the data
   * \return row batch representation of the page
   */
  inline RowBatch GetRowBatch(size_t base_rowid) const {
    RowBatch out;
    out.base_rowid  = base_rowid;
    out.ind_ptr = dmlc::BeginPtr(offset);
    out.data_ptr = dmlc::BeginPtr(data);
    out.size = offset.size() - 1;
    return out;
  }

 private:
  /*! \brief external memory column offset */
  std::vector<size_t> disk_offset_;
};
}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_BATCH_PAGE_H_
