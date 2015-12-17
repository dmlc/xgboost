/*!
 * Copyright (c) 2014 by Contributors
 * \file sparse_batch_page.h
 *   content holder of sparse batch that can be saved to disk
 *   the representation can be effectively
 *   use in external memory computation
 * \author Tianqi Chen
 */
#ifndef XGBOOST_IO_SPARSE_BATCH_PAGE_H_
#define XGBOOST_IO_SPARSE_BATCH_PAGE_H_

#include <vector>
#include <algorithm>
#include "../data.h"

namespace xgboost {
namespace io {
/*!
 * \brief storage unit of sparse batch
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
   * \brief load the by providing a list of interested segments
   *        only the interested segments are loaded
   * \param fi the input stream of the file
   * \param sorted_index_set sorted index of segments we are interested in
   * \return true of the loading as successful, false if end of file was reached
   */
  inline bool Load(utils::ISeekStream *fi,
                   const std::vector<bst_uint> &sorted_index_set) {
    if (!fi->Read(&disk_offset_)) return false;
    // setup the offset
    offset.clear(); offset.push_back(0);
    for (size_t i = 0; i < sorted_index_set.size(); ++i) {
      bst_uint fid = sorted_index_set[i];
      utils::Check(fid + 1 < disk_offset_.size(), "bad col.blob format");
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
        utils::Assert(disk_offset_[fid] > curr_offset, "fset index was not sorted");
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
        utils::Check(fi->Read(BeginPtr(data) + offset[i],
                              size_to_read * sizeof(SparseBatch::Entry)) != 0,
                     "Invalid SparsePage file");
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
  inline bool Load(utils::IStream *fi) {
    if (!fi->Read(&offset)) return false;
    utils::Check(offset.size() != 0, "Invalid SparsePage file");
    data.resize(offset.back());
    if (data.size() != 0) {
      utils::Check(fi->Read(BeginPtr(data), data.size() * sizeof(SparseBatch::Entry)) != 0,
                   "Invalid SparsePage file");
    }
    return true;
  }
  /*!
   * \brief save the data to fo, when a page was written
   *    to disk it must contain all the elements in the
   * \param fo output stream
   */
  inline void Save(utils::IStream *fo) const {
    utils::Assert(offset.size() != 0 && offset[0] == 0, "bad offset");
    utils::Assert(offset.back() == data.size(), "in consistent SparsePage");
    fo->Write(offset);
    if (data.size() != 0) {
      fo->Write(BeginPtr(data), data.size() * sizeof(SparseBatch::Entry));
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
  inline bool PushLoad(utils::IStream *fi) {
    if (!fi->Read(&disk_offset_)) return false;
    data.resize(offset.back() + disk_offset_.back());
    if (disk_offset_.back() != 0) {
      utils::Check(fi->Read(BeginPtr(data) + offset.back(),
                            disk_offset_.back() * sizeof(SparseBatch::Entry)) != 0,
                   "Invalid SparsePage file");
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
    std::memcpy(BeginPtr(data) + offset.back(),
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
    std::memcpy(BeginPtr(data) + top,
                BeginPtr(batch.data),
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
      std::memcpy(BeginPtr(data) + begin, inst.data,
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
    out.ind_ptr = BeginPtr(offset);
    out.data_ptr = BeginPtr(data);
    out.size = offset.size() - 1;
    return out;
  }

 private:
  /*! \brief external memory column offset */
  std::vector<size_t> disk_offset_;
};
/*!
 * \brief factory class for SparsePage,
 *        used in threadbuffer template
 */
class SparsePageFactory {
 public:
  SparsePageFactory(void)
      : action_load_all_(true), set_load_all_(true) {}
  inline void SetFile(const utils::FileStream &fi,
                      size_t file_begin = 0) {
    fi_ = fi;
    file_begin_ = file_begin;
  }
  inline const std::vector<bst_uint> &index_set(void) const {
    return action_index_set_;
  }
  // set index set, will be used after next before first
  inline void SetIndexSet(const std::vector<bst_uint> &index_set,
                          bool load_all) {
    set_load_all_ = load_all;
    if (!set_load_all_) {
      set_index_set_ = index_set;
      std::sort(set_index_set_.begin(), set_index_set_.end());
    }
  }
  inline bool Init(void) {
    return true;
  }
  inline void SetParam(const char *name, const char *val) {}
  inline bool LoadNext(SparsePage *val) {
    if (!action_load_all_) {
      if (action_index_set_.size() == 0) {
        return false;
      } else {
        return val->Load(&fi_, action_index_set_);
      }
    } else {
      return val->Load(&fi_);
    }
  }
  inline SparsePage *Create(void) {
    return new SparsePage();
  }
  inline void FreeSpace(SparsePage *a) {
    delete a;
  }
  inline void Destroy(void) {
    fi_.Close();
  }
  inline void BeforeFirst(void) {
    fi_.Seek(file_begin_);
    action_load_all_ = set_load_all_;
    if (!set_load_all_) {
      action_index_set_ = set_index_set_;
    }
  }

 private:
  bool action_load_all_, set_load_all_;
  size_t file_begin_;
  utils::FileStream fi_;
  std::vector<bst_uint> action_index_set_;
  std::vector<bst_uint> set_index_set_;
};
}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_SPARSE_BATCH_PAGE_H_
