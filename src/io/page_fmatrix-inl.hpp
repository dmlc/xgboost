#ifndef XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
#define XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
/*!
 * \file page_fmatrix-inl.hpp
 * sparse page manager for fmatrix
 * \author Tianqi Chen
 */
#include "../data.h"
#include "../utils/iterator.h"
#include "../utils/thread_buffer.h"
namespace xgboost {
namespace io {

class CSCMatrixManager {
 public:
  /*! \brief in memory page */
  struct Page {
   public:
    /*! \brief initialize the page */
    inline void Init(size_t size) {
      buffer.resize(size);
      num_entry = 0;
      col_index.clear();
      col_data.clear();
    }    
    /*! \brief number of used entries */
    size_t num_entry;
    /*! \brief column index */
    std::vector<bst_uint> col_index;
    /*! \brief column data */
    std::vector<ColBatch::Inst> col_data;            
    /*! \brief number of free entries */
    inline size_t NumFreeEntry(void) const {
      return buffer.size() - num_entry;
    }
    inline ColBatch::Entry* AllocEntry(size_t len) {
      ColBatch::Entry *p_data = &buffer[0] + num_entry;
      num_entry += len;
      return p_data;
    }
    /*! \brief get underlying batch */
    inline ColBatch GetBatch(void) const {
      ColBatch batch; 
      batch.col_index = &col_index[0];
      batch.col_data  = &col_data[0];
      return batch;
    }
   private:
    /*! \brief buffer space, not to be changed since ready */
    std::vector<ColBatch::Entry> buffer;
  };
  
 private:
  /*! \brief fill a page with */
  inline bool Fill(size_t cidx, Page *p_page) {
    size_t len = col_ptr_[cidx+1] - col_ptr_[cidx];
    if (p_page->NumFreeEntry() < len) return false;
    ColBatch::Entry *p_data = p_page->AllocEntry(len);
    fi->Seek(col_ptr_[cidx]);
    utils::Check(fi->Read(p_data, sizeof(ColBatch::Entry) * len) != 0,
                 "invalid column buffer format");
    p_page->col_data.push_back(ColBatch::Inst(p_data, len));
    p_page->col_index.push_back(cidx);
  }
  /*! \brief size of data content */
  size_t data_size_;
  /*! \brief input stream */
  utils::ISeekStream *fi;
  /*! \brief column pointer of CSC format */
  std::vector<size_t> col_ptr_;  
};

}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_PAGE_FMATRIX_INL_HPP_
