/*!
 *  Copyright (c) 2015 by Contributors
 * \file libsvm_parser.h
 * \brief iterator parser to parse libsvm format
 * \author Tianqi Chen
 */
#ifndef XGBOOST_IO_LIBSVM_PARSER_H_
#define XGBOOST_IO_LIBSVM_PARSER_H_

#include <vector>
#include <cstring>
#include <cctype>
#include "../utils/omp.h"
#include "../utils/utils.h"
#include "../sync/sync.h"
#include "./sparse_batch_page.h"

namespace xgboost {
namespace io {
/*! \brief page returned by libsvm parser */
struct LibSVMPage : public SparsePage {
  std::vector<float> label;
  // overload clear  
  inline void Clear() {
    SparsePage::Clear();
    label.clear();
  }
};
/*!
 * \brief libsvm parser that parses the input lines
 * and returns rows in input data
 */
class LibSVMParser : public utils::IIterator<LibSVMPage> {
 public:
  explicit LibSVMParser(dmlc::InputSplit *source,
                        int nthread)
      : bytes_read_(0), at_head_(true),
        data_ptr_(0), data_end_(0), source_(source) {
    int maxthread;
    #pragma omp parallel
    {
      maxthread = omp_get_num_threads();
    }
    maxthread = std::max(maxthread / 2, 1);
    nthread_ = std::min(maxthread, nthread);
  }
  virtual ~LibSVMParser() {
    delete source_;
  }
  virtual void BeforeFirst(void) {
    utils::Assert(at_head_, "cannot call BeforeFirst");
  }
  virtual const LibSVMPage &Value(void) const {
    return data_[data_ptr_ - 1];
  }
  virtual bool Next(void) {
    while (true) {
      while (data_ptr_ < data_end_) {
        data_ptr_ += 1;
        if (data_[data_ptr_ - 1].Size() != 0) {
          return true;
        }
      }
      if (!FillData()) break;
      data_ptr_ = 0; data_end_ = data_.size();
    }
    return false;
  }
  inline size_t bytes_read(void) const {
    return bytes_read_;
  }

 protected:
  inline bool FillData() {
    dmlc::InputSplit::Blob chunk;
    if (!source_->NextChunk(&chunk)) return false;
    int nthread;
    #pragma omp parallel num_threads(nthread_)
    {
      nthread = omp_get_num_threads();
    }
    // reserve space for data
    data_.resize(nthread);
    bytes_read_ += chunk.size;
    utils::Assert(chunk.size != 0, "LibSVMParser.FileData");
    char *head = reinterpret_cast<char*>(chunk.dptr);        
    #pragma omp parallel num_threads(nthread_)
    {
      // threadid
      int tid = omp_get_thread_num();
      size_t nstep = (chunk.size + nthread - 1) / nthread;
      size_t sbegin = std::min(tid * nstep, chunk.size);
      size_t send = std::min((tid + 1) * nstep, chunk.size);
      char *pbegin = BackFindEndLine(head + sbegin, head);
      char *pend;
      if (tid + 1 == nthread) {
        pend = head + send;
      } else {
        pend = BackFindEndLine(head + send, head);
      }
      ParseBlock(pbegin, pend, &data_[tid]);
    }
    data_ptr_ = 0;
    return true;
  }
  /*!
   * \brief parse data into out
   * \param begin beginning of buffer
   * \param end end of buffer
   */
  inline void ParseBlock(char *begin,
                         char *end,
                         LibSVMPage *out) {    
    out->Clear();
    char *p = begin;
    while (p != end) {
      while (isspace(*p) && p != end) ++p;
      if (p == end) break;
      char *head = p;
      while (isdigit(*p) && p != end) ++p;
      if (*p == ':') {
        out->data.push_back(SparseBatch::Entry(atol(head),
                                               atof(p + 1)));
      } else {
        if (out->label.size() != 0) {
          out->offset.push_back(out->data.size());
        }
        out->label.push_back(atof(head));
      }
      while (!isspace(*p) && p != end) ++p;
    }
    if (out->label.size() != 0) {
      out->offset.push_back(out->data.size());
    }
    utils::Check(out->label.size() + 1 == out->offset.size(),
                 "LibSVMParser inconsistent");
  }
  /*!
   * \brief start from bptr, go backward and find first endof line
   * \param bptr end position to go backward
   * \param begin the beginning position of buffer
   * \return position of first endof line going backward
   */
  inline char* BackFindEndLine(char *bptr,
                               char *begin) {
    for (; bptr != begin; --bptr) {
      if (*bptr == '\n' || *bptr == '\r') return bptr;
    }
    return begin;
  }
  
 private:
  // nthread
  int nthread_;
  // number of bytes readed
  size_t bytes_read_;
  // at beginning, at end of stream
  bool at_head_;
  // pointer to begin and end of data
  size_t data_ptr_, data_end_;
  // source split that provides the data
  dmlc::InputSplit *source_;
  // internal data
  std::vector<LibSVMPage> data_;
};
}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_LIBSVM_PARSER_H_
