/*!
 *  Copyright (c) 2015 by Contributors
 * \file libsvm_parser.h
 * \brief iterator parser to parse libsvm format
 * \author Tianqi Chen
 */
#ifndef DMLC_DATA_PARSER_H_
#define DMLC_DATA_PARSER_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/threadediter.h>
#include <vector>
#include "./row_block.h"

namespace dmlc {
namespace data {
/*! \brief declare thread class */
template <typename IndexType>
class ThreadedParser;
/*! \brief base class for parser to parse data */

template <typename IndexType>
class ParserImpl : public Parser<IndexType> {
 public:
  ParserImpl() : data_ptr_(0), data_end_(0) {}
  // virtual destructor
  virtual ~ParserImpl() {}
  /*! \brief implement next */
  virtual bool Next(void) {
    while (true) {
      while (data_ptr_ < data_end_) {
        data_ptr_ += 1;
        if (data_[data_ptr_ - 1].Size() != 0) {
          block_ = data_[data_ptr_ - 1].GetBlock();
          return true;
        }
      }
      if (!ParseNext(&data_)) break;
      data_ptr_ = 0;
      data_end_ = static_cast<IndexType>(data_.size());
    }
    return false;
  }
  virtual const RowBlock<IndexType> &Value(void) const {
    return block_;
  }
  /*! \return size of bytes read so far */
  virtual size_t BytesRead(void) const = 0;

 protected:
  // allow ThreadedParser to see ParseNext
  friend class ThreadedParser<IndexType>;
  /*!
   * \brief read in next several blocks of data
   * \param data vector of data to be returned
   * \return true if the data is loaded, false if reach end
   */
  virtual bool ParseNext(std::vector<RowBlockContainer<IndexType> > *data) = 0;
  /*! \brief pointer to begin and end of data */
  IndexType data_ptr_, data_end_;
  /*! \brief internal data */
  std::vector<RowBlockContainer<IndexType> > data_;
  /*! \brief internal row block */
  RowBlock<IndexType> block_;
};

#if DMLC_ENABLE_STD_THREAD

template <typename IndexType>
class ThreadedParser : public ParserImpl<IndexType> {
 public:
  explicit ThreadedParser(ParserImpl<IndexType> *base)
      : base_(base), tmp_(NULL) {
    iter_.set_max_capacity(8);
    iter_.Init([base](std::vector<RowBlockContainer<IndexType> > **dptr) {
        if (*dptr == NULL) {
          *dptr = new std::vector<RowBlockContainer<IndexType> >();
        }
        return base->ParseNext(*dptr);
      }, [base]() {base->BeforeFirst();});
  }
  virtual ~ThreadedParser(void) {
    // stop things before base is deleted
    iter_.Destroy();
    delete base_;
    delete tmp_;
  }
  virtual void BeforeFirst() {
    iter_.BeforeFirst();
  }
  /*! \brief implement next */
  using ParserImpl<IndexType>::data_ptr_;
  using ParserImpl<IndexType>::data_end_;
  virtual bool Next(void) {
    while (true) {
      while (data_ptr_ < data_end_) {
        data_ptr_ += 1;
        if ((*tmp_)[data_ptr_ - 1].Size() != 0) {
          this->block_ = (*tmp_)[data_ptr_ - 1].GetBlock();
          return true;
        }
      }
      if (tmp_ != NULL) iter_.Recycle(&tmp_);
      if (!iter_.Next(&tmp_)) break;
      data_ptr_ = 0; data_end_ = tmp_->size();
    }
    return false;
  }
  virtual size_t BytesRead(void) const {
    return base_->BytesRead();
  }

 protected:
  virtual bool ParseNext(std::vector<RowBlockContainer<IndexType> > *data) {
    LOG(FATAL) << "cannot call ParseNext"; return false;
  }

 private:
  /*! \brief the place where we get the data */
  Parser<IndexType> *base_;
  /*! \brief backend threaded iterator */
  ThreadedIter<std::vector<RowBlockContainer<IndexType> > > iter_;
  /*! \brief current chunk of data */
  std::vector<RowBlockContainer<IndexType> > *tmp_;
};
#endif  // DMLC_USE_CXX11
}  // namespace data
}  // namespace dmlc
#endif  // DMLC_DATA_PARSER_H_
