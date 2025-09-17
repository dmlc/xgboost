/**
 * Copyright 2014-2024, XGBoost Contributors
 * \file sparse_page_writer.h
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_WRITER_H_
#define XGBOOST_DATA_SPARSE_PAGE_WRITER_H_

#include <functional>  // for function
#include <string>      // for string

#include "../common/io.h"   // for AlignedResourceReadStream, AlignedFileWriteStream
#include "dmlc/registry.h"  // for Registry, FunctionRegEntryBase

namespace xgboost::data {
template<typename T>
struct SparsePageFormatReg;

/**
 * @brief Format specification of various data formats like SparsePage.
 */
template <typename T>
class SparsePageFormat {
 public:
  virtual ~SparsePageFormat() = default;
  /**
   * @brief Load all the segments into page, advance fi to end of the block.
   *
   * @param page The data to read page into.
   * @param fi the input stream of the file
   * @return true of the loading as successful, false if end of file was reached
   */
  virtual bool Read(T* page, common::AlignedResourceReadStream* fi) = 0;
  /**
   * @brief save the data to fo, when a page was written.
   *
   * @param fo output stream
   */
  virtual size_t Write(const T& page, common::AlignedFileWriteStream* fo) = 0;
};

/*!
 * \brief Create sparse page of format.
 * \return The created format functors.
 */
template<typename T>
inline SparsePageFormat<T>* CreatePageFormat(const std::string& name) {
  auto *e = ::dmlc::Registry<SparsePageFormatReg<T>>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown format type " << name;
    return nullptr;
  }
  return (e->body)();
}

/**
 * @brief Registry entry for sparse page format.
 */
template<typename T>
struct SparsePageFormatReg
    : public dmlc::FunctionRegEntryBase<SparsePageFormatReg<T>,
                                        std::function<SparsePageFormat<T>* ()>> {
};
}  // namespace xgboost::data
#endif  // XGBOOST_DATA_SPARSE_PAGE_WRITER_H_
