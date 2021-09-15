/*!
 * Copyright (c) 2014-2019 by Contributors
 * \file sparse_page_writer.h
 * \author Tianqi Chen
 */
#ifndef XGBOOST_DATA_SPARSE_PAGE_WRITER_H_
#define XGBOOST_DATA_SPARSE_PAGE_WRITER_H_

#include <xgboost/data.h>
#include <dmlc/io.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <memory>
#include <functional>

#if DMLC_ENABLE_STD_THREAD
#include <dmlc/concurrency.h>
#include <thread>
#endif  // DMLC_ENABLE_STD_THREAD

namespace xgboost {
namespace data {

template<typename T>
struct SparsePageFormatReg;

/*!
 * \brief Format specification of SparsePage.
 */
template<typename T>
class SparsePageFormat {
 public:
  /*! \brief virtual destructor */
  virtual ~SparsePageFormat() = default;
  /*!
   * \brief Load all the segments into page, advance fi to end of the block.
   * \param page The data to read page into.
   * \param fi the input stream of the file
   * \return true of the loading as successful, false if end of file was reached
   */
  virtual bool Read(T* page, dmlc::SeekStream* fi) = 0;
  /*!
   * \brief save the data to fo, when a page was written.
   * \param fo output stream
   */
  virtual size_t Write(const T& page, dmlc::Stream* fo) = 0;
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

/*!
 * \brief Registry entry for sparse page format.
 */
template<typename T>
struct SparsePageFormatReg
    : public dmlc::FunctionRegEntryBase<SparsePageFormatReg<T>,
                                        std::function<SparsePageFormat<T>* ()>> {
};

/*!
 * \brief Macro to register sparse page format.
 *
 * \code
 * // example of registering a objective
 * XGBOOST_REGISTER_SPARSE_PAGE_FORMAT(raw)
 * .describe("Raw binary data format.")
 * .set_body([]() {
 *     return new RawFormat();
 *   });
 * \endcode
 */
#define SparsePageFmt SparsePageFormat<SparsePage>
#define XGBOOST_REGISTER_SPARSE_PAGE_FORMAT(Name)                       \
  DMLC_REGISTRY_REGISTER(SparsePageFormatReg<SparsePage>, SparsePageFmt, Name)

#define CSCPageFmt SparsePageFormat<CSCPage>
#define XGBOOST_REGISTER_CSC_PAGE_FORMAT(Name)                       \
  DMLC_REGISTRY_REGISTER(SparsePageFormatReg<CSCPage>, CSCPageFmt, Name)

#define SortedCSCPageFmt SparsePageFormat<SortedCSCPage>
#define XGBOOST_REGISTER_SORTED_CSC_PAGE_FORMAT(Name)                       \
  DMLC_REGISTRY_REGISTER(SparsePageFormatReg<SortedCSCPage>, SortedCSCPageFmt, Name)

#define EllpackPageFmt SparsePageFormat<EllpackPage>
#define XGBOOST_REGISTER_ELLPACK_PAGE_FORMAT(Name)                       \
  DMLC_REGISTRY_REGISTER(SparsePageFormatReg<EllpackPage>, EllpackPageFmt, Name)

#define GHistIndexPageFmt SparsePageFormat<GHistIndexMatrix>
#define XGBOOST_REGISTER_GHIST_INDEX_PAGE_FORMAT(Name)                         \
  DMLC_REGISTRY_REGISTER(SparsePageFormatReg<GHistIndexMatrix>,                \
                         GHistIndexPageFmt, Name)

}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_WRITER_H_
