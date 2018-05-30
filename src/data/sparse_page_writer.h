/*!
 * Copyright (c) 2014 by Contributors
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
#endif

namespace xgboost {
namespace data {
/*!
 * \brief Format specification of SparsePage.
 */
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
  virtual bool Read(SparsePage* page, dmlc::SeekStream* fi) = 0;
  /*!
   * \brief read only the segments we are interested in, advance fi to end of the block.
   * \param page The page to load the data into.
   * \param fi the input stream of the file
   * \param sorted_index_set sorted index of segments we are interested in
   * \return true of the loading as successful, false if end of file was reached
   */
  virtual bool Read(SparsePage* page,
                    dmlc::SeekStream* fi,
                    const std::vector<bst_uint>& sorted_index_set) = 0;
  /*!
   * \brief save the data to fo, when a page was written.
   * \param fo output stream
   */
  virtual void Write(const SparsePage& page, dmlc::Stream* fo) = 0;
  /*!
   * \brief Create sparse page of format.
   * \return The created format functors.
   */
  static SparsePageFormat* Create(const std::string& name);
  /*!
   * \brief decide the format from cache prefix.
   * \return pair of row format, column format type of the cache prefix.
   */
  static std::pair<std::string, std::string> DecideFormat(const std::string& cache_prefix);
};

#if DMLC_ENABLE_STD_THREAD
/*!
 * \brief A threaded writer to write sparse batch page to sharded files.
 */
class SparsePageWriter {
 public:
  /*!
   * \brief constructor
   * \param name_shards name of shard files.
   * \param format_shards format of each shard.
   * \param extra_buffer_capacity Extra buffer capacity before block.
   */
  explicit SparsePageWriter(
      const std::vector<std::string>& name_shards,
      const std::vector<std::string>& format_shards,
      size_t extra_buffer_capacity);
  /*! \brief destructor, will close the files automatically */
  ~SparsePageWriter();
  /*!
   * \brief Push a write job to the writer.
   * This function won't block,
   * writing is done by another thread inside writer.
   * \param page The page to be written
   */
  void PushWrite(std::shared_ptr<SparsePage>&& page);
  /*!
   * \brief Allocate a page to store results.
   *  This function can block when the writer is too slow and buffer pages
   *  have not yet been recycled.
   * \param out_page Used to store the allocated pages.
   */
  void Alloc(std::shared_ptr<SparsePage>* out_page);

 private:
  /*! \brief number of allocated pages */
  size_t num_free_buffer_;
  /*! \brief clock_pointer */
  size_t clock_ptr_;
  /*! \brief writer threads */
  std::vector<std::unique_ptr<std::thread> > workers_;
  /*! \brief recycler queue */
  dmlc::ConcurrentBlockingQueue<std::shared_ptr<SparsePage> > qrecycle_;
  /*! \brief worker threads */
  std::vector<dmlc::ConcurrentBlockingQueue<std::shared_ptr<SparsePage> > > qworkers_;
};
#endif  // DMLC_ENABLE_STD_THREAD

/*!
 * \brief Registry entry for sparse page format.
 */
struct SparsePageFormatReg
    : public dmlc::FunctionRegEntryBase<SparsePageFormatReg,
                                        std::function<SparsePageFormat* ()> > {
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
#define XGBOOST_REGISTER_SPARSE_PAGE_FORMAT(Name)                       \
  DMLC_REGISTRY_REGISTER(::xgboost::data::SparsePageFormatReg, SparsePageFormat, Name)

}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_WRITER_H_
