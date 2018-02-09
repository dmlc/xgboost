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
 * \brief in-memory storage unit of sparse batch
 */
class SparsePage {
 public:
  /*! \brief Format of the sparse page. */
  class Format;
  /*! \brief Writer to write the sparse page to files. */
  class Writer;
  /*! \brief minimum index of all index, used as hint for compression. */
  bst_uint min_index;
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
  /*! \return estimation of memory cost of this page */
  inline size_t MemCostBytes(void) const {
    return offset.size() * sizeof(size_t) + data.size() * sizeof(SparseBatch::Entry);
  }
  /*! \brief clear the page */
  inline void Clear(void) {
    min_index = 0;
    offset.clear();
    offset.push_back(0);
    data.clear();
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
   * \brief Push row block into the page.
   * \param batch the row batch.
   */
  inline void Push(const dmlc::RowBlock<uint32_t>& batch) {
    data.reserve(data.size() + batch.offset[batch.size] - batch.offset[0]);
    offset.reserve(offset.size() + batch.size);
    CHECK(batch.index != nullptr);
    for (size_t i = 0; i < batch.size; ++i) {
      offset.push_back(offset.back() + batch.offset[i + 1] - batch.offset[i]);
    }
    for (size_t i = batch.offset[0]; i < batch.offset[batch.size]; ++i) {
      uint32_t index = batch.index[i];
      bst_float fvalue = batch.value == nullptr ? 1.0f : batch.value[i];
      data.push_back(SparseBatch::Entry(index, fvalue));
    }
    CHECK_EQ(offset.back(), data.size());
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
};

/*!
 * \brief Format specification of SparsePage.
 */
class SparsePage::Format {
 public:
  /*! \brief virtual destructor */
  virtual ~Format() {}
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
  static Format* Create(const std::string& name);
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
class SparsePage::Writer {
 public:
  /*!
   * \brief constructor
   * \param name_shards name of shard files.
   * \param format_shards format of each shard.
   * \param extra_buffer_capacity Extra buffer capacity before block.
   */
  explicit Writer(
      const std::vector<std::string>& name_shards,
      const std::vector<std::string>& format_shards,
      size_t extra_buffer_capacity);
  /*! \brief destructor, will close the files automatically */
  ~Writer();
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
                                        std::function<SparsePage::Format* ()> > {
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
#endif  // XGBOOST_DATA_SPARSE_BATCH_PAGE_H_
