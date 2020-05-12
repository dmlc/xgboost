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
   * \brief read only the segments we are interested in, advance fi to end of the block.
   * \param page The page to load the data into.
   * \param fi the input stream of the file
   * \param sorted_index_set sorted index of segments we are interested in
   * \return true of the loading as successful, false if end of file was reached
   */
  virtual bool Read(T* page,
                    dmlc::SeekStream* fi,
                    const std::vector<bst_uint>& sorted_index_set) = 0;
  /*!
   * \brief save the data to fo, when a page was written.
   * \param fo output stream
   */
  virtual void Write(const T& page, dmlc::Stream* fo) = 0;
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

#if DMLC_ENABLE_STD_THREAD
/*!
 * \brief A threaded writer to write sparse batch page to sharded files.
 * @tparam T Type of the page.
 */
template<typename T>
class SparsePageWriter {
 public:
  /*!
   * \brief constructor
   * \param name_shards name of shard files.
   * \param format_shards format of each shard.
   * \param extra_buffer_capacity Extra buffer capacity before block.
   */
  explicit SparsePageWriter(const std::vector<std::string>& name_shards,
                            const std::vector<std::string>& format_shards,
                            size_t extra_buffer_capacity)
      : num_free_buffer_(extra_buffer_capacity + name_shards.size()),
        clock_ptr_(0),
        workers_(name_shards.size()),
        qworkers_(name_shards.size()) {
    CHECK_EQ(name_shards.size(), format_shards.size());
    // start writer threads
    for (size_t i = 0; i < name_shards.size(); ++i) {
      std::string name_shard = name_shards[i];
      std::string format_shard = format_shards[i];
      auto* wqueue = &qworkers_[i];
      workers_[i].reset(new std::thread(
          [this, name_shard, format_shard, wqueue]() {
            std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(name_shard.c_str(), "w"));
            std::unique_ptr<SparsePageFormat<T>> fmt(CreatePageFormat<T>(format_shard));
            fo->Write(format_shard);
            std::shared_ptr<T> page;
            while (wqueue->Pop(&page)) {
              if (page == nullptr) break;
              fmt->Write(*page, fo.get());
              qrecycle_.Push(std::move(page));
            }
            fo.reset(nullptr);
            LOG(INFO) << "SparsePageWriter Finished writing to " << name_shard;
          }));
    }
  }

  /*! \brief destructor, will close the files automatically */
  ~SparsePageWriter() {
    for (auto& queue : qworkers_) {
      // use nullptr to signal termination.
      std::shared_ptr<T> sig(nullptr);
      queue.Push(std::move(sig));
    }
    for (auto& thread : workers_) {
      thread->join();
    }
  }

  /*!
   * \brief Push a write job to the writer.
   * This function won't block,
   * writing is done by another thread inside writer.
   * \param page The page to be written
   */
  void PushWrite(std::shared_ptr<T>&& page) {
    qworkers_[clock_ptr_].Push(std::move(page));
    clock_ptr_ = (clock_ptr_ + 1) % workers_.size();
  }

  /*!
   * \brief Allocate a page to store results.
   *  This function can block when the writer is too slow and buffer pages
   *  have not yet been recycled.
   * \param out_page Used to store the allocated pages.
   */
  void Alloc(std::shared_ptr<T>* out_page) {
    CHECK(*out_page == nullptr);
    if (num_free_buffer_ != 0) {
      out_page->reset(new T());
      --num_free_buffer_;
    } else {
      CHECK(qrecycle_.Pop(out_page));
    }
  }

 private:
  /*! \brief number of allocated pages */
  size_t num_free_buffer_;
  /*! \brief clock_pointer */
  size_t clock_ptr_;
  /*! \brief writer threads */
  std::vector<std::unique_ptr<std::thread>> workers_;
  /*! \brief recycler queue */
  dmlc::ConcurrentBlockingQueue<std::shared_ptr<T>> qrecycle_;
  /*! \brief worker threads */
  std::vector<dmlc::ConcurrentBlockingQueue<std::shared_ptr<T>>> qworkers_;
};
#endif  // DMLC_ENABLE_STD_THREAD

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
  DMLC_REGISTRY_REGISTER(SparsePageFormatReg<EllpackPage>, EllpackPageFm, Name)

}  // namespace data
}  // namespace xgboost
#endif  // XGBOOST_DATA_SPARSE_PAGE_WRITER_H_
