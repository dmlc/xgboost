/*!
 *  Copyright (c) 2015 by Contributors
 * \file cached_input_split.h
 * \brief InputSplit that reads from an existing InputSplit
 *  and cache the data into local disk, the second iteration
 *  will be reading from the local cached data
 * \author Tianqi Chen
 */
#ifndef DMLC_IO_CACHED_INPUT_SPLIT_H_
#define DMLC_IO_CACHED_INPUT_SPLIT_H_

#include <dmlc/base.h>
// this code depends on c++11

#if DMLC_ENABLE_STD_THREAD
#include <dmlc/threadediter.h>
#include <string>
#include <algorithm>
#include "./input_split_base.h"

namespace dmlc {
namespace io {
/*!
 * \brief InputSplit that reads from an existing InputSplit
 *  and cache the data into local disk, the second iteration
 *  will be reading from the local cached data
 */
class CachedInputSplit : public InputSplit {
 public:
  /*!
   * \brief constructor
   * \param base source input split
   * \param cache_file the path to cache file
   * \param reuse_exist_cache whether reuse existing cache file, if any
   */
  CachedInputSplit(InputSplitBase *base,
                   const char *cache_file,
                   bool reuse_exist_cache = true)
      : buffer_size_(InputSplitBase::kBufferSize),
        cache_file_(cache_file),
        fo_(NULL), fi_(NULL),
        base_(base), tmp_chunk_(NULL),
        iter_preproc_(NULL) {
    if (reuse_exist_cache) {
      if (!this->InitCachedIter()) {
        this->InitPreprocIter();
      }
    } else {
      this->InitPreprocIter();
    }
  }
  // destructor
  virtual ~CachedInputSplit(void) {
    // NOTE delete can handle NULL ptr
    // deletion order matters
    delete iter_preproc_;
    delete fo_;
    iter_cached_.Destroy();
    delete tmp_chunk_;
    delete base_;
    delete fi_;
  }
  virtual void BeforeFirst(void) {
    // if preprocessing did not end
    // pull data from preprocessing module
    if (iter_preproc_ != NULL) {
      if (tmp_chunk_ != NULL) {
        iter_preproc_->Recycle(&tmp_chunk_);
      }
      while (iter_preproc_->Next(&tmp_chunk_)) {
        iter_preproc_->Recycle(&tmp_chunk_);
      }
      // finalize the push out process
      delete iter_preproc_;
      delete fo_;
      iter_preproc_ = NULL;
      fo_ = NULL;
      CHECK(this->InitCachedIter())
          << "Failed to initialize CachedIter";
    } else {
      iter_cached_.BeforeFirst();
    }
    if (tmp_chunk_ != NULL) {
      iter_cached_.Recycle(&tmp_chunk_);
    }
  }
  virtual void ResetPartition(unsigned part_index, unsigned num_parts) {
    LOG(FATAL) << "ResetPartition is not supported in CachedInputSplit";
  }
  virtual void HintChunkSize(size_t chunk_size) {
    buffer_size_ = std::max(chunk_size / sizeof(size_t), buffer_size_);
  }
  // implement next record
  virtual bool NextRecord(Blob *out_rec) {
    auto *iter = iter_preproc_ != NULL ? iter_preproc_ : &iter_cached_;
    if (tmp_chunk_ == NULL) {
      if (!iter->Next(&tmp_chunk_)) return false;
    }
    while (!base_->ExtractNextRecord(out_rec, tmp_chunk_)) {
      iter->Recycle(&tmp_chunk_);
      if (!iter->Next(&tmp_chunk_)) return false;
    }
    return true;
  }
  // implement next chunk
  virtual bool NextChunk(Blob *out_chunk) {
    auto *iter = iter_preproc_ != NULL ? iter_preproc_ : &iter_cached_;
    if (tmp_chunk_ == NULL) {
      if (!iter->Next(&tmp_chunk_)) return false;
    }
    while (!base_->ExtractNextChunk(out_chunk, tmp_chunk_)) {
      iter->Recycle(&tmp_chunk_);
      if (!iter->Next(&tmp_chunk_)) return false;
    }
    return true;
  }

 private:
  /*! \brief internal buffer size */
  size_t buffer_size_;
  /*! \brief cache file path */
  std::string cache_file_;
  /*! \brief output stream to cache file*/
  dmlc::Stream *fo_;
  /*! \brief input stream from cache file */
  dmlc::SeekStream *fi_;
  /*! \brief the place where we get the data */
  InputSplitBase *base_;
  /*! \brief current chunk of data */
  InputSplitBase::Chunk *tmp_chunk_;
  /*! \brief backend thread iterator for preprocessing  */
  ThreadedIter<InputSplitBase::Chunk> *iter_preproc_;
  /*! \brief backend thread iterator for cache */
  ThreadedIter<InputSplitBase::Chunk> iter_cached_;
  /*! \brief initialize the cached iterator */
  inline void InitPreprocIter(void);
  /*!
   * \brief initialize the cached iterator
   * \return wheher the file exist and
   *  initialization is successful
   */
  inline bool InitCachedIter(void);
};

inline void CachedInputSplit:: InitPreprocIter(void) {
  fo_ = dmlc::Stream::Create(cache_file_.c_str(), "w");
  iter_preproc_ = new ThreadedIter<InputSplitBase::Chunk>();
  iter_preproc_->set_max_capacity(16);
  iter_preproc_->Init([this](InputSplitBase::Chunk **dptr) {
      if (*dptr == NULL) {
        *dptr = new InputSplitBase::Chunk(buffer_size_);
      }
      auto *p = *dptr;
      if (!p->Load(base_, buffer_size_)) return false;
      // after loading, save to disk
      size_t size = p->end - p->begin;
      fo_->Write(&size, sizeof(size));
      fo_->Write(p->begin, size);
      return true;
    });
}

inline bool CachedInputSplit::InitCachedIter(void) {
  fi_ = dmlc::SeekStream::CreateForRead(cache_file_.c_str(), true);
  if (fi_ == NULL) return false;
  iter_cached_.Init([this](InputSplitBase::Chunk **dptr) {
      if (*dptr == NULL) {
        *dptr = new InputSplitBase::Chunk(buffer_size_);
      }
      auto *p = *dptr;
      // read data from cache file
      size_t size;
      size_t nread = fi_->Read(&size, sizeof(size));
      if (nread == 0) return false;
      CHECK(nread == sizeof(size))
          << cache_file_ << " has invalid cache file format";
      p->data.resize(size / sizeof(size_t) + 1);
      p->begin = reinterpret_cast<char*>(BeginPtr(p->data));
      p->end = p->begin + size;
      CHECK(fi_->Read(p->begin, size) == size)
          << cache_file_ << " has invalid cache file format";
      return true;
    },
    [this]() { fi_->Seek(0); });
  return true;
}
}  // namespace io
}  // namespace dmlc
#endif  // DMLC_USE_CXX11
#endif  // DMLC_IO_CACHED_INPUT_SPLIT_H_
