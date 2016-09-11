/*!
 *  Copyright (c) 2015 by Contributors
 * \file single_file_split.h
 * \brief base implementation of line-spliter
 * \author Tianqi Chen
 */
#ifndef DMLC_IO_SINGLE_FILE_SPLIT_H_
#define DMLC_IO_SINGLE_FILE_SPLIT_H_

#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <string>
#include <algorithm>

namespace dmlc {
namespace io {
/*!
 * \brief line split implementation from single FILE
 * simply returns lines of files, used for stdin
 */
class SingleFileSplit : public InputSplit {
 public:
  explicit SingleFileSplit(const char *fname)
      : use_stdin_(false), buffer_size_(kBufferSize),
        chunk_begin_(NULL), chunk_end_(NULL) {
    if (!std::strcmp(fname, "stdin")) {
#ifndef DMLC_STRICT_CXX98_
      use_stdin_ = true; fp_ = stdin;
#endif
    }
    if (!use_stdin_) {
      fp_ = fopen64(fname, "rb");
      CHECK(fp_ != NULL) << "SingleFileSplit: fail to open " << fname;
    }
    buffer_.resize(kBufferSize);
  }
  virtual ~SingleFileSplit(void) {
    if (!use_stdin_) std::fclose(fp_);
  }
  virtual void BeforeFirst(void) {
    fseek(fp_, 0, SEEK_SET);
  }
  virtual void HintChunkSize(size_t chunk_size) {
    buffer_size_ = std::max(chunk_size, buffer_size_);
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, 1, size, fp_);
  }
  virtual void ResetPartition(unsigned part_index, unsigned num_parts) {
    CHECK(part_index == 0 && num_parts == 1);
    this->BeforeFirst();
  }
  virtual void Write(const void *ptr, size_t size) {
    LOG(FATAL) << "InputSplit do not support write";
  }
  virtual bool NextRecord(Blob *out_rec) {
    if (chunk_begin_ == chunk_end_) {
      if (!LoadChunk()) return false;
    }
    char *next = FindNextRecord(chunk_begin_,
                                chunk_end_);
    out_rec->dptr = chunk_begin_;
    out_rec->size = next - chunk_begin_;
    chunk_begin_ = next;
    return true;
  }
  virtual bool NextChunk(Blob *out_chunk) {
    if (chunk_begin_ == chunk_end_) {
      if (!LoadChunk()) return false;
    }
    out_chunk->dptr = chunk_begin_;
    out_chunk->size = chunk_end_ - chunk_begin_;
    chunk_begin_ = chunk_end_;
    return true;
  }
  inline bool ReadChunk(void *buf, size_t *size) {
    size_t max_size = *size;
    if (max_size <= overflow_.length()) {
      *size = 0; return true;
    }
    if (overflow_.length() != 0) {
      std::memcpy(buf, BeginPtr(overflow_), overflow_.length());
    }
    size_t olen = overflow_.length();
    overflow_.resize(0);
    size_t nread = this->Read(reinterpret_cast<char*>(buf) + olen,
                              max_size - olen);
    nread += olen;
    if (nread == 0) return false;
    if (nread != max_size) {
      *size = nread;
      return true;
    } else {
      const char *bptr = reinterpret_cast<const char*>(buf);
      // return the last position where a record starts
      const char *bend = this->FindLastRecordBegin(bptr, bptr + max_size);
      *size = bend - bptr;
      overflow_.resize(max_size - *size);
      if (overflow_.length() != 0) {
        std::memcpy(BeginPtr(overflow_), bend, overflow_.length());
      }
      return true;
    }
  }

 protected:
  inline const char* FindLastRecordBegin(const char *begin,
                                         const char *end) {
    if (begin == end) return begin;
    for (const char *p = end - 1; p != begin; --p) {
      if (*p == '\n' || *p == '\r') return p + 1;
    }
    return begin;
  }
  inline char* FindNextRecord(char *begin, char *end) {
    char *p;
    for (p = begin; p != end; ++p) {
      if (*p == '\n' || *p == '\r') break;
    }
    for (; p != end; ++p) {
      if (*p != '\n' && *p != '\r') return p;
    }
    return end;
  }
  inline bool LoadChunk(void) {
    if (buffer_.length() < buffer_size_) {
      buffer_.resize(buffer_size_);
    }
    while (true) {
      size_t size = buffer_.length();
      if (!ReadChunk(BeginPtr(buffer_), &size)) return false;
      if (size == 0) {
        buffer_.resize(buffer_.length() * 2);
      } else {
        chunk_begin_ = reinterpret_cast<char *>(BeginPtr(buffer_));
        chunk_end_ = chunk_begin_ + size;
        break;
      }
    }
    return true;
  }

 private:
  // buffer size
  static const size_t kBufferSize = 1 << 18UL;
  // file
  std::FILE *fp_;
  bool use_stdin_;
  // internal overflow
  std::string overflow_;
  // internal buffer
  std::string buffer_;
  // internal buffer size
  size_t buffer_size_;
  // beginning of chunk
  char *chunk_begin_;
  // end of chunk
  char *chunk_end_;
};
}  // namespace io
}  // namespace dmlc
#endif  // DMLC_IO_SINGLE_FILE_SPLIT_H_
