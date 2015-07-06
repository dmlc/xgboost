// Copyright by Contributors
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <string>
#include "../utils/io.h"

// implements a single no split version of DMLC
// in case we want to avoid dependency on dmlc-core

namespace xgboost {
namespace utils {
/*!
 * \brief line split implementation from single FILE
 * simply returns lines of files, used for stdin
 */
class SingleFileSplit : public dmlc::InputSplit {
 public:
  explicit SingleFileSplit(const char *fname)
      : use_stdin_(false),
        chunk_begin_(NULL), chunk_end_(NULL) {
    if (!std::strcmp(fname, "stdin")) {
#ifndef XGBOOST_STRICT_CXX98_
      use_stdin_ = true; fp_ = stdin;
#endif
    }
    if (!use_stdin_) {
      fp_ = utils::FopenCheck(fname, "rb");
    }
    buffer_.resize(kBufferSize);
  }
  virtual ~SingleFileSplit(void) {
    if (!use_stdin_) std::fclose(fp_);
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, 1, size, fp_);
  }
  virtual void Write(const void *ptr, size_t size) {
    utils::Error("cannot do write in inputsplit");
  }
  virtual void BeforeFirst(void) {
    std::fseek(fp_, 0, SEEK_SET);
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
  // beginning of chunk
  char *chunk_begin_;
  // end of chunk
  char *chunk_end_;
};

class StdFile : public dmlc::Stream {
 public:
  explicit StdFile(std::FILE *fp, bool use_stdio)
      : fp(fp), use_stdio(use_stdio) {
  }
  virtual ~StdFile(void) {
    this->Close();
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, 1, size, fp);
  }
  virtual void Write(const void *ptr, size_t size) {
    std::fwrite(ptr, size, 1, fp);
  }
  virtual void Seek(size_t pos) {
    std::fseek(fp, static_cast<long>(pos), SEEK_SET);  // NOLINT(*)
  }
  virtual size_t Tell(void) {
    return std::ftell(fp);
  }
  virtual bool AtEnd(void) const {
    return std::feof(fp) != 0;
  }
  inline void Close(void) {
    if (fp != NULL && !use_stdio) {
      std::fclose(fp); fp = NULL;
    }
  }

 private:
  std::FILE *fp;
  bool use_stdio;
};
}  // namespace utils
}  // namespace xgboost

namespace dmlc {
InputSplit* InputSplit::Create(const char *uri,
                               unsigned part,
                               unsigned nsplit,
                               const char *type) {
  using namespace std;
  using namespace xgboost;
  const char *msg = "xgboost is compiled in local mode\n"\
      "to use hdfs, s3 or distributed version, compile with make dmlc=1";
  utils::Check(strncmp(uri, "s3://", 5) != 0, msg);
  utils::Check(strncmp(uri, "hdfs://", 7) != 0, msg);
  utils::Check(nsplit == 1, msg);
  return new utils::SingleFileSplit(uri);
}

Stream *Stream::Create(const char *fname, const char * const mode, bool allow_null) {
  using namespace std;
  using namespace xgboost;
  const char *msg = "xgboost is compiled in local mode\n"\
      "to use hdfs, s3 or distributed version, compile with make dmlc=1";
  utils::Check(strncmp(fname, "s3://", 5) != 0, msg);
  utils::Check(strncmp(fname, "hdfs://", 7) != 0, msg);

  std::FILE *fp = NULL;
  bool use_stdio = false;
  using namespace std;
#ifndef XGBOOST_STRICT_CXX98_
  if (!strcmp(fname, "stdin")) {
    use_stdio = true; fp = stdin;
  }
  if (!strcmp(fname, "stdout")) {
    use_stdio = true; fp = stdout;
  }
#endif
  if (!strncmp(fname, "file://", 7)) fname += 7;
  if (!use_stdio) {
    std::string flag = mode;
    if (flag == "w") flag = "wb";
    if (flag == "r") flag = "rb";
    fp = fopen64(fname, flag.c_str());
  }
  if (fp != NULL) {
    return new utils::StdFile(fp, use_stdio);
  } else {
    utils::Check(allow_null, "fail to open file %s", fname);
    return NULL;
  }
}
}  // namespace dmlc

