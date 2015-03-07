#ifndef RABIT_LEARN_IO_UTILS_H_
#define RABIT_LEARN_IO_UTILS_H_
/*!
 * \file utils.h
 * \brief some helper utils that can be used to implement IO
 * \author Tianqi Chen
 */
namespace rabit {
namespace io {
/*! \brief buffer reader of the stream that allows you to get  */
class StreamBufferReader {
 public:
  StreamBufferReader(size_t buffer_size)
      :stream_(NULL),
       read_len_(1), read_ptr_(1) {
    buffer_.resize(buffer_size);
  }
  /*!
   * \brief set input stream
   */
  inline void set_stream(IStream *stream) {
    stream_ = stream;
    read_len_ = read_ptr_ = 1;
  }
  /*!
   * \brief allows quick read using get char
   */
  inline char GetChar(void) {
    while (true) {
      if (read_ptr_ < read_len_) {
        return buffer_[read_ptr_++];
      } else {
        read_len_ = stream_->Read(&buffer_[0], buffer_.length());
        if (read_len_ == 0) return EOF;
        read_ptr_ = 0;
      }
    }
  }
  inline bool AtEnd(void) const {
    return read_len_ == 0;
  }
  
 private:
  /*! \brief the underlying stream */
  IStream *stream_;
  /*! \brief buffer to hold data */
  std::string buffer_;
  /*! \brief length of valid data in buffer */
  size_t read_len_;
  /*! \brief pointer in the buffer */
  size_t read_ptr_;
};

/*! \brief implementation of file i/o stream */
class FileStream : public utils::ISeekStream {
 public:
  explicit FileStream(const char *fname, const char *mode)
      : use_stdio(false) {
#ifndef RABIT_STRICT_CXX98_
    if (!strcmp(fname, "stdin")) {
      use_stdio = true; fp = stdin;
    }
    if (!strcmp(fname, "stdout")) {
      use_stdio = true; fp = stdout;
    }
#endif
    if (!use_stdio) {
      fp = utils::FopenCheck(fname, mode);
    }
  }
  virtual ~FileStream(void) {
    this->Close();
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, 1, size, fp);
  }
  virtual void Write(const void *ptr, size_t size) {
    std::fwrite(ptr, size, 1, fp);
  }
  virtual void Seek(size_t pos) {
    std::fseek(fp, static_cast<long>(pos), SEEK_SET);
  }
  virtual size_t Tell(void) {
    return std::ftell(fp);
  }
  virtual bool AtEnd(void) const {
    return feof(fp) != 0;
  }
  inline void Close(void) {
    if (fp != NULL && !use_stdio) {
      std::fclose(fp); fp = NULL;
    }
  }

 private:
  FILE *fp;
  bool use_stdio;
};
}  // namespace io
}  // namespace rabit
#endif  // RABIT_LEARN_IO_UTILS_H_

