#ifndef XGBOOST_UTILS_IO_H
#define XGBOOST_UTILS_IO_H
#include <cstdio>
#include <vector>
#include <string>
#include "./utils.h"
/*!
 * \file io.h
 * \brief general stream interface for serialization, I/O
 * \author Tianqi Chen
 */
namespace xgboost {
namespace utils {
/*!
 * \brief interface of stream I/O, used to serialize model
 */
class IStream {
 public:
  /*!
   * \brief read data from stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   * \return usually is the size of data readed
   */
  virtual size_t Read(void *ptr, size_t size) = 0;
  /*!
   * \brief write data to stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   */
  virtual void Write(const void *ptr, size_t size) = 0;
  /*! \brief virtual destructor */
  virtual ~IStream(void) {}

 public:
  // helper functions to write various of data structures
  /*!
   * \brief binary serialize a vector 
   * \param vec vector to be serialized
   */
  template<typename T>
  inline void Write(const std::vector<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    this->Write(&sz, sizeof(sz));
    if (sz != 0) {
      this->Write(&vec[0], sizeof(T) * sz);
    }
  }
  /*!
   * \brief binary load a vector 
   * \param out_vec vector to be loaded
   * \return whether load is successfull
   */
  template<typename T>
  inline bool Read(std::vector<T> *out_vec) {
    uint64_t sz;
    if (this->Read(&sz, sizeof(sz)) == 0) return false;
    out_vec->resize(sz);
    if (sz != 0) {
      if (this->Read(&(*out_vec)[0], sizeof(T) * sz) == 0) return false;
    }
    return true;
  }
  /*!
   * \brief binary serialize a string
   * \param str the string to be serialized
   */ 
  inline void Write(const std::string &str) {
    uint64_t sz = static_cast<uint64_t>(str.length());
    this->Write(&sz, sizeof(sz));
    if (sz != 0) {
      this->Write(&str[0], sizeof(char) * sz);
    }
  }
  /*!
   * \brief binary load a string
   * \param out_str string to be loaded
   * \return whether load is successful
   */
  inline bool Read(std::string *out_str) {
    uint64_t sz;
    if (this->Read(&sz, sizeof(sz)) == 0) return false;
    out_str->resize(sz);
    if (sz != 0) {
      if (this->Read(&(*out_str)[0], sizeof(char) * sz) == 0) return false;
    }
    return true;
  }
};

/*! \brief implementation of file i/o stream */
class FileStream : public IStream {
 private:
  std::FILE *fp;
 public:
  explicit FileStream(std::FILE *fp) : fp(fp) {
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, size, 1, fp);
  }
  virtual void Write(const void *ptr, size_t size) {
    std::fwrite(ptr, size, 1, fp);
  }
  inline void Seek(size_t pos) {
    std::fseek(fp, 0, SEEK_SET);
  }
  inline void Close(void) {
    std::fclose(fp);
  }
};

}  // namespace utils
}  // namespace xgboost
#endif
