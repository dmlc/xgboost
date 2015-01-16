/*!
 *  Copyright (c) 2014 by Contributors
 * \file rabit_serializable.h
 * \brief defines serializable interface of rabit
 * \author Tianqi Chen
 */
#ifndef RABIT_RABIT_SERIALIZABLE_H_
#define RABIT_RABIT_SERIALIZABLE_H_
#include <vector>
#include <string>
#include "./rabit/utils.h"
namespace rabit {
/*!
 * \brief interface of stream I/O, used by ISerializable
 * \sa ISerializable
 */
class IStream {
 public:
  /*!
   * \brief reads data from a stream
   * \param ptr pointer to a memory buffer
   * \param size block size
   * \return the size of data read
   */
  virtual size_t Read(void *ptr, size_t size) = 0;
  /*!
   * \brief writes data to a stream
   * \param ptr pointer to a memory buffer
   * \param size block size
   */
  virtual void Write(const void *ptr, size_t size) = 0;
  /*! \brief virtual destructor */
  virtual ~IStream(void) {}

 public:
  // helper functions to write/read different data structures
  /*!
   * \brief writes a vector
   * \param vec vector to be written/serialized
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
   * \brief loads a vector
   * \param out_vec vector to be loaded/deserialized
   * \return whether the load was successful
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
   * \brief writes a string
   * \param str the string to be written/serialized
   */ 
  inline void Write(const std::string &str) {
    uint64_t sz = static_cast<uint64_t>(str.length());
    this->Write(&sz, sizeof(sz));
    if (sz != 0) {
      this->Write(&str[0], sizeof(char) * sz);
    }
  }
  /*!
   * \brief loads a string
   * \param out_str string to be loaded/deserialized
   * \return whether the load/deserialization was successful
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

/*! \brief interface for serializable objects */
class ISerializable {
 public:
  /*! 
  * \brief load the model from a stream
  * \param fi stream where to load the model from
  */
  virtual void Load(IStream &fi) = 0;
  /*! 
  * \brief saves the model to a stream
  * \param fo stream where to save the model to
  */
  virtual void Save(IStream &fo) const = 0;
};
}  // namespace rabit
#endif  // RABIT_RABIT_SERIALIZABLE_H_
