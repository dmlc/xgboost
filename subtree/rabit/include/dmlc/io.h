/*!
 *  Copyright (c) 2015 by Contributors
 * \file io.h
 * \brief defines serializable interface of dmlc
 */
#ifndef DMLC_IO_H_
#define DMLC_IO_H_
#include <cstdio>
#include <string>
#include <vector>

/*! \brief namespace for dmlc */
namespace dmlc {
/*!
 * \brief interface of stream I/O for serialization
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
  /*!
   * \brief generic factory function
   *    create an stream, the stream will close the underlying files
   *    upon deletion
   * \param uri the uri of the input currently we support
   *            hdfs://, s3://, and file:// by default file:// will be used
   * \param flag can be "w", "r", "a"
   */
  static IStream *Create(const char *uri, const char* const flag);
  // helper functions to write/read different data structures
  /*!
   * \brief writes a vector
   * \param vec vector to be written/serialized
   */
  template<typename T>
  inline void Write(const std::vector<T> &vec);
  /*!
   * \brief loads a vector
   * \param out_vec vector to be loaded/deserialized
   * \return whether the load was successful
   */
  template<typename T>
  inline bool Read(std::vector<T> *out_vec);
  /*!
   * \brief writes a string
   * \param str the string to be written/serialized
   */ 
  inline void Write(const std::string &str);
  /*!
   * \brief loads a string
   * \param out_str string to be loaded/deserialized
   * \return whether the load/deserialization was successful
   */
  inline bool Read(std::string *out_str);
};

/*! \brief interface of i/o stream that support seek */
class ISeekStream: public IStream {
 public:
  // virtual destructor
  virtual ~ISeekStream(void) {}
  /*! \brief seek to certain position of the file */
  virtual void Seek(size_t pos) = 0;
  /*! \brief tell the position of the stream */
  virtual size_t Tell(void) = 0;
  /*! \return whether we are at end of file */
  virtual bool AtEnd(void) const = 0;  
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

/*!
 * \brief input split header, used to create input split on input dataset
 * this class can be used to obtain filesystem invariant splits from input files
 */
class InputSplit {
 public:
  /*!
   * \brief read next line, store into out_data
   * \param out_data the string that stores the line data, \n is not included
   * \return true of next line was found, false if we read all the lines
   */
  virtual bool ReadLine(std::string *out_data) = 0;
  /*! \brief destructor*/
  virtual ~InputSplit(void) {}  
  /*!
   * \brief factory function:
   *  create input split given a uri
   * \param uri the uri of the input, can contain hdfs prefix
   * \param part_index the part id of current input
   * \param num_parts total number of splits
   */
  static InputSplit* Create(const char *uri,
                            unsigned part_index,
                            unsigned num_parts);
};

// implementations of inline functions
template<typename T>
inline void IStream::Write(const std::vector<T> &vec) {
  size_t sz = vec.size();
  this->Write(&sz, sizeof(sz));
  if (sz != 0) {
    this->Write(&vec[0], sizeof(T) * sz);
  }
}
template<typename T>
inline bool IStream::Read(std::vector<T> *out_vec) {
  size_t sz;
  if (this->Read(&sz, sizeof(sz)) == 0) return false;
  out_vec->resize(sz);
  if (sz != 0) {
    if (this->Read(&(*out_vec)[0], sizeof(T) * sz) == 0) return false;
  }
  return true;
}
inline void IStream::Write(const std::string &str) {
  size_t sz = str.length();
  this->Write(&sz, sizeof(sz));
  if (sz != 0) {
    this->Write(&str[0], sizeof(char) * sz);
  }
}
inline bool IStream::Read(std::string *out_str) {
  size_t sz;
  if (this->Read(&sz, sizeof(sz)) == 0) return false;
  out_str->resize(sz);
  if (sz != 0) {
    if (this->Read(&(*out_str)[0], sizeof(char) * sz) == 0) {
      return false;
    }
  }
  return true;
}
}  // namespace dmlc
#endif  // DMLC_IO_H_
