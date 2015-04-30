/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief defines configuration macros
 */
#ifndef DMLC_BASE_H_
#define DMLC_BASE_H_

/*! \brief whether use glog for logging*/
#ifndef DMLC_USE_GLOG
#define DMLC_USE_GLOG 0
#endif

/*! \brief whether compile with hdfs support */
#ifndef DMLC_USE_HDFS
#define DMLC_USE_HDFS 0
#endif

/*! \brief whether compile with s3 support */
#ifndef DMLC_USE_S3
#define DMLC_USE_S3 0
#endif

/*! \brief whether or not use parameter server */
#ifndef DMLC_USE_PS
#define DMLC_USE_PS 0
#endif

/*! \brief whether or not use c++11 support */
#ifndef DMLC_USE_CXX11
#define DMLC_USE_CXX11 defined(__GXX_EXPERIMENTAL_CXX0X) || __cplusplus >= 201103L || defined(_MSC_VER)
#endif

///
/// code block to handle optionally loading
///
#if !defined(__GNUC__)
#define fopen64 std::fopen
#endif
#ifdef _MSC_VER
// NOTE: sprintf_s is not equivalent to snprintf,
// they are equivalent when success, which is sufficient for our case
#define snprintf sprintf_s
#define vsnprintf vsprintf_s
#else
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#pragma message ("Warning: FILE OFFSET BITS defined to be 32 bit")
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 std::fopen
#endif

extern "C" {
#include <sys/types.h>
}
#endif

#ifdef _MSC_VER
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef __int64 int64_t;
#else
#include <inttypes.h>
#endif
#include <vector>
#include <string>

/*! \brief namespace for dmlc */
namespace dmlc {
/*!
 * \brief safely get the beginning address of a vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline T *BeginPtr(std::vector<T> &vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*! \brief get the beginning address of a vector */
template<typename T>
inline const T *BeginPtr(const std::vector<T> &vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
inline char* BeginPtr(std::string &str) {
  if (str.length() == 0) return NULL;
  return &str[0];
}
inline const char* BeginPtr(const std::string &str) {
  if (str.length() == 0) return NULL;
  return &str[0];
}
}  // namespace dmlc
#endif  // DMLC_BASE_H_
