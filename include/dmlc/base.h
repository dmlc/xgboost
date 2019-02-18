/*!
 *  Copyright (c) 2015 by Contributors
 * \file base.h
 * \brief defines configuration macros
 */
#ifndef DMLC_BASE_H_
#define DMLC_BASE_H_

/*! \brief whether use glog for logging */
#ifndef DMLC_USE_GLOG
#define DMLC_USE_GLOG 0
#endif  // DMLC_USE_GLOG

/*!
 * \brief whether throw dmlc::Error instead of
 *  directly calling abort when FATAL error occured
 *  NOTE: this may still not be perfect.
 *  do not use FATAL and CHECK in destructors
 */
#ifndef DMLC_LOG_FATAL_THROW
#define DMLC_LOG_FATAL_THROW 1
#endif  // DMLC_LOG_FATAL_THROW

/*!
 * \brief whether always log a message before throw
 * This can help identify the error that cannot be catched.
 */
#ifndef DMLC_LOG_BEFORE_THROW
#define DMLC_LOG_BEFORE_THROW 1
#endif  // DMLC_LOG_BEFORE_THROW

/*!
 * \brief Whether to use customized logger,
 * whose output can be decided by other libraries.
 */
#ifndef DMLC_LOG_CUSTOMIZE
#define DMLC_LOG_CUSTOMIZE 0
#endif  // DMLC_LOG_CUSTOMIZE

/*!
 * \brief Wheter to print stack trace for fatal error,
 * enabled on linux when using gcc.
 */
#if (!defined(DMLC_LOG_STACK_TRACE) \
     && defined(__GNUC__) && !defined(__MINGW32__) \
     && !defined(__sun) && !defined(__SVR4))
#define DMLC_LOG_STACK_TRACE 1
#endif  // guards

/*! \brief whether compile with hdfs support */
#ifndef DMLC_USE_HDFS
#define DMLC_USE_HDFS 0
#endif  // DMLC_USE_HDFS

/*! \brief whether compile with s3 support */
#ifndef DMLC_USE_S3
#define DMLC_USE_S3 0
#endif  // DMLC_USE_S3

/*! \brief whether or not use parameter server */
#ifndef DMLC_USE_PS
#define DMLC_USE_PS 0
#endif  // DMLC_USE_PS

/*! \brief whether or not use c++11 support */
#ifndef DMLC_USE_CXX11
#if defined(__GXX_EXPERIMENTAL_CXX0X__) || defined(_MSC_VER)
#define DMLC_USE_CXX11 1
#else
#define DMLC_USE_CXX11 (__cplusplus >= 201103L)
#endif  // defined(__GXX_EXPERIMENTAL_CXX0X__) || defined(_MSC_VER)
#endif  // DMLC_USE_CXX11

/*! \brief strict CXX11 support */
#ifndef DMLC_STRICT_CXX11
#if defined(_MSC_VER)
#define DMLC_STRICT_CXX11 1
#else
#define DMLC_STRICT_CXX11 (__cplusplus >= 201103L)
#endif  // defined(_MSC_VER)
#endif  // DMLC_STRICT_CXX11

/*! \brief Whether cxx11 thread local is supported */
#ifndef DMLC_CXX11_THREAD_LOCAL
#if defined(_MSC_VER)
#if (_MSC_VER >= 1900)
#define DMLC_CXX11_THREAD_LOCAL 1
#else
#define DMLC_CXX11_THREAD_LOCAL 0
#endif  // (_MSC_VER >= 1900)
#else
#define DMLC_CXX11_THREAD_LOCAL (__cplusplus >= 201103L)
#endif  // defined(_MSC_VER)
#endif  // DMLC_CXX11_THREAD_LOCAL


/*! \brief whether RTTI is enabled */
#ifndef DMLC_ENABLE_RTTI
#define DMLC_ENABLE_RTTI 1
#endif  // DMLC_ENABLE_RTTI

/// check if g++ is before 4.6
#if DMLC_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)
#if __GNUC__ == 4 && __GNUC_MINOR__ < 6
#pragma message("Will need g++-4.6 or higher to compile all"           \
                "the features in dmlc-core, "                           \
                "compile without c++0x, some features may be disabled")
#undef DMLC_USE_CXX11
#define DMLC_USE_CXX11 0
#endif  // __GNUC__ == 4 && __GNUC_MINOR__ < 6
#endif  // DMLC_USE_CXX11 && defined(__GNUC__) && !defined(__clang_version__)


/*!
 * \brief Enable std::thread related modules,
 *  Used to disable some module in mingw compile.
 */
#ifndef DMLC_ENABLE_STD_THREAD
#define DMLC_ENABLE_STD_THREAD DMLC_USE_CXX11
#endif  // DMLC_ENABLE_STD_THREAD

/*! \brief whether enable regex support, actually need g++-4.9 or higher*/
#ifndef DMLC_USE_REGEX
#define DMLC_USE_REGEX DMLC_STRICT_CXX11
#endif  // DMLC_USE_REGEX

/*! \brief helper macro to supress unused warning */
#if defined(__GNUC__)
#define DMLC_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define DMLC_ATTRIBUTE_UNUSED
#endif  // defined(__GNUC__)

/*! \brief helper macro to generate string concat */
#define DMLC_STR_CONCAT_(__x, __y) __x##__y
#define DMLC_STR_CONCAT(__x, __y) DMLC_STR_CONCAT_(__x, __y)

/*!
 * \brief Disable copy constructor and assignment operator.
 *
 * If C++11 is supported, both copy and move constructors and
 * assignment operators are deleted explicitly. Otherwise, they are
 * only declared but not implemented. Place this macro in private
 * section if C++11 is not available.
 */
#ifndef DISALLOW_COPY_AND_ASSIGN
#  if DMLC_USE_CXX11
#    define DISALLOW_COPY_AND_ASSIGN(T) \
       T(T const&) = delete; \
       T(T&&) = delete; \
       T& operator=(T const&) = delete; \
       T& operator=(T&&) = delete
#  else
#    define DISALLOW_COPY_AND_ASSIGN(T) \
       T(T const&); \
       T& operator=(T const&)
#  endif  // DMLC_USE_CXX11
#endif  // DISALLOW_COPY_AND_ASSIGN

///
/// code block to handle optionally loading
///
#if !defined(__GNUC__)
#define fopen64 std::fopen
#endif  // !defined(__GNUC__)

#if (defined __MINGW32__) && !(defined __MINGW64__)
#define fopen64 std::fopen
#endif  // (defined __MINGW32__) && !(defined __MINGW64__)

#ifdef _MSC_VER

#  if _MSC_VER < 1900
// NOTE: sprintf_s is not equivalent to snprintf,
// they are equivalent when success, which is sufficient for our case
#  define snprintf sprintf_s
#  define vsnprintf vsprintf_s
#  endif  // _MSC_VER < 1900

#else

#  ifdef _FILE_OFFSET_BITS
#    if _FILE_OFFSET_BITS == 32
#      pragma message("Warning: FILE OFFSET BITS defined to be 32 bit")
#    endif  // _FILE_OFFSET_BITS == 32
#  endif  // _FILE_OFFSET_BITS

#  ifdef __APPLE__
#    define off64_t off_t
#    define fopen64 std::fopen
#  endif  // __APPLE__

extern "C" {
#include <sys/types.h>
}
#endif  // _MSC_VER

#ifdef _MSC_VER
//! \cond Doxygen_Suppress
typedef signed char int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
//! \endcond
#else
#include <inttypes.h>
#endif  // _MSC_VER

#include <string>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER < 1900
#define noexcept_true throw ()
#define noexcept_false
#define noexcept(a) noexcept_##a
#endif  // defined(_MSC_VER) && _MSC_VER < 1900

#if DMLC_USE_CXX11
#define DMLC_THROW_EXCEPTION noexcept(false)
#define DMLC_NO_EXCEPTION  noexcept(true)
#else
#define DMLC_THROW_EXCEPTION
#define DMLC_NO_EXCEPTION
#endif  // DMLC_USE_CXX11

/*! \brief namespace for dmlc */
namespace dmlc {
/*!
 * \brief safely get the beginning address of a vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline T *BeginPtr(std::vector<T> &vec) {  // NOLINT(*)
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a const vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline const T *BeginPtr(const std::vector<T> &vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a string
 * \param str input string
 * \return beginning address of a string
 */
inline char* BeginPtr(std::string &str) {  // NOLINT(*)
  if (str.length() == 0) return NULL;
  return &str[0];
}
/*!
 * \brief get the beginning address of a const string
 * \param str input string
 * \return beginning address of a string
 */
inline const char* BeginPtr(const std::string &str) {
  if (str.length() == 0) return NULL;
  return &str[0];
}
}  // namespace dmlc

#if defined(_MSC_VER) && _MSC_VER < 1900
#define constexpr const
#define alignof __alignof
#endif  // defined(_MSC_VER) && _MSC_VER < 1900

#endif  // DMLC_BASE_H_
