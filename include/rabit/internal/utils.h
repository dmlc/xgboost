/*!
 *  Copyright (c) 2014 by Contributors
 * \file utils.h
 * \brief simple utils to support the code
 * \author Tianqi Chen
 */
#ifndef RABIT_INTERNAL_UTILS_H_
#define RABIT_INTERNAL_UTILS_H_
#define _CRT_SECURE_NO_WARNINGS
#include <string.h>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "dmlc/io.h"

#ifndef RABIT_STRICT_CXX98_
#include <cstdarg>
#endif  // RABIT_STRICT_CXX98_

#if !defined(__GNUC__) || defined(__FreeBSD__)
#define fopen64 std::fopen
#endif  // !defined(__GNUC__) || defined(__FreeBSD__)

#ifdef _MSC_VER
// NOTE: sprintf_s is not equivalent to snprintf,
// they are equivalent when success, which is sufficient for our case
#define snprintf sprintf_s
#define vsnprintf vsprintf_s

#else

#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#pragma message("Warning: FILE OFFSET BITS defined to be 32 bit")
#endif  // _FILE_OFFSET_BITS == 32
#endif  // _FILE_OFFSET_BITS

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 std::fopen
#endif  // __APPLE__

extern "C" {
#include <sys/types.h>
}
#endif  // _MSC_VER

#ifdef _MSC_VER
typedef unsigned char uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef __int64 int64_t;
#else
#include <inttypes.h>
#endif  // _MSC_VER

namespace rabit {
/*! \brief namespace for helper utils of the project */
namespace utils {

/*! \brief error message buffer length */
const int kPrintBuffer = 1 << 12;

/*! \brief we may want to keep the process alive when there are multiple workers
 * co-locate in the same process */
extern bool STOP_PROCESS_ON_ERROR;

/* \brief Case-insensitive string comparison */
inline int CompareStringsCaseInsensitive(const char* s1, const char* s2) {
#ifdef _MSC_VER
  return _stricmp(s1, s2);
#else  // _MSC_VER
  return strcasecmp(s1, s2);
#endif  // _MSC_VER
}

/* \brief parse config string too bool*/
inline bool StringToBool(const char* s) {
  return CompareStringsCaseInsensitive(s, "true") == 0 || atoi(s) != 0;
}

#ifndef RABIT_CUSTOMIZE_MSG_
/*!
 * \brief handling of Assert error, caused by inappropriate input
 * \param msg error message
 */
inline void HandleAssertError(const char *msg) {
  if (STOP_PROCESS_ON_ERROR) {
    fprintf(stderr, "AssertError:%s, shutting down process\n", msg);
    exit(-1);
  } else {
    fprintf(stderr, "AssertError:%s, rabit is configured to keep process running\n", msg);
    throw dmlc::Error(msg);
  }
}
/*!
 * \brief handling of Check error, caused by inappropriate input
 * \param msg error message
 */
inline void HandleCheckError(const char *msg) {
  if (STOP_PROCESS_ON_ERROR) {
    fprintf(stderr, "%s, shutting down process\n", msg);
    exit(-1);
  } else {
    fprintf(stderr, "%s, rabit is configured to keep process running\n", msg);
    throw dmlc::Error(msg);
  }
}
inline void HandlePrint(const char *msg) {
  printf("%s", msg);
}

inline void HandleLogInfo(const char *fmt, ...) {
  std::string msg(kPrintBuffer, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(&msg[0], kPrintBuffer, fmt, args);
  va_end(args);
  fprintf(stdout, "%s", msg.c_str());
  fflush(stdout);
}
#else
#ifndef RABIT_STRICT_CXX98_
// include declarations, some one must implement this
void HandleAssertError(const char *msg);
void HandleCheckError(const char *msg);
void HandlePrint(const char *msg);
#endif  // RABIT_STRICT_CXX98_
#endif  // RABIT_CUSTOMIZE_MSG_
#ifdef RABIT_STRICT_CXX98_
// these function pointers are to be assigned
extern "C" void (*Printf)(const char *fmt, ...);
extern "C" int (*SPrintf)(char *buf, size_t size, const char *fmt, ...);
extern "C" void (*Assert)(int exp, const char *fmt, ...);
extern "C" void (*Check)(int exp, const char *fmt, ...);
extern "C" void (*Error)(const char *fmt, ...);
#else
/*! \brief printf, prints messages to the console */
inline void Printf(const char *fmt, ...) {
  std::string msg(kPrintBuffer, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(&msg[0], kPrintBuffer, fmt, args);
  va_end(args);
  HandlePrint(msg.c_str());
}
/*! \brief portable version of snprintf */
inline int SPrintf(char *buf, size_t size, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vsnprintf(buf, size, fmt, args);
  va_end(args);
  return ret;
}

/*! \brief assert a condition is true, use this to handle debug information */
inline void Assert(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleAssertError(msg.c_str());
  }
}

/*!\brief same as assert, but this is intended to be used as a message for users */
inline void Check(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}

/*! \brief report error message, same as check */
inline void Error(const char *fmt, ...) {
  {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    HandleCheckError(msg.c_str());
  }
}
#endif  // RABIT_STRICT_CXX98_

/*! \brief replace fopen, report error when the file open fails */
inline std::FILE *FopenCheck(const char *fname, const char *flag) {
  std::FILE *fp = fopen64(fname, flag);
  Check(fp != NULL, "can not open file \"%s\"\n", fname);
  return fp;
}
}  // namespace utils
// easy utils that can be directly accessed in xgboost
/*! \brief get the beginning address of a vector */
template<typename T>
inline T *BeginPtr(std::vector<T> &vec) {  // NOLINT(*)
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*! \brief get the beginning address of a vector */
template<typename T>
inline const T *BeginPtr(const std::vector<T> &vec) {  // NOLINT(*)
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
inline char* BeginPtr(std::string &str) {  // NOLINT(*)
  if (str.length() == 0) return NULL;
  return &str[0];
}
inline const char* BeginPtr(const std::string &str) {
  if (str.length() == 0) return NULL;
  return &str[0];
}
}  // namespace rabit
#endif  // RABIT_INTERNAL_UTILS_H_
