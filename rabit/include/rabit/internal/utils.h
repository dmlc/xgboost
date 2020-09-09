/*!
 *  Copyright (c) 2014 by Contributors
 * \file utils.h
 * \brief simple utils to support the code
 * \author Tianqi Chen
 */
#ifndef RABIT_INTERNAL_UTILS_H_
#define RABIT_INTERNAL_UTILS_H_

#include <rabit/base.h>
#include <cstring>
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

#include <cinttypes>

namespace rabit {
/*! \brief namespace for helper utils of the project */
namespace utils {

/*! \brief error message buffer length */
const int kPrintBuffer = 1 << 12;

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

/*! \brief assert a condition is true, use this to handle debug information */
inline void Assert(bool exp, const char *fmt, ...) {
  if (!exp) {
    std::string msg(kPrintBuffer, '\0');
    va_list args;
    va_start(args, fmt);
    vsnprintf(&msg[0], kPrintBuffer, fmt, args);
    va_end(args);
    LOG(FATAL) << msg;
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
    LOG(FATAL) << msg;
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
    LOG(FATAL) << msg;
  }
}
}  // namespace utils
}  // namespace rabit
#endif  // RABIT_INTERNAL_UTILS_H_
