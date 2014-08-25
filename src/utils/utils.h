#ifndef XGBOOST_UTILS_UTILS_H_
#define XGBOOST_UTILS_UTILS_H_
/*!
 * \file utils.h
 * \brief simple utils to support the code
 * \author Tianqi Chen
 */
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#ifdef _MSC_VER
#define fopen64 fopen
// temporal solution for MSVC
inline int snprintf(char *ptr, size_t sz, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int ret = vsprintf(ptr, fmt, args);
  va_end(args);
  return ret;
}
#else
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#warning "FILE OFFSET BITS defined to be 32 bit"
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 fopen
#endif

#define _FILE_OFFSET_BITS 64
extern "C" {
#include <sys/types.h>
};
#endif

#ifdef _MSC_VER
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
typedef long int64_t;
#else
#include <inttypes.h>
#endif

namespace xgboost {
/*! \brief namespace for helper utils of the project */
namespace utils {

/*! \brief assert an condition is true, use this to handle debug information */
inline void Assert(bool exp, const char *fmt, ...) {
  if (!exp) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "AssertError:");
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(-1);
  }
}

/*!\brief same as assert, but this is intended to be used as message for user*/
inline void Check(bool exp, const char *fmt, ...) {
  if (!exp) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(-1);
  }
}

/*! \brief report error message, same as check */
inline void Error(const char *fmt, ...) {
  {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(-1);
  }
}

/*! \brief replace fopen, report error when the file open fails */
inline FILE *FopenCheck(const char *fname, const char *flag) {
  FILE *fp = fopen64(fname, flag);
  Check(fp != NULL, "can not open file \"%s\"\n", fname);
  return fp;
}

}  // namespace utils
}  // namespace xgboost
#endif  // XGBOOST_UTILS_UTILS_H_
