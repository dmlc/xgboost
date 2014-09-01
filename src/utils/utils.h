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
#include <string>
#include <cstdlib>
#ifdef _MSC_VER
#define fopen64 fopen
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
#define fopen64 fopen
#endif

#define _FILE_OFFSET_BITS 64
extern "C" {
#include <sys/types.h>
}
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
/*! \brief error message buffer length */
const int kPrintBuffer = 1 << 12;

#ifndef XGBOOST_CUSTOMIZE_MSG_
/*! 
 * \brief handling of Assert error, caused by in-apropriate input
 * \param msg error message 
 */
inline void HandleAssertError(const char *msg) {
  fprintf(stderr, "AssertError:%s\n", msg);
  exit(-1);
}
/*! 
 * \brief handling of Check error, caused by in-apropriate input
 * \param msg error message 
 */
inline void HandleCheckError(const char *msg) {
  fprintf(stderr, "%s\n", msg);
  exit(-1);
}
inline void HandlePrint(const char *msg) {
  printf("%s", msg);
}
#else
// include declarations, some one must implement this
void HandleAssertError(const char *msg);
void HandleCheckError(const char *msg);
void HandlePrint(const char *msg);
#endif

/*! \brief printf, print message to the console */
inline void Printf(const char *fmt, ...) {
  std::string msg(kPrintBuffer, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(&msg[0], kPrintBuffer, fmt, args);
  va_end(args);
  HandlePrint(msg.c_str());
}

/*! \brief assert an condition is true, use this to handle debug information */
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

/*!\brief same as assert, but this is intended to be used as message for user*/
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

/*! \brief replace fopen, report error when the file open fails */
inline FILE *FopenCheck(const char *fname, const char *flag) {
  FILE *fp = fopen64(fname, flag);
  Check(fp != NULL, "can not open file \"%s\"\n", fname);
  return fp;
}

}  // namespace utils
}  // namespace xgboost
#endif  // XGBOOST_UTILS_UTILS_H_
