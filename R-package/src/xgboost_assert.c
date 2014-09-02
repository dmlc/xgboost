#include <stdio.h>
#include <stdarg.h>
#include <Rinternals.h>

// implements error handling
void XGBoostAssert_R(int exp, const char *fmt, ...) {
  char buf[1024];
  if (exp == 0) {
    va_list args;  
    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    va_end(args);
    error("AssertError:%s\n", buf);
  }  
}
void XGBoostCheck_R(int exp, const char *fmt, ...) {
  char buf[1024];
  if (exp == 0) {
    va_list args;  
    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    va_end(args);
    error("%s\n", buf);
  }
}
int XGBoostSPrintf_R(char *buf, size_t size, const char *fmt, ...) {
  int ret;
  va_list args;  
  va_start(args, fmt);
  ret = vsnprintf(buf, size, fmt, args);
  va_end(args);
  return ret;
}
