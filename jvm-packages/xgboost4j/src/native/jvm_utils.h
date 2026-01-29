/**
 *  Copyright 2014-2025, XGBoost Contributors
 */
#ifndef XGBOOST_NATIVE_JVM_UTILS_H_
#define XGBOOST_NATIVE_JVM_UTILS_H_

#include <jni.h>

#include "xgboost/logging.h"  // for Check

#define JVM_CHECK_CALL(__expr) \
  {                            \
    int __errcode = (__expr);  \
    if (__errcode != 0) {      \
      return __errcode;        \
    }                          \
  }

JavaVM *&GlobalJvm();
void setHandle(JNIEnv *jenv, jlongArray jhandle, void *handle);

template <typename T>
T CheckJvmCall(T const &v, JNIEnv *jenv) {
  if (!v) {
    CHECK(jenv->ExceptionOccurred());
    jenv->ExceptionDescribe();
  }
  return v;
}

#endif  // XGBOOST_NATIVE_JVM_UTILS_H_
