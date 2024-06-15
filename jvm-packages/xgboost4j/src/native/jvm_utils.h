#ifndef JVM_UTILS_H_
#define JVM_UTILS_H_

#define JVM_CHECK_CALL(__expr)                                                 \
  {                                                                            \
    int __errcode = (__expr);                                                  \
    if (__errcode != 0) {                                                      \
      return __errcode;                                                        \
    }                                                                          \
  }

JavaVM*& GlobalJvm();
void setHandle(JNIEnv *jenv, jlongArray jhandle, void* handle);

#endif  // JVM_UTILS_H_
