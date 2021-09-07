//
// Created by bobwang on 2021/9/8.
//

#ifndef XGBOOST_USE_CUDA

namespace xgboost {
namespace jni {
XGB_DLL int XGDeviceQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass jcls,
                                                          jobject jiter,
                                                          jfloat jmissing,
                                                          jint jmax_bin, jint jnthread,
                                                          jlongArray jout) {
  API_BEGIN();
  common::AssertGPUSupport();
  API_END();
}
} // namespace jni
} // namespace xgboost
#endif
