/**
 * Copyright 2021-2024, XGBoost Contributors
 */
#ifndef XGBOOST_USE_CUDA

#include <jni.h>

#include "../../../../src/c_api/c_api_error.h"
#include "../../../../src/common/common.h"

namespace xgboost::jni {
XGB_DLL int XGQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass jcls, jobject jdata_iter,
                                                    jlongArray jref, char const *config,
                                                    jlongArray jout) {
  API_BEGIN();
  common::AssertGPUSupport();
  API_END();
}
}  // namespace xgboost::jni
#endif  // XGBOOST_USE_CUDA
