/**
 * Copyright 2021-2026, XGBoost Contributors
 */
#ifndef XGBOOST_USE_CUDA

#include <jni.h>

#include "../../../../src/c_api/c_api_error.h"
#include "../../../../src/common/common.h"
#include "xgboost4j.h"

namespace xgboost::jni {
int QdmFromCallback(JNIEnv *, jobject, jlongArray, char const *, bool, jlongArray) {
  API_BEGIN();
  common::AssertGPUSupport();
  API_END();
}
}  // namespace xgboost::jni

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CudaSetDevice
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CudaSetDevice(JNIEnv *, jclass,
                                                                            jint) {
  API_BEGIN();
  xgboost::common::AssertGPUSupport();
  API_END();
}

#endif  // XGBOOST_USE_CUDA
