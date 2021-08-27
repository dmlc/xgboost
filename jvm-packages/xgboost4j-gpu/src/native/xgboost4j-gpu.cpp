#include <xgboost/c_api.h>
#include "./xgboost4j-gpu.h"
#include "jvm_utils.h"

namespace xgboost {
namespace jni {
    jint XGDeviceQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass jcls,
                                                       jobject jiter,
                                                       jfloat jmissing,
                                                       jint jmax_bin, jint jnthread,
                                                       jlongArray jout);
}
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBGetLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI_XGBGpuGetLastError
  (JNIEnv *jenv, jclass jcls) {
  jstring jresult = 0;
  const char* result = XGBGetLastError();
  if (result != NULL) {
    jresult = jenv->NewStringUTF(result);
  }
  return jresult;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDeviceQuantileDMatrixCreateFromCallback
 * Signature: (Ljava/util/Iterator;FII[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI_XGDeviceQuantileDMatrixCreateFromCallback
    (JNIEnv *jenv, jclass jcls, jobject jiter, jfloat jmissing, jint jmax_bin,
     jint jnthread, jlongArray jout) {
  return xgboost::jni::XGDeviceQuantileDMatrixCreateFromCallbackImpl(
      jenv, jcls, jiter, jmissing, jmax_bin, jnthread, jout);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSetInfoFromInterface
 * Signature: (JLjava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI_XGDMatrixSetInfoFromInterface
    (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jstring jjson_columns) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  const char* field = jenv->GetStringUTFChars(jfield, 0);
  const char* cjson_columns = jenv->GetStringUTFChars(jjson_columns, 0);

  int ret = XGDMatrixSetInfoFromInterface(handle, field, cjson_columns);
  JVM_CHECK_CALL(ret);
  //release
  if (field) jenv->ReleaseStringUTFChars(jfield, field);
  if (cjson_columns) jenv->ReleaseStringUTFChars(jjson_columns, cjson_columns);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixCreateFromArrayInterfaceColumns
 * Signature: (Ljava/lang/String;FI[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI_XGDMatrixCreateFromArrayInterfaceColumns
  (JNIEnv *jenv, jclass jcls, jstring jjson_columns, jfloat jmissing, jint jnthread, jlongArray jout) {
  DMatrixHandle result;
  const char* cjson_columns = jenv->GetStringUTFChars(jjson_columns, nullptr);
  int ret = XGDMatrixCreateFromArrayInterfaceColumns(
    cjson_columns, jmissing, jnthread, &result);
  JVM_CHECK_CALL(ret);
  if (cjson_columns) {
    jenv->ReleaseStringUTFChars(jjson_columns, cjson_columns);
  }

  setHandle(jenv, jout, result);
  return ret;
}
