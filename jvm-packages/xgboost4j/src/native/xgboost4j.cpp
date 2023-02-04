/**
  Copyright (c) 2014-2023 by Contributors
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "./xgboost4j.h"

#include <rabit/c_api.h>
#include <xgboost/base.h>
#include <xgboost/c_api.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "../../../src/c_api/c_api_utils.h"

#define JVM_CHECK_CALL(__expr)                                                 \
  {                                                                            \
    int __errcode = (__expr);                                                  \
    if (__errcode != 0) {                                                      \
      return __errcode;                                                        \
    }                                                                          \
  }

// helper functions
// set handle
void setHandle(JNIEnv *jenv, jlongArray jhandle, void* handle) {
#ifdef __APPLE__
  jlong out = (long) handle;
#else
  int64_t out = (int64_t) handle;
#endif
  jenv->SetLongArrayRegion(jhandle, 0, 1, &out);
}

JavaVM*& GlobalJvm() {
  static JavaVM* vm;
  return vm;
}

// overrides JNI on load
jint JNI_OnLoad(JavaVM *vm, void *reserved) {
  GlobalJvm() = vm;
  return JNI_VERSION_1_6;
}

XGB_EXTERN_C int XGBoost4jCallbackDataIterNext(
    DataIterHandle data_handle,
    XGBCallbackSetData* set_function,
    DataHolderHandle set_function_handle) {
  jobject jiter = static_cast<jobject>(data_handle);
  JNIEnv* jenv;
  int jni_status = GlobalJvm()->GetEnv((void **)&jenv, JNI_VERSION_1_6);
  if (jni_status == JNI_EDETACHED) {
    GlobalJvm()->AttachCurrentThread(reinterpret_cast<void **>(&jenv), nullptr);
  } else {
    CHECK(jni_status == JNI_OK);
  }
  try {
    jclass iterClass = jenv->FindClass("java/util/Iterator");
    jmethodID hasNext = jenv->GetMethodID(iterClass,
                                          "hasNext", "()Z");
    jmethodID next = jenv->GetMethodID(iterClass,
                                       "next", "()Ljava/lang/Object;");
    int ret_value;
    if (jenv->CallBooleanMethod(jiter, hasNext)) {
      ret_value = 1;
      jobject batch = jenv->CallObjectMethod(jiter, next);
      if (batch == nullptr) {
        CHECK(jenv->ExceptionOccurred());
        jenv->ExceptionDescribe();
        return -1;
      }

      jclass batchClass = jenv->GetObjectClass(batch);
      jlongArray joffset = (jlongArray)jenv->GetObjectField(
          batch, jenv->GetFieldID(batchClass, "rowOffset", "[J"));
      jfloatArray jlabel = (jfloatArray)jenv->GetObjectField(
          batch, jenv->GetFieldID(batchClass, "label", "[F"));
      jfloatArray jweight = (jfloatArray)jenv->GetObjectField(
          batch, jenv->GetFieldID(batchClass, "weight", "[F"));
      jintArray jindex = (jintArray)jenv->GetObjectField(
          batch, jenv->GetFieldID(batchClass, "featureIndex", "[I"));
      jfloatArray jvalue = (jfloatArray)jenv->GetObjectField(
          batch, jenv->GetFieldID(batchClass, "featureValue", "[F"));
      jint jcols = jenv->GetIntField(
          batch, jenv->GetFieldID(batchClass, "featureCols", "I"));
      XGBoostBatchCSR cbatch;
      cbatch.size = jenv->GetArrayLength(joffset) - 1;
      cbatch.columns = jcols;
      cbatch.offset = reinterpret_cast<jlong *>(
          jenv->GetLongArrayElements(joffset, 0));
      if (jlabel != nullptr) {
        cbatch.label = jenv->GetFloatArrayElements(jlabel, 0);
        CHECK_EQ(jenv->GetArrayLength(jlabel), static_cast<long>(cbatch.size))
            << "batch.label.length must equal batch.numRows()";
      } else {
        cbatch.label = nullptr;
      }
      if (jweight != nullptr) {
        cbatch.weight = jenv->GetFloatArrayElements(jweight, 0);
        CHECK_EQ(jenv->GetArrayLength(jweight), static_cast<long>(cbatch.size))
            << "batch.weight.length must equal batch.numRows()";
      } else {
        cbatch.weight = nullptr;
      }
      long max_elem = cbatch.offset[cbatch.size];
      cbatch.index = (int*) jenv->GetIntArrayElements(jindex, 0);
      cbatch.value = jenv->GetFloatArrayElements(jvalue, 0);

      CHECK_EQ(jenv->GetArrayLength(jindex), max_elem)
          << "batch.index.length must equal batch.offset.back()";
      CHECK_EQ(jenv->GetArrayLength(jvalue), max_elem)
          << "batch.index.length must equal batch.offset.back()";
      // cbatch is ready
      CHECK_EQ((*set_function)(set_function_handle, cbatch), 0)
          << XGBGetLastError();
      // release the elements.
      jenv->ReleaseLongArrayElements(
          joffset, reinterpret_cast<jlong *>(cbatch.offset), 0);
      jenv->DeleteLocalRef(joffset);
      if (jlabel != nullptr) {
        jenv->ReleaseFloatArrayElements(jlabel, cbatch.label, 0);
        jenv->DeleteLocalRef(jlabel);
      }
      if (jweight != nullptr) {
        jenv->ReleaseFloatArrayElements(jweight, cbatch.weight, 0);
        jenv->DeleteLocalRef(jweight);
      }
      jenv->ReleaseIntArrayElements(jindex, (jint*) cbatch.index, 0);
      jenv->DeleteLocalRef(jindex);
      jenv->ReleaseFloatArrayElements(jvalue, cbatch.value, 0);
      jenv->DeleteLocalRef(jvalue);
      jenv->DeleteLocalRef(batch);
      jenv->DeleteLocalRef(batchClass);
      ret_value = 1;
    } else {
      ret_value = 0;
    }
    jenv->DeleteLocalRef(iterClass);
    // only detach if it is a async call.
    if (jni_status == JNI_EDETACHED) {
      GlobalJvm()->DetachCurrentThread();
    }
    return ret_value;
  } catch(dmlc::Error const& e) {
    // only detach if it is a async call.
    if (jni_status == JNI_EDETACHED) {
      GlobalJvm()->DetachCurrentThread();
    }
    LOG(FATAL) << e.what();
    return -1;
  }
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBGetLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBGetLastError
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
 * Method:    XGDMatrixCreateFromDataIter
 * Signature: (Ljava/util/Iterator;Ljava/lang/String;[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromDataIter
  (JNIEnv *jenv, jclass jcls, jobject jiter, jstring jcache_info, jlongArray jout) {
  DMatrixHandle result;
  const char* cache_info = nullptr;
  if (jcache_info != nullptr) {
    cache_info = jenv->GetStringUTFChars(jcache_info, 0);
  }
  int ret = XGDMatrixCreateFromDataIter(
      jiter, XGBoost4jCallbackDataIterNext, cache_info, &result);
  JVM_CHECK_CALL(ret);
  if (cache_info) {
    jenv->ReleaseStringUTFChars(jcache_info, cache_info);
  }
  setHandle(jenv, jout, result);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixCreateFromFile
 * Signature: (Ljava/lang/String;I[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromFile
  (JNIEnv *jenv, jclass jcls, jstring jfname, jint jsilent, jlongArray jout) {
  DMatrixHandle result;
  const char* fname = jenv->GetStringUTFChars(jfname, 0);
  int ret = XGDMatrixCreateFromFile(fname, jsilent, &result);
  JVM_CHECK_CALL(ret);
  if (fname) {
    jenv->ReleaseStringUTFChars(jfname, fname);
  }
  setHandle(jenv, jout, result);
  return ret;
}

namespace {
/**
 * \brief Create from sparse matrix.
 *
 * \param maker Indirect call to XGBoost C function for creating CSC and CSR.
 *
 * \return Status
 */
template <typename Fn>
jint MakeJVMSparseInput(JNIEnv *jenv, jlongArray jindptr, jintArray jindices, jfloatArray jdata,
                        jfloat jmissing, jint jnthread, Fn &&maker, jlongArray jout) {
  DMatrixHandle result;

  jlong *indptr = jenv->GetLongArrayElements(jindptr, nullptr);
  jint *indices = jenv->GetIntArrayElements(jindices, nullptr);
  jfloat *data = jenv->GetFloatArrayElements(jdata, nullptr);
  bst_ulong nindptr = static_cast<bst_ulong>(jenv->GetArrayLength(jindptr));
  bst_ulong nelem = static_cast<bst_ulong>(jenv->GetArrayLength(jdata));

  std::string sindptr, sindices, sdata;
  CHECK_EQ(indptr[nindptr - 1], nelem);
  using IndPtrT = std::conditional_t<std::is_convertible<jlong *, long *>::value, long, long long>;
  using IndT =
      std::conditional_t<std::is_convertible<jint *, std::int32_t *>::value, std::int32_t, long>;
  xgboost::detail::MakeSparseFromPtr(
      static_cast<IndPtrT const *>(indptr), static_cast<IndT const *>(indices),
      static_cast<float const *>(data), nindptr, &sindptr, &sindices, &sdata);

  xgboost::Json jconfig{xgboost::Object{}};
  auto missing = static_cast<float>(jmissing);
  auto n_threads = static_cast<std::int32_t>(jnthread);
  // Construct configuration
  jconfig["nthread"] = xgboost::Integer{n_threads};
  jconfig["missing"] = xgboost::Number{missing};
  std::string config;
  xgboost::Json::Dump(jconfig, &config);

  jint ret = maker(sindptr.c_str(), sindices.c_str(), sdata.c_str(), config.c_str(), &result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);

  // Release
  jenv->ReleaseLongArrayElements(jindptr, indptr, 0);
  jenv->ReleaseIntArrayElements(jindices, indices, 0);
  jenv->ReleaseFloatArrayElements(jdata, data, 0);
  return ret;
}
}  // anonymous namespace

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixCreateFromCSR
 * Signature: ([J[I[FIFI[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromCSR(
    JNIEnv *jenv, jclass jcls, jlongArray jindptr, jintArray jindices, jfloatArray jdata, jint jcol,
    jfloat jmissing, jint jnthread, jlongArray jout) {
  using CSTR = char const *;
  return MakeJVMSparseInput(
      jenv, jindptr, jindices, jdata, jmissing, jnthread,
      [&](CSTR sindptr, CSTR sindices, CSTR sdata, CSTR sconfig, DMatrixHandle *result) {
        return XGDMatrixCreateFromCSR(sindptr, sindices, sdata, static_cast<std::int32_t>(jcol),
                                      sconfig, result);
      },
      jout);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixCreateFromCSC
 * Signature: ([J[I[FIFI[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromCSC(
    JNIEnv *jenv, jclass jcls, jlongArray jindptr, jintArray jindices, jfloatArray jdata, jint jrow,
    jfloat jmissing, jint jnthread, jlongArray jout) {
  using CSTR = char const *;
  return MakeJVMSparseInput(
      jenv, jindptr, jindices, jdata, jmissing, jnthread,
      [&](CSTR sindptr, CSTR sindices, CSTR sdata, CSTR sconfig, DMatrixHandle *result) {
        return XGDMatrixCreateFromCSC(sindptr, sindices, sdata, static_cast<bst_ulong>(jrow),
                                      sconfig, result);
      },
      jout);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixCreateFromMatRef
 * Signature: (JIIF)J
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromMatRef
  (JNIEnv *jenv, jclass jcls, jlong jdataRef, jint jnrow, jint jncol, jfloat jmiss, jlongArray jout) {
  DMatrixHandle result;
  bst_ulong nrow = (bst_ulong)jnrow;
  bst_ulong ncol = (bst_ulong)jncol;
  jint ret = (jint) XGDMatrixCreateFromMat((float const *)jdataRef, nrow, ncol, jmiss, &result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
  return ret;
}


/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixCreateFromMat
 * Signature: ([FIIF)J
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromMat
  (JNIEnv *jenv, jclass jcls, jfloatArray jdata, jint jnrow, jint jncol, jfloat jmiss, jlongArray jout) {
  DMatrixHandle result;
  jfloat* data = jenv->GetFloatArrayElements(jdata, 0);
  bst_ulong nrow = (bst_ulong)jnrow;
  bst_ulong ncol = (bst_ulong)jncol;
  jint ret = (jint) XGDMatrixCreateFromMat((float const *)data, nrow, ncol, jmiss, &result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
  //release
  jenv->ReleaseFloatArrayElements(jdata, data, 0);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSliceDMatrix
 * Signature: (J[I)J
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSliceDMatrix
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jintArray jindexset, jlongArray jout) {
  DMatrixHandle result;
  DMatrixHandle handle = (DMatrixHandle) jhandle;

  jint* indexset = jenv->GetIntArrayElements(jindexset, 0);
  bst_ulong len = (bst_ulong)jenv->GetArrayLength(jindexset);

  // default to not allowing slicing with group ID specified -- feel free to add if necessary
  jint ret = (jint) XGDMatrixSliceDMatrixEx(handle, (int const *)indexset, len, &result, 0);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
  //release
  jenv->ReleaseIntArrayElements(jindexset, indexset, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixFree
 * Signature: (J)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixFree
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  int ret = XGDMatrixFree(handle);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSaveBinary
 * Signature: (JLjava/lang/String;I)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSaveBinary
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname, jint jsilent) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  const char* fname = jenv->GetStringUTFChars(jfname, 0);
  int ret = XGDMatrixSaveBinary(handle, fname, jsilent);
  JVM_CHECK_CALL(ret);
  if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSetFloatInfo
 * Signature: (JLjava/lang/String;[F)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSetFloatInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jfloatArray jarray) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  const char*  field = jenv->GetStringUTFChars(jfield, 0);

  jfloat* array = jenv->GetFloatArrayElements(jarray, NULL);
  bst_ulong len = (bst_ulong)jenv->GetArrayLength(jarray);
  int ret = XGDMatrixSetFloatInfo(handle, field, (float const *)array, len);
  JVM_CHECK_CALL(ret);
  //release
  if (field) jenv->ReleaseStringUTFChars(jfield, field);
  jenv->ReleaseFloatArrayElements(jarray, array, 0);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSetUIntInfo
 * Signature: (JLjava/lang/String;[I)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSetUIntInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jintArray jarray) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  const char*  field = jenv->GetStringUTFChars(jfield, 0);
  jint* array = jenv->GetIntArrayElements(jarray, NULL);
  bst_ulong len = (bst_ulong)jenv->GetArrayLength(jarray);
  int ret = XGDMatrixSetUIntInfo(handle, (char const *)field, (unsigned int const *)array, len);
  JVM_CHECK_CALL(ret);
  //release
  if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
  jenv->ReleaseIntArrayElements(jarray, array, 0);

  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixGetFloatInfo
 * Signature: (JLjava/lang/String;)[F
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixGetFloatInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jobjectArray jout) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  const char*  field = jenv->GetStringUTFChars(jfield, 0);
  bst_ulong len;
  float *result;
  int ret = XGDMatrixGetFloatInfo(handle, field, &len, (const float**) &result);
  JVM_CHECK_CALL(ret);
  if (field) jenv->ReleaseStringUTFChars(jfield, field);

  jsize jlen = (jsize) len;
  jfloatArray jarray = jenv->NewFloatArray(jlen);
  jenv->SetFloatArrayRegion(jarray, 0, jlen, (jfloat *) result);
  jenv->SetObjectArrayElement(jout, 0, (jobject) jarray);

  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixGetUIntInfo
 * Signature: (JLjava/lang/String;)[I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixGetUIntInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jobjectArray jout) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  const char*  field = jenv->GetStringUTFChars(jfield, 0);
  bst_ulong len;
  unsigned int *result;
  int ret = (jint) XGDMatrixGetUIntInfo(handle, field, &len, (const unsigned int **) &result);
  JVM_CHECK_CALL(ret);
  if (field) jenv->ReleaseStringUTFChars(jfield, field);

  jsize jlen = (jsize) len;
  jintArray jarray = jenv->NewIntArray(jlen);
  jenv->SetIntArrayRegion(jarray, 0, jlen, (jint *) result);
  jenv->SetObjectArrayElement(jout, 0, jarray);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixNumRow
 * Signature: (J)J
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixNumRow
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlongArray jout) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  bst_ulong result[1];
  int ret = (jint) XGDMatrixNumRow(handle, result);
  JVM_CHECK_CALL(ret);
  jenv->SetLongArrayRegion(jout, 0, 1, (const jlong *) result);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixNumNonMissing
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixNumNonMissing(
    JNIEnv *jenv, jclass, jlong jhandle, jlongArray jout) {
  DMatrixHandle handle = reinterpret_cast<DMatrixHandle>(jhandle);
  CHECK(handle);
  bst_ulong result[1];
  auto ret = static_cast<jint>(XGDMatrixNumNonMissing(handle, result));
  jlong jresult[1]{static_cast<jlong>(result[0])};
  jenv->SetLongArrayRegion(jout, 0, 1, jresult);
  JVM_CHECK_CALL(ret);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterCreate
 * Signature: ([J)J
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterCreate
  (JNIEnv *jenv, jclass jcls, jlongArray jhandles, jlongArray jout) {
  std::vector<DMatrixHandle> handles;
  if (jhandles != nullptr) {
    size_t len = jenv->GetArrayLength(jhandles);
    jlong *cjhandles = jenv->GetLongArrayElements(jhandles, 0);
    for (size_t i = 0; i < len; ++i) {
      handles.push_back((DMatrixHandle) cjhandles[i]);
    }
    jenv->ReleaseLongArrayElements(jhandles, cjhandles, 0);
  }
  BoosterHandle result;
  int ret = XGBoosterCreate(dmlc::BeginPtr(handles), handles.size(), &result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterFree
 * Signature: (J)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterFree
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
    BoosterHandle handle = (BoosterHandle) jhandle;
    return XGBoosterFree(handle);
}


/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSetParam
 * Signature: (JLjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSetParam
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jname, jstring jvalue) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  const char* name = jenv->GetStringUTFChars(jname, 0);
  const char* value = jenv->GetStringUTFChars(jvalue, 0);
  int ret = XGBoosterSetParam(handle, name, value);
  JVM_CHECK_CALL(ret);
  //release
  if (name) jenv->ReleaseStringUTFChars(jname, name);
  if (value) jenv->ReleaseStringUTFChars(jvalue, value);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterUpdateOneIter
 * Signature: (JIJ)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterUpdateOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jint jiter, jlong jdtrain) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  DMatrixHandle dtrain = (DMatrixHandle) jdtrain;
  return XGBoosterUpdateOneIter(handle, jiter, dtrain);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterBoostOneIter
 * Signature: (JJ[F[F)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterBoostOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jdtrain, jfloatArray jgrad, jfloatArray jhess) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  DMatrixHandle dtrain = (DMatrixHandle) jdtrain;
  jfloat* grad = jenv->GetFloatArrayElements(jgrad, 0);
  jfloat* hess = jenv->GetFloatArrayElements(jhess, 0);
  bst_ulong len = (bst_ulong)jenv->GetArrayLength(jgrad);
  int ret = XGBoosterBoostOneIter(handle, dtrain, grad, hess, len);
  JVM_CHECK_CALL(ret);
  //release
  jenv->ReleaseFloatArrayElements(jgrad, grad, 0);
  jenv->ReleaseFloatArrayElements(jhess, hess, 0);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterEvalOneIter
 * Signature: (JI[J[Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterEvalOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jint jiter, jlongArray jdmats, jobjectArray jevnames, jobjectArray jout) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  std::vector<DMatrixHandle> dmats;
  std::vector<std::string> evnames;
  std::vector<const char*> evchars;

  size_t len =  static_cast<size_t>(jenv->GetArrayLength(jdmats));
  // put handle from jhandles to chandles
  jlong* cjdmats = jenv->GetLongArrayElements(jdmats, 0);
  for (size_t i = 0; i < len; ++i) {
    dmats.push_back((DMatrixHandle) cjdmats[i]);
    jstring jevname = (jstring)jenv->GetObjectArrayElement(jevnames, i);
    const char *s =jenv->GetStringUTFChars(jevname, 0);
    evnames.push_back(std::string(s, jenv->GetStringLength(jevname)));
    if (s != nullptr) jenv->ReleaseStringUTFChars(jevname, s);
  }
  jenv->ReleaseLongArrayElements(jdmats, cjdmats, 0);
  for (size_t i = 0; i < len; ++i) {
    evchars.push_back(evnames[i].c_str());
  }
  const char* result;
  int ret = XGBoosterEvalOneIter(handle, jiter,
                                 dmlc::BeginPtr(dmats),
                                 dmlc::BeginPtr(evchars),
                                 len, &result);
  JVM_CHECK_CALL(ret);
  jstring jinfo = nullptr;
  if (result != nullptr) {
    jinfo = jenv->NewStringUTF(result);
  }
  jenv->SetObjectArrayElement(jout, 0, jinfo);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterPredict
 * Signature: (JJIJ)[F
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterPredict
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jdmat, jint joption_mask, jint jntree_limit, jobjectArray jout) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  DMatrixHandle dmat = (DMatrixHandle) jdmat;
  bst_ulong len;
  float *result;
  int ret = XGBoosterPredict(handle, dmat, joption_mask, (unsigned int) jntree_limit,
                             /* training = */ 0,  // Currently this parameter is not supported by JVM
                             &len, (const float **) &result);
  JVM_CHECK_CALL(ret);
  if (len) {
    jsize jlen = (jsize) len;
    jfloatArray jarray = jenv->NewFloatArray(jlen);
    jenv->SetFloatArrayRegion(jarray, 0, jlen, (jfloat *) result);
    jenv->SetObjectArrayElement(jout, 0, jarray);
  }
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterLoadModel
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterLoadModel
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  const char* fname = jenv->GetStringUTFChars(jfname, 0);

  int ret = XGBoosterLoadModel(handle, fname);
  JVM_CHECK_CALL(ret);
  if (fname) {
    jenv->ReleaseStringUTFChars(jfname,fname);
  }
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSaveModel
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSaveModel
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  const char*  fname = jenv->GetStringUTFChars(jfname, 0);

  int ret = XGBoosterSaveModel(handle, fname);
  JVM_CHECK_CALL(ret);
  if (fname) {
    jenv->ReleaseStringUTFChars(jfname, fname);
  }
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterLoadModelFromBuffer
 * Signature: (J[B)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterLoadModelFromBuffer
    (JNIEnv *jenv, jclass jcls, jlong jhandle, jbyteArray jbytes) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  jbyte* buffer = jenv->GetByteArrayElements(jbytes, 0);
  int ret = XGBoosterLoadModelFromBuffer(
      handle, buffer, jenv->GetArrayLength(jbytes));
  JVM_CHECK_CALL(ret);
  jenv->ReleaseByteArrayElements(jbytes, buffer, 0);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSaveModelToBuffer
 * Signature: (JLjava/lang/String;[[B)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSaveModelToBuffer
  (JNIEnv * jenv, jclass jcls, jlong jhandle, jstring jformat, jobjectArray jout) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  const char *format = jenv->GetStringUTFChars(jformat, 0);
  bst_ulong len = 0;
  const char *result{nullptr};
  xgboost::Json config {xgboost::Object{}};
  config["format"] = std::string{format};
  std::string config_str;
  xgboost::Json::Dump(config, &config_str);

  int ret = XGBoosterSaveModelToBuffer(handle, config_str.c_str(), &len, &result);
  JVM_CHECK_CALL(ret);
  if (result) {
    jbyteArray jarray = jenv->NewByteArray(len);
    jenv->SetByteArrayRegion(jarray, 0, len, (jbyte *)result);
    jenv->SetObjectArrayElement(jout, 0, jarray);
  }
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterDumpModelEx
 * Signature: (JLjava/lang/String;ILjava/lang/String;[[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterDumpModelEx
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfmap, jint jwith_stats, jstring jformat, jobjectArray jout) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  const char *fmap = jenv->GetStringUTFChars(jfmap, 0);
  const char *format = jenv->GetStringUTFChars(jformat, 0);
  bst_ulong len = 0;
  char **result;

  int ret = XGBoosterDumpModelEx(handle, fmap, jwith_stats, format, &len, (const char ***) &result);
  JVM_CHECK_CALL(ret);

  jsize jlen = (jsize) len;
  jobjectArray jinfos = jenv->NewObjectArray(jlen, jenv->FindClass("java/lang/String"), jenv->NewStringUTF(""));
  for(int i=0 ; i<jlen; i++) {
    jenv->SetObjectArrayElement(jinfos, i, jenv->NewStringUTF((const char*) result[i]));
  }
  jenv->SetObjectArrayElement(jout, 0, jinfos);

  if (fmap) jenv->ReleaseStringUTFChars(jfmap, (const char *)fmap);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterDumpModelExWithFeatures
 * Signature: (J[Ljava/lang/String;ILjava/lang/String;[[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterDumpModelExWithFeatures
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jobjectArray jfeature_names, jint jwith_stats,
    jstring jformat, jobjectArray jout) {

  BoosterHandle handle = (BoosterHandle) jhandle;
  bst_ulong feature_num = (bst_ulong)jenv->GetArrayLength(jfeature_names);

  std::vector<std::string> feature_names;
  std::vector<char*> feature_names_char;

  std::string feature_type_q = "q";
  std::vector<char*> feature_types_char;

  for (bst_ulong i = 0; i < feature_num; ++i) {
    jstring jfeature_name = (jstring)jenv->GetObjectArrayElement(jfeature_names, i);
    const char *s = jenv->GetStringUTFChars(jfeature_name, 0);
    feature_names.push_back(std::string(s, jenv->GetStringLength(jfeature_name)));
    if (s != nullptr) jenv->ReleaseStringUTFChars(jfeature_name, s);
    if (feature_names.back().length() == 0) feature_names.pop_back();
  }

  for (size_t i = 0; i < feature_names.size(); ++i) {
    feature_names_char.push_back(&feature_names[i][0]);
    feature_types_char.push_back(&feature_type_q[0]);
  }

  const char *format = jenv->GetStringUTFChars(jformat, 0);
  bst_ulong len = 0;
  char **result;

  int ret = XGBoosterDumpModelExWithFeatures(handle, feature_num,
                                             (const char **) dmlc::BeginPtr(feature_names_char),
                                             (const char **) dmlc::BeginPtr(feature_types_char),
                                             jwith_stats, format, &len, (const char ***) &result);
  JVM_CHECK_CALL(ret);

  jsize jlen = (jsize) len;
  jobjectArray jinfos = jenv->NewObjectArray(jlen, jenv->FindClass("java/lang/String"), jenv->NewStringUTF(""));
  for(int i=0 ; i<jlen; i++) {
    jenv->SetObjectArrayElement(jinfos, i, jenv->NewStringUTF((const char*) result[i]));
  }
  jenv->SetObjectArrayElement(jout, 0, jinfos);

  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterGetAttrNames
 * Signature: (J[[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterGetAttrNames
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jobjectArray jout) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  bst_ulong len = 0;
  char **result;
  int ret = XGBoosterGetAttrNames(handle, &len, (const char ***) &result);
  JVM_CHECK_CALL(ret);

  jsize jlen = (jsize) len;
  jobjectArray jinfos = jenv->NewObjectArray(jlen, jenv->FindClass("java/lang/String"), jenv->NewStringUTF(""));
  for(int i=0 ; i<jlen; i++) {
    jenv->SetObjectArrayElement(jinfos, i, jenv->NewStringUTF((const char*) result[i]));
  }
  jenv->SetObjectArrayElement(jout, 0, jinfos);

  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterGetAttr
 * Signature: (JLjava/lang/String;[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterGetAttr
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jkey, jobjectArray jout) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  const char* key = jenv->GetStringUTFChars(jkey, 0);
  const char* result;
  int success;
  int ret = XGBoosterGetAttr(handle, key, &result, &success);
  JVM_CHECK_CALL(ret);
  //release
  if (key) jenv->ReleaseStringUTFChars(jkey, key);

  if (success > 0) {
    jstring jret = jenv->NewStringUTF(result);
    jenv->SetObjectArrayElement(jout, 0, jret);
  }

  return ret;
};

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSetAttr
 * Signature: (JLjava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSetAttr
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jkey, jstring jvalue) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  const char* key = jenv->GetStringUTFChars(jkey, 0);
  const char* value = jenv->GetStringUTFChars(jvalue, 0);
  int ret = XGBoosterSetAttr(handle, key, value);
  JVM_CHECK_CALL(ret);
  //release
  if (key) jenv->ReleaseStringUTFChars(jkey, key);
  if (value) jenv->ReleaseStringUTFChars(jvalue, value);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterLoadRabitCheckpoint
 * Signature: (J[I)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterLoadRabitCheckpoint
  (JNIEnv *jenv , jclass jcls, jlong jhandle, jintArray jout) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  int version;
  int ret = XGBoosterLoadRabitCheckpoint(handle, &version);
  JVM_CHECK_CALL(ret);
  jint jversion = version;
  jenv->SetIntArrayRegion(jout, 0, 1, &jversion);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSaveRabitCheckpoint
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSaveRabitCheckpoint
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  return XGBoosterSaveRabitCheckpoint(handle);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterGetNumFeature
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterGetNumFeature
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlongArray jout) {
  BoosterHandle handle = (BoosterHandle) jhandle;
  bst_ulong num_feature;
  int ret = XGBoosterGetNumFeature(handle, &num_feature);
  JVM_CHECK_CALL(ret);
  jlong jnum_feature = num_feature;
  jenv->SetLongArrayRegion(jout, 0, 1, &jnum_feature);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorInit
 * Signature: ([Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorInit
  (JNIEnv *jenv, jclass jcls, jobjectArray jargs) {
  xgboost::Json config{xgboost::Object{}};
  bst_ulong len = (bst_ulong)jenv->GetArrayLength(jargs);
  assert(len % 2 == 0);
  for (bst_ulong i = 0; i < len / 2; ++i) {
    jstring key = (jstring)jenv->GetObjectArrayElement(jargs, 2 * i);
    std::string key_str(jenv->GetStringUTFChars(key, 0), jenv->GetStringLength(key));
    jstring value = (jstring)jenv->GetObjectArrayElement(jargs, 2 * i + 1);
    std::string value_str(jenv->GetStringUTFChars(value, 0), jenv->GetStringLength(value));
    config[key_str] = xgboost::String(value_str);
  }
  std::string json_str;
  xgboost::Json::Dump(config, &json_str);
  JVM_CHECK_CALL(XGCommunicatorInit(json_str.c_str()));
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorFinalize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorFinalize
  (JNIEnv *jenv, jclass jcls) {
  JVM_CHECK_CALL(XGCommunicatorFinalize());
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorPrint
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorPrint
  (JNIEnv *jenv, jclass jcls, jstring jmsg) {
  std::string str(jenv->GetStringUTFChars(jmsg, 0),
                  jenv->GetStringLength(jmsg));
  JVM_CHECK_CALL(XGCommunicatorPrint(str.c_str()));
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorGetRank
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorGetRank
  (JNIEnv *jenv, jclass jcls, jintArray jout) {
  jint rank = XGCommunicatorGetRank();
  jenv->SetIntArrayRegion(jout, 0, 1, &rank);
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorGetWorldSize
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorGetWorldSize
  (JNIEnv *jenv, jclass jcls, jintArray jout) {
  jint out = XGCommunicatorGetWorldSize();
  jenv->SetIntArrayRegion(jout, 0, 1, &out);
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorAllreduce
 * Signature: (Ljava/nio/ByteBuffer;III)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorAllreduce
  (JNIEnv *jenv, jclass jcls, jobject jsendrecvbuf, jint jcount, jint jenum_dtype, jint jenum_op) {
  void *ptr_sendrecvbuf = jenv->GetDirectBufferAddress(jsendrecvbuf);
  JVM_CHECK_CALL(XGCommunicatorAllreduce(ptr_sendrecvbuf, (size_t) jcount, jenum_dtype, jenum_op));
  return 0;
}

namespace xgboost {
namespace jni {
  XGB_DLL int XGDeviceQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass jcls,
                                                            jobject jiter,
                                                            jfloat jmissing,
                                                            jint jmax_bin, jint jnthread,
                                                            jlongArray jout);
  XGB_DLL int XGQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass jcls,
                                                      jobject jdata_iter, jobject jref_iter,
                                                      char const *config, jlongArray jout);
} // namespace jni
} // namespace xgboost

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDeviceQuantileDMatrixCreateFromCallback
 * Signature: (Ljava/util/Iterator;FII[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDeviceQuantileDMatrixCreateFromCallback
    (JNIEnv *jenv, jclass jcls, jobject jiter, jfloat jmissing, jint jmax_bin,
     jint jnthread, jlongArray jout) {
  return xgboost::jni::XGDeviceQuantileDMatrixCreateFromCallbackImpl(
      jenv, jcls, jiter, jmissing, jmax_bin, jnthread, jout);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGQuantileDMatrixCreateFromCallback
 * Signature: (Ljava/util/Iterator;Ljava/util/Iterator;Ljava/lang/String;[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGQuantileDMatrixCreateFromCallback
    (JNIEnv *jenv, jclass jcls, jobject jdata_iter, jobject jref_iter, jstring jconf, jlongArray jout) {
  char const *conf = jenv->GetStringUTFChars(jconf, 0);
  return xgboost::jni::XGQuantileDMatrixCreateFromCallbackImpl(jenv, jcls, jdata_iter, jref_iter,
                                                               conf, jout);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSetInfoFromInterface
 * Signature: (JLjava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSetInfoFromInterface
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
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromArrayInterfaceColumns
  (JNIEnv *jenv, jclass jcls, jstring jjson_columns, jfloat jmissing, jint jnthread, jlongArray jout) {
  DMatrixHandle result;
  const char* cjson_columns = jenv->GetStringUTFChars(jjson_columns, nullptr);
  xgboost::Json config{xgboost::Object{}};
  auto missing = static_cast<float>(jmissing);
  auto n_threads = static_cast<int32_t>(jnthread);
  config["missing"] = xgboost::Number(missing);
  config["nthread"] = xgboost::Integer(n_threads);
  std::string config_str;
  xgboost::Json::Dump(config, &config_str);
  int ret = XGDMatrixCreateFromCudaColumnar(cjson_columns, config_str.c_str(),
                                            &result);
  JVM_CHECK_CALL(ret);
  if (cjson_columns) {
    jenv->ReleaseStringUTFChars(jjson_columns, cjson_columns);
  }

  setHandle(jenv, jout, result);
  return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSetStrFeatureInfo
    (JNIEnv *jenv, jclass jclz, jlong jhandle, jstring jfield, jobjectArray jvalues) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  const char* field = jenv->GetStringUTFChars(jfield, 0);
  int size = jenv->GetArrayLength(jvalues);

  // tmp storage for java strings
  std::vector<std::string> values;
  for (int i = 0; i < size; i++) {
    jstring jstr = (jstring)(jenv->GetObjectArrayElement(jvalues, i));
    const char *value = jenv->GetStringUTFChars(jstr, 0);
    values.emplace_back(value);
    if (value) jenv->ReleaseStringUTFChars(jstr, value);
  }

  std::vector<char const*> c_values;
  c_values.resize(size);
  std::transform(values.cbegin(), values.cend(),
                 c_values.begin(),
                 [](auto const &str) { return str.c_str(); });

  int ret = XGDMatrixSetStrFeatureInfo(handle, field, c_values.data(), size);
  JVM_CHECK_CALL(ret);

  if (field) jenv->ReleaseStringUTFChars(jfield, field);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixGetStrFeatureInfo
 * Signature: (JLjava/lang/String;[J[[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixGetStrFeatureInfo
  (JNIEnv *jenv, jclass jclz, jlong jhandle, jstring jfield, jlongArray joutLenArray,
     jobjectArray joutValueArray) {
  DMatrixHandle handle = (DMatrixHandle) jhandle;
  const char *field = jenv->GetStringUTFChars(jfield, 0);

  bst_ulong out_len = 0;
  char const **c_out_features;
  int ret = XGDMatrixGetStrFeatureInfo(handle, field, &out_len, &c_out_features);

  jlong jlen = (jlong) out_len;
  jenv->SetLongArrayRegion(joutLenArray, 0, 1, &jlen);

  jobjectArray jinfos = jenv->NewObjectArray(jlen, jenv->FindClass("java/lang/String"),
                                             jenv->NewStringUTF(""));
  for (int i = 0; i < jlen; i++) {
    jenv->SetObjectArrayElement(jinfos, i, jenv->NewStringUTF(c_out_features[i]));
  }
  jenv->SetObjectArrayElement(joutValueArray, 0, jinfos);

  JVM_CHECK_CALL(ret);
  if (field) jenv->ReleaseStringUTFChars(jfield, field);
  return ret;
}
