/**
 *  Copyright 2014-2024, XGBoost Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
*/

#include "./xgboost4j.h"

#include <xgboost/base.h>
#include <xgboost/c_api.h>
#include <xgboost/json.h>
#include <xgboost/logging.h>
#include <xgboost/string_view.h>  // for StringView

#include <algorithm>  // for copy_n
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>  // for unique_ptr
#include <string>
#include <type_traits>
#include <vector>

#include "jvm_utils.h"  // for JVM_CHECK_CALL
#include "../../../../src/c_api/c_api_error.h"
#include "../../../../src/c_api/c_api_utils.h"
#include "../../../../src/data/array_interface.h"  // for ArrayInterface

// helper functions
// set handle
void setHandle(JNIEnv *jenv, jlongArray jhandle, void *handle) {
#ifdef __APPLE__
  jlong out = (long)handle;
#else
  int64_t out = (int64_t)handle;
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

namespace {
template <typename T>
using Deleter = std::function<void(T *)>;
}  // anonymous namespace

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

      std::unique_ptr<XGBoostBatchCSR, Deleter<XGBoostBatchCSR>> cbatch{
          [&] {
            auto ptr = new XGBoostBatchCSR;
            auto &cbatch = *ptr;

            // Init
            cbatch.size = jenv->GetArrayLength(joffset) - 1;
            cbatch.columns = jcols;
            cbatch.offset = reinterpret_cast<jlong *>(jenv->GetLongArrayElements(joffset, nullptr));

            if (jlabel != nullptr) {
              cbatch.label = jenv->GetFloatArrayElements(jlabel, nullptr);
              CHECK_EQ(jenv->GetArrayLength(jlabel), static_cast<long>(cbatch.size))
                  << "batch.label.length must equal batch.numRows()";
            } else {
              cbatch.label = nullptr;
            }

            if (jweight != nullptr) {
              cbatch.weight = jenv->GetFloatArrayElements(jweight, nullptr);
              CHECK_EQ(jenv->GetArrayLength(jweight), static_cast<long>(cbatch.size))
                  << "batch.weight.length must equal batch.numRows()";
            } else {
              cbatch.weight = nullptr;
            }

            auto max_elem = cbatch.offset[cbatch.size];
            cbatch.index = (int *)jenv->GetIntArrayElements(jindex, nullptr);
            cbatch.value = jenv->GetFloatArrayElements(jvalue, nullptr);
            CHECK_EQ(jenv->GetArrayLength(jindex), max_elem)
                << "batch.index.length must equal batch.offset.back()";
            CHECK_EQ(jenv->GetArrayLength(jvalue), max_elem)
                << "batch.index.length must equal batch.offset.back()";
            return ptr;
          }(),
          [&](XGBoostBatchCSR *ptr) {
            auto &cbatch = *ptr;
            jenv->ReleaseLongArrayElements(joffset, reinterpret_cast<jlong *>(cbatch.offset), 0);
            jenv->DeleteLocalRef(joffset);

            if (jlabel) {
              jenv->ReleaseFloatArrayElements(jlabel, cbatch.label, 0);
              jenv->DeleteLocalRef(jlabel);
            }
            if (jweight) {
              jenv->ReleaseFloatArrayElements(jweight, cbatch.weight, 0);
              jenv->DeleteLocalRef(jweight);
            }

            jenv->ReleaseIntArrayElements(jindex, (jint *)cbatch.index, 0);
            jenv->DeleteLocalRef(jindex);

            jenv->ReleaseFloatArrayElements(jvalue, cbatch.value, 0);
            jenv->DeleteLocalRef(jvalue);

            delete ptr;
          }};

      CHECK_EQ((*set_function)(set_function_handle, *cbatch), 0) << XGBGetLastError();

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
  if (result) {
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
  (JNIEnv *jenv, jclass jcls, jobject jiter, jstring jcache_info, jfloat jmissing, jlongArray jout) {
  DMatrixHandle result;
  std::unique_ptr<char const, Deleter<char const>> cache_info;
  if (jcache_info != nullptr) {
    cache_info = {jenv->GetStringUTFChars(jcache_info, nullptr), [&](char const *ptr) {
                    jenv->ReleaseStringUTFChars(jcache_info, ptr);
                  }};
  }
  auto missing = static_cast<float>(jmissing);
  int ret =
      XGDMatrixCreateFromDataIter(jiter, XGBoost4jCallbackDataIterNext, cache_info.get(),
                                  missing,&result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixCreateFromFile
 * Signature: (Ljava/lang/String;I[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromFile(
    JNIEnv *jenv, jclass jcls, jstring jfname, jint jsilent, jlongArray jout) {
  std::unique_ptr<char const, Deleter<char const>> fname{jenv->GetStringUTFChars(jfname, nullptr),
                                                         [&](char const *ptr) {
                                                           jenv->ReleaseStringUTFChars(jfname, ptr);
                                                         }};
  DMatrixHandle result;
  int ret = XGDMatrixCreateFromFile(fname.get(), jsilent, &result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
  return ret;
}

namespace {
using JavaIndT =
    std::conditional_t<std::is_convertible<jint *, std::int32_t *>::value, std::int32_t, long>;
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

  std::unique_ptr<jlong, Deleter<jlong>> indptr{jenv->GetLongArrayElements(jindptr, nullptr),
                                                [&](jlong *ptr) {
                                                  jenv->ReleaseLongArrayElements(jindptr, ptr, 0);
                                                }};
  std::unique_ptr<jint, Deleter<jint>> indices{jenv->GetIntArrayElements(jindices, nullptr),
                                               [&](jint *ptr) {
                                                 jenv->ReleaseIntArrayElements(jindices, ptr, 0);
                                               }};
  std::unique_ptr<jfloat, Deleter<jfloat>> data{jenv->GetFloatArrayElements(jdata, nullptr),
                                                [&](jfloat *ptr) {
                                                  jenv->ReleaseFloatArrayElements(jdata, ptr, 0);
                                                }};

  bst_ulong nindptr = static_cast<bst_ulong>(jenv->GetArrayLength(jindptr));
  bst_ulong nelem = static_cast<bst_ulong>(jenv->GetArrayLength(jdata));

  std::string sindptr, sindices, sdata;
  CHECK_EQ(indptr.get()[nindptr - 1], nelem);
  using IndPtrT = std::conditional_t<std::is_convertible<jlong *, long *>::value, long, long long>;
  xgboost::detail::MakeSparseFromPtr(
      static_cast<IndPtrT const *>(indptr.get()), static_cast<JavaIndT const *>(indices.get()),
      static_cast<float const *>(data.get()), nindptr, &sindptr, &sindices, &sdata);

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
  std::unique_ptr<jfloat, Deleter<jfloat>> data{jenv->GetFloatArrayElements(jdata, 0), [&](jfloat* ptr) {
    jenv->ReleaseFloatArrayElements(jdata, ptr, 0);
  }};

  bst_ulong nrow = (bst_ulong)jnrow;
  bst_ulong ncol = (bst_ulong)jncol;
  jint ret =
      XGDMatrixCreateFromMat(static_cast<float const *>(data.get()), nrow, ncol, jmiss, &result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
  return ret;
}

namespace {
// Workaround int is not the same as jint. For some reason, if constexpr couldn't dispatch
// the following.
template <typename T>
auto SliceDMatrixWinWar(DMatrixHandle handle, T *ptr, std::size_t len, DMatrixHandle *result) {
  // default to not allowing slicing with group ID specified -- feel free to add if necessary
  return XGDMatrixSliceDMatrixEx(handle, ptr, len, result, 0);
}

template <>
auto SliceDMatrixWinWar<long>(DMatrixHandle handle, long *ptr, std::size_t len, DMatrixHandle *result) {
  std::vector<std::int32_t> copy(len);
  std::copy_n(ptr, len, copy.begin());
  // default to not allowing slicing with group ID specified -- feel free to add if necessary
  return XGDMatrixSliceDMatrixEx(handle, copy.data(), len, result, 0);
}
}  // namespace

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSliceDMatrix
 * Signature: (J[I)J
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSliceDMatrix(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jintArray jindexset, jlongArray jout) {
  DMatrixHandle result;
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);

  std::unique_ptr<jint, Deleter<jint>> indexset{jenv->GetIntArrayElements(jindexset, nullptr),
                                                [&](jint *ptr) {
                                                  jenv->ReleaseIntArrayElements(jindexset, ptr, 0);
                                                }};
  auto len = static_cast<bst_ulong>(jenv->GetArrayLength(jindexset));
  auto ret = SliceDMatrixWinWar(handle, indexset.get(), len, &result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
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
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSaveBinary(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname, jint jsilent) {
  DMatrixHandle handle = reinterpret_cast<DMatrixHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> fname{
      jenv->GetStringUTFChars(jfname, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jfname, ptr);
        }
      }};
  int ret = XGDMatrixSaveBinary(handle, fname.get(), jsilent);
  JVM_CHECK_CALL(ret);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSetFloatInfo
 * Signature: (JLjava/lang/String;[F)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSetFloatInfo(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jfloatArray jarray) {
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> field{
      jenv->GetStringUTFChars(jfield, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jfield, ptr);
        }
      }};
  std::unique_ptr<jfloat, Deleter<jfloat>> array{jenv->GetFloatArrayElements(jarray, nullptr),
                                                 [&](jfloat *ptr) {
                                                   jenv->ReleaseFloatArrayElements(jarray, ptr, 0);
                                                 }};

  bst_ulong len = (bst_ulong)jenv->GetArrayLength(jarray);
  auto str = xgboost::linalg::Make1dInterface(array.get(), len);
  return XGDMatrixSetInfoFromInterface(handle, field.get(), str.c_str());
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSetUIntInfo
 * Signature: (JLjava/lang/String;[I)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSetUIntInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jintArray jarray) {
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> field{
      jenv->GetStringUTFChars(jfield, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jfield, ptr);
        }
      }};
  std::unique_ptr<jint, Deleter<jint>> array{jenv->GetIntArrayElements(jarray, nullptr),
                                             [&](jint *ptr) {
                                               jenv->ReleaseIntArrayElements(jarray, ptr, 0);
                                             }};
  bst_ulong len = (bst_ulong)jenv->GetArrayLength(jarray);
  auto str = xgboost::linalg::Make1dInterface(array.get(), len);
  return XGDMatrixSetInfoFromInterface(handle, field.get(), str.c_str());
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixGetFloatInfo
 * Signature: (JLjava/lang/String;)[F
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixGetFloatInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jobjectArray jout) {
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> field{
      jenv->GetStringUTFChars(jfield, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jfield, ptr);
        }
      }};
  bst_ulong len;
  float *result;
  int ret = XGDMatrixGetFloatInfo(handle, field.get(), &len, (const float**) &result);
  JVM_CHECK_CALL(ret);

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
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> field{
      jenv->GetStringUTFChars(jfield, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jfield, ptr);
        }
      }};
  bst_ulong len;
  unsigned int *result;
  int ret = (jint)XGDMatrixGetUIntInfo(handle, field.get(), &len, (const unsigned int **)&result);
  JVM_CHECK_CALL(ret);

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
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);
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
    std::unique_ptr<jlong, Deleter<jlong>> cjhandles{
        jenv->GetLongArrayElements(jhandles, nullptr), [&](jlong *ptr) {
          jenv->ReleaseLongArrayElements(jhandles, ptr, 0);
        }};
    for (size_t i = 0; i < len; ++i) {
      handles.push_back(reinterpret_cast<DMatrixHandle>(cjhandles.get()[i]));
    }
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
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterFree(JNIEnv *jenv,
                                                                            jclass jcls,
                                                                            jlong jhandle) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  return XGBoosterFree(handle);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSetParam
 * Signature: (JLjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSetParam(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jname, jstring jvalue) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> name{jenv->GetStringUTFChars(jname, nullptr),
                                                        [&](char const *ptr) {
                                                          if (ptr) {
                                                            jenv->ReleaseStringUTFChars(jname, ptr);
                                                          }
                                                        }};
  std::unique_ptr<char const, Deleter<char const>> value{
      jenv->GetStringUTFChars(jvalue, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jvalue, ptr);
        }
      }};
  int ret = XGBoosterSetParam(handle, name.get(), value.get());
  JVM_CHECK_CALL(ret);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterUpdateOneIter
 * Signature: (JIJ)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterUpdateOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jint jiter, jlong jdtrain) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  auto dtrain = reinterpret_cast<DMatrixHandle>(jdtrain);
  return XGBoosterUpdateOneIter(handle, jiter, dtrain);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterTrainOneIter
 * Signature: (JJI[F[F)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterTrainOneIter(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jdtrain, jint jiter, jfloatArray jgrad,
    jfloatArray jhess) {
  API_BEGIN();
  auto handle = reinterpret_cast<BoosterHandle *>(jhandle);
  auto dtrain = reinterpret_cast<DMatrixHandle *>(jdtrain);
  CHECK(handle);
  CHECK(dtrain);
  bst_ulong n_samples{0};
  JVM_CHECK_CALL(XGDMatrixNumRow(dtrain, &n_samples));

  bst_ulong len = static_cast<bst_ulong>(jenv->GetArrayLength(jgrad));
  std::unique_ptr<jfloat, Deleter<jfloat>> grad{jenv->GetFloatArrayElements(jgrad, nullptr),
                                                [&](jfloat *ptr) {
                                                  jenv->ReleaseFloatArrayElements(jgrad, ptr, 0);
                                                }};
  std::unique_ptr<jfloat, Deleter<jfloat>> hess{jenv->GetFloatArrayElements(jhess, nullptr),
                                                [&](jfloat *ptr) {
                                                  jenv->ReleaseFloatArrayElements(jhess, ptr, 0);
                                                }};
  CHECK(grad);
  CHECK(hess);

  xgboost::bst_target_t n_targets{1};
  if (len != n_samples && n_samples != 0) {
    CHECK_EQ(len % n_samples, 0) << "Invalid size of gradient.";
    n_targets = len / n_samples;
  }

  auto ctx = xgboost::detail::BoosterCtx(handle);
  auto [s_grad, s_hess] = xgboost::detail::MakeGradientInterface(
      ctx, grad.get(), hess.get(), xgboost::linalg::kC, n_samples, n_targets);
  return XGBoosterTrainOneIter(handle, dtrain, static_cast<std::int32_t>(jiter), s_grad.c_str(),
                               s_hess.c_str());
  API_END();
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterEvalOneIter
 * Signature: (JI[J[Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterEvalOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jint jiter, jlongArray jdmats, jobjectArray jevnames, jobjectArray jout) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::vector<DMatrixHandle> dmats;
  std::vector<std::string> evnames;
  std::vector<const char*> evchars;

  size_t len =  static_cast<size_t>(jenv->GetArrayLength(jdmats));
  // put handle from jhandles to chandles
  std::unique_ptr<jlong, Deleter<jlong>> cjdmats{
      jenv->GetLongArrayElements(jdmats, nullptr), [&](jlong *ptr) {
        jenv->ReleaseLongArrayElements(jdmats, ptr, 0);
      }};
  for (size_t i = 0; i < len; ++i) {
    dmats.push_back(reinterpret_cast<DMatrixHandle>(cjdmats.get()[i]));
    jstring jevname = (jstring)jenv->GetObjectArrayElement(jevnames, i);
    std::unique_ptr<char const, Deleter<char const>> s{jenv->GetStringUTFChars(jevname, nullptr),
                                                       [&](char const *ptr) {
                                                         jenv->ReleaseStringUTFChars(jevname, ptr);
                                                       }};
    evnames.emplace_back(s.get(), jenv->GetStringLength(jevname));
  }

  for (size_t i = 0; i < len; ++i) {
    evchars.push_back(evnames[i].c_str());
  }
  const char *result;
  int ret = XGBoosterEvalOneIter(handle, jiter, dmlc::BeginPtr(dmats), dmlc::BeginPtr(evchars), len,
                                 &result);
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
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  auto dmat = reinterpret_cast<DMatrixHandle>(jdmat);
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
 * Method:    XGBoosterPredictFromDense
 * Signature: (J[FJJFIII[F[[F)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterPredictFromDense(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jfloatArray jdata, jlong num_rows, jlong num_features,
    jfloat missing, jint iteration_begin, jint iteration_end, jint predict_type,
    jfloatArray jmargin, jobjectArray jout) {
  API_BEGIN();
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);

  /**
   * Create array interface.
   */
  namespace linalg = xgboost::linalg;
  jfloat *data = jenv->GetFloatArrayElements(jdata, nullptr);
  xgboost::Context ctx;
  auto t_data = linalg::MakeTensorView(
      ctx.Device(),
      xgboost::common::Span{data, static_cast<std::size_t>(num_rows * num_features)}, num_rows,
      num_features);
  auto s_array = linalg::ArrayInterfaceStr(t_data);

  /**
   * Create configuration object.
   */
  xgboost::Json config{xgboost::Object{}};
  config["cache_id"] = xgboost::Integer{};
  config["type"] = xgboost::Integer{static_cast<std::int32_t>(predict_type)};
  config["iteration_begin"] = xgboost::Integer{static_cast<xgboost::bst_layer_t>(iteration_begin)};
  config["iteration_end"] = xgboost::Integer{static_cast<xgboost::bst_layer_t>(iteration_end)};
  config["missing"] = xgboost::Number{static_cast<float>(missing)};
  config["strict_shape"] = xgboost::Boolean{true};
  std::string s_config;
  xgboost::Json::Dump(config, &s_config);

  /**
   * Handle base margin
   */
  BoosterHandle proxy{nullptr};

  float *margin{nullptr};
  if (jmargin) {
    margin = jenv->GetFloatArrayElements(jmargin, nullptr);
    JVM_CHECK_CALL(XGProxyDMatrixCreate(&proxy));
    auto str = xgboost::linalg::Make1dInterface(margin, jenv->GetArrayLength(jmargin));
    JVM_CHECK_CALL(XGDMatrixSetInfoFromInterface(proxy, "base_margin", str.c_str()));
  }

  bst_ulong const *out_shape;
  bst_ulong out_dim;
  float const *result;
  auto ret = XGBoosterPredictFromDense(handle, s_array.c_str(), s_config.c_str(), proxy, &out_shape,
                                       &out_dim, &result);

  jenv->ReleaseFloatArrayElements(jdata, data, 0);
  if (proxy) {
    XGDMatrixFree(proxy);
    jenv->ReleaseFloatArrayElements(jmargin, margin, 0);
  }

  if (ret != 0) {
    return ret;
  }

  std::size_t n{1};
  for (std::size_t i = 0; i < out_dim; ++i) {
    n *= out_shape[i];
  }

  jfloatArray jarray = jenv->NewFloatArray(n);

  jenv->SetFloatArrayRegion(jarray, 0, n, result);
  jenv->SetObjectArrayElement(jout, 0, jarray);

  API_END();
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterLoadModel
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterLoadModel(JNIEnv *jenv,
                                                                                 jclass jcls,
                                                                                 jlong jhandle,
                                                                                 jstring jfname) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> fname{jenv->GetStringUTFChars(jfname, nullptr),
                                                         [&](char const *ptr) {
                                                           jenv->ReleaseStringUTFChars(jfname, ptr);
                                                         }};
  return XGBoosterLoadModel(handle, fname.get());
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSaveModel
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSaveModel(JNIEnv *jenv,
                                                                                 jclass jcls,
                                                                                 jlong jhandle,
                                                                                 jstring jfname) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> fname{
      jenv->GetStringUTFChars(jfname, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jfname, ptr);
        }
      }};
  return XGBoosterSaveModel(handle, fname.get());
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterLoadModelFromBuffer
 * Signature: (J[B)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterLoadModelFromBuffer(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jbyteArray jbytes) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<jbyte, Deleter<jbyte>> buffer{jenv->GetByteArrayElements(jbytes, nullptr),
                                                [&](jbyte *ptr) {
                                                  jenv->ReleaseByteArrayElements(jbytes, ptr, 0);
                                                }};
  return XGBoosterLoadModelFromBuffer(handle, buffer.get(), jenv->GetArrayLength(jbytes));
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSaveModelToBuffer
 * Signature: (JLjava/lang/String;[[B)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSaveModelToBuffer
  (JNIEnv * jenv, jclass jcls, jlong jhandle, jstring jformat, jobjectArray jout) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> format{
      jenv->GetStringUTFChars(jformat, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jformat, ptr);
        }
      }};
  bst_ulong len = 0;
  const char *result{nullptr};
  xgboost::Json config{xgboost::Object{}};
  config["format"] = std::string{format.get()};
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
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> fmap{jenv->GetStringUTFChars(jfmap, nullptr),
                                                        [&](char const *ptr) {
                                                          if (ptr) {
                                                            jenv->ReleaseStringUTFChars(jfmap, ptr);
                                                          }
                                                        }};
  std::unique_ptr<char const, Deleter<char const>> format{
      jenv->GetStringUTFChars(jformat, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jformat, ptr);
        }
      }};
  bst_ulong len = 0;
  char const **result;

  int ret = XGBoosterDumpModelEx(handle, fmap.get(), jwith_stats, format.get(), &len, &result);
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
 * Method:    XGBoosterDumpModelExWithFeatures
 * Signature: (J[Ljava/lang/String;ILjava/lang/String;[[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterDumpModelExWithFeatures
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jobjectArray jfeature_names, jint jwith_stats,
    jstring jformat, jobjectArray jout) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  bst_ulong feature_num = (bst_ulong)jenv->GetArrayLength(jfeature_names);

  std::vector<std::string> feature_names;
  std::vector<char const*> feature_names_char;

  std::string feature_type_q = "q";
  std::vector<char const *> feature_types_char;

  for (bst_ulong i = 0; i < feature_num; ++i) {
    jstring jfeature_name = (jstring)jenv->GetObjectArrayElement(jfeature_names, i);
    std::unique_ptr<char const, Deleter<char const>> s{
        jenv->GetStringUTFChars(jfeature_name, nullptr), [&](char const *ptr) {
          if (ptr != nullptr) {
            jenv->ReleaseStringUTFChars(jfeature_name, ptr);
          }
        }};
    feature_names.emplace_back(s.get(), jenv->GetStringLength(jfeature_name));

    if (feature_names.back().length() == 0) {
      feature_names.pop_back();
    }
  }

  for (size_t i = 0; i < feature_names.size(); ++i) {
    feature_names_char.push_back(feature_names[i].c_str());
    feature_types_char.push_back(feature_type_q.c_str());
  }

  std::unique_ptr<char const, Deleter<char const>> format{
      jenv->GetStringUTFChars(jformat, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jformat, ptr);
        }
      }};
  bst_ulong len = 0;
  char **result;

  int ret = XGBoosterDumpModelExWithFeatures(
      handle, feature_num, (const char **)dmlc::BeginPtr(feature_names_char),
      (const char **)dmlc::BeginPtr(feature_types_char), jwith_stats, format.get(), &len,
      (const char ***)&result);
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
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterGetAttr(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jkey, jobjectArray jout) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> key{jenv->GetStringUTFChars(jkey, nullptr),
                                                       [&](char const *ptr) {
                                                         if (ptr) {
                                                           jenv->ReleaseStringUTFChars(jkey, ptr);
                                                         }
                                                       }};

  const char *result;
  int success;
  int ret = XGBoosterGetAttr(handle, key.get(), &result, &success);
  JVM_CHECK_CALL(ret);

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
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSetAttr(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jkey, jstring jvalue) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> key{jenv->GetStringUTFChars(jkey, nullptr),
                                                       [&](char const *ptr) {
                                                         if (ptr) {
                                                           jenv->ReleaseStringUTFChars(jkey, ptr);
                                                         }
                                                       }};
  std::unique_ptr<char const, Deleter<char const>> value{
      jenv->GetStringUTFChars(jvalue, nullptr), [&](char const *ptr) {
        if (ptr) {
          jenv->ReleaseStringUTFChars(jvalue, ptr);
        }
      }};
  return XGBoosterSetAttr(handle, key.get(), value.get());
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterGetNumFeature
 * Signature: (J[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterGetNumFeature(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jlongArray jout) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  bst_ulong num_feature;
  int ret = XGBoosterGetNumFeature(handle, &num_feature);
  JVM_CHECK_CALL(ret);
  jlong jnum_feature = num_feature;
  jenv->SetLongArrayRegion(jout, 0, 1, &jnum_feature);
  return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterGetNumBoostedRound(
    JNIEnv *jenv, jclass, jlong jhandle, jintArray jout) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::int32_t n_rounds{0};
  auto ret = XGBoosterBoostedRounds(handle, &n_rounds);
  JVM_CHECK_CALL(ret);
  jint jn_rounds = n_rounds;
  jenv->SetIntArrayRegion(jout, 0, 1, &jn_rounds);
  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorInit
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorInit(JNIEnv *jenv,
                                                                               jclass jcls,
                                                                               jstring jargs) {
  xgboost::Json config{xgboost::Object{}};
  std::unique_ptr<char const, Deleter<char const>> args{jenv->GetStringUTFChars(jargs, nullptr),
                                                        [&](char const *ptr) {
                                                          if (ptr) {
                                                            jenv->ReleaseStringUTFChars(jargs, ptr);
                                                          }
                                                        }};
  return XGCommunicatorInit(args.get());
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    TrackerCreate
 * Signature: (Ljava/lang/String;IIIJ[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_TrackerCreate(
    JNIEnv *jenv, jclass, jstring host, jint n_workers, jint port, jint sortby, jlong timeout,
    jlongArray jout) {
  using namespace xgboost;  // NOLINT

  TrackerHandle handle;
  Json config{Object{}};
  std::unique_ptr<char const, Deleter<char const>> p_shost{jenv->GetStringUTFChars(host, nullptr),
                                                           [&](char const *ptr) {
                                                             jenv->ReleaseStringUTFChars(host, ptr);
                                                           }};
  std::string shost{p_shost.get(),
                    static_cast<std::string::size_type>(jenv->GetStringLength(host))};
  if (!shost.empty()) {
    config["host"] = shost;
  }
  config["port"] = Integer{static_cast<Integer::Int>(port)};
  config["n_workers"] = Integer{static_cast<Integer::Int>(n_workers)};
  config["timeout"] = Integer{static_cast<Integer::Int>(timeout)};
  config["sortby"] = Integer{static_cast<Integer::Int>(sortby)};
  config["dmlc_communicator"] = String{"rabit"};
  std::string sconfig = Json::Dump(config);
  JVM_CHECK_CALL(XGTrackerCreate(sconfig.c_str(), &handle));
  setHandle(jenv, jout, handle);

  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    TrackerRun
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_TrackerRun(JNIEnv *, jclass,
                                                                         jlong jhandle) {
  auto handle = reinterpret_cast<TrackerHandle>(jhandle);
  JVM_CHECK_CALL(XGTrackerRun(handle, nullptr));
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    TrackerWaitFor
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_TrackerWaitFor(JNIEnv *, jclass,
                                                                             jlong jhandle,
                                                                             jlong timeout) {
  using namespace xgboost;  // NOLINT

  auto handle = reinterpret_cast<TrackerHandle>(jhandle);
  Json config{Object{}};
  config["timeout"] = Integer{static_cast<Integer::Int>(timeout)};
  std::string sconfig = Json::Dump(config);
  JVM_CHECK_CALL(XGTrackerWaitFor(handle, sconfig.c_str()));
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    TrackerWorkerArgs
 * Signature: (JJ[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_TrackerWorkerArgs(
    JNIEnv *jenv, jclass, jlong jhandle, jlong timeout, jobjectArray jout) {
  using namespace xgboost;  // NOLINT

  Json config{Object{}};
  config["timeout"] = Integer{static_cast<Integer::Int>(timeout)};
  std::string sconfig = Json::Dump(config);
  auto handle = reinterpret_cast<TrackerHandle>(jhandle);
  char const *args;
  JVM_CHECK_CALL(XGTrackerWorkerArgs(handle, &args));
  auto jargs = Json::Load(StringView{args});

  jstring jret = jenv->NewStringUTF(args);
  jenv->SetObjectArrayElement(jout, 0, jret);
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    TrackerFree
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_TrackerFree(JNIEnv *, jclass,
                                                                          jlong jhandle) {
  auto handle = reinterpret_cast<TrackerHandle>(jhandle);
  JVM_CHECK_CALL(XGTrackerFree(handle));
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorFinalize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorFinalize(JNIEnv *,
                                                                                   jclass) {
  JVM_CHECK_CALL(XGCommunicatorFinalize());
  return 0;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    CommunicatorPrint
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_CommunicatorPrint(JNIEnv *jenv,
                                                                                jclass jcls,
                                                                                jstring jmsg) {
  std::unique_ptr<char const, Deleter<char const>> msg{jenv->GetStringUTFChars(jmsg, nullptr),
                                                       [&](char const *ptr) {
                                                         if (ptr) {
                                                           jenv->ReleaseStringUTFChars(jmsg, ptr);
                                                         }
                                                       }};
  std::string str(msg.get(), jenv->GetStringLength(jmsg));
  return XGCommunicatorPrint(str.c_str());
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

namespace xgboost::jni {
XGB_DLL int XGQuantileDMatrixCreateFromCallbackImpl(JNIEnv *jenv, jclass jcls, jobject jdata_iter,
                                                    jobject jref_iter, char const *config,
                                                    jlongArray jout);
}  // namespace xgboost::jni

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGQuantileDMatrixCreateFromCallback
 * Signature: (Ljava/util/Iterator;[JLjava/lang/String;[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGQuantileDMatrixCreateFromCallback(
    JNIEnv *jenv, jclass jcls, jobject jdata_iter, jlongArray jref, jstring jconf,
    jlongArray jout) {
  std::unique_ptr<char const, Deleter<char const>> conf{jenv->GetStringUTFChars(jconf, nullptr),
                                                        [&](char const *ptr) {
                                                          jenv->ReleaseStringUTFChars(jconf, ptr);
                                                        }};
  return xgboost::jni::XGQuantileDMatrixCreateFromCallbackImpl(jenv, jcls, jdata_iter, jref,
                                                               conf.get(), jout);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixSetInfoFromInterface
 * Signature: (JLjava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSetInfoFromInterface(
    JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jstring jjson_columns) {
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> field{jenv->GetStringUTFChars(jfield, nullptr),
                                                         [&](char const *ptr) {
                                                           jenv->ReleaseStringUTFChars(jfield, ptr);
                                                         }};
  std::unique_ptr<char const, Deleter<char const>> cjson_columns{
      jenv->GetStringUTFChars(jjson_columns, nullptr), [&](char const *ptr) {
        jenv->ReleaseStringUTFChars(jjson_columns, ptr);
      }};

  return XGDMatrixSetInfoFromInterface(handle, field.get(), cjson_columns.get());
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixCreateFromArrayInterfaceColumns
 * Signature: (Ljava/lang/String;FI[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixCreateFromArrayInterfaceColumns
  (JNIEnv *jenv, jclass jcls, jstring jjson_columns, jfloat jmissing, jint jnthread, jlongArray jout) {
  DMatrixHandle result;
  std::unique_ptr<char const, Deleter<char const>> cjson_columns{
      jenv->GetStringUTFChars(jjson_columns, nullptr), [&](char const *ptr) {
        jenv->ReleaseStringUTFChars(jjson_columns, ptr);
      }};
  xgboost::Json config{xgboost::Object{}};
  auto missing = static_cast<float>(jmissing);
  auto n_threads = static_cast<int32_t>(jnthread);
  config["missing"] = xgboost::Number(missing);
  config["nthread"] = xgboost::Integer(n_threads);
  std::string config_str;
  xgboost::Json::Dump(config, &config_str);
  int ret = XGDMatrixCreateFromCudaColumnar(cjson_columns.get(), config_str.c_str(), &result);
  JVM_CHECK_CALL(ret);
  setHandle(jenv, jout, result);
  return ret;
}

JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixSetStrFeatureInfo
    (JNIEnv *jenv, jclass jclz, jlong jhandle, jstring jfield, jobjectArray jvalues) {
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> field{jenv->GetStringUTFChars(jfield, nullptr),
                                                         [&](char const *ptr) {
                                                           jenv->ReleaseStringUTFChars(jfield, ptr);
                                                         }};
  int size = jenv->GetArrayLength(jvalues);

  // tmp storage for java strings
  std::vector<std::string> values;
  for (int i = 0; i < size; i++) {
    jstring jstr = (jstring)(jenv->GetObjectArrayElement(jvalues, i));
    std::unique_ptr<char const, Deleter<char const>> value{jenv->GetStringUTFChars(jstr, nullptr),
                                                           [&](char const *ptr) {
                                                             jenv->ReleaseStringUTFChars(jstr, ptr);
                                                           }};
    values.emplace_back(value.get());
  }

  std::vector<char const *> c_values;
  c_values.resize(size);
  std::transform(values.cbegin(), values.cend(), c_values.begin(),
                 [](auto const &str) { return str.c_str(); });

  return XGDMatrixSetStrFeatureInfo(handle, field.get(), c_values.data(), size);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixGetStrFeatureInfo
 * Signature: (JLjava/lang/String;[J[[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixGetStrFeatureInfo(
    JNIEnv *jenv, jclass jclz, jlong jhandle, jstring jfield, jlongArray joutLenArray,
    jobjectArray joutValueArray) {
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> field{jenv->GetStringUTFChars(jfield, nullptr),
                                                         [&](char const *ptr) {
                                                           jenv->ReleaseStringUTFChars(jfield, ptr);
                                                         }};

  bst_ulong out_len = 0;
  char const **c_out_features;
  int ret = XGDMatrixGetStrFeatureInfo(handle, field.get(), &out_len, &c_out_features);

  jlong jlen = (jlong)out_len;
  jenv->SetLongArrayRegion(joutLenArray, 0, 1, &jlen);

  jobjectArray jinfos =
      jenv->NewObjectArray(jlen, jenv->FindClass("java/lang/String"), jenv->NewStringUTF(""));
  for (int i = 0; i < jlen; i++) {
    jenv->SetObjectArrayElement(jinfos, i, jenv->NewStringUTF(c_out_features[i]));
  }
  jenv->SetObjectArrayElement(joutValueArray, 0, jinfos);

  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSetStrFeatureInfo
 * Signature: (JLjava/lang/String;[Ljava/lang/String;])I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterSetStrFeatureInfo(
    JNIEnv *jenv, jclass jclz, jlong jhandle, jstring jfield,
    jobjectArray jfeatures) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);

  std::unique_ptr<char const, Deleter<char const>> field{jenv->GetStringUTFChars(jfield, nullptr),
                                                         [&](char const *ptr) {
                                                           jenv->ReleaseStringUTFChars(jfield, ptr);
                                                         }};
  bst_ulong feature_num = (bst_ulong)jenv->GetArrayLength(jfeatures);

  std::vector<std::string> features;
  std::vector<char const*> features_char;

  for (bst_ulong i = 0; i < feature_num; ++i) {
    jstring jfeature = (jstring)jenv->GetObjectArrayElement(jfeatures, i);
    std::unique_ptr<char const, Deleter<char const>> s{
        jenv->GetStringUTFChars(jfeature, nullptr), [&](char const *ptr) {
          if (ptr) {
            jenv->ReleaseStringUTFChars(jfeature, ptr);
          }
        }};
    features.emplace_back(s.get(), jenv->GetStringLength(jfeature));
  }

  for (size_t i = 0; i < features.size(); ++i) {
    features_char.push_back(features[i].c_str());
  }

  return XGBoosterSetStrFeatureInfo(handle, field.get(), dmlc::BeginPtr(features_char),
                                    feature_num);
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGBoosterSetGtrFeatureInfo
 * Signature: (JLjava/lang/String;[Ljava/lang/String;])I
 */
JNIEXPORT jint JNICALL
Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGBoosterGetStrFeatureInfo(
    JNIEnv *jenv, jclass jclz, jlong jhandle, jstring jfield,
    jobjectArray jout) {
  auto handle = reinterpret_cast<BoosterHandle>(jhandle);
  std::unique_ptr<char const, Deleter<char const>> field{jenv->GetStringUTFChars(jfield, nullptr),
                                                         [&](char const *ptr) {
                                                           jenv->ReleaseStringUTFChars(jfield, ptr);
                                                         }};

  bst_ulong feature_num = (bst_ulong)jenv->GetArrayLength(jout);

  const char **features;
  std::vector<char *> features_char;

  int ret =
      XGBoosterGetStrFeatureInfo(handle, field.get(), &feature_num, (const char ***)&features);
  JVM_CHECK_CALL(ret);

  for (bst_ulong i = 0; i < feature_num; i++) {
    jstring jfeature = jenv->NewStringUTF(features[i]);
    jenv->SetObjectArrayElement(jout, i, jfeature);
  }

  return ret;
}

/*
 * Class:     ml_dmlc_xgboost4j_java_XGBoostJNI
 * Method:    XGDMatrixGetQuantileCut
 * Signature: (J[[J[[F)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_java_XGBoostJNI_XGDMatrixGetQuantileCut(
    JNIEnv *jenv, jclass, jlong jhandle, jobjectArray j_indptr, jobjectArray j_values) {
  using namespace xgboost;  // NOLINT
  auto handle = reinterpret_cast<DMatrixHandle>(jhandle);

  char const *str_indptr;
  char const *str_data;
  Json config{Object{}};
  auto str_config = Json::Dump(config);

  auto ret = XGDMatrixGetQuantileCut(handle, str_config.c_str(), &str_indptr, &str_data);

  ArrayInterface<1> indptr{StringView{str_indptr}};
  ArrayInterface<1> data{StringView{str_data}};
  CHECK_GE(indptr.Shape<0>(), 2);

  // Cut ptr
  auto j_indptr_array = jenv->NewLongArray(indptr.Shape<0>());
  CHECK_EQ(indptr.type, ArrayInterfaceHandler::Type::kU8);
  CHECK_LT(indptr(indptr.Shape<0>() - 1),
           static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
  static_assert(sizeof(jlong) == sizeof(std::uint64_t));
  jenv->SetLongArrayRegion(j_indptr_array, 0, indptr.Shape<0>(),
                           static_cast<jlong const *>(indptr.data));
  jenv->SetObjectArrayElement(j_indptr, 0, j_indptr_array);

  // Cut values
  auto n_cuts = indptr(indptr.Shape<0>() - 1);
  jfloatArray jcuts_array = jenv->NewFloatArray(n_cuts);
  CHECK_EQ(data.type, ArrayInterfaceHandler::Type::kF4);
  jenv->SetFloatArrayRegion(jcuts_array, 0, n_cuts, static_cast<float const *>(data.data));
  jenv->SetObjectArrayElement(j_values, 0, jcuts_array);

  return ret;
}
