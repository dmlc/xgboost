/*
 Copyright (c) 2014 by Contributors 
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

#include <jni.h>
#include "../wrapper/xgboost_wrapper.h"
#include "xgboost4j_wrapper.h"

JNIEXPORT jstring JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBGetLastError
  (JNIEnv *jenv, jclass jcls) {
    jstring jresult = 0 ;
    char* result = 0;
    result = (char *)XGBGetLastError();
    if (result) jresult = jenv->NewStringUTF((const char *)result);
    return jresult;
}

JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixCreateFromFile
  (JNIEnv *jenv, jclass jcls, jstring jfname, jint jsilent, jlongArray jout) {
    jint jresult = 0 ;
    char *fname = (char *) 0 ;
    int silent;
    void* result[1];
    unsigned long out[1];
    
    fname = (char *)jenv->GetStringUTFChars(jfname, 0);
    
    silent = (int)jsilent; 
    jresult = (jint) XGDMatrixCreateFromFile((char const *)fname, silent, result);
    
    
    *(void **)&out[0] = *result;

    if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
    
    jenv->SetLongArrayRegion(jout, 0, 1, (const jlong *) out);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixCreateFromCSR
 * Signature: ([J[J[F)J
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixCreateFromCSR
  (JNIEnv *jenv, jclass jcls, jlongArray jindptr, jintArray jindices, jfloatArray jdata, jlongArray jout) {
    jint jresult = 0 ;
    bst_ulong nindptr ;
    bst_ulong nelem;
    void *result[1];
    unsigned long out[1];
  
    jlong* indptr = jenv->GetLongArrayElements(jindptr, 0);
    jint* indices = jenv->GetIntArrayElements(jindices, 0);
    jfloat* data = jenv->GetFloatArrayElements(jdata, 0); 
    nindptr = (bst_ulong)jenv->GetArrayLength(jindptr); 
    nelem = (bst_ulong)jenv->GetArrayLength(jdata); 

    jresult = (jint) XGDMatrixCreateFromCSR((unsigned long const *)indptr, (unsigned int const *)indices, (float const *)data, nindptr, nelem, result);    
    *(void **)&out[0] = *result;
    jenv->SetLongArrayRegion(jout, 0, 1, (const jlong *) out);
    
    //release
    jenv->ReleaseLongArrayElements(jindptr, indptr, 0);
    jenv->ReleaseIntArrayElements(jindices, indices, 0);
    jenv->ReleaseFloatArrayElements(jdata, data, 0);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixCreateFromCSC
 * Signature: ([J[J[F)J
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixCreateFromCSC
  (JNIEnv *jenv, jclass jcls, jlongArray jindptr, jintArray jindices, jfloatArray jdata, jlongArray jout) {
    jint jresult = 0;
    bst_ulong nindptr ;
    bst_ulong nelem;
    void *result[1];
    unsigned long out[1];
  
    jlong* indptr = jenv->GetLongArrayElements(jindptr, NULL);
    jint* indices = jenv->GetIntArrayElements(jindices, 0);
    jfloat* data = jenv->GetFloatArrayElements(jdata, NULL);
    nindptr = (bst_ulong)jenv->GetArrayLength(jindptr); 
    nelem = (bst_ulong)jenv->GetArrayLength(jdata); 

    jresult = (jint) XGDMatrixCreateFromCSC((unsigned long const *)indptr, (unsigned int const *)indices, (float const *)data, nindptr, nelem, result);  
    *(void **)&out[0] = *result;
    jenv->SetLongArrayRegion(jout, 0, 1, (const jlong *) out);
    
    //release
    jenv->ReleaseLongArrayElements(jindptr, indptr, 0);
    jenv->ReleaseIntArrayElements(jindices, indices, 0);
    jenv->ReleaseFloatArrayElements(jdata, data, 0);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixCreateFromMat
 * Signature: ([FIIF)J
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixCreateFromMat
  (JNIEnv *jenv, jclass jcls, jfloatArray jdata, jint jnrow, jint jncol, jfloat jmiss, jlongArray jout) {
    jint jresult = 0 ;
    bst_ulong nrow ;
    bst_ulong ncol ;
    float miss ;
    void *result[1];
    unsigned long out[1];
  

    jfloat* data = jenv->GetFloatArrayElements(jdata, 0);
    nrow = (bst_ulong)jnrow; 
    ncol = (bst_ulong)jncol; 
    miss = (float)jmiss; 
    
    jresult = (jint) XGDMatrixCreateFromMat((float const *)data, nrow, ncol, miss, result);
    *(void **)&out[0] = *result;
    jenv->SetLongArrayRegion(jout, 0, 1, (const jlong *) out);
  
    //release
    jenv->ReleaseFloatArrayElements(jdata, data, 0);
  
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSliceDMatrix
 * Signature: (J[I)J
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSliceDMatrix
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jintArray jindexset, jlongArray jout) {
    jint jresult = 0 ;
    void *handle = (void *) 0 ;
    bst_ulong len;
    void *result[1];
    unsigned long out[1];

    jint* indexset = jenv->GetIntArrayElements(jindexset, 0);
    handle = *(void **)&jhandle; 
    len = (bst_ulong)jenv->GetArrayLength(jindexset); 

    jresult = (jint) XGDMatrixSliceDMatrix(handle, (int const *)indexset, len, result);
    *(void **)&out[0] = *result;
    jenv->SetLongArrayRegion(jout, 0, 1, (const jlong *) out);
    
    //release
    jenv->ReleaseIntArrayElements(jindexset, indexset, 0);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixFree
 * Signature: (J)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixFree
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    handle = *(void **)&jhandle; 
    jresult = (jint) XGDMatrixFree(handle);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSaveBinary
 * Signature: (JLjava/lang/String;I)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSaveBinary
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname, jint jsilent) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    char *fname = (char *) 0 ;
    int silent ;
    handle = *(void **)&jhandle; 
    fname = 0;
    fname = (char *)jenv->GetStringUTFChars(jfname, 0);

    silent = (int)jsilent; 
    jresult = (jint) XGDMatrixSaveBinary(handle, (char const *)fname, silent);
    if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSetFloatInfo
 * Signature: (JLjava/lang/String;[F)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSetFloatInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jfloatArray jarray) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    char *field = (char *) 0 ;
    bst_ulong len;


    handle = *(void **)&jhandle; 

    field = (char *)jenv->GetStringUTFChars(jfield, 0);

    
    jfloat* array = jenv->GetFloatArrayElements(jarray, NULL);
    len = (bst_ulong)jenv->GetArrayLength(jarray); 
    jresult = (jint) XGDMatrixSetFloatInfo(handle, (char const *)field, (float const *)array, len);
    
    //release
    if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
    jenv->ReleaseFloatArrayElements(jarray, array, 0);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSetUIntInfo
 * Signature: (JLjava/lang/String;[I)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSetUIntInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jintArray jarray) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    char *field = (char *) 0 ;
    bst_ulong len ;
    handle = *(void **)&jhandle; 
    field = 0;
    field = (char *)jenv->GetStringUTFChars(jfield, 0);

    
    jint* array = jenv->GetIntArrayElements(jarray, NULL);
    len = (bst_ulong)jenv->GetArrayLength(jarray);

    jresult = (jint) XGDMatrixSetUIntInfo(handle, (char const *)field, (unsigned int const *)array, len);
    //release
    if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
    jenv->ReleaseIntArrayElements(jarray, array, 0);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSetGroup
 * Signature: (J[I)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSetGroup
  (JNIEnv * jenv, jclass jcls, jlong jhandle, jintArray jarray) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    bst_ulong len ;

    handle = *(void **)&jhandle;
    jint* array = jenv->GetIntArrayElements(jarray, NULL);
    len = (bst_ulong)jenv->GetArrayLength(jarray); 

    jresult = (jint) XGDMatrixSetGroup(handle, (unsigned int const *)array, len);
    
    //release
    jenv->ReleaseIntArrayElements(jarray, array, 0);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixGetFloatInfo
 * Signature: (JLjava/lang/String;)[F
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixGetFloatInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jobjectArray jout) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    char *field = (char *) 0 ;
    bst_ulong len[1];
    *len = 0;
    float *result[1];

    handle = *(void **)&jhandle; 
    field = 0;
    if (jfield) {
        field = (char *)jenv->GetStringUTFChars(jfield, 0);
        if (!field) return 0;
    }
    
    jresult = (jint) XGDMatrixGetFloatInfo(handle, (char const *)field, len, (const float **) result);
 
    if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
  
    jsize jlen = (jsize)*len;
    jfloatArray jarray = jenv->NewFloatArray(jlen);
    jenv->SetFloatArrayRegion(jarray, 0, jlen, (jfloat *) *result);
    jenv->SetObjectArrayElement(jout, 0, (jobject) jarray);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixGetUIntInfo
 * Signature: (JLjava/lang/String;)[I
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixGetUIntInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jobjectArray jout) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    char *field = (char *) 0 ;
    bst_ulong len[1];
    *len = 0;
    unsigned int *result[1];

    handle = *(void **)&jhandle; 
    field = (char *)jenv->GetStringUTFChars(jfield, 0);
    
    jresult = (jint) XGDMatrixGetUIntInfo(handle, (char const *)field, len, (const unsigned int **) result);
 
    if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
  
    jsize jlen = (jsize)*len;
    jintArray jarray = jenv->NewIntArray(jlen);
    jenv->SetIntArrayRegion(jarray, 0, jlen, (jint *) *result);
    jenv->SetObjectArrayElement(jout, 0, jarray);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixNumRow
 * Signature: (J)J
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixNumRow
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlongArray jout) {
    jint jresult = 0 ;
    void *handle = (void *) 0 ;
    bst_ulong result[1];
    handle = *(void **)&jhandle; 
    jresult = (jint) XGDMatrixNumRow(handle, result);
    jenv->SetLongArrayRegion(jout, 0, 1, (const jlong *) result);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterCreate
 * Signature: ([J)J
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterCreate
  (JNIEnv *jenv, jclass jcls, jlongArray jhandles, jlongArray jout) {
    jint jresult = 0;
    void **handles = 0;
    bst_ulong len = 0;
    void *result[1];
    jlong* cjhandles = 0;
    unsigned long out[1];
    
    if(jhandles) {
        len = (bst_ulong)jenv->GetArrayLength(jhandles);
        handles = new void*[len];
        //put handle from jhandles to chandles
        cjhandles = jenv->GetLongArrayElements(jhandles, 0);
        for(bst_ulong i=0; i<len; i++) {
            handles[i] = *(void **)&cjhandles[i];
        }
    }
    
    jresult = (jint) XGBoosterCreate(handles, len, result);
    
    //release
    if(jhandles) {
        delete[] handles;
        jenv->ReleaseLongArrayElements(jhandles, cjhandles, 0);
    }
    
    *(void **)&out[0] = *result; 
    jenv->SetLongArrayRegion(jout, 0, 1, (const jlong *) out);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterFree
 * Signature: (J)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterFree
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
    void *handle = (void *) 0 ;
    handle = *(void **)&jhandle; 
    return (jint) XGBoosterFree(handle);
}


/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterSetParam
 * Signature: (JLjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterSetParam
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jname, jstring jvalue) {
    jint jresult = -1;
    void *handle = (void *) 0 ;
    char *name = (char *) 0 ;
    char *value = (char *) 0 ;
    handle = *(void **)&jhandle; 
    
    name = (char *)jenv->GetStringUTFChars(jname, 0);
    value = (char *)jenv->GetStringUTFChars(jvalue, 0);

    jresult = (jint) XGBoosterSetParam(handle, (char const *)name, (char const *)value);
    if (name) jenv->ReleaseStringUTFChars(jname, (const char *)name);
    if (value) jenv->ReleaseStringUTFChars(jvalue, (const char *)value);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterUpdateOneIter
 * Signature: (JIJ)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterUpdateOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jint jiter, jlong jdtrain) {
    void *handle = (void *) 0 ;
    int iter ;
    void *dtrain = (void *) 0 ;
    handle = *(void **)&jhandle; 
    iter = (int)jiter; 
    dtrain = *(void **)&jdtrain; 
    return (jint) XGBoosterUpdateOneIter(handle, iter, dtrain);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterBoostOneIter
 * Signature: (JJ[F[F)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterBoostOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jdtrain, jfloatArray jgrad, jfloatArray jhess) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    void *dtrain = (void *) 0 ;
    bst_ulong len ;

    handle = *(void **)&jhandle; 
    dtrain = *(void **)&jdtrain;    
    jfloat* grad = jenv->GetFloatArrayElements(jgrad, 0);
    jfloat* hess = jenv->GetFloatArrayElements(jhess, 0);
    len = (bst_ulong)jenv->GetArrayLength(jgrad);
    jresult = (jint) XGBoosterBoostOneIter(handle, dtrain, grad, hess, len);
    
    //release
    jenv->ReleaseFloatArrayElements(jgrad, grad, 0);
    jenv->ReleaseFloatArrayElements(jhess, hess, 0);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterEvalOneIter
 * Signature: (JI[J[Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterEvalOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jint jiter, jlongArray jdmats, jobjectArray jevnames, jobjectArray jout) {
    jint jresult = 0 ;
    void *handle = (void *) 0 ;
    int iter ;
    void **dmats = 0;
    char **evnames = 0;
    bst_ulong len ;
    char *result[1];
    
    handle = *(void **)&jhandle; 
    iter = (int)jiter; 
    len = (bst_ulong)jenv->GetArrayLength(jdmats); 
    
    
    if(len > 0) {
        dmats = new void*[len];
        evnames = new char*[len];
    }
    
    //put handle from jhandles to chandles
    jlong* cjdmats = jenv->GetLongArrayElements(jdmats, 0);
    for(bst_ulong i=0; i<len; i++) {
        dmats[i] = *(void **)&cjdmats[i];
    }
    
    //transfer jObjectArray to char**
    for(bst_ulong i=0; i<len; i++) {
        jstring jevname = (jstring)jenv->GetObjectArrayElement(jevnames, i);
        evnames[i] = (char *)jenv->GetStringUTFChars(jevname, 0);
    }
    
    jresult = (jint) XGBoosterEvalOneIter(handle, iter, dmats, (char const *(*))evnames, len, (const char **) result);
    
    if(len > 0) {
        delete[] dmats;
        //release string chars
        for(bst_ulong i=0; i<len; i++) {
            jstring jevname = (jstring)jenv->GetObjectArrayElement(jevnames, i);
            jenv->ReleaseStringUTFChars(jevname, (const char*)evnames[i]);
        }        
        delete[] evnames;
        jenv->ReleaseLongArrayElements(jdmats, cjdmats, 0);
    }
    
    jstring jinfo = 0;
    if (*result) jinfo = jenv->NewStringUTF((const char *) *result);
    jenv->SetObjectArrayElement(jout, 0, jinfo);
  
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterPredict
 * Signature: (JJIJ)[F
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterPredict
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jdmat, jint joption_mask, jlong jntree_limit, jobjectArray jout) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    void *dmat = (void *) 0 ;
    int option_mask ;
    unsigned int ntree_limit ;
    bst_ulong len[1];
    *len = 0;
    float *result[1];
    
    handle = *(void **)&jhandle; 
    dmat = *(void **)&jdmat; 
    option_mask = (int)joption_mask; 
    ntree_limit = (unsigned int)jntree_limit; 

    jresult = (jint) XGBoosterPredict(handle, dmat, option_mask, ntree_limit, len, (const float **) result);
    
    jsize jlen = (jsize)*len;
    jfloatArray jarray = jenv->NewFloatArray(jlen);
    jenv->SetFloatArrayRegion(jarray, 0, jlen, (jfloat *) *result);
    jenv->SetObjectArrayElement(jout, 0, jarray);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterLoadModel
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterLoadModel
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    char *fname = (char *) 0 ;
    handle = *(void **)&jhandle; 
    
    fname = (char *)jenv->GetStringUTFChars(jfname, 0);

    
    jresult = (jint) XGBoosterLoadModel(handle,(char const *)fname);
    if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterSaveModel
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterSaveModel
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    char *fname = (char *) 0 ;
    handle = *(void **)&jhandle; 
    fname = 0;
    fname = (char *)jenv->GetStringUTFChars(jfname, 0);

    jresult = (jint) XGBoosterSaveModel(handle, (char const *)fname);
    if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
    
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterLoadModelFromBuffer
 * Signature: (JJJ)V
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterLoadModelFromBuffer
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jbuf, jlong jlen) {
    void *handle = (void *) 0 ;
    void *buf = (void *) 0 ;
    bst_ulong len ;
    handle = *(void **)&jhandle; 
    buf = *(void **)&jbuf; 
    len = (bst_ulong)jlen; 
    return (jint) XGBoosterLoadModelFromBuffer(handle, (void const *)buf, len);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterGetModelRaw
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterGetModelRaw
  (JNIEnv * jenv, jclass jcls, jlong jhandle, jobjectArray jout) {
    jint jresult = 0 ;
    jstring jinfo = 0;
    void *handle = (void *) 0 ;
    bst_ulong len[1];
    *len = 0;
    char *result[1];
    handle = *(void **)&jhandle; 

    jresult = (jint)XGBoosterGetModelRaw(handle, len, (const char **) result);
    if (*result){
        jinfo = jenv->NewStringUTF((const char *) *result);
        jenv->SetObjectArrayElement(jout, 0, jinfo);
    }
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterDumpModel
 * Signature: (JLjava/lang/String;I)[Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterDumpModel
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfmap, jint jwith_stats, jobjectArray jout) {
    jint jresult = 0;
    void *handle = (void *) 0 ;
    char *fmap = (char *) 0 ;
    int with_stats ;
    bst_ulong len[1];
    *len = 0;
    
    char **result[1];
    handle = *(void **)&jhandle; 
    fmap = 0;
    if (jfmap) {
        fmap = (char *)jenv->GetStringUTFChars(jfmap, 0);
        if (!fmap) return 0;
    }
    with_stats = (int)jwith_stats;

    jresult = (jint) XGBoosterDumpModel(handle, (const char *)fmap, with_stats, len, (const char ***) result);
    
    jsize jlen = (jsize)*len;
    jobjectArray jinfos = jenv->NewObjectArray(jlen, jenv->FindClass("java/lang/String"), jenv->NewStringUTF(""));
    for(int i=0 ; i<jlen; i++) {
        jenv->SetObjectArrayElement(jinfos, i, jenv->NewStringUTF((const char*) result[0][i]));
    }
    jenv->SetObjectArrayElement(jout, 0, jinfos);
    
    if (fmap) jenv->ReleaseStringUTFChars(jfmap, (const char *)fmap);
    
    return jresult;
}