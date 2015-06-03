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

JNIEXPORT jlong JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixCreateFromFile
  (JNIEnv *jenv, jclass jcls, jstring jfname, jint jsilent) {
    jlong jresult = 0 ;
    char *fname = (char *) 0 ;
    int silent;
    void *result = 0 ;
    fname = 0;
    if (jfname) {
        fname = (char *)jenv->GetStringUTFChars(jfname, 0);
        if (!fname) return 0;
    }
    silent = (int)jsilent; 
    result = (void *)XGDMatrixCreateFromFile((char const *)fname, silent);
    *(void **)&jresult = result; 
    if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixCreateFromCSR
 * Signature: ([J[J[F)J
 */
JNIEXPORT jlong JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixCreateFromCSR
  (JNIEnv *jenv, jclass jcls, jlongArray jindptr, jlongArray jindices, jfloatArray jdata) {
    jlong jresult = 0 ;
    bst_ulong *indptr = (bst_ulong *) 0 ;
    unsigned int *indices = (unsigned int *) 0 ;
    float *data = (float *) 0 ;
    bst_ulong nindptr ;
    bst_ulong nelem;
    void *result = 0 ;
  
    (void)jenv;
    (void)jcls;
    jlong* cjindptr = jenv->GetLongArrayElements(jindptr, NULL);
    jlong* cjindices = jenv->GetLongArrayElements(jindices, NULL);
    jfloat* cjdata = jenv->GetFloatArrayElements(jdata, NULL);
  
    indptr = (bst_ulong *)cjindptr; 
    indices = (unsigned int *)cjindices; 
    data = (float *)cjdata; 
    nindptr = (bst_ulong)jenv->GetArrayLength(jindptr); 
    nelem = (bst_ulong)jenv->GetArrayLength(jdata); 
    result = (void *)XGDMatrixCreateFromCSR((unsigned long const *)indptr, (unsigned int const *)indices, (float const *)data, nindptr, nelem);
    *(void **)&jresult = result; 
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixCreateFromCSC
 * Signature: ([J[J[F)J
 */
JNIEXPORT jlong JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixCreateFromCSC
  (JNIEnv *jenv, jclass jcls, jlongArray jindptr, jlongArray jindices, jfloatArray jdata) {
    jlong jresult = 0 ;
    bst_ulong *indptr = (bst_ulong *) 0 ;
    unsigned int *indices = (unsigned int *) 0 ;
    float *data = (float *) 0 ;
    bst_ulong nindptr ;
    bst_ulong nelem;
    void *result = 0 ;
  
    (void)jenv;
    (void)jcls;
    jlong* cjindptr = jenv->GetLongArrayElements(jindptr, NULL);
    jlong* cjindices = jenv->GetLongArrayElements(jindices, NULL);
    jfloat* cjdata = jenv->GetFloatArrayElements(jdata, NULL);
  
    indptr = (bst_ulong *)cjindptr; 
    indices = (unsigned int *)cjindices; 
    data = (float *)cjdata; 
    nindptr = (bst_ulong)jenv->GetArrayLength(jindptr); 
    nelem = (bst_ulong)jenv->GetArrayLength(jdata); 
    result = (void *)XGDMatrixCreateFromCSC((unsigned long const *)indptr, (unsigned int const *)indices, (float const *)data, nindptr, nelem);
    *(void **)&jresult = result; 
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixCreateFromMat
 * Signature: ([FIIF)J
 */
JNIEXPORT jlong JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixCreateFromMat
  (JNIEnv *jenv, jclass jcls, jfloatArray jdata, jint jnrow, jint jncol, jfloat jmiss) {
    jlong jresult = 0 ;
  float *data = (float *) 0 ;
  bst_ulong nrow ;
  bst_ulong ncol ;
  float miss ;
  void *result = 0 ;
  
  (void)jenv;
  (void)jcls;
  jfloat* cjdata = jenv->GetFloatArrayElements(jdata, NULL);
  data = (float *)cjdata; 
  nrow = (bst_ulong)jnrow; 
  ncol = (bst_ulong)jncol; 
  miss = (float)jmiss; 
  result = (void *)XGDMatrixCreateFromMat((float const *)data, nrow, ncol, miss);
  *(void **)&jresult = result; 
  return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSliceDMatrix
 * Signature: (J[I)J
 */
JNIEXPORT jlong JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSliceDMatrix
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jintArray jindexset) {
    jlong jresult = 0 ;
    void *handle = (void *) 0 ;
    int *indexset = (int *) 0 ;
    bst_ulong len;
    void *result = 0 ;
    jint* cjindexset = jenv->GetIntArrayElements(jindexset, NULL);
    handle = *(void **)&jhandle; 
    indexset = (int *)cjindexset; 
    len = (bst_ulong)jenv->GetArrayLength(jindexset); 
    result = (void *)XGDMatrixSliceDMatrix(handle, (int const *)indexset, len);
    *(void **)&jresult = result; 
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixFree
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
    void *handle = (void *) 0 ;
    handle = *(void **)&jhandle; 
    XGDMatrixFree(handle);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSaveBinary
 * Signature: (JLjava/lang/String;I)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSaveBinary
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname, jint jsilent) {
    void *handle = (void *) 0 ;
    char *fname = (char *) 0 ;
    int silent ;
    handle = *(void **)&jhandle; 
    fname = 0;
    if (jfname) {
        fname = (char *)jenv->GetStringUTFChars(jfname, 0);
        if (!fname) return ;
    }
    silent = (int)jsilent; 
    XGDMatrixSaveBinary(handle, (char const *)fname, silent);
    if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSetFloatInfo
 * Signature: (JLjava/lang/String;[F)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSetFloatInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jfloatArray jarray) {
    void *handle = (void *) 0 ;
    char *field = (char *) 0 ;
    float *array = (float *) 0 ;
    bst_ulong len;


    handle = *(void **)&jhandle; 
    field = 0;
    if (jfield) {
        field = (char *)jenv->GetStringUTFChars(jfield, 0);
        if (!field) return ;
    }
    
    jfloat* cjarray = jenv->GetFloatArrayElements(jarray, NULL);
    array = (float *)cjarray; 
    len = (bst_ulong)jenv->GetArrayLength(jarray); 
    XGDMatrixSetFloatInfo(handle, (char const *)field, (float const *)array, len);
    if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSetUIntInfo
 * Signature: (JLjava/lang/String;[I)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSetUIntInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield, jintArray jarray) {
    void *handle = (void *) 0 ;
    char *field = (char *) 0 ;
    unsigned int *array = (unsigned int *) 0 ;
    bst_ulong len ;
    handle = *(void **)&jhandle; 
    field = 0;
    if (jfield) {
        field = (char *)jenv->GetStringUTFChars(jfield, 0);
        if (!field) return ;
    }
    
    jint* cjarray = jenv->GetIntArrayElements(jarray, NULL);
    array = *(unsigned int **)&cjarray; 
    len = (bst_ulong)jenv->GetArrayLength(jarray); 
    XGDMatrixSetUIntInfo(handle, (char const *)field, (unsigned int const *)array, len);
    if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixSetGroup
 * Signature: (J[I)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixSetGroup
  (JNIEnv * jenv, jclass jcls, jlong jhandle, jintArray jarray) {
    void *handle = (void *) 0 ;
    unsigned int *array = (unsigned int *) 0 ;
    bst_ulong len ;
    handle = *(void **)&jhandle;
    jint* cjarray = jenv->GetIntArrayElements(jarray, NULL);
    array = (unsigned int *)cjarray; 
    len = (bst_ulong)jenv->GetArrayLength(jarray); 
    XGDMatrixSetGroup(handle, (unsigned int const *)array, len);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixGetFloatInfo
 * Signature: (JLjava/lang/String;)[F
 */
JNIEXPORT jfloatArray JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixGetFloatInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield) {
    void *handle = (void *) 0 ;
    char *field = (char *) 0 ;
    bst_ulong len[1];
    *len = 0;
    float *result = 0 ;
      handle = *(void **)&jhandle; 
    field = 0;
    if (jfield) {
        field = (char *)jenv->GetStringUTFChars(jfield, 0);
        if (!field) return 0;
    }
    
    result = (float *)XGDMatrixGetFloatInfo((void const *)handle, (char const *)field, len);
 
    if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
  
    jsize jlen = (jsize)*len;
    jfloatArray jresult = jenv->NewFloatArray(jlen);
    jenv->SetFloatArrayRegion(jresult, 0, jlen, (jfloat *)result);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixGetUIntInfo
 * Signature: (JLjava/lang/String;)[I
 */
JNIEXPORT jintArray JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixGetUIntInfo
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfield) {
    void *handle = (void *) 0 ;
    char *field = (char *) 0 ;
    bst_ulong len[1];
    *len = 0;
    unsigned int *result = 0 ;
      handle = *(void **)&jhandle; 
    field = 0;
    if (jfield) {
        field = (char *)jenv->GetStringUTFChars(jfield, 0);
        if (!field) return 0;
    }
    
    result = (unsigned int *)XGDMatrixGetUIntInfo((void const *)handle, (char const *)field, len);
 
    if (field) jenv->ReleaseStringUTFChars(jfield, (const char *)field);
  
    jsize jlen = (jsize)*len;
    jintArray jresult = jenv->NewIntArray(jlen);
    jenv->SetIntArrayRegion(jresult, 0, jlen, (jint *)result);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGDMatrixNumRow
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGDMatrixNumRow
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
    jlong jresult = 0 ;
    void *handle = (void *) 0 ;
    bst_ulong result;
    handle = *(void **)&jhandle; 
    result = (bst_ulong)XGDMatrixNumRow((void const *)handle);
    jresult = (jlong)result; 
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterCreate
 * Signature: ([J)J
 */
JNIEXPORT jlong JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterCreate
  (JNIEnv *jenv, jclass jcls, jlongArray jhandles) {
    jlong jresult = 0 ;
    void **handles;
    bst_ulong len = 0;
    void *result = 0 ;

    
    if(jhandles) {
        len = (bst_ulong)jenv->GetArrayLength(jhandles);
        handles = new void*[len];
        //put handle from jhandles to chandles
        jlong* cjhandles = jenv->GetLongArrayElements(jhandles, NULL);
        for(jsize i=0; i<len; i++) {
            handles[i] = *(void **)&cjhandles[i];
        }
    }
    
    result = (void *)XGBoosterCreate(handles, len);
    
    delete[] handles;
    
    *(void **)&jresult = result; 
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterFree
  (JNIEnv *jenv, jclass jcls, jlong jhandle) {
    void *handle = (void *) 0 ;
    handle = *(void **)&jhandle; 
    XGBoosterFree(handle);
}


/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterSetParam
 * Signature: (JLjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterSetParam
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jname, jstring jvalue) {
    void *handle = (void *) 0 ;
    char *name = (char *) 0 ;
    char *value = (char *) 0 ;
    handle = *(void **)&jhandle; 
    
    name = 0;
    if (jname) {
        name = (char *)jenv->GetStringUTFChars(jname, 0);
        if (!name) return ;
    }
    
    value = 0;
    if (jvalue) {
        value = (char *)jenv->GetStringUTFChars(jvalue, 0);
        if (!value) return ;
    }
    XGBoosterSetParam(handle, (char const *)name, (char const *)value);
    if (name) jenv->ReleaseStringUTFChars(jname, (const char *)name);
    if (value) jenv->ReleaseStringUTFChars(jvalue, (const char *)value);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterUpdateOneIter
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterUpdateOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jint jiter, jlong jdtrain) {
    void *handle = (void *) 0 ;
    int iter ;
    void *dtrain = (void *) 0 ;
    handle = *(void **)&jhandle; 
    iter = (int)jiter; 
    dtrain = *(void **)&jdtrain; 
    XGBoosterUpdateOneIter(handle, iter, dtrain);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterBoostOneIter
 * Signature: (JJ[F[F)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterBoostOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jdtrain, jfloatArray jgrad, jfloatArray jhess) {
    void *handle = (void *) 0 ;
    void *dtrain = (void *) 0 ;
    float *grad = (float *) 0 ;
    float *hess = (float *) 0 ;
    bst_ulong len ;
    handle = *(void **)&jhandle; 
    dtrain = *(void **)&jdtrain;
    
    jfloat* cjgrad = jenv->GetFloatArrayElements(jgrad, NULL);
    jfloat* cjhess = jenv->GetFloatArrayElements(jhess, NULL);
    grad = (float *)cjgrad; 
    hess = (float *)cjhess; 
    len = (bst_ulong)jenv->GetArrayLength(jgrad);
    XGBoosterBoostOneIter(handle, dtrain, grad, hess, len);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterEvalOneIter
 * Signature: (JI[J[Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterEvalOneIter
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jint jiter, jlongArray jdmats, jobjectArray jevnames) {
    jstring jresult = 0 ;
    void *handle = (void *) 0 ;
    int iter ;
    void **dmats ;
    char **evnames ;
    bst_ulong len ;
    char *result = 0 ;
    
    handle = *(void **)&jhandle; 
    iter = (int)jiter; 
    len = (bst_ulong)jenv->GetArrayLength(jdmats); 
    
    
    if(len > 0) {
        dmats = new void*[len];
        evnames = new char*[len];
    }
    
    //put handle from jhandles to chandles
    jlong* cjdmats = jenv->GetLongArrayElements(jdmats, NULL);
    for(jsize i=0; i<len; i++) {
        dmats[i] = *(void **)&cjdmats[i];
    }
    
    //transfer jObjectArray to char**
    for(jsize i=0; i<len; i++) {
        jstring jevname = (jstring)jenv->GetObjectArrayElement(jevnames, i);
        evnames[i] = (char *)jenv->GetStringUTFChars(jevname, 0);
    }
    
    result = (char *)XGBoosterEvalOneIter(handle, iter, dmats, (char const *(*))evnames, len);
    
    if(len > 0) {
        delete[] dmats;
        //release string chars
        for(jsize i=0; i<len; i++) {
            jstring jevname = (jstring)jenv->GetObjectArrayElement(jevnames, i);
            jenv->ReleaseStringUTFChars(jevname, (const char*)evnames[i]);
        }        
        delete[] evnames;
    }
    
    if (result) jresult = jenv->NewStringUTF((const char *)result);
  
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterPredict
 * Signature: (JJIJ)[F
 */
JNIEXPORT jfloatArray JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterPredict
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jdmat, jint joption_mask, jlong jntree_limit) {
    void *handle = (void *) 0 ;
    void *dmat = (void *) 0 ;
    int option_mask ;
    unsigned int ntree_limit ;
    bst_ulong len[1];
    *len = 0;
    float *result = 0 ;
    handle = *(void **)&jhandle; 
    dmat = *(void **)&jdmat; 
    option_mask = (int)joption_mask; 
    ntree_limit = (unsigned int)jntree_limit; 

    result = (float *)XGBoosterPredict(handle, dmat, option_mask, ntree_limit, len);
    
    jsize jlen = (jsize)*len;
    jfloatArray jresult = jenv->NewFloatArray(jlen);
    jenv->SetFloatArrayRegion(jresult, 0, jlen, (jfloat *)result);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterLoadModel
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterLoadModel
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname) {
    void *handle = (void *) 0 ;
    char *fname = (char *) 0 ;
    handle = *(void **)&jhandle; 
    fname = 0;
    if (jfname) {
        fname = (char *)jenv->GetStringUTFChars(jfname, 0);
        if (!fname) return ;
    }
    XGBoosterLoadModel(handle,(char const *)fname);
    if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterSaveModel
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterSaveModel
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfname) {
    void *handle = (void *) 0 ;
    char *fname = (char *) 0 ;
    handle = *(void **)&jhandle; 
    fname = 0;
    if (jfname) {
        fname = (char *)jenv->GetStringUTFChars(jfname, 0);
        if (!fname) return ;
    }
    XGBoosterSaveModel(handle, (char const *)fname);
    if (fname) jenv->ReleaseStringUTFChars(jfname, (const char *)fname);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterLoadModelFromBuffer
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterLoadModelFromBuffer
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jlong jbuf, jlong jlen) {
    void *handle = (void *) 0 ;
    void *buf = (void *) 0 ;
    bst_ulong len ;
    handle = *(void **)&jhandle; 
    buf = *(void **)&jbuf; 
    len = (bst_ulong)jlen; 
    XGBoosterLoadModelFromBuffer(handle, (void const *)buf, len);
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterGetModelRaw
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterGetModelRaw
  (JNIEnv * jenv, jclass jcls, jlong jhandle) {
    jstring jresult = 0 ;
    void *handle = (void *) 0 ;
    bst_ulong len[1];
    *len = 0;
    char *result = 0 ;
    handle = *(void **)&jhandle; 

    result = (char *)XGBoosterGetModelRaw(handle, len);
    if (result) jresult = jenv->NewStringUTF((const char *)result);
    return jresult;
}

/*
 * Class:     org_dmlc_xgboost4j_wrapper_XgboostJNI
 * Method:    XGBoosterDumpModel
 * Signature: (JLjava/lang/String;I)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_org_dmlc_xgboost4j_wrapper_XgboostJNI_XGBoosterDumpModel
  (JNIEnv *jenv, jclass jcls, jlong jhandle, jstring jfmap, jint jwith_stats) {    
    void *handle = (void *) 0 ;
    char *fmap = (char *) 0 ;
    int with_stats ;
    bst_ulong len[1];
    *len = 0;
    
    char **result = 0 ;
    handle = *(void **)&jhandle; 
    fmap = 0;
    if (jfmap) {
        fmap = (char *)jenv->GetStringUTFChars(jfmap, 0);
        if (!fmap) return 0;
    }
    with_stats = (int)jwith_stats;

    result = (char **)XGBoosterDumpModel(handle, (char const *)fmap, with_stats, len);
    
    jsize jlen = (jsize)*len;
    jobjectArray jresult = jenv->NewObjectArray(jlen, jenv->FindClass("java/lang/String"), jenv->NewStringUTF(""));
    for(int i=0 ; i<jlen; i++) {
        jenv->SetObjectArrayElement(jresult, i, jenv->NewStringUTF((const char*)result[i]));
    }
    
    if (fmap) jenv->ReleaseStringUTFChars(jfmap, (const char *)fmap);
    return jresult;
}