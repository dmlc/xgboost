#include <vector>
#include <string>
#include <cstring>
#include "xgboost_wrapper.h"
#include "xgboost_R.h"
#include "../src/utils/utils.h"
#include "../src/utils/omp.h"

using namespace xgboost;

extern "C" {
  void _DMatrixFinalizer(SEXP ext) {    
    if (R_ExternalPtrAddr(ext) == NULL) return;
    XGDMatrixFree(R_ExternalPtrAddr(ext));
    R_ClearExternalPtr(ext);
  }
  SEXP XGDMatrixCreateFromFile_R(SEXP fname, SEXP silent) {
    void *handle = XGDMatrixCreateFromFile(CHAR(asChar(fname)), asInteger(silent));
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  void XGDMatrixSaveBinary_R(SEXP handle, SEXP fname, SEXP silent) {
    XGDMatrixSaveBinary(R_ExternalPtrAddr(handle),
                        CHAR(asChar(fname)), asInteger(silent));
  }
  void XGDMatrixSetInfo_R(SEXP handle, SEXP field, SEXP array) {
    int len = length(array);
    const char *name = CHAR(asChar(field));
    if (!strcmp("group", name)) {
      std::vector<unsigned> vec(len);
      #pragma omp parallel for schedule(static)      
      for (int i = 0; i < len; ++i) {
        vec[i] = static_cast<unsigned>(INTEGER(array)[i]);
      }
      XGDMatrixSetGroup(R_ExternalPtrAddr(handle), &vec[0], len);
      return;
    }
    {
      std::vector<float> vec(len);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < len; ++i) {
        vec[i] = REAL(array)[i];
      }
      XGDMatrixSetFloatInfo(R_ExternalPtrAddr(handle), 
                            CHAR(asChar(field)),
                            &vec[0], len);
    }
  }
  SEXP XGDMatrixGetInfo_R(SEXP handle, SEXP field) {
    size_t olen;
    const float *res = XGDMatrixGetFloatInfo(R_ExternalPtrAddr(handle),
                                             CHAR(asChar(field)), &olen);
    SEXP ret = PROTECT(allocVector(REALSXP, olen));
    for (size_t i = 0; i < olen; ++i) {
      REAL(ret)[i] = res[i];
    }
    UNPROTECT(1);
    return ret;
  }
  // functions related to booster
  void _BoosterFinalizer(SEXP ext) {    
    if (R_ExternalPtrAddr(ext) == NULL) return;
    XGBoosterFree(R_ExternalPtrAddr(ext));
    R_ClearExternalPtr(ext);
  }
  SEXP XGBoosterCreate_R(SEXP dmats) {
    int len = length(dmats);
    std::vector<void*> dvec;
    for (int i = 0; i < len; ++i){
      dvec.push_back(R_ExternalPtrAddr(VECTOR_ELT(dmats, i)));
    }
    void *handle = XGBoosterCreate(&dvec[0], dvec.size());
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _BoosterFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  }
  void XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val) {
    XGBoosterSetParam(R_ExternalPtrAddr(handle),
                      CHAR(asChar(name)),
                      CHAR(asChar(val)));
  }
  void XGBoosterUpdateOneIter_R(SEXP handle, SEXP iter, SEXP dtrain) {
    XGBoosterUpdateOneIter(R_ExternalPtrAddr(handle),
                           asInteger(iter),
                           R_ExternalPtrAddr(dtrain));
  }
  void XGBoosterBoostOneIter_R(SEXP handle, SEXP dtrain, SEXP grad, SEXP hess) {
    utils::Check(length(grad) == length(hess), "gradient and hess must have same length");
    int len = length(grad);
    std::vector<float> tgrad(len), thess(len);
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < len; ++j) {
      tgrad[j] = REAL(grad)[j];
      thess[j] = REAL(hess)[j];
    }
    XGBoosterBoostOneIter(R_ExternalPtrAddr(handle),
                          R_ExternalPtrAddr(dtrain),
                          &tgrad[0], &thess[0], len);
  }
  SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames) {
    utils::Check(length(dmats) == length(evnames), "dmats and evnams must have same length");
    int len = length(dmats);
    std::vector<void*> vec_dmats;
    std::vector<std::string> vec_names;
    std::vector<const char*> vec_sptr;
    for (int i = 0; i < len; ++i){
      vec_dmats.push_back(R_ExternalPtrAddr(VECTOR_ELT(dmats, i)));
      vec_names.push_back(std::string(CHAR(asChar(VECTOR_ELT(evnames, i)))));
      vec_sptr.push_back(vec_names.back().c_str());
    }
    return mkString(XGBoosterEvalOneIter(R_ExternalPtrAddr(handle),
                                         asInteger(iter),
                                         &vec_dmats[0], &vec_sptr[0], len)); 
  }
  SEXP XGBoosterPredict_R(SEXP handle, SEXP dmat, SEXP output_margin) {
    size_t olen;
    const float *res = XGBoosterPredict(R_ExternalPtrAddr(handle),
                                        R_ExternalPtrAddr(dmat),
                                        asInteger(output_margin),
                                        &olen);
    SEXP ret = PROTECT(allocVector(REALSXP, olen));
    for (size_t i = 0; i < olen; ++i) {
      REAL(ret)[i] = res[i];
    }
    UNPROTECT(1);
    return ret;
  }
  void XGBoosterLoadModel_R(SEXP handle, SEXP fname) {
    XGBoosterLoadModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname)));
  }
  void XGBoosterSaveModel_R(SEXP handle, SEXP fname) {
    XGBoosterSaveModel(R_ExternalPtrAddr(handle), CHAR(asChar(fname)));
  }
  void XGBoosterDumpModel_R(SEXP handle, SEXP fname, SEXP fmap) {
    size_t olen;
    const char **res = XGBoosterDumpModel(R_ExternalPtrAddr(handle),
                                          CHAR(asChar(fmap)),
                                          &olen);
    FILE *fo = utils::FopenCheck(CHAR(asChar(fname)), "w");
    for (size_t i = 0; i < olen; ++i) {
      fprintf(fo, "booster[%lu]:\n", i);
      fprintf(fo, "%s\n", res[i]);
    }
    fclose(fo);
  }
}
