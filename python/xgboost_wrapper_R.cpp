#include "xgboost_wrapper_R.h"
#include "xgboost_wrapper.h"
#include "../src/utils/utils.h"
using namespace xgboost;

extern "C" {
  void _DMatrixFinalizer(SEXP ext) {    
    if (R_ExternalPtrAddr(ext) == NULL) return;
    XGDMatrixFree(R_ExternalPtrAddr(ext));
    R_ClearExternalPtr(ext);
  }
  SEXP XGDMatrixCreateFromFile_R(SEXP fname) {
    void *handle = XGDMatrixCreateFromFile(CHAR(asChar(fname)), 0);
    SEXP ret = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ret, _DMatrixFinalizer, TRUE);
    UNPROTECT(1);
    return ret;
  } 
}
