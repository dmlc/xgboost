/* Copyright (c) 2015 by Contributors
 *
 * This file was initially generated using the following R command:
 * tools::package_native_routine_registration_skeleton('.', con = 'src/init.c', character_only = F)
 * and edited to conform to xgboost C linter requirements. For details, see
 * https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Registering-native-routines
 */
#include <R.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <R_ext/Rdynload.h>

/* FIXME:
Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP XGBoosterBoostOneIter_R(SEXP, SEXP, SEXP, SEXP);
extern SEXP XGBoosterCreate_R(SEXP);
extern SEXP XGBoosterDumpModel_R(SEXP, SEXP, SEXP, SEXP);
extern SEXP XGBoosterEvalOneIter_R(SEXP, SEXP, SEXP, SEXP);
extern SEXP XGBoosterGetAttrNames_R(SEXP);
extern SEXP XGBoosterGetAttr_R(SEXP, SEXP);
extern SEXP XGBoosterLoadModelFromRaw_R(SEXP, SEXP);
extern SEXP XGBoosterLoadModel_R(SEXP, SEXP);
extern SEXP XGBoosterSaveJsonConfig_R(SEXP handle);
extern SEXP XGBoosterLoadJsonConfig_R(SEXP handle, SEXP value);
extern SEXP XGBoosterSerializeToBuffer_R(SEXP handle);
extern SEXP XGBoosterUnserializeFromBuffer_R(SEXP handle, SEXP raw);
extern SEXP XGBoosterModelToRaw_R(SEXP);
extern SEXP XGBoosterPredict_R(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP XGBoosterSaveModel_R(SEXP, SEXP);
extern SEXP XGBoosterSetAttr_R(SEXP, SEXP, SEXP);
extern SEXP XGBoosterSetParam_R(SEXP, SEXP, SEXP);
extern SEXP XGBoosterUpdateOneIter_R(SEXP, SEXP, SEXP);
extern SEXP XGCheckNullPtr_R(SEXP);
extern SEXP XGDMatrixCreateFromCSC_R(SEXP, SEXP, SEXP, SEXP);
extern SEXP XGDMatrixCreateFromFile_R(SEXP, SEXP);
extern SEXP XGDMatrixCreateFromMat_R(SEXP, SEXP);
extern SEXP XGDMatrixGetInfo_R(SEXP, SEXP);
extern SEXP XGDMatrixNumCol_R(SEXP);
extern SEXP XGDMatrixNumRow_R(SEXP);
extern SEXP XGDMatrixSaveBinary_R(SEXP, SEXP, SEXP);
extern SEXP XGDMatrixSetInfo_R(SEXP, SEXP, SEXP);
extern SEXP XGDMatrixSliceDMatrix_R(SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"XGBoosterBoostOneIter_R",     (DL_FUNC) &XGBoosterBoostOneIter_R,     4},
  {"XGBoosterCreate_R",           (DL_FUNC) &XGBoosterCreate_R,           1},
  {"XGBoosterDumpModel_R",        (DL_FUNC) &XGBoosterDumpModel_R,        4},
  {"XGBoosterEvalOneIter_R",      (DL_FUNC) &XGBoosterEvalOneIter_R,      4},
  {"XGBoosterGetAttrNames_R",     (DL_FUNC) &XGBoosterGetAttrNames_R,     1},
  {"XGBoosterGetAttr_R",          (DL_FUNC) &XGBoosterGetAttr_R,          2},
  {"XGBoosterLoadModelFromRaw_R", (DL_FUNC) &XGBoosterLoadModelFromRaw_R, 2},
  {"XGBoosterLoadModel_R",        (DL_FUNC) &XGBoosterLoadModel_R,        2},
  {"XGBoosterSaveJsonConfig_R",   (DL_FUNC) &XGBoosterSaveJsonConfig_R,   1},
  {"XGBoosterLoadJsonConfig_R",   (DL_FUNC) &XGBoosterLoadJsonConfig_R,   2},
  {"XGBoosterSerializeToBuffer_R",     (DL_FUNC) &XGBoosterSerializeToBuffer_R,     1},
  {"XGBoosterUnserializeFromBuffer_R", (DL_FUNC) &XGBoosterUnserializeFromBuffer_R, 2},
  {"XGBoosterModelToRaw_R",       (DL_FUNC) &XGBoosterModelToRaw_R,       1},
  {"XGBoosterPredict_R",          (DL_FUNC) &XGBoosterPredict_R,          5},
  {"XGBoosterSaveModel_R",        (DL_FUNC) &XGBoosterSaveModel_R,        2},
  {"XGBoosterSetAttr_R",          (DL_FUNC) &XGBoosterSetAttr_R,          3},
  {"XGBoosterSetParam_R",         (DL_FUNC) &XGBoosterSetParam_R,         3},
  {"XGBoosterUpdateOneIter_R",    (DL_FUNC) &XGBoosterUpdateOneIter_R,    3},
  {"XGCheckNullPtr_R",            (DL_FUNC) &XGCheckNullPtr_R,            1},
  {"XGDMatrixCreateFromCSC_R",    (DL_FUNC) &XGDMatrixCreateFromCSC_R,    4},
  {"XGDMatrixCreateFromFile_R",   (DL_FUNC) &XGDMatrixCreateFromFile_R,   2},
  {"XGDMatrixCreateFromMat_R",    (DL_FUNC) &XGDMatrixCreateFromMat_R,    2},
  {"XGDMatrixGetInfo_R",          (DL_FUNC) &XGDMatrixGetInfo_R,          2},
  {"XGDMatrixNumCol_R",           (DL_FUNC) &XGDMatrixNumCol_R,           1},
  {"XGDMatrixNumRow_R",           (DL_FUNC) &XGDMatrixNumRow_R,           1},
  {"XGDMatrixSaveBinary_R",       (DL_FUNC) &XGDMatrixSaveBinary_R,       3},
  {"XGDMatrixSetInfo_R",          (DL_FUNC) &XGDMatrixSetInfo_R,          3},
  {"XGDMatrixSliceDMatrix_R",     (DL_FUNC) &XGDMatrixSliceDMatrix_R,     2},
  {NULL, NULL, 0}
};

#if defined(_WIN32)
__declspec(dllexport)
#endif  // defined(_WIN32)
void R_init_xgboost(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
