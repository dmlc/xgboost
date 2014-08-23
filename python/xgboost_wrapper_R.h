#ifndef XGBOOST_WRAPPER_R_H_
#define XGBOOST_WRAPPER_R_H_
/*!
 * \file xgboost_wrapper_R.h
 * \author Tianqi Chen
 * \brief R wrapper of xgboost
 */
extern "C" {
#include <Rinternals.h>
}

extern "C" {
  /*!
   * \brief load a data matrix 
   * \return a loaded data matrix
   */
  SEXP XGDMatrixCreateFromFile_R(SEXP fname);

};
#endif  // XGBOOST_WRAPPER_H_
