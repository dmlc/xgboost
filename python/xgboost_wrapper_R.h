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
   * \fname name of the content
   * \return a loaded data matrix
   */
  SEXP XGDMatrixCreateFromFile_R(SEXP fname);
  /*! 
   * \brief create xgboost learner 
   * \param dmats a list of dmatrix handles that will be cached
   */  
  SEXP XGBoosterCreate_R(SEXP dmats);
  /*! 
   * \brief set parameters 
   * \param handle handle
   * \param name  parameter name
   * \param val value of parameter
   */
  void XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val);
  /*! 
   * \brief update the model in one round using dtrain
   * \param handle handle
   * \param iter current iteration rounds
   * \param dtrain training data
   */
  void XGBoosterUpdateOneIter_R(SEXP ext, SEXP iter, SEXP dtrain);
  /*!
   * \brief get evaluation statistics for xgboost
   * \param handle handle
   * \param iter current iteration rounds
   * \param dmats list of handles to dmatrices
   * \param evname name of evaluation
   * \return the string containing evaluation stati
   */
  SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames);
};
#endif  // XGBOOST_WRAPPER_H_
