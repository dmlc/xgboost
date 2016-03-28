/*!
 * Copyright 2014 (c) by Contributors
 * \file xgboost_wrapper_R.h
 * \author Tianqi Chen
 * \brief R wrapper of xgboost
 */
#ifndef XGBOOST_R_H_ // NOLINT(*)
#define XGBOOST_R_H_ // NOLINT(*)


#include <Rinternals.h>
#include <R_ext/Random.h>
#include <Rmath.h>

#include <xgboost/c_api.h>

/*!
 * \brief check whether a handle is NULL
 * \param handle
 * \return whether it is null ptr
 */
XGB_DLL SEXP XGCheckNullPtr_R(SEXP handle);

/*!
 * \brief load a data matrix
 * \param fname name of the content
 * \param silent whether print messages
 * \return a loaded data matrix
 */
XGB_DLL SEXP XGDMatrixCreateFromFile_R(SEXP fname, SEXP silent);

/*!
 * \brief create matrix content from dense matrix
 * This assumes the matrix is stored in column major format
 * \param data R Matrix object
 * \param missing which value to represent missing value
 * \return created dmatrix
 */
XGB_DLL SEXP XGDMatrixCreateFromMat_R(SEXP mat,
                                      SEXP missing);
/*!
 * \brief create a matrix content from CSC format
 * \param indptr pointer to column headers
 * \param indices row indices
 * \param data content of the data
 * \return created dmatrix
 */
XGB_DLL SEXP XGDMatrixCreateFromCSC_R(SEXP indptr,
                                      SEXP indices,
                                      SEXP data);

/*!
 * \brief create a new dmatrix from sliced content of existing matrix
 * \param handle instance of data matrix to be sliced
 * \param idxset index set
 * \return a sliced new matrix
 */
XGB_DLL SEXP XGDMatrixSliceDMatrix_R(SEXP handle, SEXP idxset);

/*!
 * \brief load a data matrix into binary file
 * \param handle a instance of data matrix
 * \param fname file name
 * \param silent print statistics when saving
 */
XGB_DLL void XGDMatrixSaveBinary_R(SEXP handle, SEXP fname, SEXP silent);

/*!
 * \brief set information to dmatrix
 * \param handle a instance of data matrix
 * \param field field name, can be label, weight
 * \param array pointer to float vector
 */
XGB_DLL void XGDMatrixSetInfo_R(SEXP handle, SEXP field, SEXP array);

/*!
 * \brief get info vector from matrix
 * \param handle a instance of data matrix
 * \param field field name
 * \return info vector
 */
XGB_DLL SEXP XGDMatrixGetInfo_R(SEXP handle, SEXP field);

/*!
 * \brief return number of rows
 * \param handle an instance of data matrix
 */
XGB_DLL SEXP XGDMatrixNumRow_R(SEXP handle);

/*!
 * \brief return number of columns
 * \param handle an instance of data matrix
 */
XGB_DLL SEXP XGDMatrixNumCol_R(SEXP handle);

/*!
 * \brief create xgboost learner
 * \param dmats a list of dmatrix handles that will be cached
 */
XGB_DLL SEXP XGBoosterCreate_R(SEXP dmats);

/*!
 * \brief set parameters
 * \param handle handle
 * \param name  parameter name
 * \param val value of parameter
 */
XGB_DLL void XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val);

/*!
 * \brief update the model in one round using dtrain
 * \param handle handle
 * \param iter current iteration rounds
 * \param dtrain training data
 */
XGB_DLL void XGBoosterUpdateOneIter_R(SEXP ext, SEXP iter, SEXP dtrain);

/*!
 * \brief update the model, by directly specify gradient and second order gradient,
 *        this can be used to replace UpdateOneIter, to support customized loss function
 * \param handle handle
 * \param dtrain training data
 * \param grad gradient statistics
 * \param hess second order gradient statistics
 */
XGB_DLL void XGBoosterBoostOneIter_R(SEXP handle, SEXP dtrain, SEXP grad, SEXP hess);

/*!
 * \brief get evaluation statistics for xgboost
 * \param handle handle
 * \param iter current iteration rounds
 * \param dmats list of handles to dmatrices
 * \param evname name of evaluation
 * \return the string containing evaluation stati
 */
XGB_DLL SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames);

/*!
 * \brief make prediction based on dmat
 * \param handle handle
 * \param dmat data matrix
 * \param option_mask output_margin:1 predict_leaf:2
 * \param ntree_limit limit number of trees used in prediction
 */
XGB_DLL SEXP XGBoosterPredict_R(SEXP handle, SEXP dmat, SEXP option_mask, SEXP ntree_limit);
/*!
 * \brief load model from existing file
 * \param handle handle
 * \param fname file name
 */
XGB_DLL void XGBoosterLoadModel_R(SEXP handle, SEXP fname);

/*!
 * \brief save model into existing file
 * \param handle handle
 * \param fname file name
 */
XGB_DLL void XGBoosterSaveModel_R(SEXP handle, SEXP fname);

/*!
 * \brief load model from raw array
 * \param handle handle
 */
XGB_DLL void XGBoosterLoadModelFromRaw_R(SEXP handle, SEXP raw);

/*!
 * \brief save model into R's raw array
 * \param handle handle
 * \return raw array
   */
XGB_DLL SEXP XGBoosterModelToRaw_R(SEXP handle);

/*!
 * \brief dump model into a string
 * \param handle handle
 * \param fmap  name to fmap can be empty string
 * \param with_stats whether dump statistics of splits
 */
XGB_DLL SEXP XGBoosterDumpModel_R(SEXP handle, SEXP fmap, SEXP with_stats);
#endif  // XGBOOST_WRAPPER_R_H_ // NOLINT(*)
