/*!
 * Copyright 2014-2022 by XGBoost Contributors
 * \file xgboost_R.h
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
 * \brief Set global configuration
 * \param json_str a JSON string representing the list of key-value pairs
 * \return R_NilValue
 */
XGB_DLL SEXP XGBSetGlobalConfig_R(SEXP json_str);

/*!
 * \brief Get global configuration
 * \return JSON string
 */
XGB_DLL SEXP XGBGetGlobalConfig_R();

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
 * \param n_threads Number of threads used to construct DMatrix from dense matrix.
 * \return created dmatrix
 */
XGB_DLL SEXP XGDMatrixCreateFromMat_R(SEXP mat,
                                      SEXP missing,
                                      SEXP n_threads);
/*!
 * \brief create a matrix content from CSC format
 * \param indptr pointer to column headers
 * \param indices row indices
 * \param data content of the data
 * \param num_row numer of rows (when it's set to 0, then guess from data)
 * \param missing which value to represent missing value
 * \param n_threads Number of threads used to construct DMatrix from csc matrix.
 * \return created dmatrix
 */
XGB_DLL SEXP XGDMatrixCreateFromCSC_R(SEXP indptr, SEXP indices, SEXP data, SEXP num_row,
                                      SEXP missing, SEXP n_threads);

/*!
 * \brief create a matrix content from CSR format
 * \param indptr pointer to row headers
 * \param indices column indices
 * \param data content of the data
 * \param num_col numer of columns (when it's set to 0, then guess from data)
 * \param missing which value to represent missing value
 * \param n_threads Number of threads used to construct DMatrix from csr matrix.
 * \return created dmatrix
 */
XGB_DLL SEXP XGDMatrixCreateFromCSR_R(SEXP indptr, SEXP indices, SEXP data, SEXP num_col,
                                      SEXP missing, SEXP n_threads);

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
 * \return R_NilValue
 */
XGB_DLL SEXP XGDMatrixSaveBinary_R(SEXP handle, SEXP fname, SEXP silent);

/*!
 * \brief set information to dmatrix
 * \param handle a instance of data matrix
 * \param field field name, can be label, weight
 * \param array pointer to float vector
 * \return R_NilValue
 */
XGB_DLL SEXP XGDMatrixSetInfo_R(SEXP handle, SEXP field, SEXP array);

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
 * \brief create xgboost learner, saving the pointer into an existing R object
 * \param dmats a list of dmatrix handles that will be cached
 * \param R_handle a clean R external pointer (not holding any object)
 */
XGB_DLL SEXP XGBoosterCreateInEmptyObj_R(SEXP dmats, SEXP R_handle);

/*!
 * \brief set parameters
 * \param handle handle
 * \param name  parameter name
 * \param val value of parameter
 * \return R_NilValue
 */
XGB_DLL SEXP XGBoosterSetParam_R(SEXP handle, SEXP name, SEXP val);

/*!
 * \brief update the model in one round using dtrain
 * \param handle handle
 * \param iter current iteration rounds
 * \param dtrain training data
 * \return R_NilValue
 */
XGB_DLL SEXP XGBoosterUpdateOneIter_R(SEXP ext, SEXP iter, SEXP dtrain);

/*!
 * \brief update the model, by directly specify gradient and second order gradient,
 *        this can be used to replace UpdateOneIter, to support customized loss function
 * \param handle handle
 * \param dtrain training data
 * \param grad gradient statistics
 * \param hess second order gradient statistics
 * \return R_NilValue
 */
XGB_DLL SEXP XGBoosterBoostOneIter_R(SEXP handle, SEXP dtrain, SEXP grad, SEXP hess);

/*!
 * \brief get evaluation statistics for xgboost
 * \param handle handle
 * \param iter current iteration rounds
 * \param dmats list of handles to dmatrices
 * \param evname name of evaluation
 * \return the string containing evaluation stats
 */
XGB_DLL SEXP XGBoosterEvalOneIter_R(SEXP handle, SEXP iter, SEXP dmats, SEXP evnames);

/*!
 * \brief Run prediction on DMatrix, replacing `XGBoosterPredict_R`
 * \param handle handle
 * \param dmat data matrix
 * \param json_config See `XGBoosterPredictFromDMatrix` in xgboost c_api.h
 *
 * \return A list containing 2 vectors, first one for shape while second one for prediction result.
 */
XGB_DLL SEXP XGBoosterPredictFromDMatrix_R(SEXP handle, SEXP dmat, SEXP json_config);
/*!
 * \brief load model from existing file
 * \param handle handle
 * \param fname file name
 * \return R_NilValue
 */
XGB_DLL SEXP XGBoosterLoadModel_R(SEXP handle, SEXP fname);

/*!
 * \brief save model into existing file
 * \param handle handle
 * \param fname file name
 * \return R_NilValue
 */
XGB_DLL SEXP XGBoosterSaveModel_R(SEXP handle, SEXP fname);

/*!
 * \brief load model from raw array
 * \param handle handle
 * \return R_NilValue
 */
XGB_DLL SEXP XGBoosterLoadModelFromRaw_R(SEXP handle, SEXP raw);

/*!
 * \brief Save model into R's raw array
 *
 * \param handle handle
 * \param json_config JSON encoded string storing parameters for the function.  Following
 *                    keys are expected in the JSON document:
 *
 *     "format": str
 *       - json: Output booster will be encoded as JSON.
 *       - ubj:  Output booster will be encoded as Univeral binary JSON.
 *       - deprecated: Output booster will be encoded as old custom binary format.  Do now use
 *         this format except for compatibility reasons.
 *
 * \return Raw array
 */
XGB_DLL SEXP XGBoosterSaveModelToRaw_R(SEXP handle, SEXP json_config);

/*!
 * \brief Save internal parameters as a JSON string
 * \param handle handle
 * \return JSON string
 */

XGB_DLL SEXP XGBoosterSaveJsonConfig_R(SEXP handle);
/*!
 * \brief Load the JSON string returnd by XGBoosterSaveJsonConfig_R
 * \param handle handle
 * \param value JSON string
 * \return R_NilValue
 */
XGB_DLL SEXP XGBoosterLoadJsonConfig_R(SEXP handle, SEXP value);

/*!
  * \brief Memory snapshot based serialization method.  Saves everything states
  *        into buffer.
  * \param handle handle to booster
  */
XGB_DLL SEXP XGBoosterSerializeToBuffer_R(SEXP handle);

/*!
 * \brief Memory snapshot based serialization method.  Loads the buffer returned
 *        from `XGBoosterSerializeToBuffer'.
 * \param handle handle to booster
 * \return raw byte array
 */
XGB_DLL SEXP XGBoosterUnserializeFromBuffer_R(SEXP handle, SEXP raw);

/*!
 * \brief dump model into a string
 * \param handle handle
 * \param fmap  name to fmap can be empty string
 * \param with_stats whether dump statistics of splits
 * \param dump_format the format to dump the model in
 */
XGB_DLL SEXP XGBoosterDumpModel_R(SEXP handle, SEXP fmap, SEXP with_stats, SEXP dump_format);

/*!
 * \brief get learner attribute value
 * \param handle handle
 * \param name  attribute name
 * \return character containing attribute value
 */
XGB_DLL SEXP XGBoosterGetAttr_R(SEXP handle, SEXP name);

/*!
 * \brief set learner attribute value
 * \param handle handle
 * \param name  attribute name
 * \param val attribute value; NULL value would delete an attribute
 * \return R_NilValue
 */
XGB_DLL SEXP XGBoosterSetAttr_R(SEXP handle, SEXP name, SEXP val);

/*!
 * \brief get the names of learner attributes
 * \return string vector containing attribute names
 */
XGB_DLL SEXP XGBoosterGetAttrNames_R(SEXP handle);

/*!
 * \brief Get feature scores from the model.
 * \param json_config See `XGBoosterFeatureScore` in xgboost c_api.h
 * \return A vector with the first element as feature names, second element as shape of
 *         feature scores and thrid element as feature scores.
 */
XGB_DLL SEXP XGBoosterFeatureScore_R(SEXP handle, SEXP json_config);

#endif  // XGBOOST_WRAPPER_R_H_ // NOLINT(*)
