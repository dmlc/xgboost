/*!
 * Copyright (c) 2014 by Contributors
 * \file xgboost_wrapper.h
 * \author Tianqi Chen
 * \brief a C style wrapper of xgboost
 *  can be used to create wrapper of other languages
 */
#ifndef XGBOOST_WRAPPER_H_
#define XGBOOST_WRAPPER_H_

#ifdef __cplusplus
#define XGB_EXTERN_C extern "C"
#endif

#if defined(_MSC_VER) || defined(_WIN32)
#define XGB_DLL XGB_EXTERN_C __declspec(dllexport)
#else
#define XGB_DLL XGB_EXTERN_C
#endif
// manually define unsign long
typedef unsigned long bst_ulong;  // NOLINT(*)

/*! \brief handle to DMatrix */
typedef void *DMatrixHandle;
/*! \brief handle to Booster */
typedef void *BoosterHandle;

/*!
 * \brief get string message of the last error
 *
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  XGBGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 * \return const char* error inforomation
 */
XGB_DLL const char *XGBGetLastError();

/*!
 * \brief load a data matrix
 * \param fname the name of the file
 * \param silent whether print messages during loading
 * \param out a loaded data matrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromFile(const char *fname,
                                    int silent,
                                    DMatrixHandle *out);

/*!
 * \brief create a matrix content from csr format
 * \param indptr pointer to row headers
 * \param indices findex
 * \param data fvalue
 * \param nindptr number of rows in the matix + 1
 * \param nelem number of nonzero elements in the matrix
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSR(const bst_ulong *indptr,
                                   const unsigned *indices,
                                   const float *data,
                                   bst_ulong nindptr,
                                   bst_ulong nelem,
                                   DMatrixHandle *out);
/*!
 * \brief create a matrix content from CSC format
 * \param col_ptr pointer to col headers
 * \param indices findex
 * \param data fvalue
 * \param nindptr number of rows in the matix + 1
 * \param nelem number of nonzero elements in the matrix
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSC(const bst_ulong *col_ptr,
                                   const unsigned *indices,
                                   const float *data,
                                   bst_ulong nindptr,
                                   bst_ulong nelem,
                                   DMatrixHandle *out);
/*!
 * \brief create matrix content from dense matrix
 * \param data pointer to the data space
 * \param nrow number of rows
 * \param ncol number columns
 * \param missing which value to represent missing value
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromMat(const float *data,
                                   bst_ulong nrow,
                                   bst_ulong ncol,
                                   float  missing,
                                   DMatrixHandle *out);
/*!
 * \brief create a new dmatrix from sliced content of existing matrix
 * \param handle instance of data matrix to be sliced
 * \param idxset index set
 * \param len length of index set
 * \param out a sliced new matrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSliceDMatrix(DMatrixHandle handle,
                                  const int *idxset,
                                  bst_ulong len,
                                  DMatrixHandle *out);
/*!
 * \brief free space in data matrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixFree(void *handle);
/*!
 * \brief load a data matrix into binary file
 * \param handle a instance of data matrix
 * \param fname file name
 * \param silent print statistics when saving
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSaveBinary(DMatrixHandle handle,
                                const char *fname, int silent);
/*!
 * \brief set float vector to a content in info
 * \param handle a instance of data matrix
 * \param field field name, can be label, weight
 * \param array pointer to float vector
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetFloatInfo(DMatrixHandle handle,
                                  const char *field,
                                  const float *array,
                                  bst_ulong len);
/*!
 * \brief set uint32 vector to a content in info
 * \param handle a instance of data matrix
 * \param field field name
 * \param array pointer to float vector
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetUIntInfo(DMatrixHandle handle,
                                 const char *field,
                                 const unsigned *array,
                                 bst_ulong len);
/*!
 * \brief set label of the training matrix
 * \param handle a instance of data matrix
 * \param group pointer to group size
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetGroup(DMatrixHandle handle,
                              const unsigned *group,
                              bst_ulong len);
/*!
 * \brief get float info vector from matrix
 * \param handle a instance of data matrix
 * \param field field name
 * \param out_len used to set result length
 * \param out_dptr pointer to the result
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
                                  const char *field,
                                  bst_ulong* out_len,
                                  const float **out_dptr);
/*!
 * \brief get uint32 info vector from matrix
 * \param handle a instance of data matrix
 * \param field field name
 * \param out_ptr pointer to the result
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                                 const char *field,
                                 bst_ulong* out_len,
                                 const unsigned **out_dptr);
/*!
 * \brief get number of rows
 * \param handle the handle to the DMatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixNumRow(DMatrixHandle handle,
                            bst_ulong *out);
// --- start XGBoost class
/*!
 * \brief create xgboost learner
 * \param dmats matrices that are set to be cached
 * \param len length of dmats
 * \param out handle to the result booster
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterCreate(void* dmats[],
                            bst_ulong len,
                            BoosterHandle *out);
/*!
 * \brief free obj in handle
 * \param handle handle to be freed
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterFree(BoosterHandle handle);

/*!
 * \brief set parameters
 * \param handle handle
 * \param name  parameter name
 * \param val value of parameter
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSetParam(BoosterHandle handle,
                              const char *name,
                              const char *value);
/*!
 * \brief update the model in one round using dtrain
 * \param handle handle
 * \param iter current iteration rounds
 * \param dtrain training data
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterUpdateOneIter(BoosterHandle handle,
                                   int iter,
                                   DMatrixHandle dtrain);
/*!
 * \brief update the model, by directly specify gradient and second order gradient,
 *        this can be used to replace UpdateOneIter, to support customized loss function
 * \param handle handle
 * \param dtrain training data
 * \param grad gradient statistics
 * \param hess second order gradient statistics
 * \param len length of grad/hess array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterBoostOneIter(BoosterHandle handle,
                                  DMatrixHandle dtrain,
                                  float *grad,
                                  float *hess,
                                  bst_ulong len);
/*!
 * \brief get evaluation statistics for xgboost
 * \param handle handle
 * \param iter current iteration rounds
 * \param dmats pointers to data to be evaluated
 * \param evnames pointers to names of each data
 * \param len length of dmats
 * \param out_result the string containing evaluation statistics
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterEvalOneIter(BoosterHandle handle,
                                 int iter,
                                 DMatrixHandle dmats[],
                                 const char *evnames[],
                                 bst_ulong len,
                                 const char **out_result);
/*!
 * \brief make prediction based on dmat
 * \param handle handle
 * \param dmat data matrix
 * \param option_mask bit-mask of options taken in prediction, possible values
 *          0:normal prediction
 *          1:output margin instead of transformed value
 *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
 * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
 *    when the parameter is set to 0, we will use all the trees
 * \param out_len used to store length of returning result
 * \param out_result used to set a pointer to array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredict(BoosterHandle handle,
                             DMatrixHandle dmat,
                             int option_mask,
                             unsigned ntree_limit,
                             bst_ulong *out_len,
                             const float **out_result);
/*!
 * \brief load model from existing file
 * \param handle handle
 * \param fname file name
* \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadModel(BoosterHandle handle,
                               const char *fname);
/*!
 * \brief save model into existing file
 * \param handle handle
 * \param fname file name
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSaveModel(BoosterHandle handle,
                               const char *fname);
/*!
 * \brief load model from in memory buffer
 * \param handle handle
 * \param buf pointer to the buffer
 * \param len the length of the buffer
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadModelFromBuffer(BoosterHandle handle,
                                         const void *buf,
                                         bst_ulong len);
/*!
 * \brief save model into binary raw bytes, return header of the array
 * user must copy the result out, before next xgboost call
 * \param handle handle
 * \param out_len the argument to hold the output length
 * \param out_dptr the argument to hold the output data pointer
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetModelRaw(BoosterHandle handle,
                                 bst_ulong *out_len,
                                 const char **out_dptr);
/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fmap  name to fmap can be empty string
 * \param with_stats whether to dump with statistics
 * \param out_len length of output array
 * \param out_dump_array pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModel(BoosterHandle handle,
                               const char *fmap,
                               int with_stats,
                               bst_ulong *out_len,
                               const char ***out_dump_array);
#endif  // XGBOOST_WRAPPER_H_
