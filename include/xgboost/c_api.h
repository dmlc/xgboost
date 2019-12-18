/*!
 * Copyright (c) 2015 by Contributors
 * \file c_api.h
 * \author Tianqi Chen
 * \brief C API of XGBoost, used for interfacing to other languages.
 */
#ifndef XGBOOST_C_API_H_
#define XGBOOST_C_API_H_

#ifdef __cplusplus
#define XGB_EXTERN_C extern "C"
#include <cstdio>
#include <cstdint>
#else
#define XGB_EXTERN_C
#include <stdio.h>
#include <stdint.h>
#endif  // __cplusplus

#if defined(_MSC_VER) || defined(_WIN32)
#define XGB_DLL XGB_EXTERN_C __declspec(dllexport)
#else
#define XGB_DLL XGB_EXTERN_C
#endif  // defined(_MSC_VER) || defined(_WIN32)

// manually define unsigned long
typedef uint64_t bst_ulong;  // NOLINT(*)


/*! \brief handle to DMatrix */
typedef void *DMatrixHandle;  // NOLINT(*)
/*! \brief handle to Booster */
typedef void *BoosterHandle;  // NOLINT(*)
/*! \brief handle to a data iterator */
typedef void *DataIterHandle;  // NOLINT(*)
/*! \brief handle to a internal data holder. */
typedef void *DataHolderHandle;  // NOLINT(*)

/*! \brief Mini batch used in XGBoost Data Iteration */
typedef struct {  // NOLINT(*)
  /*! \brief number of rows in the minibatch */
  size_t size;
  /*! \brief row pointer to the rows in the data */
#ifdef __APPLE__
  /* Necessary as Java on MacOS defines jlong as long int
   * and gcc defines int64_t as long long int. */
  long* offset; // NOLINT(*)
#else
  int64_t* offset;  // NOLINT(*)
#endif  // __APPLE__
  /*! \brief labels of each instance */
  float* label;
  /*! \brief weight of each instance, can be NULL */
  float* weight;
  /*! \brief feature index */
  int* index;
  /*! \brief feature values */
  float* value;
} XGBoostBatchCSR;

/*!
 * \brief Return the version of the XGBoost library being currently used.
 *
 *  The output variable is only written if it's not NULL.
 *
 * \param major Store the major version number
 * \param minor Store the minor version number
 * \param patch Store the patch (revision) number
 */
XGB_DLL void XGBoostVersion(int* major, int* minor, int* patch);

/*!
 * \brief Callback to set the data to handle,
 * \param handle The handle to the callback.
 * \param batch The data content to be set.
 */
XGB_EXTERN_C typedef int XGBCallbackSetData(  // NOLINT(*)
    DataHolderHandle handle, XGBoostBatchCSR batch);

/*!
 * \brief The data reading callback function.
 *  The iterator will be able to give subset of batch in the data.
 *
 *  If there is data, the function will call set_function to set the data.
 *
 * \param data_handle The handle to the callback.
 * \param set_function The batch returned by the iterator
 * \param set_function_handle The handle to be passed to set function.
 * \return 0 if we are reaching the end and batch is not returned.
 */
XGB_EXTERN_C typedef int XGBCallbackDataIterNext(  // NOLINT(*)
    DataIterHandle data_handle, XGBCallbackSetData *set_function,
    DataHolderHandle set_function_handle);

/*!
 * \brief get string message of the last error
 *
 *  all function in this file will return 0 when success
 *  and -1 when an error occurred,
 *  XGBGetLastError can be called to retrieve the error
 *
 *  this function is thread safe and can be called by different thread
 * \return const char* error information
 */
XGB_DLL const char *XGBGetLastError(void);

/*!
 * \brief register callback function for LOG(INFO) messages -- helpful messages
 *        that are not errors.
 * Note: this function can be called by multiple threads. The callback function
 *       will run on the thread that registered it
 * \return 0 for success, -1 for failure
 */
XGB_DLL int XGBRegisterLogCallback(void (*callback)(const char*));

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
 * \brief Create a DMatrix from a data iterator.
 * \param data_handle The handle to the data.
 * \param callback The callback to get the data.
 * \param cache_info Additional information about cache file, can be null.
 * \param out The created DMatrix
 * \return 0 when success, -1 when failure happens.
 */
XGB_DLL int XGDMatrixCreateFromDataIter(
    DataIterHandle data_handle,
    XGBCallbackDataIterNext* callback,
    const char* cache_info,
    DMatrixHandle *out);

/*!
 * \brief create a matrix content from CSR format
 * \param indptr pointer to row headers
 * \param indices findex
 * \param data fvalue
 * \param nindptr number of rows in the matrix + 1
 * \param nelem number of nonzero elements in the matrix
 * \param num_col number of columns; when it's set to 0, then guess from data
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSREx(const size_t* indptr,
                                     const unsigned* indices,
                                     const float* data,
                                     size_t nindptr,
                                     size_t nelem,
                                     size_t num_col,
                                     DMatrixHandle* out);
/*!
 * \brief create a matrix content from CSC format
 * \param col_ptr pointer to col headers
 * \param indices findex
 * \param data fvalue
 * \param nindptr number of rows in the matrix + 1
 * \param nelem number of nonzero elements in the matrix
 * \param num_row number of rows; when it's set to 0, then guess from data
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSCEx(const size_t* col_ptr,
                                     const unsigned* indices,
                                     const float* data,
                                     size_t nindptr,
                                     size_t nelem,
                                     size_t num_row,
                                     DMatrixHandle* out);

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
                                   float missing,
                                   DMatrixHandle *out);
/*!
 * \brief create matrix content from dense matrix
 * \param data pointer to the data space
 * \param nrow number of rows
 * \param ncol number columns
 * \param missing which value to represent missing value
 * \param out created dmatrix
 * \param nthread number of threads (up to maximum cores available, if <=0 use all cores)
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromMat_omp(const float *data,  // NOLINT
                                       bst_ulong nrow, bst_ulong ncol,
                                       float missing, DMatrixHandle *out,
                                       int nthread);
/*!
 * \brief create matrix content from python data table
 * \param data pointer to pointer to column data
 * \param feature_stypes pointer to strings
 * \param nrow number of rows
 * \param ncol number columns
 * \param out created dmatrix
 * \param nthread number of threads (up to maximum cores available, if <=0 use all cores)
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromDT(void** data,
                                  const char ** feature_stypes,
                                  bst_ulong nrow,
                                  bst_ulong ncol,
                                  DMatrixHandle* out,
                                  int nthread);
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
 * \brief create a new dmatrix from sliced content of existing matrix
 * \param handle instance of data matrix to be sliced
 * \param idxset index set
 * \param len length of index set
 * \param out a sliced new matrix
 * \param allow_groups allow slicing of an array with groups
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSliceDMatrixEx(DMatrixHandle handle,
                                    const int *idxset,
                                    bst_ulong len,
                                    DMatrixHandle *out,
                                    int allow_groups);
/*!
 * \brief free space in data matrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixFree(DMatrixHandle handle);
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
 * \param array pointer to unsigned int vector
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetUIntInfo(DMatrixHandle handle,
                                 const char *field,
                                 const unsigned *array,
                                 bst_ulong len);

/*!
 * \brief (deprecated) Use XGDMatrixSetUIntInfo instead. Set group of the training matrix
 * \param handle a instance of data matrix
 * \param group pointer to group size
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetGroup(DMatrixHandle handle,
                              const unsigned *group,
                              bst_ulong len);

/*!
 * \brief get float info vector from matrix.
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
 * \param out_len The length of the field.
 * \param out_dptr pointer to the result
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                                 const char *field,
                                 bst_ulong* out_len,
                                 const unsigned **out_dptr);
/*!
 * \brief get number of rows.
 * \param handle the handle to the DMatrix
 * \param out The address to hold number of rows.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixNumRow(DMatrixHandle handle,
                            bst_ulong *out);
/*!
 * \brief get number of columns
 * \param handle the handle to the DMatrix
 * \param out The output of number of columns
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixNumCol(DMatrixHandle handle,
                            bst_ulong *out);
// --- start XGBoost class
/*!
 * \brief create xgboost learner
 * \param dmats matrices that are set to be cached
 * \param len length of dmats
 * \param out handle to the result booster
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterCreate(const DMatrixHandle dmats[],
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
 * \param value value of parameter
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
 *          4:output feature contributions to individual predictions
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
/*
 * Short note for serialization APIs.  There are 3 different sets of serialization API.
 *
 * - Functions with the term "Model" handles saving/loading XGBoost model like trees or
 *   linear weights.  Striping out parameters configuration like training algorithms or
 *   CUDA device ID helps user to reuse the trained model for different tasks, examples
 *   are prediction, training continuation or interpretation.
 *
 * - Functions with the term "Config" handles save/loading configuration.  It helps user
 *   to study the internal of XGBoost.  Also user can use the load method for specifying
 *   paramters in a structured way.  These functions are introduced in 1.0.0, and are not
 *   yet stable.
 *
 * - Functions with the term "Serialization" are combined of above two.  They are used in
 *   situations like check-pointing, or continuing training task in distributed
 *   environment.  In these cases the task must be carried out without any user
 *   intervention.
 */

/*!
 * \brief Load model from existing file
 * \param handle handle
 * \param fname file name
* \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadModel(BoosterHandle handle,
                               const char *fname);
/*!
 * \brief Save model into existing file
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
XGB_DLL int XGBoosterGetModelRaw(BoosterHandle handle, bst_ulong *out_len,
                                 const char **out_dptr);

/*!
 * \brief Memory snapshot based serialization method.  Saves everything states
 * into buffer.
 *
 * \param handle handle
 * \param out_len the argument to hold the output length
 * \param out_dptr the argument to hold the output data pointer
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSerializeToBuffer(BoosterHandle handle, bst_ulong *out_len,
                                       const char **out_dptr);
/*!
 * \brief Memory snapshot based serialization method.  Loads the buffer returned
 *        from `XGBoosterSerializeToBuffer'.
 *
 * \param handle handle
 * \param buf pointer to the buffer
 * \param len the length of the buffer
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterUnserializeFromBuffer(BoosterHandle handle,
                                           const void *buf, bst_ulong len);

/*!
 * \brief Initialize the booster from rabit checkpoint.
 *  This is used in distributed training API.
 * \param handle handle
 * \param version The output version of the model.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadRabitCheckpoint(BoosterHandle handle,
                                         int* version);

/*!
 * \brief Save the current checkpoint to rabit.
 * \param handle handle
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSaveRabitCheckpoint(BoosterHandle handle);


/*!
 * \brief Save XGBoost's internal configuration into a JSON document.  Currently the
 *        support is experimental, function signature may change in the future without
 *        notice.
 *
 * \param handle handle to Booster object.
 * \param out_str A valid pointer to array of characters.  The characters array is
 *                allocated and managed by XGBoost, while pointer to that array needs to
 *                be managed by caller.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSaveJsonConfig(BoosterHandle handle, bst_ulong *out_len,
                                    char const **out_str);
/*!
 * \brief Load XGBoost's internal configuration from a JSON document.  Currently the
 *        support is experimental, function signature may change in the future without
 *        notice.
 *
 * \param handle handle to Booster object.
 * \param json_parameters string representation of a JSON document.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadJsonConfig(BoosterHandle handle,
                                    char const *json_parameters);

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

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fmap  name to fmap can be empty string
 * \param with_stats whether to dump with statistics
 * \param format the format to dump the model in
 * \param out_len length of output array
 * \param out_dump_array pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelEx(BoosterHandle handle,
                                 const char *fmap,
                                 int with_stats,
                                 const char *format,
                                 bst_ulong *out_len,
                                 const char ***out_dump_array);

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fnum number of features
 * \param fname names of features
 * \param ftype types of features
 * \param with_stats whether to dump with statistics
 * \param out_len length of output array
 * \param out_models pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelWithFeatures(BoosterHandle handle,
                                           int fnum,
                                           const char **fname,
                                           const char **ftype,
                                           int with_stats,
                                           bst_ulong *out_len,
                                           const char ***out_models);

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fnum number of features
 * \param fname names of features
 * \param ftype types of features
 * \param with_stats whether to dump with statistics
 * \param format the format to dump the model in
 * \param out_len length of output array
 * \param out_models pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelExWithFeatures(BoosterHandle handle,
                                             int fnum,
                                             const char **fname,
                                             const char **ftype,
                                             int with_stats,
                                             const char *format,
                                             bst_ulong *out_len,
                                             const char ***out_models);

/*!
 * \brief Get string attribute from Booster.
 * \param handle handle
 * \param key The key of the attribute.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetAttr(BoosterHandle handle,
                             const char* key,
                             const char** out,
                             int *success);
/*!
 * \brief Set or delete string attribute.
 *
 * \param handle handle
 * \param key The key of the attribute.
 * \param value The value to be saved.
 *              If nullptr, the attribute would be deleted.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSetAttr(BoosterHandle handle,
                             const char* key,
                             const char* value);
/*!
 * \brief Get the names of all attribute from Booster.
 * \param handle handle
 * \param out_len the argument to hold the output length
 * \param out pointer to hold the output attribute stings
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetAttrNames(BoosterHandle handle,
                                  bst_ulong* out_len,
                                  const char*** out);
#endif  // XGBOOST_C_API_H_
