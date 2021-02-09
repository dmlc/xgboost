/*!
 * Copyright (c) 2015~2020 by Contributors
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
#define XGB_DLL XGB_EXTERN_C __attribute__ ((visibility ("default")))
#endif  // defined(_MSC_VER) || defined(_WIN32)

// manually define unsigned long
typedef uint64_t bst_ulong;  // NOLINT(*)

/*! \brief handle to DMatrix */
typedef void *DMatrixHandle;  // NOLINT(*)
/*! \brief handle to Booster */
typedef void *BoosterHandle;  // NOLINT(*)

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
 * \brief Set global configuration (collection of parameters that apply globally). This function
 *        accepts the list of key-value pairs representing the global-scope parameters to be
 *        configured. The list of key-value pairs are passed in as a JSON string.
 * \param json_str a JSON string representing the list of key-value pairs. The JSON object shall
 *                 be flat: no value can be a JSON object or an array.
 * \return 0 for success, -1 for failure
 */
XGB_DLL int XGBSetGlobalConfig(const char* json_str);

/*!
 * \brief Get current global configuration (collection of parameters that apply globally).
 * \param json_str pointer to received returned global configuration, represented as a JSON string.
 * \return 0 for success, -1 for failure
 */
XGB_DLL int XGBGetGlobalConfig(const char** json_str);

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
 * \brief create a matrix content from CSR format
 * \param indptr pointer to row headers
 * \param indices findex
 * \param data fvalue
 * \param nindptr number of rows in the matrix + 1
 * \param nelem number of nonzero elements in the matrix
 * \param num_col number of columns; when it's set to kAdapterUnknownSize, then guess from data
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
 * \brief Create a matrix from CSR matrix.
 * \param indptr  JSON encoded __array_interface__ to row pointers in CSR.
 * \param indices JSON encoded __array_interface__ to column indices in CSR.
 * \param data    JSON encoded __array_interface__ to values in CSR.
 * \param num_col Number of columns.
 * \param json_config JSON encoded configuration.  Required values are:
 *
 *          - missing
 *          - nthread
 *
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSR(char const *indptr,
                                   char const *indices, char const *data,
                                   bst_ulong ncol,
                                   char const* json_config,
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

/*
 * ========================== Begin data callback APIs =========================
 *
 * Short notes for data callback
 *
 * There are 2 sets of data callbacks for DMatrix.  The first one is currently exclusively
 * used by JVM packages.  It uses `XGBoostBatchCSR` to accept batches for CSR formated
 * input, and concatenate them into 1 final big CSR.  The related functions are:
 *
 * - XGBCallbackSetData
 * - XGBCallbackDataIterNext
 * - XGDMatrixCreateFromDataIter
 *
 * Another set is used by Quantile based DMatrix (used by hist algorithm) for reducing
 * memory usage.  Currently only GPU implementation is available.  It accept foreign data
 * iterators as callbacks and works similar to external memory.  For GPU Hist, the data is
 * first compressed by quantile sketching then merged.  This is particular useful for
 * distributed setting as it eliminates 2 copies of data.  1 by a `concat` from external
 * library to make the data into a blob for normal DMatrix initialization, another by the
 * internal CSR copy of DMatrix.  Related functions are:
 *
 * - XGProxyDMatrixCreate
 * - XGDMatrixCallbackNext
 * - DataIterResetCallback
 * - XGDeviceQuantileDMatrixSetDataCudaArrayInterface
 * - XGDeviceQuantileDMatrixSetDataCudaColumnar
 * - ... (data setters)
 */

/*  ==== First set of callback functions, used exclusively by JVM packages. ==== */

/*! \brief handle to a external data iterator */
typedef void *DataIterHandle;  // NOLINT(*)
/*! \brief handle to a internal data holder. */
typedef void *DataHolderHandle;  // NOLINT(*)


/*! \brief Mini batch used in XGBoost Data Iteration */
typedef struct {  // NOLINT(*)
  /*! \brief number of rows in the minibatch */
  size_t size;
  /* \brief number of columns in the minibatch. */
  size_t columns;
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

/*  == Second set of callback functions, used by constructing Quantile based DMatrix. ===
 *
 * Short note for how to use the second set of callback for GPU Hist tree method.
 *
 * Step 0: Define a data iterator with 2 methods `reset`, and `next`.
 * Step 1: Create a DMatrix proxy by `XGProxyDMatrixCreate` and hold the handle.
 * Step 2: Pass the iterator handle, proxy handle and 2 methods into
 *         `XGDeviceQuantileDMatrixCreateFromCallback`.
 * Step 3: Call appropriate data setters in `next` functions.
 *
 * See test_iterative_device_dmatrix.cu or Python interface for examples.
 */

/*!
 * \brief Create a DMatrix proxy for setting data, can be free by XGDMatrixFree.
 *
 * \param out      The created Device Quantile DMatrix
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGProxyDMatrixCreate(DMatrixHandle* out);

/*!
 * \brief Callback function prototype for getting next batch of data.
 *
 * \param iter  A handler to the user defined iterator.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_EXTERN_C typedef int XGDMatrixCallbackNext(DataIterHandle iter);  // NOLINT(*)

/*!
 * \brief Callback function prototype for reseting external iterator
 */
XGB_EXTERN_C typedef void DataIterResetCallback(DataIterHandle handle); // NOLINT(*)

/*!
 * \brief Create a device DMatrix with data iterator.
 *
 * \param iter     A handle to external data iterator.
 * \param proxy    A DMatrix proxy handle created by `XGProxyDMatrixCreate`.
 * \param reset    Callback function reseting the iterator state.
 * \param next     Callback function yieling the next batch of data.
 * \param missing  Which value to represent missing value
 * \param nthread  Number of threads to use, 0 for default.
 * \param max_bin  Maximum number of bins for building histogram.
 * \param out      The created Device Quantile DMatrix
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDeviceQuantileDMatrixCreateFromCallback(
    DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback *reset,
    XGDMatrixCallbackNext *next, float missing, int nthread, int max_bin,
    DMatrixHandle *out);
/*!
 * \brief Set data on a DMatrix proxy.
 *
 * \param handle          A DMatrix proxy created by XGProxyDMatrixCreate
 * \param c_interface_str Null terminated JSON document string representation of CUDA
 *                        array interface.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDeviceQuantileDMatrixSetDataCudaArrayInterface(
    DMatrixHandle handle,
    const char* c_interface_str);
/*!
 * \brief Set data on a DMatrix proxy.
 *
 * \param handle          A DMatrix proxy created by XGProxyDMatrixCreate
 * \param c_interface_str Null terminated JSON document string representation of CUDA
 *                        array interface, with an array of columns.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDeviceQuantileDMatrixSetDataCudaColumnar(
    DMatrixHandle handle,
    const char* c_interface_str);
/*
 * ==========================- End data callback APIs ==========================
 */



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
 * \brief Set content in array interface to a content in info.
 * \param handle a instance of data matrix
 * \param field field name.
 * \param c_interface_str JSON string representation of array interface.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetInfoFromInterface(DMatrixHandle handle,
                                          char const* field,
                                          char const* c_interface_str);

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
 * \brief Set string encoded information of all features.
 *
 * Accepted fields are:
 *   - feature_name
 *   - feature_type
 *
 * \param handle    An instance of data matrix
 * \param field     Feild name
 * \param features  Pointer to array of strings.
 * \param size      Size of `features` pointer (number of strings passed in).
 *
 * \return 0 when success, -1 when failure happens
 *
 * \code
 *
 *   char const* feat_names [] {"feat_0", "feat_1"};
 *   XGDMatrixSetStrFeatureInfo(handle, "feature_name", feat_names, 2);
 *
 *   // i for integer, q for quantitive.  Similarly "int" and "float" are also recognized.
 *   char const* feat_types [] {"i", "q"};
 *   XGDMatrixSetStrFeatureInfo(handle, "feature_type", feat_types, 2);
 *
 * \endcode
 */
XGB_DLL int XGDMatrixSetStrFeatureInfo(DMatrixHandle handle, const char *field,
                                       const char **features,
                                       const bst_ulong size);

/*!
 * \brief Get string encoded information of all features.
 *
 * Accepted fields are:
 *   - feature_name
 *   - feature_type
 *
 * Caller is responsible for copying out the data, before next call to any API function of
 * XGBoost.
 *
 * \param handle       An instance of data matrix
 * \param field        Feild name
 * \param size         Size of output pointer `features` (number of strings returned).
 * \param out_features Address of a pointer to array of strings.  Result is stored in
 *                     thread local memory.
 *
 * \return 0 when success, -1 when failure happens
 *
 * \code
 *
 *  char const **c_out_features = NULL;
 *  bst_ulong out_size = 0;
 *
 *  // Asumming the feature names are already set by `XGDMatrixSetStrFeatureInfo`.
 *  XGDMatrixGetStrFeatureInfo(handle, "feature_name", &out_size,
 *                             &c_out_features)
 *
 *  for (bst_ulong i = 0; i < out_size; ++i) {
 *    // Here we are simply printing the string.  Copy it out if the feature name is
 *    // useful after printing.
 *    printf("feature %lu: %s\n", i, c_out_features[i]);
 *  }
 *
 * \endcode
 */
XGB_DLL int XGDMatrixGetStrFeatureInfo(DMatrixHandle handle, const char *field,
                                       bst_ulong *size,
                                       const char ***out_features);

/*!
 * \brief Set meta info from dense matrix.  Valid field names are:
 *
 *  - label
 *  - weight
 *  - base_margin
 *  - group
 *  - label_lower_bound
 *  - label_upper_bound
 *  - feature_weights
 *
 * \param handle An instance of data matrix
 * \param field  Feild name
 * \param data   Pointer to consecutive memory storing data.
 * \param size   Size of the data, this is relative to size of type.  (Meaning NOT number
 *               of bytes.)
 * \param type   Indicator of data type.  This is defined in xgboost::DataType enum class.
 *
 *    float    = 1
 *    double   = 2
 *    uint32_t = 3
 *    uint64_t = 4
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetDenseInfo(DMatrixHandle handle, const char *field,
                                  void *data, bst_ulong size, int type);

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
 * \brief Slice a model using boosting index. The slice m:n indicates taking all trees
 *        that were fit during the boosting rounds m, (m+1), (m+2), ..., (n-1).
 *
 * \param handle Booster to be sliced.
 * \param begin_layer start of the slice
 * \param end_layer end of the slice; end_layer=0 is equivalent to
 *                  end_layer=num_boost_round
 * \param step step size of the slice
 * \param out Sliced booster.
 *
 * \return 0 when success, -1 when failure happens, -2 when index is out of bound.
 */
XGB_DLL int XGBoosterSlice(BoosterHandle handle, int begin_layer,
                           int end_layer, int step,
                           BoosterHandle *out);

/*!
 * \brief Get number of boosted rounds from gradient booster.  When process_type is
 *        update, this number might drop due to removed tree.
 * \param handle Handle to booster.
 * \param out Pointer to output integer.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterBoostedRounds(BoosterHandle handle, int* out);

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
 * \brief get number of features
 * \param out number of features
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetNumFeature(BoosterHandle handle,
                                   bst_ulong *out);

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
 * \brief make prediction based on dmat (deprecated, use `XGBoosterPredictFromDMatrix` instead)
 * \param handle handle
 * \param dmat data matrix
 * \param option_mask bit-mask of options taken in prediction, possible values
 *          0:normal prediction
 *          1:output margin instead of transformed value
 *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
 *          4:output feature contributions to individual predictions
 * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
 *    when the parameter is set to 0, we will use all the trees
 * \param training Whether the prediction function is used as part of a training loop.
 *    Prediction can be run in 2 scenarios:
 *    1. Given data matrix X, obtain prediction y_pred from the model.
 *    2. Obtain the prediction for computing gradients. For example, DART booster performs dropout
 *       during training, and the prediction result will be different from the one obtained by normal
 *       inference step due to dropped trees.
 *    Set training=false for the first scenario. Set training=true for the second scenario.
 *    The second scenario applies when you are defining a custom objective function.
 * \param out_len used to store length of returning result
 * \param out_result used to set a pointer to array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredict(BoosterHandle handle,
                             DMatrixHandle dmat,
                             int option_mask,
                             unsigned ntree_limit,
                             int training,
                             bst_ulong *out_len,
                             const float **out_result);
/*!
 * \brief Make prediction from DMatrix, replacing `XGBoosterPredict`.
 *
 * \param handle Booster handle
 * \param dmat   DMatrix handle
 * \param c_json_config String encoded predict configuration in JSON format.
 *
 *    "type": [0, 5]
 *      0: normal prediction
 *      1: output margin
 *      2: predict contribution
 *      3: predict approxmated contribution
 *      4: predict feature interaction
 *      5: predict leaf
 *    "training": bool
 *      Whether the prediction function is used as part of a training loop.  **Not used
 *      for inplace prediction**.
 *
 *      Prediction can be run in 2 scenarios:
 *        1. Given data matrix X, obtain prediction y_pred from the model.
 *        2. Obtain the prediction for computing gradients. For example, DART booster performs dropout
 *           during training, and the prediction result will be different from the one obtained by normal
 *           inference step due to dropped trees.
 *      Set training=false for the first scenario. Set training=true for the second
 *      scenario.  The second scenario applies when you are defining a custom objective
 *      function.
 *    "iteration_begin": int
 *      Beginning iteration of prediction.
 *    "iteration_end": int
 *      End iteration of prediction.  Set to 0 this will become the size of tree model.
 *    "strict_shape": bool
 *      Whether should we reshape the output with stricter rules.  If set to true,
 *      normal/margin/contrib/interaction predict will output consistent shape
 *      disregarding the use of multi-class model, and leaf prediction will output 4-dim
 *      array representing: (n_samples, n_iterations, n_classes, n_trees_in_forest)
 *
 *   Run a normal prediction with strict output shape, 2 dim for softprob , 1 dim for others.
 *   \code
 *      {
 *         "type": 0,
 *         "training": False,
 *         "iteration_begin": 0,
 *         "iteration_end": 0,
 *         "strict_shape": true,
 *     }
 *   \endcode
 *
 * \param out_shape Shape of output prediction (copy before use).
 * \param out_dim   Dimension of output prediction.
 * \param out_result Buffer storing prediction value (copy before use).
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromDMatrix(BoosterHandle handle,
                                        DMatrixHandle dmat,
                                        char const* c_json_config,
                                        bst_ulong const **out_shape,
                                        bst_ulong *out_dim,
                                        float const **out_result);
/*
 * \brief Inplace prediction from CPU dense matrix.
 *
 * \param handle        Booster handle.
 * \param values        JSON encoded __array_interface__ to values.
 * \param c_json_config See `XGBoosterPredictFromDMatrix` for more info.
 *
 *   Additional fields for inplace prediction are:
 *     "missing": float
 *
 * \param m             An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 *
 * \param out_shape     See `XGBoosterPredictFromDMatrix` for more info.
 * \param out_dim       See `XGBoosterPredictFromDMatrix` for more info.
 * \param out_result    See `XGBoosterPredictFromDMatrix` for more info.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromDense(BoosterHandle handle,
                                      char const *values,
                                      char const *c_json_config,
                                      DMatrixHandle m,
                                      bst_ulong const **out_shape,
                                      bst_ulong *out_dim,
                                      const float **out_result);

/*
 * \brief Inplace prediction from CPU CSR matrix.
 *
 * \param handle        Booster handle.
 * \param indptr        JSON encoded __array_interface__ to row pointer in CSR.
 * \param indices       JSON encoded __array_interface__ to column indices in CSR.
 * \param values        JSON encoded __array_interface__ to values in CSR..
 * \param ncol          Number of features in data.
 * \param c_json_config See `XGBoosterPredictFromDMatrix` for more info.
 *   Additional fields for inplace prediction are:
 *     "missing": float
 *
 * \param m             An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 *
 * \param out_shape     See `XGBoosterPredictFromDMatrix` for more info.
 * \param out_dim       See `XGBoosterPredictFromDMatrix` for more info.
 * \param out_result    See `XGBoosterPredictFromDMatrix` for more info.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromCSR(BoosterHandle handle, char const *indptr,
                                    char const *indices, char const *values,
                                    bst_ulong ncol,
                                    char const *c_json_config, DMatrixHandle m,
                                    bst_ulong const **out_shape,
                                    bst_ulong *out_dim,
                                    const float **out_result);

/*
 * \brief Inplace prediction from CUDA Dense matrix (cupy in Python).
 *
 * \param handle        Booster handle
 * \param values        JSON encoded __cuda_array_interface__ to values.
 * \param c_json_config See `XGBoosterPredictFromDMatrix` for more info.
 *   Additional fields for inplace prediction are:
 *     "missing": float
 *
 * \param m             An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 * \param out_shape     See `XGBoosterPredictFromDMatrix` for more info.
 * \param out_dim       See `XGBoosterPredictFromDMatrix` for more info.
 * \param out_result    See `XGBoosterPredictFromDMatrix` for more info.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromCudaArray(
    BoosterHandle handle, char const *values, char const *c_json_config,
    DMatrixHandle m, bst_ulong const **out_shape, bst_ulong *out_dim,
    const float **out_result);

/*
 * \brief Inplace prediction from CUDA dense dataframe (cuDF in Python).
 *
 * \param handle        Booster handle
 * \param values        List of __cuda_array_interface__ for all columns encoded in JSON list.
 * \param c_json_config See `XGBoosterPredictFromDMatrix` for more info.
 *   Additional fields for inplace prediction are:
 *     "missing": float
 *
 * \param m             An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 * \param out_shape     See `XGBoosterPredictFromDMatrix` for more info.
 * \param out_dim       See `XGBoosterPredictFromDMatrix` for more info.
 * \param out_result    See `XGBoosterPredictFromDMatrix` for more info.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromCudaColumnar(
    BoosterHandle handle, char const *values, char const *c_json_config,
    DMatrixHandle m, bst_ulong const **out_shape, bst_ulong *out_dim,
    const float **out_result);


/*
 * ========================== Begin Serialization APIs =========================
 */
/*
 * Short note for serialization APIs.  There are 3 different sets of serialization API.
 *
 * - Functions with the term "Model" handles saving/loading XGBoost model like trees or
 *   linear weights.  Striping out parameters configuration like training algorithms or
 *   CUDA device ID.  These functions are designed to let users reuse the trained model
 *   for different tasks, examples are prediction, training continuation or model
 *   interpretation.
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
 * \param fname File URI or file name.
* \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadModel(BoosterHandle handle,
                               const char *fname);
/*!
 * \brief Save model into existing file
 * \param handle handle
 * \param fname File URI or file name.
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
 * \param out_len length of output string
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
/*
 * =========================== End Serialization APIs ==========================
 */


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
