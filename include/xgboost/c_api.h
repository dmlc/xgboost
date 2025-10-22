/**
 * Copyright 2015-2025, XGBoost Contributors
 *
 * @brief C API of XGBoost, used to interface with other high-level languages.
 */
#ifndef XGBOOST_C_API_H_
#define XGBOOST_C_API_H_

#ifdef __cplusplus
#define XGB_EXTERN_C extern "C"
#include <cstddef>
#include <cstdio>
#include <cstdint>
#else
#define XGB_EXTERN_C
#include <stddef.h>
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

/**
 * @mainpage
 *
 * @brief XGBoost C API reference.
 *
 * For the official document page see:
 * <a href="https://xgboost.readthedocs.io/en/stable/c.html">XGBoost C Package</a>.
 */

/**
 * @defgroup Library Library
 *
 * These functions are used to obtain general information about XGBoost including version,
 * build info and current global configuration.
 *
 * @{
 */

/** @brief handle to DMatrix */
typedef void *DMatrixHandle;  // NOLINT(*)
/** @brief handle to Booster */
typedef void *BoosterHandle;  // NOLINT(*)

/**
 * @brief Return the version of the XGBoost library.
 *
 *   The output variable is only written if it's not NULL.
 *
 * @param major Store the major version number.
 * @param minor Store the minor version number.
 * @param patch Store the patch (revision) number.
 */
XGB_DLL void XGBoostVersion(int* major, int* minor, int* patch);

/**
 * @brief Get compile information of the shared XGBoost library.
 *
 * @param out string encoded JSON object containing build flags and dependency versions.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBuildInfo(char const **out);

/**
 * @brief Get the string message of the last error.
 *
 *   Most functions in XGBoost returns 0 when success and non-zero when an error
 *   occurred. In the case of error, @ref XGBGetLastError can be used to retrieve the
 *   error message
 *
 *   This function is thread safe.
 *
 * @return The error message from the last error.
 */
XGB_DLL const char *XGBGetLastError();

/**
 * @brief register callback function for LOG(INFO) messages -- helpful messages
 *        that are not errors.
 *
 * @note This function can be called by multiple threads. The callback function
 *       will run on the thread that registered it.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBRegisterLogCallback(void (*callback)(const char*));

/**
 * @brief Set global configuration (collection of parameters that apply globally). This function
 *        accepts the list of key-value pairs representing the global-scope parameters to be
 *        configured. The list of key-value pairs are passed in as a JSON string.
 * @param config a JSON string representing the list of key-value pairs. The JSON object shall
 *                 be flat: no value can be a JSON object or an array.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBSetGlobalConfig(char const *config);

/**
 * @brief Get current global configuration (collection of parameters that apply globally).
 * @param out_config pointer to received returned global configuration, represented as a JSON string.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBGetGlobalConfig(char const **out_config);

/**@}*/

/**
 * @defgroup DMatrix DMatrix
 *
 * @brief DMatrix is the basic data storage for XGBoost used by all XGBoost algorithms
 *        including both training, prediction and explanation. There are a few variants of
 *        `DMatrix` including normal `DMatrix`, which is a CSR matrix, `QuantileDMatrix`,
 *        which is used by histogram-based tree methods for saving memory, and lastly the
 *        experimental external-memory-based DMatrix, which reads data in batches during
 *        training. For the last two variants, see the @ref Streaming group.
 *
 * @{
 */

/**
 * @brief load a data matrix
 *
 * @deprecated since 2.0.0
 * @see XGDMatrixCreateFromURI()
 * @param fname the name of the file
 * @param silent whether print messages during loading
 * @param out a loaded data matrix
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromFile(const char *fname, int silent, DMatrixHandle *out);

/**
 * @brief load a data matrix
 *
 * @param config JSON encoded parameters for DMatrix construction.  Accepted fields are:
 *   - uri: The URI of the input file. The URI parameter `format` is required when loading text data.
 *          @verbatim embed:rst:leading-asterisk
 *            See :doc:`/tutorials/input_format` for more info.
 *          @endverbatim
 *   - silent (optional): Whether to print message during loading. Default to true.
 *   - data_split_mode (optional): Whether the file was split by row or column beforehand for distributed computing. Default to row.
 * @param out a loaded data matrix
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromURI(char const *config, DMatrixHandle *out);

/**
 * @brief Create a DMatrix from columnar data. (table)
 *
 * A special type of input to the `DMatrix` is the columnar format, which refers to
 * column-based dataframes. XGBoost can accept both numeric data types like integers and
 * floats, along with the categorical type, called dictionary in arrow's term. The
 * addition of categorical type is introduced in 3.1.0. The dataframe is represented by a
 * list array interfaces with one object for each column.
 *
 * A categorical type is represented by 3 buffers, the validity mask, the names of the
 * categories (called index for most of the dataframe implementation), and the codes used
 * to represent the categories in the rows. XGBoost consumes a categorical column by
 * accepting two JSON-encoded arrow arrays in a list. The first item in the list is a JSON
 * object with `{"offsets": IntegerArray, "values": StringArray }` representing the string
 * names defined by the arrow columnar format. The second buffer is an masked integer
 * array that stores the categorical codes along with the validity mask:
 *
 * @code{javascript}
 * [
 *   // categorical column, represented as an array (list)
 *   [
 *     {
 *       'offsets':
 *       {
 *         'data': (129412626415808, True),
 *         'typestr': '<i4', 'version': 3, 'strides': None, 'shape': (3,), 'mask': None
 *       },
 *       'values':
 *       {
 *         'data': (129412626416000, True),
 *         'typestr': '<i1', 'version': 3, 'strides': None, 'shape': (7,), 'mask': None
 *       }
 *     },
 *     {
 *       'data': (106200854378448, True),
 *       'typestr': '<i1', 'version': 3, 'strides': None, 'shape': (2,), 'mask': None
 *     }
 *   ],
 *   // numeric column, represented as an object, same number of rows as the previous column (2)
 *   {
 *     'data': (106200854378448, True),
 *     'typestr': '<f4', 'version': 3, 'strides': None, 'shape': (2,), 'mask': None
 *   }
 * ]
 * @endcode
 *
 * As for numeric inputs, it's the same as dense array.
 *
 * @param data   A list of JSON-encoded array interfaces.
 * @param config See @ref XGDMatrixCreateFromDense for details.
 * @param out    The created DMatrix.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromColumnar(char const *data, char const *config, DMatrixHandle *out);

/**
 * @example c-api-demo.c
 */
/**
 * @brief Create a DMatrix from CSR matrix.
 * @param indptr  JSON encoded __array_interface__ to row pointers in CSR.
 * @param indices JSON encoded __array_interface__ to column indices in CSR.
 * @param data    JSON encoded __array_interface__ to values in CSR.
 * @param ncol    The number of columns.
 * @param config  See @ref XGDMatrixCreateFromDense for details.
 * @param out     The created dmatrix
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSR(char const *indptr, char const *indices, char const *data,
                                   bst_ulong ncol, char const *config, DMatrixHandle *out);

/**
 * @brief Create a DMatrix from dense array.
 *
 * The array interface is defined in https://numpy.org/doc/2.1/reference/arrays.interface.html
 * We encode the interface as a JSON object.
 *
 * @param data   JSON encoded __array_interface__ to array values.
 * @param config JSON encoded configuration.  Required values are:
 *   - missing: Which value to represent missing value.
 *   - nthread (optional): Number of threads used for initializing DMatrix.
 *   - data_split_mode (optional): Whether the data was split by row or column beforehand. Default to row.
 * @param out The created DMatrix
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromDense(char const *data, char const *config, DMatrixHandle *out);

/**
 * @brief Create a DMatrix from a CSC matrix.
 *
 * @param indptr  JSON encoded __array_interface__ to column pointers in CSC.
 * @param indices JSON encoded __array_interface__ to row indices in CSC.
 * @param data    JSON encoded __array_interface__ to values in CSC.
 * @param nrow    The number of rows in the matrix.
 * @param config  See @ref XGDMatrixCreateFromDense for details.
 * @param out     The created dmatrix.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSC(char const *indptr, char const *indices, char const *data,
                                   bst_ulong nrow, char const *config, DMatrixHandle *out);


/**
 * @brief create matrix content from dense matrix
 * @param data pointer to the data space
 * @param nrow number of rows
 * @param ncol number columns
 * @param missing which value to represent missing value
 * @param out created dmatrix
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromMat(const float *data,
                                   bst_ulong nrow,
                                   bst_ulong ncol,
                                   float missing,
                                   DMatrixHandle *out);
/**
 * @brief create matrix content from dense matrix
 * @param data pointer to the data space
 * @param nrow number of rows
 * @param ncol number columns
 * @param missing which value to represent missing value
 * @param out created dmatrix
 * @param nthread number of threads (up to maximum cores available, if <=0 use all cores)
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromMat_omp(const float *data,  // NOLINT
                                       bst_ulong nrow, bst_ulong ncol,
                                       float missing, DMatrixHandle *out,
                                       int nthread);

/**
 * @brief Create DMatrix from CUDA columnar format. (cuDF)
 *
 * See @ref XGDMatrixCreateFromColumnar for a brief description of the columnar format.
 *
 * @param data   A list of JSON-encoded array interfaces.
 * @param config See @ref XGDMatrixCreateFromDense for details.
 * @param out    Created dmatrix
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCudaColumnar(char const *data, char const *config,
                                            DMatrixHandle *out);

/**
 * @brief Create DMatrix from CUDA array.
 * @param data JSON encoded __cuda_array_interface__ for array data.
 * @param config JSON encoded configuration.  Required values are:
 *   - missing: Which value to represent missing value.
 *   - nthread (optional): Number of threads used for initializing DMatrix.
 *   - data_split_mode (optional): Whether the data was split by row or column beforehand. Default to row.
 * @param out created dmatrix
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCudaArrayInterface(char const *data, char const *config,
                                                  DMatrixHandle *out);

/**
 * @defgroup Streaming Streaming
 * @ingroup DMatrix
 *
 * @brief Quantile DMatrix and external memory DMatrix can be created from batches of
 *        data.
 *
 * There are 2 sets of data callbacks for DMatrix.  The first one is currently exclusively
 * used by JVM packages.  It uses `XGBoostBatchCSR` to accept batches for CSR formated
 * input, and concatenate them into 1 final big CSR.  The related functions are:
 *
 * - @ref XGBCallbackSetData
 * - @ref XGBCallbackDataIterNext
 * - @ref XGDMatrixCreateFromDataIter
 *
 * Another set is used by external data iterator. It accepts foreign data iterators as
 * callbacks.  There are 2 different senarios where users might want to pass in callbacks
 * instead of raw data.  First it's the Quantile DMatrix used by the hist and GPU-based
 * hist tree method. For this case, the data is first compressed by quantile sketching
 * then merged.  This is particular useful for distributed setting as it eliminates 2
 * copies of data. First one by a `concat` from external library to make the data into a
 * blob for normal DMatrix initialization, another one by the internal CSR copy of
 * DMatrix.
 *
 * The second use case is external memory support where users can pass a custom data
 * iterator into XGBoost for loading data in batches. For both cases, the iterator is only
 * used during the construction of the DMatrix and can be safely freed after construction
 * finishes. There are short notes on each of the use cases in respected DMatrix factory
 * function.
 *
 * Related functions are:
 *
 * # Factory functions
 * - @ref XGDMatrixCreateFromCallback for external memory
 * - @ref XGQuantileDMatrixCreateFromCallback for quantile DMatrix
 * - @ref XGExtMemQuantileDMatrixCreateFromCallback for External memory Quantile DMatrix
 *
 * # Proxy that callers can use to pass data to XGBoost
 * - @ref XGProxyDMatrixCreate
 * - @ref XGDMatrixCallbackNext
 * - @ref DataIterResetCallback
 * - @ref XGProxyDMatrixSetDataCudaArrayInterface
 * - @ref XGProxyDMatrixSetDataColumnar
 * - @ref XGProxyDMatrixSetDataCudaColumnar
 * - @ref XGProxyDMatrixSetDataDense
 * - @ref XGProxyDMatrixSetDataCSR
 * - ... (data setters)
 *
 * @{
 */

/*  ==== First set of callback functions, used exclusively by JVM packages. ==== */

/** @brief handle to a external data iterator */
typedef void *DataIterHandle;  // NOLINT(*)
/** @brief handle to an internal data holder. */
typedef void *DataHolderHandle;  // NOLINT(*)


/** @brief Mini batch used in XGBoost Data Iteration */
typedef struct {  // NOLINT(*)
  /** @brief number of rows in the minibatch */
  size_t size;
  /** @brief number of columns in the minibatch. */
  size_t columns;
  /** @brief row pointer to the rows in the data */
#ifdef __APPLE__
  /* Necessary as Java on MacOS defines jlong as long int
   * and gcc defines int64_t as long long int. */
  long* offset; // NOLINT(*)
#else
  int64_t* offset;  // NOLINT(*)
#endif  // __APPLE__
  /** @brief labels of each instance */
  float* label;
  /** @brief weight of each instance, can be NULL */
  float* weight;
  /** @brief feature index */
  int* index;
  /** @brief feature values */
  float* value;
} XGBoostBatchCSR;

/**
 * @brief Callback to set the data to handle,
 * @param handle The handle to the callback.
 * @param batch The data content to be set.
 */
XGB_EXTERN_C typedef int XGBCallbackSetData(  // NOLINT(*)
    DataHolderHandle handle, XGBoostBatchCSR batch);

/**
 * @brief The data reading callback function.
 *  The iterator will be able to give subset of batch in the data.
 *
 *  If there is data, the function will call set_function to set the data.
 *
 * @param data_handle The handle to the callback.
 * @param set_function The batch returned by the iterator
 * @param set_function_handle The handle to be passed to set function.
 * @return 0 if we are reaching the end and batch is not returned.
 */
XGB_EXTERN_C typedef int XGBCallbackDataIterNext(  // NOLINT(*)
    DataIterHandle data_handle, XGBCallbackSetData *set_function,
    DataHolderHandle set_function_handle);

/**
 * @brief Create a DMatrix from a data iterator.
 * @param data_handle The handle to the data.
 * @param callback The callback to get the data.
 * @param cache_info Additional information about cache file, can be null.
 * @param missing Which value to represent missing value.
 * @param out The created DMatrix
 * @return 0 when success, -1 when failure happens.
 */
XGB_DLL int XGDMatrixCreateFromDataIter(
    DataIterHandle data_handle,
    XGBCallbackDataIterNext* callback,
    const char* cache_info,
    float missing,
    DMatrixHandle *out);

/**
 * Second set of callback functions, used by constructing Quantile DMatrix or external
 * memory DMatrix using a custom iterator.
 */

/**
 * @brief Create a DMatrix proxy for setting data, can be freed by @ref XGDMatrixFree.
 *
 * The DMatrix proxy is only a temporary reference (wrapper) to the actual user data. For
 * instance, if a dense matrix (like a numpy array) is passed into the proxy DMatrix via
 * the @ref XGProxyDMatrixSetDataDense method, then the proxy DMatrix holds only a
 * reference and the input array cannot be freed until the next iteration starts, signaled
 * by a call to the @ref XGDMatrixCallbackNext by XGBoost. It's called `ProxyDMatrix`
 * because it reuses the interface of the DMatrix class in XGBoost, but it's just a mid
 * interface for the @ref XGDMatrixCreateFromCallback and related constructors to consume
 * various user input types.
 *
 * @code{.unparsed}
 *   User inputs -> Proxy DMatrix (wrapper) -> Actual DMatrix
 * @endcode
 *
 * @param out The created Proxy DMatrix.
 *
 * @return 0 when success, -1 when failure happens.
 */
XGB_DLL int XGProxyDMatrixCreate(DMatrixHandle* out);

/**
 * @brief Callback function prototype for getting next batch of data.
 *
 * @param iter  A handler to the user defined iterator.
 *
 * @return 0 when success, -1 when failure happens.
 */
XGB_EXTERN_C typedef int XGDMatrixCallbackNext(DataIterHandle iter);  // NOLINT(*)

/**
 * @brief Callback function prototype for resetting the external iterator.
 */
XGB_EXTERN_C typedef void DataIterResetCallback(DataIterHandle handle); // NOLINT(*)


/**
 * @brief Create an external memory DMatrix with data iterator.
 *
 * Short note for how to use second set of callback for external memory data support:
 *
 * - Step 0: Define a data iterator with 2 methods `reset`, and `next`.
 * - Step 1: Create a DMatrix proxy by \ref XGProxyDMatrixCreate and hold the handle.
 * - Step 2: Pass the iterator handle, proxy handle and 2 methods into
 *           \ref XGDMatrixCreateFromCallback, along with other parameters encoded as a JSON object.
 * - Step 3: Call appropriate data setters in `next` functions.
 *
 * @param iter    A handle to external data iterator.
 * @param proxy   A DMatrix proxy handle created by \ref XGProxyDMatrixCreate.
 * @param reset   Callback function resetting the iterator state.
 * @param next    Callback function yielding the next batch of data.
 * @param config  JSON encoded parameters for DMatrix construction.  Accepted fields are:
 *   - missing:      Which value to represent missing value
 *   - cache_prefix: The path of cache file, caller must initialize all the directories in this path.
 *   - nthread (optional): Number of threads used for initializing DMatrix.
 * @param[out] out      The created external memory DMatrix
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy,
                                        DataIterResetCallback *reset, XGDMatrixCallbackNext *next,
                                        char const *config, DMatrixHandle *out);
/**
 * @example external_memory.c
 */

/**
 * @brief Create a Quantile DMatrix with a data iterator.
 *
 * Short note for how to use the second set of callback for (GPU)Hist tree method:
 *
 * - Step 0: Define a data iterator with 2 methods `reset`, and `next`.
 * - Step 1: Create a DMatrix proxy by @ref XGProxyDMatrixCreate and hold the handle.
 * - Step 2: Pass the iterator handle, proxy handle and 2 methods into
 *           @ref XGQuantileDMatrixCreateFromCallback.
 * - Step 3: Call appropriate data setters in `next` functions.
 *
 * See test_iterative_dmatrix.cu or Python interface for examples.
 *
 * @param iter     A handle to external data iterator.
 * @param proxy    A DMatrix proxy handle created by @ref XGProxyDMatrixCreate.
 * @param ref      Reference DMatrix for providing quantile information.
 * @param reset    Callback function resetting the iterator state.
 * @param next     Callback function yielding the next batch of data.
 * @param config   JSON encoded parameters for DMatrix construction.  Accepted fields are:
 *   - missing:      Which value to represent missing value
 *   - nthread (optional): Number of threads used for initializing DMatrix.
 *   - max_bin (optional): Maximum number of bins for building histogram. Must be consistent with
 *                         the corresponding booster training parameter.
 *   - max_quantile_blocks (optional): For GPU-based inputs, XGBoost handles incoming
 *       batches with multiple growing substreams. This parameter sets the maximum number
 *       of batches before XGBoost can cut the sub-stream and create a new one. This can
 *       help bound the memory usage. By default, XGBoost grows new sub-streams
 *       exponentially until batches are exhausted. Only used for the training dataset and
 *       the default is None (unbounded).
 * @param out      The created Quantile DMatrix.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGQuantileDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy,
                                                DataIterHandle ref, DataIterResetCallback *reset,
                                                XGDMatrixCallbackNext *next, char const *config,
                                                DMatrixHandle *out);

/**
 * @brief Create a Quantile DMatrix backed by external memory.
 *
 * @since 3.0.0
 *
 * @note This is experimental and subject to change.
 *
 * @verbatim embed:rst:leading-asterisk
 *    See :doc:`/tutorials/external_memory` for more info.
 * @endverbatim
 *
 * @param iter     A handle to external data iterator.
 * @param proxy    A DMatrix proxy handle created by @ref XGProxyDMatrixCreate.
 * @param ref      Reference DMatrix for providing quantile information.
 * @param reset    Callback function resetting the iterator state.
 * @param next     Callback function yielding the next batch of data.
 * @param config   JSON encoded parameters for DMatrix construction.  Accepted fields are:
 *   - missing:      Which value to represent missing value
 *   - cache_prefix: The path of cache file, caller must initialize all the directories in this path.
 *   - nthread (optional): Number of threads used for initializing DMatrix.
 *   - max_bin (optional): Maximum number of bins for building histogram. Must be consistent with
 *                         the corresponding booster training parameter.
 *   - on_host (optional): Whether the data should be placed on host memory. Used by GPU inputs.
 *   - min_cache_page_bytes (optional): The minimum number of bytes for each internal GPU
 *      page. Set to 0 to disable page concatenation. Automatic configuration if the
 *      parameter is not provided or set to None.
 *   - max_quantile_blocks (optional): For GPU-based inputs, XGBoost handles incoming
 *       batches with multiple growing substreams. This parameter sets the maximum number
 *       of batches before XGBoost can cut the sub-stream and create a new one. This can
 *       help bound the memory usage. By default, XGBoost grows new sub-streams
 *       exponentially until batches are exhausted. Only used for the training dataset and
 *       the default is None (unbounded).
 * - cache_host_ratio (optioinal): For GPU-based inputs, XGBoost can split the cache into
 *      host and device portitions to reduce the data transfer overhead. This parameter
 *      specifies the size of host cache compared to the size of the entire cache:
 *      `host / (host + device)`.
 * @param out The created Quantile DMatrix.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGExtMemQuantileDMatrixCreateFromCallback(DataIterHandle iter, DMatrixHandle proxy,
                                                      DataIterHandle ref,
                                                      DataIterResetCallback *reset,
                                                      XGDMatrixCallbackNext *next,
                                                      char const *config, DMatrixHandle *out);

/**
 * @brief Set data on a DMatrix proxy.
 *
 * @param handle  A DMatrix proxy created by @ref XGProxyDMatrixCreate
 * @param data    Null terminated JSON document string representation of CUDA
 *                array interface.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGProxyDMatrixSetDataCudaArrayInterface(DMatrixHandle handle, const char *data);

/**
 * @brief Set columnar (table) data on a DMatrix proxy.
 *
 * @param handle A DMatrix proxy created by @ref XGProxyDMatrixCreate
 * @param data   See @ref XGDMatrixCreateFromColumnar for details.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGProxyDMatrixSetDataColumnar(DMatrixHandle handle, char const *data);

/**
 * @brief Set CUDA-based columnar (table) data on a DMatrix proxy.
 *
 * @param handle A DMatrix proxy created by @ref XGProxyDMatrixCreate
 * @param data   See @ref XGDMatrixCreateFromColumnar for details.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGProxyDMatrixSetDataCudaColumnar(DMatrixHandle handle, const char *data);

/**
 * @brief Set data on a DMatrix proxy.
 *
 * @param handle  A DMatrix proxy created by @ref XGProxyDMatrixCreate
 * @param data    Null terminated JSON document string representation of array
 *                interface.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGProxyDMatrixSetDataDense(DMatrixHandle handle, char const *data);

/**
 * @brief Set data on a DMatrix proxy.
 *
 * @param handle        A DMatrix proxy created by \ref XGProxyDMatrixCreate
 * @param indptr        JSON encoded __array_interface__ to row pointer in CSR.
 * @param indices       JSON encoded __array_interface__ to column indices in CSR.
 * @param data          JSON encoded __array_interface__ to values in CSR..
 * @param ncol          The number of columns of input CSR matrix.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGProxyDMatrixSetDataCSR(DMatrixHandle handle, char const *indptr,
                                     char const *indices, char const *data,
                                     bst_ulong ncol);

/** @} */  // End of Streaming

/**
 * @brief create a new dmatrix from sliced content of existing matrix
 * @param handle instance of data matrix to be sliced
 * @param idxset index set
 * @param len length of index set
 * @param out a sliced new matrix
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSliceDMatrix(DMatrixHandle handle,
                                  const int *idxset,
                                  bst_ulong len,
                                  DMatrixHandle *out);
/**
 * @brief create a new dmatrix from sliced content of existing matrix
 * @param handle instance of data matrix to be sliced
 * @param idxset index set
 * @param len length of index set
 * @param out a sliced new matrix
 * @param allow_groups allow slicing of an array with groups
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSliceDMatrixEx(DMatrixHandle handle,
                                    const int *idxset,
                                    bst_ulong len,
                                    DMatrixHandle *out,
                                    int allow_groups);
/**
 * @brief Free a DMatrix object.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixFree(DMatrixHandle handle);
/**
 * @example c-api-demo.c inference.c external_memory.c
 */

/**
 * @brief Save the DMatrix object into a file. `QuantileDMatrix` and external memory
 *        DMatrix are not supported.
 *
 * @param handle a instance of data matrix
 * @param fname File name
 * @param silent print statistics when saving
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSaveBinary(DMatrixHandle handle,
                                const char *fname, int silent);

/**
 * @brief Set content in array interface to a content in info.
 *
 * @param handle An instance of data matrix
 * @param field  Field name.
 * @param data   JSON encoded __array_interface__ to values in the dense matrix/vector.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetInfoFromInterface(DMatrixHandle handle, char const *field,
                                          char const *data);

/**
 * @brief set float vector to a content in info
 * @param handle a instance of data matrix
 * @param field field name, can be label, weight
 * @param array pointer to float vector
 * @param len length of array
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetFloatInfo(DMatrixHandle handle, const char *field, const float *array,
                                  bst_ulong len);
/**
 * @deprecated since 2.1.0
 *
 * Use @ref XGDMatrixSetInfoFromInterface instead.
 */
XGB_DLL int XGDMatrixSetUIntInfo(DMatrixHandle handle, const char *field, const unsigned *array,
                                 bst_ulong len);

/**
 * @brief Set string encoded information of all features.
 *
 * Accepted fields are:
 *   - feature_name
 *   - feature_type
 *
 * @param handle    An instance of data matrix
 * @param field     Field name
 * @param features  Pointer to array of strings.
 * @param size      Size of `features` pointer (number of strings passed in).
 *
 * @return 0 when success, -1 when failure happens
 *
 * @code{c}
 *
 *   char const* feat_names [] {"feat_0", "feat_1"};
 *   XGDMatrixSetStrFeatureInfo(handle, "feature_name", feat_names, 2);
 *
 *   // i for integer, q for quantitive, c for categorical.  Similarly "int" and "float"
 *   // are also recognized.
 *   char const* feat_types [] {"i", "q"};
 *   XGDMatrixSetStrFeatureInfo(handle, "feature_type", feat_types, 2);
 *
 * @endcode
 */
XGB_DLL int XGDMatrixSetStrFeatureInfo(DMatrixHandle handle, const char *field,
                                       const char **features,
                                       const bst_ulong size);

/**
 * @brief Get string encoded information of all features.
 *
 * Accepted fields are:
 *   - feature_name
 *   - feature_type
 *
 * Caller is responsible for copying out the data, before next call to any API function of
 * XGBoost.
 *
 * @param handle       An instance of data matrix
 * @param field        Field name
 * @param size         Size of output pointer `features` (number of strings returned).
 * @param out_features Address of a pointer to array of strings.  Result is stored in
 *                     thread local memory.
 *
 * @return 0 when success, -1 when failure happens
 *
 * @code{c}
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
 * @endcode
 */
XGB_DLL int XGDMatrixGetStrFeatureInfo(DMatrixHandle handle, const char *field,
                                       bst_ulong *size,
                                       const char ***out_features);

/**
 * @deprecated since 2.1.0
 *
 * Use @ref XGDMatrixSetInfoFromInterface instead.
 */
XGB_DLL int XGDMatrixSetDenseInfo(DMatrixHandle handle, const char *field, void const *data,
                                  bst_ulong size, int type);

/**
 * @brief get float info vector from matrix.
 * @param handle a instance of data matrix
 * @param field field name
 * @param out_len used to set result length
 * @param out_dptr pointer to the result
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixGetFloatInfo(const DMatrixHandle handle, const char *field, bst_ulong *out_len,
                                  const float **out_dptr);
/**
 * @example c-api-demo.c
 */

/**
 * @brief get uint32 info vector from matrix
 * @param handle a instance of data matrix
 * @param field field name
 * @param out_len The length of the field.
 * @param out_dptr pointer to the result
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                                 const char *field,
                                 bst_ulong* out_len,
                                 const unsigned **out_dptr);
/**
 * @brief Get the number of rows from a DMatrix.
 *
 * @param handle the handle to the DMatrix
 * @param out The address to hold number of rows.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixNumRow(DMatrixHandle handle, bst_ulong *out);
/**
 * @brief Get the number of columns from a DMatrix.
 *
 * @param handle the handle to the DMatrix
 * @param out The output of number of columns
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixNumCol(DMatrixHandle handle, bst_ulong *out);

/**
 * @brief Get number of valid values from a DMatrix.
 *
 * @param handle the handle to the DMatrix
 * @param out The output of number of non-missing values
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixNumNonMissing(DMatrixHandle handle, bst_ulong *out);

/**
 * @brief Get the data split mode from DMatrix.
 *
 * @param handle the handle to the DMatrix
 * @param out The output of the data split mode
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixDataSplitMode(DMatrixHandle handle, bst_ulong *out);

/**
 * @brief Get the predictors from DMatrix as CSR matrix for testing.  If this is a
 *        quantized DMatrix, quantized values are returned instead.
 *
 * Unlike most of XGBoost C functions, caller of `XGDMatrixGetDataAsCSR` is required to
 * allocate the memory for return buffer instead of using thread local memory from
 * XGBoost. This is to avoid allocating a huge memory buffer that can not be freed until
 * exiting the thread.
 *
 * @since 1.7.0
 *
 * @param handle the handle to the DMatrix
 * @param config JSON configuration string. At the moment it should be an empty document,
 *               preserved for future use.
 * @param out_indptr  indptr of output CSR matrix.
 * @param out_indices Column index of output CSR matrix.
 * @param out_data    Data value of CSR matrix.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixGetDataAsCSR(DMatrixHandle const handle, char const *config,
                                  bst_ulong *out_indptr, unsigned *out_indices, float *out_data);

/**
 * @brief Export the quantile cuts used for training histogram-based models like `hist` and
 *        `approx`. Useful for model compression.
 *
 * @since 2.0.0
 *
 * @param handle the handle to the DMatrix
 * @param config JSON configuration string. At the moment it should be an empty document,
 *               preserved for future use.
 *
 * @param out_indptr indptr of output CSC matrix represented by a JSON encoded
 *                   __(cuda_)array_interface__.
 * @param out_data   Data value of CSC matrix represented by a JSON encoded
 *                   __(cuda_)array_interface__.
 */
XGB_DLL int XGDMatrixGetQuantileCut(DMatrixHandle const handle, char const *config,
                                     char const **out_indptr, char const **out_data);

/** @} */  // End of DMatrix

/**
 * @defgroup Booster Booster
 *
 * @brief The `Booster` class is the gradient-boosted model for XGBoost.
 *
 * During training, the booster object has many caches for improved performance. In
 * addition to gradient and prediction, it also includes runtime buffers like leaf
 * partitions. These buffers persist with the Booster object until either XGBoosterReset()
 * is called or the booster is deleted by the XGBoosterFree().
 *
 * @{
 */

/**
 * @brief Create a XGBoost learner (booster)
 *
 * @param dmats matrices that are set to be cached by the booster.
 * @param len length of dmats
 * @param out handle to the result booster
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterCreate(const DMatrixHandle dmats[], bst_ulong len, BoosterHandle *out);
/**
 * @example c-api-demo.c
 */

/**
 * @brief Delete the booster.
 *
 * @param handle The handle to be freed.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterFree(BoosterHandle handle);
/**
 * @example c-api-demo.c inference.c external_memory.c
 */

/**
 * @brief Reset the booster object to release data caches used for training.
 *
 * @since 3.0.0
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterReset(BoosterHandle handle);

/**
 * @brief Slice a model using boosting index. The slice m:n indicates taking all trees
 *        that were fit during the boosting rounds m, (m+1), (m+2), ..., (n-1).
 *
 * @param handle Booster to be sliced.
 * @param begin_layer start of the slice
 * @param end_layer end of the slice; end_layer=0 is equivalent to
 *                  end_layer=num_boost_round
 * @param step step size of the slice
 * @param out Sliced booster.
 *
 * @return 0 when success, -1 when failure happens, -2 when index is out of bound.
 */
XGB_DLL int XGBoosterSlice(BoosterHandle handle, int begin_layer,
                           int end_layer, int step,
                           BoosterHandle *out);

/**
 * @brief Get number of boosted rounds from gradient booster.  When process_type is
 *        update, this number might drop due to removed tree.
 * @param handle Handle to booster.
 * @param out Pointer to output integer.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterBoostedRounds(BoosterHandle handle, int* out);

/**
 * @brief set parameters
 * @param handle handle
 * @param name  parameter name
 * @param value value of parameter
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSetParam(BoosterHandle handle,
                              const char *name,
                              const char *value);
/**
 * @example c-api-demo.c
 */

/**
 * @brief get number of features
 * @param handle Handle to booster.
 * @param out number of features
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetNumFeature(BoosterHandle handle, bst_ulong *out);
/**
 * @example c-api-demo.c
 */

/**
 * @brief update the model in one round using dtrain
 * @param handle handle
 * @param iter current iteration rounds
 * @param dtrain training data
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterUpdateOneIter(BoosterHandle handle, int iter, DMatrixHandle dtrain);
/**
 * @example c-api-demo.c
 */

/**
 * @deprecated since 2.1.0
 */
XGB_DLL int XGBoosterBoostOneIter(BoosterHandle handle, DMatrixHandle dtrain, float *grad,
                                  float *hess, bst_ulong len);

/**
 * @brief Update a model with gradient and Hessian. This is used for training with a
 *        custom objective function.
 *
 * @since 2.0.0
 *
 * @param handle handle
 * @param dtrain The training data.
 * @param iter   The current iteration round. When training continuation is used, the count
 *               should restart.
 * @param grad   Json encoded __(cuda)_array_interface__ for gradient.
 * @param hess   Json encoded __(cuda)_array_interface__ for Hessian.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterTrainOneIter(BoosterHandle handle, DMatrixHandle dtrain, int iter,
                                  char const *grad, char const *hess);

/**
 * @brief get evaluation statistics for xgboost
 * @param handle handle
 * @param iter current iteration rounds
 * @param dmats pointers to data to be evaluated
 * @param evnames pointers to names of each data
 * @param len length of dmats
 * @param out_result the string containing evaluation statistics
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterEvalOneIter(BoosterHandle handle, int iter, DMatrixHandle dmats[],
                                 const char *evnames[], bst_ulong len, const char **out_result);
/**
 * @example c-api-demo.c
 */

/**
 * @defgroup Prediction Prediction
 * @ingroup Booster
 *
 * @brief These functions are used for running prediction and explanation algorithms.
 *
 * @{
 */

/**
 * @brief make prediction based on dmat (deprecated, use \ref XGBoosterPredictFromDMatrix instead)
 * \deprecated
 * \see XGBoosterPredictFromDMatrix()
 *
 * @param handle handle
 * @param dmat data matrix
 * @param option_mask bit-mask of options taken in prediction, possible values
 *          0:normal prediction
 *          1:output margin instead of transformed value
 *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
 *          4:output feature contributions to individual predictions
 * @param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
 *    when the parameter is set to 0, we will use all the trees
 * @param training Whether the prediction function is used as part of a training loop.
 *    Prediction can be run in 2 scenarios:
 *    1. Given data matrix X, obtain prediction y_pred from the model.
 *    2. Obtain the prediction for computing gradients. For example, DART booster performs dropout
 *       during training, and the prediction result will be different from the one obtained by normal
 *       inference step due to dropped trees.
 *    Set training=false for the first scenario. Set training=true for the second scenario.
 *    The second scenario applies when you are defining a custom objective function.
 * @param out_len used to store length of returning result
 * @param out_result used to set a pointer to array
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredict(BoosterHandle handle,
                             DMatrixHandle dmat,
                             int option_mask,
                             unsigned ntree_limit,
                             int training,
                             bst_ulong *out_len,
                             const float **out_result);

/**
 * @brief Make prediction from DMatrix, replacing \ref XGBoosterPredict.
 *
 * @param handle Booster handle
 * @param dmat   DMatrix handle
 * @param config String encoded predict configuration in JSON format, with following
 *                      available fields in the JSON object:
 *
 *    "type": [0, 6]
 *      - 0: normal prediction
 *      - 1: output margin
 *      - 2: predict contribution
 *      - 3: predict approximated contribution
 *      - 4: predict feature interaction
 *      - 5: predict approximated feature interaction
 *      - 6: predict leaf
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
 *      End iteration of prediction.  Set to 0 this will become the size of tree model (all the trees).
 *    "strict_shape": bool
 *      Whether should we reshape the output with stricter rules.  If set to true,
 *      normal/margin/contrib/interaction predict will output consistent shape
 *      disregarding the use of multi-class model, and leaf prediction will output 4-dim
 *      array representing: (n_samples, n_iterations, n_classes, n_trees_in_forest)
 *
 *   Example JSON input for running a normal prediction with strict output shape, 2 dim
 *   for softprob , 1 dim for others.
 *   @code{javascript}
 *      {
 *         "type": 0,
 *         "training": false,
 *         "iteration_begin": 0,
 *         "iteration_end": 0,
 *         "strict_shape": true
 *      }
 *   @endcode
 *
 * @param out_shape Shape of output prediction (copy before use).
 * @param out_dim   Dimension of output prediction.
 * @param out_result Buffer storing prediction value (copy before use).
 *
 * @return 0 when success, -1 when failure happens
 *
 * @see XGBoosterPredictFromDense XGBoosterPredictFromCSR XGBoosterPredictFromCudaArray XGBoosterPredictFromCudaColumnar
 */
XGB_DLL int XGBoosterPredictFromDMatrix(BoosterHandle handle, DMatrixHandle dmat,
                                        char const *config, bst_ulong const **out_shape,
                                        bst_ulong *out_dim, float const **out_result);
/**
 * @example inference.c
 */

/**
 * @brief Inplace prediction from CPU dense matrix.
 *
 * \note If the booster is configured to run on a CUDA device, XGBoost falls back to run
 *       prediction with DMatrix with a performance warning.
 *
 * @param handle        Booster handle.
 * @param values        JSON encoded __array_interface__ to values.
 * @param config        See \ref XGBoosterPredictFromDMatrix for more info.
 *   Additional fields for inplace prediction are:
 *     - "missing": float
 * @param m             An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 *
 * @param out_shape     See \ref XGBoosterPredictFromDMatrix for more info.
 * @param out_dim       See \ref XGBoosterPredictFromDMatrix for more info.
 * @param out_result    See \ref XGBoosterPredictFromDMatrix for more info.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromDense(BoosterHandle handle, char const *values, char const *config,
                                      DMatrixHandle m, bst_ulong const **out_shape,
                                      bst_ulong *out_dim, const float **out_result);
/**
 * @example inference.c
 */

/**
 * @brief Inplace prediction from CPU columnar data. (Table)
 *
 * @note If the booster is configured to run on a CUDA device, XGBoost falls back to run
 *       prediction with DMatrix with a performance warning.
 *
 * @param handle        Booster handle.
 * @param data          See @ref XGDMatrixCreateFromColumnar for more info.
 * @param config        See @ref XGBoosterPredictFromDMatrix for more info.
 *   Additional fields for inplace prediction are:
 *     - "missing": float
 * @param m             An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 *
 * @param out_shape     See @ref XGBoosterPredictFromDMatrix for more info.
 * @param out_dim       See @ref XGBoosterPredictFromDMatrix for more info.
 * @param out_result    See @ref XGBoosterPredictFromDMatrix for more info.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromColumnar(BoosterHandle handle, char const *values,
                                         char const *config, DMatrixHandle m,
                                         bst_ulong const **out_shape, bst_ulong *out_dim,
                                         const float **out_result);

/**
 * @brief Inplace prediction from CPU CSR matrix.
 *
 * \note If the booster is configured to run on a CUDA device, XGBoost falls back to run
 *       prediction with DMatrix with a performance warning.
 *
 * @param handle        Booster handle.
 * @param indptr        JSON encoded __array_interface__ to row pointer in CSR.
 * @param indices       JSON encoded __array_interface__ to column indices in CSR.
 * @param values        JSON encoded __array_interface__ to values in CSR..
 * @param ncol          Number of features in data.
 * @param config        See \ref XGBoosterPredictFromDMatrix for more info.
 *   Additional fields for inplace prediction are:
 *     - "missing": float
 * @param m             An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 *
 * @param out_shape     See \ref XGBoosterPredictFromDMatrix for more info.
 * @param out_dim       See \ref XGBoosterPredictFromDMatrix for more info.
 * @param out_result    See \ref XGBoosterPredictFromDMatrix for more info.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromCSR(BoosterHandle handle, char const *indptr, char const *indices,
                                    char const *values, bst_ulong ncol, char const *config,
                                    DMatrixHandle m, bst_ulong const **out_shape,
                                    bst_ulong *out_dim, const float **out_result);

/**
 * @brief Inplace prediction from CUDA Dense matrix (cupy in Python).
 *
 * @note If the booster is configured to run on a CPU, XGBoost falls back to run
 *       prediction with DMatrix with a performance warning.
 *
 * @param handle        Booster handle
 * @param values        JSON encoded __cuda_array_interface__ to values.
 * @param config        See @ref XGBoosterPredictFromDMatrix for more info.
 *   Additional fields for inplace prediction are:
 *     - "missing": float
 * @param proxy         An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 * @param out_shape     See @ref XGBoosterPredictFromDMatrix for more info.
 * @param out_dim       See @ref XGBoosterPredictFromDMatrix for more info.
 * @param out_result    See @ref XGBoosterPredictFromDMatrix for more info.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromCudaArray(BoosterHandle handle, char const *values,
                                          char const *config, DMatrixHandle proxy,
                                          bst_ulong const **out_shape, bst_ulong *out_dim,
                                          const float **out_result);

/**
 * @brief Inplace prediction from CUDA dense dataframe (cuDF in Python).
 *
 * @note If the booster is configured to run on a CPU, XGBoost falls back to run
 *       prediction with DMatrix with a performance warning.
 *
 * @param handle        Booster handle
 * @param data          See @ref XGDMatrixCreateFromColumnar for more info.
 * @param config        See @ref XGBoosterPredictFromDMatrix for more info.
 *   Additional fields for inplace prediction are:
 *     - "missing": float
 * @param proxy         An optional (NULL if not available) proxy DMatrix instance
 *                      storing meta info.
 * @param out_shape     See @ref XGBoosterPredictFromDMatrix for more info.
 * @param out_dim       See @ref XGBoosterPredictFromDMatrix for more info.
 * @param out_result    See @ref XGBoosterPredictFromDMatrix for more info.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredictFromCudaColumnar(BoosterHandle handle, char const *data,
                                             char const *config, DMatrixHandle proxy,
                                             bst_ulong const **out_shape, bst_ulong *out_dim,
                                             const float **out_result);

/**@}*/  // End of Prediction


/**
 * @defgroup Serialization Serialization
 * @ingroup Booster
 *
 * @brief There are multiple ways to serialize a Booster object depending on the use case.
 *
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
 *   parameters in a structured way.  These functions were introduced in 1.0.0.
 *
 * - Functions with the term "Serialization" are combination of above two.  They are used
 *   in situations like check-pointing, or continuing training task in a distributed
 *   environment.  In these cases the task must be carried out without any user
 *   intervention.
 *
 * @{
 */

/**
 * @brief Load the model from an existing file
 *
 * @param handle handle
 * @param fname File name. The string must be UTF-8 encoded.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadModel(BoosterHandle handle,
                               const char *fname);
/**
 * @brief Save the model into an existing file
 *
 * @param handle handle
 * @param fname File name. The string must be UTF-8 encoded.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSaveModel(BoosterHandle handle,
                               const char *fname);
/**
 * @brief load model from in memory buffer
 *
 * @param handle handle
 * @param buf pointer to the buffer
 * @param len the length of the buffer
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadModelFromBuffer(BoosterHandle handle,
                                         const void *buf,
                                         bst_ulong len);

/**
 * @brief Save model into raw bytes, return header of the array.  User must copy the
 *        result out, before next xgboost call
 *
 * @param handle handle
 * @param config JSON encoded string storing parameters for the function.  Following
 *               keys are expected in the JSON document:
 *               - "format": str
 *                 - json: Output booster will be encoded as JSON.
 *                 - ubj:  Output booster will be encoded as Universal binary JSON.
 *                   this format except for compatibility reasons.
 * @param out_len  The argument to hold the output length
 * @param out_dptr The argument to hold the output data pointer
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSaveModelToBuffer(BoosterHandle handle, char const *config, bst_ulong *out_len,
                                       char const **out_dptr);

/**
 * @brief Memory snapshot based serialization method.  Saves everything states
 * into buffer.
 *
 * @param handle handle
 * @param out_len the argument to hold the output length
 * @param out_dptr the argument to hold the output data pointer
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSerializeToBuffer(BoosterHandle handle, bst_ulong *out_len,
                                       const char **out_dptr);
/**
 * @brief Memory snapshot based serialization method.  Loads the buffer returned
 *        from \ref XGBoosterSerializeToBuffer.
 *
 * @param handle handle
 * @param buf pointer to the buffer
 * @param len the length of the buffer
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterUnserializeFromBuffer(BoosterHandle handle,
                                           const void *buf, bst_ulong len);

/**
 * @brief Save XGBoost's internal configuration into a JSON document.  Currently the
 *        support is experimental, function signature may change in the future without
 *        notice.
 *
 * @param handle handle to Booster object.
 * @param out_len length of output string
 * @param out_str A valid pointer to array of characters.  The characters array is
 *                allocated and managed by XGBoost, while pointer to that array needs to
 *                be managed by caller.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSaveJsonConfig(BoosterHandle handle, bst_ulong *out_len,
                                    char const **out_str);
/**
 * @brief Load XGBoost's internal configuration from a JSON document.  Currently the
 *        support is experimental, function signature may change in the future without
 *        notice.
 *
 * @param handle handle to Booster object.
 * @param config string representation of a JSON document.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadJsonConfig(BoosterHandle handle, char const *config);
/**@}*/  // End of Serialization

/**
 * @brief dump model, return array of strings representing model dump
 * @param handle handle
 * @param fmap  name to fmap can be empty string
 * @param with_stats whether to dump with statistics
 * @param out_len length of output array
 * @param out_dump_array pointer to hold representing dump of each model
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModel(BoosterHandle handle,
                               const char *fmap,
                               int with_stats,
                               bst_ulong *out_len,
                               const char ***out_dump_array);

/**
 * @brief dump model, return array of strings representing model dump
 * @param handle handle
 * @param fmap  name to fmap can be empty string
 * @param with_stats whether to dump with statistics
 * @param format the format to dump the model in
 * @param out_len length of output array
 * @param out_dump_array pointer to hold representing dump of each model
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelEx(BoosterHandle handle,
                                 const char *fmap,
                                 int with_stats,
                                 const char *format,
                                 bst_ulong *out_len,
                                 const char ***out_dump_array);

/**
 * @brief dump model, return array of strings representing model dump
 * @param handle handle
 * @param fnum number of features
 * @param fname names of features
 * @param ftype types of features
 * @param with_stats whether to dump with statistics
 * @param out_len length of output array
 * @param out_models pointer to hold representing dump of each model
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelWithFeatures(BoosterHandle handle,
                                           int fnum,
                                           const char **fname,
                                           const char **ftype,
                                           int with_stats,
                                           bst_ulong *out_len,
                                           const char ***out_models);

/**
 * @brief dump model, return array of strings representing model dump
 * @param handle handle
 * @param fnum number of features
 * @param fname names of features
 * @param ftype types of features
 * @param with_stats whether to dump with statistics
 * @param format the format to dump the model in
 * @param out_len length of output array
 * @param out_models pointer to hold representing dump of each model
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelExWithFeatures(BoosterHandle handle,
                                             int fnum,
                                             const char **fname,
                                             const char **ftype,
                                             int with_stats,
                                             const char *format,
                                             bst_ulong *out_len,
                                             const char ***out_models);

/**
 * @brief Get string attribute from Booster.
 * @param handle handle
 * @param key The key of the attribute.
 * @param out The result attribute, can be NULL if the attribute do not exist.
 * @param success Whether the result is contained in out.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetAttr(BoosterHandle handle,
                             const char* key,
                             const char** out,
                             int *success);
/**
 * @brief Set or delete string attribute.
 *
 * @param handle handle
 * @param key The key of the attribute.
 * @param value The value to be saved.
 *              If nullptr, the attribute would be deleted.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSetAttr(BoosterHandle handle,
                             const char* key,
                             const char* value);
/**
 * @brief Get the names of all attribute from Booster.
 * @param handle handle
 * @param out_len the argument to hold the output length
 * @param out pointer to hold the output attribute stings
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetAttrNames(BoosterHandle handle,
                                  bst_ulong* out_len,
                                  const char*** out);

/**
 * @brief Set string encoded feature info in Booster, similar to the feature
 *        info in DMatrix.
 *
 * Accepted fields are:
 *   - feature_name
 *   - feature_type
 *
 * @param handle    An instance of Booster
 * @param field     Field name
 * @param features  Pointer to array of strings.
 * @param size      Size of `features` pointer (number of strings passed in).
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSetStrFeatureInfo(BoosterHandle handle, const char *field,
                                       const char **features,
                                       const bst_ulong size);

/**
 * @brief Get string encoded feature info from Booster, similar to the feature info
 *        in DMatrix.
 *
 * Accepted field names are:
 *   - feature_name
 *   - feature_type
 *
 * Caller is responsible for copying out the data, before the next call to any API
 * function of XGBoost.
 *
 * @param handle       An instance of Booster
 * @param field        Field name
 * @param len          Size of output pointer `features` (number of strings returned).
 * @param out_features Address of a pointer to array of strings. Result is stored in
 *        thread local memory.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetStrFeatureInfo(BoosterHandle handle, const char *field,
                                       bst_ulong *len,
                                       const char ***out_features);

/**
 * @brief Calculate feature scores for tree models.  When used on linear model, only the
 * `weight` importance type is defined, and output scores is a row major matrix with shape
 * [n_features, n_classes] for multi-class model.  For tree model, out_n_feature is always
 * equal to out_n_scores and has multiple definitions of importance type.
 *
 * @param handle          An instance of Booster
 * @param config          Parameters for computing scores encoded as JSON.  Accepted JSON keys are:
 *   - importance_type: A JSON string with following possible values:
 *       * 'weight': the number of times a feature is used to split the data across all trees.
 *       * 'gain': the average gain across all splits the feature is used in.
 *       * 'cover': the average coverage across all splits the feature is used in.
 *       * 'total_gain': the total gain across all splits the feature is used in.
 *       * 'total_cover': the total coverage across all splits the feature is used in.
 *   - feature_map: An optional JSON string with URI or path to the feature map file.
 *   - feature_names: An optional JSON array with string names for each feature.
 *
 * @param out_n_features  Length of output feature names.
 * @param out_features    An array of string as feature names, ordered the same as output scores.
 * @param out_dim         Dimension of output feature scores.
 * @param out_shape       Shape of output feature scores with length of `out_dim`.
 * @param out_scores      An array of floating point as feature scores with shape of `out_shape`.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterFeatureScore(BoosterHandle handle, const char *config,
                                  bst_ulong *out_n_features, char const ***out_features,
                                  bst_ulong *out_dim, bst_ulong const **out_shape,
                                  float const **out_scores);
/**@}*/  // End of Booster

/**
 * @defgroup Collective Collective
 *
 * @brief Experimental support for exposing internal communicator in XGBoost.
 *
 * @note This is still under development.
 *
 * The collective communicator in XGBoost evolved from the `rabit` project of dmlc but has
 * changed significantly since its adoption. It consists of a tracker and a set of
 * workers. The tracker is responsible for bootstrapping the communication group and
 * handling centralized tasks like logging. The workers are actual communicators
 * performing collective tasks like allreduce.
 *
 * To use the collective implementation, one needs to first create a tracker with
 * corresponding parameters, then get the arguments for workers using
 * XGTrackerWorkerArgs().  The obtained arguments can then be passed to the
 * XGCommunicatorInit() function. Call to XGCommunicatorInit() must be accompanied with a
 * XGCommunicatorFinalize() call for cleanups. Please note that the communicator uses
 * `std::thread` in C++, which has undefined behavior in a C++ destructor due to the
 * runtime shutdown sequence. It's preferable to call XGCommunicatorFinalize() before the
 * runtime is shutting down. This requirement is similar to a Python thread or socket,
 * which should not be relied upon in a `__del__` function.
 *
 * Since it's used as a part of XGBoost, errors will be returned when a XGBoost function
 * is called, for instance, training a booster might return a connection error.
 *
 * @{
 */

/**
 * @brief Handle to the tracker.
 *
 *   There are currently two types of tracker in XGBoost, first one is `rabit`, while the
 *   other one is `federated`.  `rabit` is used for normal collective communication, while
 *   `federated` is used for federated learning.
 *
 */
typedef void *TrackerHandle; /* NOLINT */

/**
 * @brief Create a new tracker.
 *
 * @param config JSON encoded parameters.
 *
 *   - dmlc_communicator: String, the type of tracker to create. Available options are
 *                        `rabit` and `federated`. See @ref TrackerHandle for more info.
 *   - n_workers: Integer, the number of workers.
 *   - port: (Optional) Integer, the port this tracker should listen to.
 *   - timeout: (Optional) Integer, timeout in seconds for various networking
                 operations. Default is 300 seconds.
 *
 *   Some configurations are `rabit` specific:
 *
 *   - host: (Optional) String, Used by the the `rabit` tracker to specify the address of the host.
 *           This can be useful when the communicator cannot reliably obtain the host address.
 *   - sortby: (Optional) Integer.
 *     + 0: Sort workers by their host name.
 *     + 1: Sort workers by task IDs.
 *
 *   Some `federated` specific configurations:
 *   - federated_secure: Boolean, whether this is a secure server. False for testing.
 *   - server_key_path: Path to the server key. Used only if this is a secure server.
 *   - server_cert_path: Path to the server certificate. Used only if this is a secure server.
 *   - client_cert_path: Path to the client certificate. Used only if this is a secure server.
 *
 * @param handle The handle to the created tracker.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGTrackerCreate(char const *config, TrackerHandle *handle);

/**
 * @brief Get the arguments needed for running workers. This should be called after
 *        XGTrackerRun().
 *
 * @param handle The handle to the tracker.
 * @param args The arguments returned as a JSON document.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGTrackerWorkerArgs(TrackerHandle handle, char const **args);

/**
 * @brief Start the tracker. The tracker runs in the background and this function returns
 *        once the tracker is started.
 *
 * @param handle The handle to the tracker.
 * @param config Unused at the moment, preserved for the future.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGTrackerRun(TrackerHandle handle, char const *config);

/**
 * @brief Wait for the tracker to finish, should be called after XGTrackerRun(). This
 *        function will block until the tracker task is finished or timeout is reached.
 *
 * @param handle The handle to the tracker.
 * @param config JSON encoded configuration. No argument is required yet, preserved for
 *        the future.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGTrackerWaitFor(TrackerHandle handle, char const *config);

/**
 * @brief Free a tracker instance. This should be called after XGTrackerWaitFor(). If the
 *        tracker is not properly waited, this function will shutdown all connections with
 *        the tracker, potentially leading to undefined behavior.
 *
 * @param handle The handle to the tracker.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGTrackerFree(TrackerHandle handle);

/**
 * @brief Initialize the collective communicator.
 *
 *  Currently the communicator API is experimental, function signatures may change in the future
 *  without notice.
 *
 *  Call this once in the worker process before using anything. Please make sure
 *  XGCommunicatorFinalize() is called after use. The initialized commuicator is a global
 *  thread-local variable.
 *
 * @param config JSON encoded configuration. Accepted JSON keys are:
 *   - dmlc_communicator: The type of the communicator, this should match the tracker type.
 *     * rabit: Use Rabit. This is the default if the type is unspecified.
 *     * federated: Use the gRPC interface for Federated Learning.
 *
 * Only applicable to the `rabit` communicator:
 *   - dmlc_tracker_uri: Hostname or IP address of the tracker.
 *   - dmlc_tracker_port: Port number of the tracker.
 *   - dmlc_task_id: ID of the current task, can be used to obtain deterministic rank assignment.
 *   - dmlc_retry: The number of retries for connection failure.
 *   - dmlc_timeout: Timeout in seconds.
 *   - dmlc_nccl_path: Path to the nccl shared library `libnccl.so`.
 *
 * Only applicable to the `federated` communicator (use upper case for environment variables, use
 * lower case for runtime configuration):
 *   - federated_server_address: Address of the federated server.
 *   - federated_world_size: Number of federated workers.
 *   - federated_rank: Rank of the current worker.
 *   - federated_server_cert_path: Server certificate file path. Only needed for the SSL mode.
 *   - federated_client_key_path: Client key file path. Only needed for the SSL mode.
 *   - federated_client_cert_path: Client certificate file path. Only needed for the SSL mode.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGCommunicatorInit(char const* config);

/**
 * @brief Finalize the collective communicator.
 *
 * Call this function after you have finished all jobs.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGCommunicatorFinalize(void);

/**
 * @brief Get rank of the current process.
 *
 * @return Rank of the worker.
 */
XGB_DLL int XGCommunicatorGetRank(void);

/**
 * @brief Get the total number of processes.
 *
 * @return Total world size.
 */
XGB_DLL int XGCommunicatorGetWorldSize(void);

/**
 * @brief Get if the communicator is distributed.
 *
 * @return True if the communicator is distributed.
 */
XGB_DLL int XGCommunicatorIsDistributed(void);

/**
 * @brief Print the message to the tracker.
 *
 * This function can be used to communicate the information of the progress to the user
 * who monitors the tracker.
 *
 * @param message The message to be printed.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGCommunicatorPrint(char const *message);

/**
 * @brief Get the name of the processor.
 *
 * @param name_str Pointer to received returned processor name.
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGCommunicatorGetProcessorName(const char** name_str);

/**
 * @brief Broadcast a memory region to all others from root. This function is NOT
 *        thread-safe.
 *
 * Example:
 * @code
 *   int a = 1;
 *   Broadcast(&a, sizeof(a), root);
 * @endcode
 *
 * @param send_receive_buffer Pointer to the send or receive buffer.
 * @param size Size of the data in bytes.
 * @param root The process rank to broadcast from.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGCommunicatorBroadcast(void *send_receive_buffer, size_t size, int root);

/**
 * @brief Perform in-place allreduce. This function is NOT thread-safe.
 *
 * Example Usage: the following code gives sum of the result
 * @code
 *     enum class Op {
 *         kMax = 0, kMin = 1, kSum = 2, kBitwiseAND = 3, kBitwiseOR = 4, kBitwiseXOR = 5
 *     };
 *     std::vector<int> data(10);
 *     ...
 *     Allreduce(data.data(), data.size(), DataType:kInt32, Op::kSum);
 *     ...
 * @endcode

 * @param send_receive_buffer Buffer for both sending and receiving data.
 * @param count Number of elements to be reduced.
 * @param data_type Enumeration of data type, see xgboost::collective::DataType in communicator.h.
 * @param op Enumeration of operation type, see xgboost::collective::Operation in communicator.h.
 *
 * @return 0 when success, -1 when failure happens
 */
XGB_DLL int XGCommunicatorAllreduce(void *send_receive_buffer, size_t count, int data_type, int op);

/**@}*/  // End of Collective
#endif  // XGBOOST_C_API_H_
