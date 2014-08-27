#ifndef XGBOOST_WRAPPER_H_
#define XGBOOST_WRAPPER_H_
/*!
 * \file xgboost_wrapperh
 * \author Tianqi Chen
 * \brief a C style wrapper of xgboost
 *  can be used to create wrapper of other languages
 */
#include <cstdio>
// define uint64_t
typedef unsigned long uint64_t;

extern "C" {
  /*!
   * \brief load a data matrix 
   * \return a loaded data matrix
   */
  void* XGDMatrixCreateFromFile(const char *fname, int silent);
  /*! 
   * \brief create a matrix content from csr format
   * \param indptr pointer to row headers
   * \param indices findex
   * \param data fvalue
   * \param nindptr number of rows in the matix + 1 
   * \param nelem number of nonzero elements in the matrix
   * \return created dmatrix
   */
  void* XGDMatrixCreateFromCSR(const uint64_t *indptr,
                               const unsigned *indices,
                               const float *data,
                               uint64_t nindptr,
                               uint64_t nelem);
  /*!
   * \brief create matrix content from dense matrix
   * \param data pointer to the data space
   * \param nrow number of rows
   * \param ncol number columns
   * \param missing which value to represent missing value
   * \return created dmatrix
   */
  void* XGDMatrixCreateFromMat(const float *data,
                               uint64_t nrow,
                               uint64_t ncol,
                               float  missing);
  /*!
   * \brief create a new dmatrix from sliced content of existing matrix
   * \param handle instance of data matrix to be sliced
   * \param idxset index set
   * \param len length of index set
   * \return a sliced new matrix
   */
  void* XGDMatrixSliceDMatrix(void *handle,
                              const int *idxset,
                              uint64_t len);
  /*!
   * \brief free space in data matrix
   */
  void XGDMatrixFree(void *handle);
  /*!
   * \brief load a data matrix into binary file
   * \param handle a instance of data matrix
   * \param fname file name
   * \param silent print statistics when saving
   */
  void XGDMatrixSaveBinary(void *handle, const char *fname, int silent);
  /*!
   * \brief set float vector to a content in info
   * \param handle a instance of data matrix
   * \param field field name, can be label, weight
   * \param array pointer to float vector
   * \param len length of array
   */
  void XGDMatrixSetFloatInfo(void *handle, const char *field, const float *array, uint64_t len);
  /*!
   * \brief set uint32 vector to a content in info
   * \param handle a instance of data matrix
   * \param field field name
   * \param array pointer to float vector
   * \param len length of array
   */
  void XGDMatrixSetUIntInfo(void *handle, const char *field, const unsigned *array, uint64_t len);
  /*!
   * \brief set label of the training matrix
   * \param handle a instance of data matrix
   * \param group pointer to group size
   * \param len length of array
   */
  void XGDMatrixSetGroup(void *handle, const unsigned *group, uint64_t len);
  /*!
   * \brief get float info vector from matrix
   * \param handle a instance of data matrix
   * \param field field name
   * \param out_len used to set result length
   * \return pointer to the result
   */
  const float* XGDMatrixGetFloatInfo(const void *handle, const char *field, uint64_t* out_len);
  /*!
   * \brief get uint32 info vector from matrix
   * \param handle a instance of data matrix
   * \param field field name
   * \param out_len used to set result length
   * \return pointer to the result
   */
  const unsigned* XGDMatrixGetUIntInfo(const void *handle, const char *field, uint64_t* out_len);
  /*!
   * \brief return number of rows
   */
  uint64_t XGDMatrixNumRow(const void *handle);
  // --- start XGBoost class
  /*! 
   * \brief create xgboost learner 
   * \param dmats matrices that are set to be cached
   * \param len length of dmats
   */
  void *XGBoosterCreate(void* dmats[], uint64_t len);
  /*! 
   * \brief free obj in handle 
   * \param handle handle to be freed
   */
  void XGBoosterFree(void* handle);
  /*! 
   * \brief set parameters 
   * \param handle handle
   * \param name  parameter name
   * \param val value of parameter
   */    
  void XGBoosterSetParam(void *handle, const char *name, const char *value);
  /*! 
   * \brief update the model in one round using dtrain
   * \param handle handle
   * \param iter current iteration rounds
   * \param dtrain training data
   */
  void XGBoosterUpdateOneIter(void *handle, int iter, void *dtrain);
  /*!
   * \brief update the model, by directly specify gradient and second order gradient,
   *        this can be used to replace UpdateOneIter, to support customized loss function
   * \param handle handle
   * \param dtrain training data
   * \param grad gradient statistics
   * \param hess second order gradient statistics
   * \param len length of grad/hess array
   */
  void XGBoosterBoostOneIter(void *handle, void *dtrain,
                             float *grad, float *hess, uint64_t len);
  /*!
   * \brief get evaluation statistics for xgboost
   * \param handle handle
   * \param iter current iteration rounds
   * \param dmats pointers to data to be evaluated
   * \param evnames pointers to names of each data
   * \param len length of dmats
   * \return the string containing evaluation stati
   */
  const char *XGBoosterEvalOneIter(void *handle, int iter, void *dmats[],
                                   const char *evnames[], uint64_t len);
  /*!
   * \brief make prediction based on dmat
   * \param handle handle
   * \param dmat data matrix
   * \param output_margin whether only output raw margin value
   * \param len used to store length of returning result
   */
  const float *XGBoosterPredict(void *handle, void *dmat, int output_margin, uint64_t *len);
  /*!
   * \brief load model from existing file
   * \param handle handle
   * \param fname file name
   */
  void XGBoosterLoadModel(void *handle, const char *fname);
  /*!
   * \brief save model into existing file
   * \param handle handle
   * \param fname file name
   */
  void XGBoosterSaveModel(const void *handle, const char *fname);
  /*!
   * \brief dump model, return array of strings representing model dump
   * \param handle handle
   * \param fmap  name to fmap can be empty string
   * \param out_len length of output array
   * \return char *data[], representing dump of each model
   */
  const char **XGBoosterDumpModel(void *handle, const char *fmap,
                                  uint64_t *out_len);
};
#endif  // XGBOOST_WRAPPER_H_
