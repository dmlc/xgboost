#ifndef XGBOOST_WRAPPER_H_
#define XGBOOST_WRAPPER_H_
/*!
 * \file xgboost_wrapperh
 * \author Tianqi Chen
 * \brief a C style wrapper of xgboost
 *  can be used to create wrapper of other languages
 */
#include <cstdio>

extern "C" {
  /*!
   * \brief load a data matrix 
   * \return a loaded data matrix
   */
  void* XGDMatrixCreateFromFile(const char *fname, int silent);
  /*! 
   * \brief create a matrix content from csr format
   * \param handle a instance of data matrix
   * \param indptr pointer to row headers
   * \param indices findex
   * \param data fvalue
   * \param nindptr number of rows in the matix + 1 
   * \param nelem number of nonzero elements in the matrix
   * \return created dmatrix
   */
  void* XGDMatrixCreateFromCSR(const size_t *indptr,
                               const unsigned *indices,
                               const float *data,
                               size_t nindptr,
                               size_t nelem);
  /*!
   * \brief create matrix content from dense matrix
   * \param handle a instance of data matrix
   * \param data pointer to the data space
   * \param nrow number of rows
   * \param ncol number columns
   * \param missing which value to represent missing value
   * \return created dmatrix
   */
  void* XGDMatrixCreateFromMat(const float *data,
                               size_t nrow,
                               size_t ncol,
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
                              size_t len);
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
  void XGDMatrixSetFloatInfo(void *handle, const char *field, const float *array, size_t len);
  /*!
   * \brief set label of the training matrix
   * \param handle a instance of data matrix
   * \param group pointer to group size
   * \param len length of array
   */
  void XGDMatrixSetGroup(void *handle, const unsigned *group, size_t len);
  /*!
   * \brief get float info vector from matrix
   * \param handle a instance of data matrix
   * \param field field name
   * \param out_len used to set result length
   * \return pointer to the label
   */
  const float* XGDMatrixGetFloatInfo(const void *handle, const char *field, size_t* out_len);
  /*!
   * \brief return number of rows
   */
  size_t XGDMatrixNumRow(const void *handle);
  // --- start XGBoost class
  /*! 
   * \brief create xgboost learner 
   * \param dmats matrices that are set to be cached
   * \param len length of dmats
   */
  void *XGBoosterCreate(void* dmats[], size_t len);
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
                             float *grad, float *hess, size_t len);
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
                                   const char *evnames[], size_t len);
  /*!
   * \brief make prediction based on dmat
   * \param handle handle
   * \param dmat data matrix
   * \param output_margin whether only output raw margin value
   * \param len used to store length of returning result
   */
  const float *XGBoosterPredict(void *handle, void *dmat, int output_margin, size_t *len);
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
                                  size_t *out_len);
};
#endif  // XGBOOST_WRAPPER_H_
