#ifndef XGBOOST_WRAPPER_H_
#define XGBOOST_WRAPPER_H_
/*!
 * \file xgboost_wrapperh
 * \author Tianqi Chen
 * \brief a C style wrapper of xgboost
 *  can be used to create wrapper of other languages
 */
#include <cstdio>
#ifdef _MSC_VER
#define XGB_DLL __declspec(dllexport)
#else
#define XGB_DLL
#endif
// manually define unsign long
typedef unsigned long bst_ulong;


extern "C" {
  /*!
   * \brief load a data matrix 
   * \return a loaded data matrix
   */
  XGB_DLL void* XGDMatrixCreateFromFile(const char *fname, int silent);
  /*!
   * \brief create a matrix content from csr format
   * \param indptr pointer to row headers
   * \param indices findex
   * \param data fvalue
   * \param nindptr number of rows in the matix + 1 
   * \param nelem number of nonzero elements in the matrix
   * \return created dmatrix
   */
  XGB_DLL void* XGDMatrixCreateFromCSR(const bst_ulong *indptr,
                                       const unsigned *indices,
                                       const float *data,
                                       bst_ulong nindptr,
                                       bst_ulong nelem);
  /*!
   * \brief create a matrix content from CSC format
   * \param col_ptr pointer to col headers
   * \param indices findex
   * \param data fvalue
   * \param nindptr number of rows in the matix + 1 
   * \param nelem number of nonzero elements in the matrix
   * \return created dmatrix
   */
  XGB_DLL void* XGDMatrixCreateFromCSC(const bst_ulong *col_ptr,
                                       const unsigned *indices,
                                       const float *data,
                                       bst_ulong nindptr,
                                       bst_ulong nelem);  
  /*!
   * \brief create matrix content from dense matrix
   * \param data pointer to the data space
   * \param nrow number of rows
   * \param ncol number columns
   * \param missing which value to represent missing value
   * \return created dmatrix
   */
  XGB_DLL void* XGDMatrixCreateFromMat(const float *data,
                                       bst_ulong nrow,
                                       bst_ulong ncol,
                                       float  missing);
  /*!
   * \brief create a new dmatrix from sliced content of existing matrix
   * \param handle instance of data matrix to be sliced
   * \param idxset index set
   * \param len length of index set
   * \return a sliced new matrix
   */
  XGB_DLL void* XGDMatrixSliceDMatrix(void *handle,
                                      const int *idxset,
                                      bst_ulong len);
  /*!
   * \brief free space in data matrix
   */
  XGB_DLL void XGDMatrixFree(void *handle);
  /*!
   * \brief load a data matrix into binary file
   * \param handle a instance of data matrix
   * \param fname file name
   * \param silent print statistics when saving
   */
  XGB_DLL void XGDMatrixSaveBinary(void *handle, const char *fname, int silent);
  /*!
   * \brief set float vector to a content in info
   * \param handle a instance of data matrix
   * \param field field name, can be label, weight
   * \param array pointer to float vector
   * \param len length of array
   */
  XGB_DLL void XGDMatrixSetFloatInfo(void *handle, const char *field, const float *array, bst_ulong len);
  /*!
   * \brief set uint32 vector to a content in info
   * \param handle a instance of data matrix
   * \param field field name
   * \param array pointer to float vector
   * \param len length of array
   */
  XGB_DLL void XGDMatrixSetUIntInfo(void *handle, const char *field, const unsigned *array, bst_ulong len);
  /*!
   * \brief set label of the training matrix
   * \param handle a instance of data matrix
   * \param group pointer to group size
   * \param len length of array
   */
  XGB_DLL void XGDMatrixSetGroup(void *handle, const unsigned *group, bst_ulong len);
  /*!
   * \brief get float info vector from matrix
   * \param handle a instance of data matrix
   * \param field field name
   * \param out_len used to set result length
   * \return pointer to the result
   */
  XGB_DLL const float* XGDMatrixGetFloatInfo(const void *handle, const char *field, bst_ulong* out_len);
  /*!
   * \brief get uint32 info vector from matrix
   * \param handle a instance of data matrix
   * \param field field name
   * \param out_len used to set result length
   * \return pointer to the result
   */
  XGB_DLL const unsigned* XGDMatrixGetUIntInfo(const void *handle, const char *field, bst_ulong* out_len);
  /*!
   * \brief return number of rows
   */
  XGB_DLL bst_ulong XGDMatrixNumRow(const void *handle);
  // --- start XGBoost class
  /*! 
   * \brief create xgboost learner 
   * \param dmats matrices that are set to be cached
   * \param len length of dmats
   */
  XGB_DLL void *XGBoosterCreate(void* dmats[], bst_ulong len);
  /*! 
   * \brief free obj in handle 
   * \param handle handle to be freed
   */
  XGB_DLL void XGBoosterFree(void* handle);
  /*! 
   * \brief set parameters 
   * \param handle handle
   * \param name  parameter name
   * \param val value of parameter
   */    
  XGB_DLL void XGBoosterSetParam(void *handle, const char *name, const char *value);
  /*! 
   * \brief update the model in one round using dtrain
   * \param handle handle
   * \param iter current iteration rounds
   * \param dtrain training data
   */
  XGB_DLL void XGBoosterUpdateOneIter(void *handle, int iter, void *dtrain);
  /*!
   * \brief update the model, by directly specify gradient and second order gradient,
   *        this can be used to replace UpdateOneIter, to support customized loss function
   * \param handle handle
   * \param dtrain training data
   * \param grad gradient statistics
   * \param hess second order gradient statistics
   * \param len length of grad/hess array
   */
  XGB_DLL void XGBoosterBoostOneIter(void *handle, void *dtrain,
                                     float *grad, float *hess, bst_ulong len);
  /*!
   * \brief get evaluation statistics for xgboost
   * \param handle handle
   * \param iter current iteration rounds
   * \param dmats pointers to data to be evaluated
   * \param evnames pointers to names of each data
   * \param len length of dmats
   * \return the string containing evaluation stati
   */
  XGB_DLL const char *XGBoosterEvalOneIter(void *handle, int iter, void *dmats[],
                                           const char *evnames[], bst_ulong len);
  /*!
   * \brief make prediction based on dmat
   * \param handle handle
   * \param dmat data matrix
   * \param output_margin whether only output raw margin value
   * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
   *    when the parameter is set to 0, we will use all the trees
   * \param len used to store length of returning result
   */
  XGB_DLL const float *XGBoosterPredict(void *handle, void *dmat, int output_margin, unsigned ntree_limit, bst_ulong *len);
  /*!
   * \brief load model from existing file
   * \param handle handle
   * \param fname file name
   */
  XGB_DLL void XGBoosterLoadModel(void *handle, const char *fname);
  /*!
   * \brief save model into existing file
   * \param handle handle
   * \param fname file name
   */
  XGB_DLL void XGBoosterSaveModel(const void *handle, const char *fname);
  /*!
   * \brief dump model, return array of strings representing model dump
   * \param handle handle
   * \param fmap  name to fmap can be empty string
   * \param out_len length of output array
   * \return char *data[], representing dump of each model
   */
  XGB_DLL const char **XGBoosterDumpModel(void *handle, const char *fmap,
                                          bst_ulong *out_len);
}
#endif  // XGBOOST_WRAPPER_H_
