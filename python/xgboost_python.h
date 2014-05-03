#ifndef XGBOOST_PYTHON_H
#define XGBOOST_PYTHON_H
/*!
 * \file xgboost_regrank_data.h
 * \brief python wrapper for xgboost, using ctypes, 
 *        hides everything behind functions
 *      use c style interface
 */
#include "../booster/xgboost_data.h"
extern "C"{
    /*! \brief type of row entry */
    typedef xgboost::booster::FMatrixS::REntry XGEntry;
    
    /*! 
     * \brief create a data matrix 
     * \return a new data matrix
     */
    void* XGDMatrixCreate(void);
    /*! 
     * \brief free space in data matrix
     */
    void XGDMatrixFree(void *handle);
    /*! 
     * \brief load a data matrix from text file or buffer(if exists)
     * \param handle a instance of data matrix
     * \param fname file name 
     * \param silent print statistics when loading
     */
    void XGDMatrixLoad(void *handle, const char *fname, int silent);
    /*!
     * \brief load a data matrix into binary file
     * \param handle a instance of data matrix
     * \param fname file name 
     * \param silent print statistics when saving
     */
    void XGDMatrixSaveBinary(void *handle, const char *fname, int silent);
    /*! 
     * \brief set matrix content from csr format
     * \param handle a instance of data matrix
     * \param indptr pointer to row headers
     * \param indices findex
     * \param data    fvalue
     * \param nindptr number of rows in the matix + 1 
     * \param nelem number of nonzero elements in the matrix
     */
    void XGDMatrixParseCSR( void *handle, 
                            const size_t *indptr,
                            const unsigned *indices,
                            const float *data,
                            size_t nindptr,
                            size_t nelem );
    /*! 
     * \brief set label of the training matrix
     * \param handle a instance of data matrix
     * \param data array of row content 
     * \param len length of array
     */    
    void XGDMatrixSetLabel( void *handle, const float *label, size_t len );        
    /*! 
     * \brief get label set from matrix
     * \param handle a instance of data matrix
     * \param len used to set result length
     */
    const float* XGDMatrixGetLabel( const void *handle, size_t* len );
    /*! 
     * \brief add row 
     * \param handle a instance of data matrix
     * \param data array of row content 
     * \param len length of array
     */
    void XGDMatrixAddRow(void *handle, const XGEntry *data, size_t len);
    /*! 
     * \brief create a booster
     */
    void* XGBoostCreate(void);

    /*! 
     * \brief create a booster
     */
    void* XGBoost(void);
    
};
#endif

