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
     * \brief add row 
     * \param handle a instance of data matrix
     * \param fname file name 
     * \return a new data matrix
     */
    void XGDMatrixPush(void *handle, const XGEntry *data, int len);
    
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

