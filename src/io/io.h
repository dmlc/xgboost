#ifndef XGBOOST_IO_IO_H_
#define XGBOOST_IO_IO_H_
/*!
 * \file io.h
 * \brief handles input data format of xgboost
 *    I/O module handles a specific DMatrix format
 * \author Tianqi Chen
 */
#include "../data.h"
#include "../learner/dmatrix.h"

namespace xgboost {
/*! \brief namespace related to data format */
namespace io {
/*! \brief DMatrix object that I/O module support save/load */
typedef learner::DMatrix DataMatrix;
/*!
 * \brief load DataMatrix from stream
 * \param fname file name to be loaded
 * \param silent whether print message during loading
 * \param savebuffer whether temporal buffer the file if the file is in text format
 * \return a loaded DMatrix
 */
DataMatrix* LoadDataMatrix(const char *fname, bool silent = false, bool savebuffer = true);
/*!
 * \brief save DataMatrix into stream, 
 *  note: the saved dmatrix format may not be in exactly same as input
 *  SaveDMatrix will choose the best way to materialize the dmatrix.
 * \param dmat the dmatrix to be saved
 * \param fname file name to be savd
 * \param silent whether print message during saving
 */
void SaveDataMatrix(const DataMatrix &dmat, const char *fname, bool silent = false);

}  // namespace io
}  // namespace xgboost
#endif  // XGBOOST_IO_IO_H_
