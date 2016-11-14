// Copyright (c) 2016 by Contributors
#ifndef XGBOOST_C_API_INTEGRITY_TESTS_H_
#define XGBOOST_C_API_INTEGRITY_TESTS_H_

namespace xgboost {
/*!
* \brief This class is for running individual integrity tests on c++ functionality.
*          Its use is to prevent future code changes accidentally breaking important
*          functionality. This is useful for internal tests that are not possible at
*          the client, i.e., not possible within R/Python (e.g., if the state of
*          an object that is not visible to the client needs testing)
*/
class XGIntegrityTests {
 public:
    /*!
    * \brief If a DMatrix is "sliced" via XGDMatrixSliceDMatrix and the DMatrix has
    *          associated groups defined, the sliced groups must have the correct
    *          structure.  This method will test that the group strutures created are
    *          correct!
    */
    static void DMatrixGroupSlices();
};
}  // namespace xgboost
#endif  // XGBOOST_C_API_INTEGRITY_TESTS_H_
