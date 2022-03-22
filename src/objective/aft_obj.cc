/*!
 * Copyright 2019-2020 by Contributors
 * \file aft_obj.cc
 * \brief Definition of AFT loss for survival analysis.
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

// Dummy file to keep the CUDA conditional compile trick.

#include <dmlc/registry.h>
namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(aft_obj);

}  // namespace obj
}  // namespace xgboost

#ifndef XGBOOST_USE_CUDA
#include "aft_obj.cu"
#endif  // XGBOOST_USE_CUDA
