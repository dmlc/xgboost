/*!
 * Copyright 2019-2020 by Contributors
 * \file survival_util.cc
 * \brief Utility functions, useful for implementing objective and metric functions for survival
 *        analysis
 * \author Avinash Barnwal, Hyunsu Cho and Toby Hocking
 */

#include <dmlc/registry.h>
#include "survival_util.h"

namespace xgboost {
namespace common {

DMLC_REGISTER_PARAMETER(AFTParam);

}  // namespace common
}  // namespace xgboost
