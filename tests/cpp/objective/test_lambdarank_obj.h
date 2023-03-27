/**
 * Copyright 2023, XGBoost Contributors
 */
#ifndef XGBOOST_OBJECTIVE_TEST_LAMBDARANK_OBJ_H_
#define XGBOOST_OBJECTIVE_TEST_LAMBDARANK_OBJ_H_
#include <gtest/gtest.h>
#include <xgboost/data.h>                           // for MetaInfo
#include <xgboost/host_device_vector.h>             // for HostDeviceVector
#include <xgboost/linalg.h>                         // for All
#include <xgboost/objective.h>                      // for ObjFunction

#include <memory>                                   // for shared_ptr, make_shared
#include <numeric>                                  // for iota
#include <vector>                                   // for vector

#include "../../../src/common/ranking_utils.h"      // for LambdaRankParam, MAPCache
#include "../../../src/objective/lambdarank_obj.h"  // for MAPStat
#include "../helpers.h"                             // for EmptyDMatrix

namespace xgboost::obj {
/**
 * \brief Initialize test data for make pair tests.
 */
void InitMakePairTest(Context const* ctx, MetaInfo* out_info, HostDeviceVector<float>* out_predt);
}  // namespace xgboost::obj
#endif  // XGBOOST_OBJECTIVE_TEST_LAMBDARANK_OBJ_H_
