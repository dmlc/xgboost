/**
 * Copyright (c) 2023, XGBoost Contributors
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
void TestMAPStat(Context const* ctx);

inline void TestNDCGJsonIO(Context const* ctx) {
  std::unique_ptr<xgboost::ObjFunction> obj{ObjFunction::Create("rank:ndcg", ctx)};

  obj->Configure(Args{});
  Json j_obj{Object()};
  obj->SaveConfig(&j_obj);

  ASSERT_EQ(get<String>(j_obj["name"]), "rank:ndcg");
  auto const& j_param = j_obj["lambdarank_param"];

  ASSERT_EQ(get<String>(j_param["ndcg_exp_gain"]), "1");
  ASSERT_EQ(get<String>(j_param["lambdarank_num_pair_per_sample"]),
            std::to_string(ltr::LambdaRankParam::NotSet()));
}

void TestNDCGGPair(Context const* ctx);

void TestUnbiasedNDCG(Context const* ctx);

void TestMAPGPair(Context const* ctx);

/**
 * \brief Initialize test data for make pair tests.
 */
void InitMakePairTest(Context const* ctx, MetaInfo* out_info, HostDeviceVector<float>* out_predt);
}  // namespace xgboost::obj
#endif  // XGBOOST_OBJECTIVE_TEST_LAMBDARANK_OBJ_H_
