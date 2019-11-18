// Copyright by Contributors
#include <xgboost/objective.h>
#include <xgboost/generic_parameters.h>
#include "../helpers.h"
#include "../../src/common/json_experimental.h"

namespace xgboost {

TEST(Objective, DeclareUnifiedTest(PairwiseRankingGPair)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::GenericParameter lparam = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<xgboost::ObjFunction> obj {
    xgboost::ObjFunction::Create("rank:pairwise", &lparam)
  };
  obj->Configure(args);
  CheckConfigReload(obj, "rank:pairwise");

  // Test with setting sample weight to second query group
  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {2.0f, 0.0f},
                          {0, 2, 4},
                          {1.9f, -1.9f, 0.0f, 0.0f},
                          {1.995f, 1.995f, 0.0f, 0.0f});

  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {1.0f, 1.0f},
                          {0, 2, 4},
                          {0.95f, -0.95f,  0.95f, -0.95f},
                          {0.9975f, 0.9975f, 0.9975f, 0.9975f});

  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

TEST(Objective, DeclareUnifiedTest(NDCG_Json_IO)) {
  xgboost::GenericParameter tparam;
  tparam.UpdateAllowUnknown(Args{});

  std::unique_ptr<xgboost::ObjFunction> obj {
    xgboost::ObjFunction::Create("rank:ndcg", &tparam)
  };

  obj->Configure(Args{});
  xgboost::experimental::Document j_obj;
  obj->SaveConfig(&(j_obj.GetObject()));

  ASSERT_EQ((*j_obj.GetObject().FindMemberByKey("name")).GetString(), "rank:ndcg");;

  auto j_param = *(j_obj.GetObject().FindMemberByKey("lambda_rank_param"));
  ASSERT_TRUE(j_param.IsObject());
  ASSERT_EQ((*j_param.FindMemberByKey("num_pairsample")).GetString(), "1");
  ASSERT_EQ((*j_param.FindMemberByKey("fix_list_weight")).GetString(), "0");
}

TEST(Objective, DeclareUnifiedTest(PairwiseRankingGPairSameLabels)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::GenericParameter lparam = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<ObjFunction> obj {
    ObjFunction::Create("rank:pairwise", &lparam)
  };
  obj->Configure(args);
  // No computation of gradient/hessian, as there is no diversity in labels
  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {1,   1, 1, 1},
                          {2.0f, 0.0f},
                          {0, 2, 4},
                          {0.0f, 0.0f, 0.0f, 0.0f},
                          {0.0f, 0.0f, 0.0f, 0.0f});

  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

TEST(Objective, DeclareUnifiedTest(NDCGRankingGPair)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::GenericParameter lparam = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<xgboost::ObjFunction> obj {
    xgboost::ObjFunction::Create("rank:ndcg", &lparam)
  };
  obj->Configure(args);
  CheckConfigReload(obj, "rank:ndcg");

  // Test with setting sample weight to second query group
  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {2.0f, 0.0f},
                          {0, 2, 4},
                          {0.7f, -0.7f, 0.0f, 0.0f},
                          {0.74f, 0.74f, 0.0f, 0.0f});

  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {1.0f, 1.0f},
                          {0, 2, 4},
                          {0.35f, -0.35f,  0.35f, -0.35f},
                          {0.368f, 0.368f, 0.368f, 0.368f});
  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

}  // namespace xgboost
