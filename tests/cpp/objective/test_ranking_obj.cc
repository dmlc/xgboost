// Copyright by Contributors
#include <xgboost/context.h>
#include <xgboost/json.h>
#include <xgboost/objective.h>

#include "../helpers.h"

namespace xgboost {

TEST(Objective, DeclareUnifiedTest(PairwiseRankingGPair)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::Context ctx = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("rank:pairwise", &ctx)};
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

TEST(Objective, DeclareUnifiedTest(PairwiseRankingGPairSameLabels)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::Context ctx = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<ObjFunction> obj{ObjFunction::Create("rank:pairwise", &ctx)};
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

TEST(Objective, DeclareUnifiedTest(MAPRankingGPair)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::Context ctx = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("rank:map", &ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, "rank:map");

  // Test with setting sample weight to second query group
  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {2.0f, 0.0f},
                          {0, 2, 4},
                          {0.95f, -0.95f,  0.0f, 0.0f},
                          {0.9975f, 0.9975f, 0.0f, 0.0f});

  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {1.0f, 1.0f},
                          {0, 2, 4},
                          {0.475f, -0.475f,  0.475f, -0.475f},
                          {0.4988f, 0.4988f, 0.4988f, 0.4988f});
  ASSERT_NO_THROW(obj->DefaultEvalMetric());
}

}  // namespace xgboost
