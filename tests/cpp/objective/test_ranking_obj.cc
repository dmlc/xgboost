// Copyright by Contributors
#include <xgboost/objective.h>
#include <xgboost/generic_parameters.h>
#include "../helpers.h"
#include <xgboost/json.h>

namespace xgboost {

TEST(Objective, PairwiseRankingGPair) {
  xgboost::GenericParameter tparam;
  std::vector<std::pair<std::string, std::string>> args;
  tparam.InitAllowUnknown(args);

  std::unique_ptr<xgboost::ObjFunction> obj {
    xgboost::ObjFunction::Create("rank:pairwise", &tparam)
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

TEST(Objective, NDCG_Json_IO) {
  xgboost::GenericParameter tparam;
  tparam.InitAllowUnknown(Args{});

  std::unique_ptr<xgboost::ObjFunction> obj {
    xgboost::ObjFunction::Create("rank:ndcg", &tparam)
  };

  obj->Configure(Args{});
  Json j_obj {Object()};
  obj->SaveConfig(&j_obj);

  ASSERT_EQ(get<String>(j_obj["name"]), "rank:ndcg");;

  auto const& j_param = j_obj["lambda_rank_param"];

  ASSERT_EQ(get<String>(j_param["num_pairsample"]), "1");
  ASSERT_EQ(get<String>(j_param["fix_list_weight"]), "0");
}

}  // namespace xgboost
