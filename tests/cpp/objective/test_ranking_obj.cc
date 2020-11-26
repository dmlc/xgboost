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

TEST(Objective, DeclareUnifiedTest(NDCG_JsonIO)) {
  xgboost::Context ctx;
  ctx.UpdateAllowUnknown(Args{});

  std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("lambdamart:ndcg", &ctx)};

  obj->Configure(Args{});
  Json j_obj {Object()};
  obj->SaveConfig(&j_obj);

  ASSERT_EQ(get<String>(j_obj["name"]), "lambdamart:ndcg");;

  auto const& j_param = j_obj["ndcg_param"];

  ASSERT_EQ(get<String>(j_param["ndcg_label_type"]), "relevance");
  ASSERT_EQ(get<String>(j_param["ndcg_truncation"]), "1");
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

TEST(Objective, DeclareUnifiedTest(NDCGRankingGPair)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::Context ctx = xgboost::CreateEmptyGenericParam(GPUIDX);

  std::unique_ptr<xgboost::ObjFunction> obj{xgboost::ObjFunction::Create("lambdamart:ndcg", &ctx)};
  obj->Configure(args);
  CheckConfigReload(obj, "lambdamart:ndcg");

  // No gain in swapping 2 documents.
  CheckRankingObjFunction(obj, {1, 1, 1, 1}, {1, 1, 1, 1}, {1.0f, 1.0f}, {0, 2, 4},
                          {0.0f, -0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f});

  HostDeviceVector<float> predts{0, 1, 0, 1};
  MetaInfo info;
  info.labels = linalg::Tensor<float, 2>{{0, 1, 0, 1}, {4, 1}, GPUIDX};
  info.group_ptr_ = {0, 2, 4};
  info.num_row_ = 4;
  HostDeviceVector<GradientPair> gpairs;
  obj->GetGradient(predts, info, 0, &gpairs);
  ASSERT_EQ(gpairs.Size(), predts.Size());

  {
    predts = {1, 0, 1, 0};
    HostDeviceVector<GradientPair> gpairs;
    obj->GetGradient(predts, info, 0, &gpairs);
    for (size_t i = 0; i < gpairs.Size(); ++i) {
      ASSERT_GT(gpairs.HostSpan()[i].GetHess(), 0);
    }
    ASSERT_LT(gpairs.HostSpan()[1].GetGrad(), 0);
    ASSERT_LT(gpairs.HostSpan()[3].GetGrad(), 0);

    ASSERT_GT(gpairs.HostSpan()[0].GetGrad(), 0);
    ASSERT_GT(gpairs.HostSpan()[2].GetGrad(), 0);

    info.weights_ = {2, 3};
    HostDeviceVector<GradientPair> weighted_gpairs;
    obj->GetGradient(predts, info, 0, &weighted_gpairs);
    auto const& h_gpairs = gpairs.ConstHostSpan();
    auto const& h_weighted_gpairs = weighted_gpairs.ConstHostSpan();
    for (size_t i : {0ul, 1ul}) {
      ASSERT_FLOAT_EQ(h_weighted_gpairs[i].GetGrad(), h_gpairs[i].GetGrad() * 2.0f);
      ASSERT_FLOAT_EQ(h_weighted_gpairs[i].GetHess(), h_gpairs[i].GetHess() * 2.0f);
    }
    for (size_t i : {2ul, 3ul}) {
      ASSERT_FLOAT_EQ(h_weighted_gpairs[i].GetGrad(), h_gpairs[i].GetGrad() * 3.0f);
      ASSERT_FLOAT_EQ(h_weighted_gpairs[i].GetHess(), h_gpairs[i].GetHess() * 3.0f);
    }
  }

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
