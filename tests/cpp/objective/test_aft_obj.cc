/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <limits>
#include <cmath>

#include "xgboost/objective.h"
#include "xgboost/logging.h"
#include "../helpers.h"
#include "../../../src/common/survival_util.h"

namespace xgboost {
namespace common {

TEST(Objective, DeclareUnifiedTest(AFTObjConfiguration)) {
  auto ctx = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<ObjFunction> objective(ObjFunction::Create("survival:aft", &ctx));
  objective->Configure({ {"aft_loss_distribution", "logistic"},
                          {"aft_loss_distribution_scale", "5"} });

  // Configuration round-trip test
  Json j_obj{ Object() };
  objective->SaveConfig(&j_obj);
  EXPECT_EQ(get<String>(j_obj["name"]), "survival:aft");
  auto aft_param_json = j_obj["aft_loss_param"];
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution"]), "logistic");
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution_scale"]), "5");
}

/**
 * Verify that gradient pair (gpair) is computed correctly for various prediction values.
 * Reference values obtained from
 * https://github.com/avinashbarnwal/GSOC-2019/blob/master/AFT/R/combined_assignment.R
 **/

// Generate prediction value ranging from 2**1 to 2**15, using grid points in log scale
// Then check prediction against the reference values
static inline void CheckGPairOverGridPoints(
                      ObjFunction* obj,
                      bst_float true_label_lower_bound,
                      bst_float true_label_upper_bound,
                      const std::string& dist_type,
                      const std::vector<bst_float>& expected_grad,
                      const std::vector<bst_float>& expected_hess,
                      float ftol = 1e-4f) {
  const int num_point = 20;
  const double log_y_low = 1.0;
  const double log_y_high = 15.0;

  obj->Configure({ {"aft_loss_distribution", dist_type},
                   {"aft_loss_distribution_scale", "1"} });

  MetaInfo info;
  info.num_row_ = num_point;
  info.labels_lower_bound_.HostVector()
    = std::vector<bst_float>(num_point, true_label_lower_bound);
  info.labels_upper_bound_.HostVector()
    = std::vector<bst_float>(num_point, true_label_upper_bound);
  info.weights_.HostVector() = std::vector<bst_float>();
  std::vector<bst_float> preds(num_point);
  for (int i = 0; i < num_point; ++i) {
    preds[i] = std::log(std::pow(2.0, i * (log_y_high - log_y_low) / (num_point - 1) + log_y_low));
  }

  HostDeviceVector<GradientPair> out_gpair;
  obj->GetGradient(HostDeviceVector<bst_float>(preds), info, 1, &out_gpair);
  const auto& gpair = out_gpair.HostVector();
  CHECK_EQ(num_point, expected_grad.size());
  CHECK_EQ(num_point, expected_hess.size());
  for (int i = 0; i < num_point; ++i) {
    EXPECT_NEAR(gpair[i].GetGrad(), expected_grad[i], ftol);
    EXPECT_NEAR(gpair[i].GetHess(), expected_hess[i], ftol);
  }
}

TEST(Objective, DeclareUnifiedTest(AFTObjGPairUncensoredLabels)) {
  auto ctx = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<ObjFunction> obj(ObjFunction::Create("survival:aft", &ctx));

  CheckGPairOverGridPoints(obj.get(), 100.0f, 100.0f, "normal",
    { -3.9120f, -3.4013f, -2.8905f, -2.3798f, -1.8691f, -1.3583f, -0.8476f, -0.3368f, 0.1739f,
      0.6846f, 1.1954f, 1.7061f, 2.2169f, 2.7276f, 3.2383f, 3.7491f, 4.2598f, 4.7706f, 5.2813f,
      5.7920f },
    { 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f,
      1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f, 1.0000f });
  CheckGPairOverGridPoints(obj.get(), 100.0f, 100.0f, "logistic",
    { -0.9608f, -0.9355f, -0.8948f, -0.8305f, -0.7327f, -0.5910f, -0.4001f, -0.1668f, 0.0867f,
      0.3295f, 0.5354f, 0.6927f, 0.8035f, 0.8773f, 0.9245f, 0.9540f, 0.9721f, 0.9832f, 0.9899f,
      0.9939f },
    { 0.0384f, 0.0624f, 0.0997f, 0.1551f, 0.2316f, 0.3254f, 0.4200f, 0.4861f, 0.4962f, 0.4457f,
      0.3567f, 0.2601f, 0.1772f, 0.1152f, 0.0726f, 0.0449f, 0.0275f, 0.0167f, 0.0101f, 0.0061f });
  CheckGPairOverGridPoints(obj.get(), 100.0f, 100.0f, "extreme",
    { -15.0000f, -15.0000f, -15.0000f, -9.8028f, -5.4822f, -2.8897f, -1.3340f, -0.4005f, 0.1596f,
      0.4957f, 0.6974f, 0.8184f, 0.8910f, 0.9346f, 0.9608f, 0.9765f, 0.9859f, 0.9915f, 0.9949f,
      0.9969f },
    { 15.0000f, 15.0000f, 15.0000f, 10.8028f, 6.4822f, 3.8897f, 2.3340f, 1.4005f, 0.8404f, 0.5043f,
      0.3026f, 0.1816f, 0.1090f, 0.0654f, 0.0392f, 0.0235f, 0.0141f, 0.0085f, 0.0051f, 0.0031f });
}

TEST(Objective, DeclareUnifiedTest(AFTObjGPairLeftCensoredLabels)) {
  auto ctx = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<ObjFunction> obj(ObjFunction::Create("survival:aft", &ctx));

  CheckGPairOverGridPoints(obj.get(), 0.0f, 20.0f, "normal",
    { 0.0285f, 0.0832f, 0.1951f, 0.3804f, 0.6403f, 0.9643f, 1.3379f, 1.7475f, 2.1828f, 2.6361f,
      3.1023f, 3.5779f, 4.0603f, 4.5479f, 5.0394f, 5.5340f, 6.0309f, 6.5298f, 7.0303f, 7.5326f },
    { 0.0663f, 0.1559f, 0.2881f, 0.4378f, 0.5762f, 0.6878f, 0.7707f, 0.8300f, 0.8719f, 0.9016f,
      0.9229f, 0.9385f, 0.9501f, 0.9588f, 0.9656f, 0.9709f, 0.9751f, 0.9785f, 0.9813f, 0.9877f });
  CheckGPairOverGridPoints(obj.get(), 0.0f, 20.0f, "logistic",
    { 0.0909f, 0.1428f, 0.2174f, 0.3164f, 0.4355f, 0.5625f, 0.6818f, 0.7812f, 0.8561f, 0.9084f,
      0.9429f, 0.9650f, 0.9787f, 0.9871f, 0.9922f, 0.9953f, 0.9972f, 0.9983f, 0.9990f, 0.9994f },
    { 0.0826f, 0.1224f, 0.1701f, 0.2163f, 0.2458f, 0.2461f, 0.2170f, 0.1709f, 0.1232f, 0.0832f,
      0.0538f, 0.0338f, 0.0209f, 0.0127f, 0.0077f, 0.0047f, 0.0028f, 0.0017f, 0.0010f, 0.0006f });
  CheckGPairOverGridPoints(obj.get(), 0.0f, 20.0f, "extreme",
    { 0.0005f, 0.0149f, 0.1011f, 0.2815f, 0.4881f, 0.6610f, 0.7847f, 0.8665f, 0.9183f, 0.9504f,
      0.9700f, 0.9820f, 0.9891f, 0.9935f, 0.9961f, 0.9976f, 0.9986f, 0.9992f, 0.9995f, 0.9997f },
    { 0.0041f, 0.0747f, 0.2731f, 0.4059f, 0.3829f, 0.2901f, 0.1973f, 0.1270f, 0.0793f, 0.0487f,
      0.0296f, 0.0179f, 0.0108f, 0.0065f, 0.0039f, 0.0024f, 0.0014f, 0.0008f, 0.0005f, 0.0003f });
}

TEST(Objective, DeclareUnifiedTest(AFTObjGPairRightCensoredLabels)) {
  auto ctx = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<ObjFunction> obj(ObjFunction::Create("survival:aft", &ctx));

  CheckGPairOverGridPoints(obj.get(), 60.0f, std::numeric_limits<float>::infinity(), "normal",
    { -3.6583f, -3.1815f, -2.7135f, -2.2577f, -1.8190f, -1.4044f, -1.0239f, -0.6905f, -0.4190f,
      -0.2209f, -0.0973f, -0.0346f, -0.0097f, -0.0021f, -0.0004f, -0.0000f, -0.0000f, -0.0000f,
      -0.0000f, -0.0000f },
    { 0.9407f, 0.9259f, 0.9057f, 0.8776f, 0.8381f, 0.7821f, 0.7036f, 0.5970f, 0.4624f, 0.3128f,
      0.1756f, 0.0780f, 0.0265f, 0.0068f, 0.0013f, 0.0002f, 0.0000f, 0.0000f, 0.0000f, 0.0000f });
  CheckGPairOverGridPoints(obj.get(), 60.0f, std::numeric_limits<float>::infinity(), "logistic",
    { -0.9677f, -0.9474f, -0.9153f, -0.8663f, -0.7955f, -0.7000f, -0.5834f, -0.4566f, -0.3352f,
      -0.2323f, -0.1537f, -0.0982f, -0.0614f, -0.0377f, -0.0230f, -0.0139f, -0.0084f, -0.0051f,
      -0.0030f, -0.0018f },
    { 0.0312f, 0.0499f, 0.0776f, 0.1158f, 0.1627f, 0.2100f, 0.2430f, 0.2481f, 0.2228f, 0.1783f,
      0.1300f, 0.0886f, 0.0576f, 0.0363f, 0.0225f, 0.0137f, 0.0083f, 0.0050f, 0.0030f, 0.0018f });
  CheckGPairOverGridPoints(obj.get(), 60.0f, std::numeric_limits<float>::infinity(), "extreme",
    { -15.0000f, -15.0000f, -10.8018f, -6.4817f, -3.8893f, -2.3338f, -1.4004f, -0.8403f, -0.5042f,
      -0.3026f, -0.1816f, -0.1089f, -0.0654f, -0.0392f, -0.0235f, -0.0141f, -0.0085f, -0.0051f,
      -0.0031f, -0.0018f },
    { 15.0000f, 15.0000f, 10.8018f, 6.4817f, 3.8893f, 2.3338f, 1.4004f, 0.8403f, 0.5042f, 0.3026f,
      0.1816f, 0.1089f, 0.0654f, 0.0392f, 0.0235f, 0.0141f, 0.0085f, 0.0051f, 0.0031f, 0.0018f });
}

TEST(Objective, DeclareUnifiedTest(AFTObjGPairIntervalCensoredLabels)) {
  auto ctx = CreateEmptyGenericParam(GPUIDX);
  std::unique_ptr<ObjFunction> obj(ObjFunction::Create("survival:aft", &ctx));

  CheckGPairOverGridPoints(obj.get(), 16.0f, 200.0f, "normal",
    { -2.4435f, -1.9965f, -1.5691f, -1.1679f, -0.7990f, -0.4649f, -0.1596f, 0.1336f, 0.4370f,
      0.7682f, 1.1340f, 1.5326f, 1.9579f, 2.4035f, 2.8639f, 3.3351f, 3.8143f, 4.2995f, 4.7891f,
      5.2822f },
    { 0.8909f, 0.8579f, 0.8134f, 0.7557f, 0.6880f, 0.6221f, 0.5789f, 0.5769f, 0.6171f, 0.6818f,
      0.7500f, 0.8088f, 0.8545f, 0.8884f, 0.9131f, 0.9312f, 0.9446f, 0.9547f, 0.9624f, 0.9684f });
  CheckGPairOverGridPoints(obj.get(), 16.0f, 200.0f, "logistic",
    { -0.8790f, -0.8112f, -0.7153f, -0.5893f, -0.4375f, -0.2697f, -0.0955f, 0.0800f, 0.2545f,
      0.4232f, 0.5768f, 0.7054f, 0.8040f, 0.8740f, 0.9210f, 0.9513f, 0.9703f, 0.9820f, 0.9891f,
      0.9934f },
    { 0.1086f, 0.1588f, 0.2176f, 0.2745f, 0.3164f, 0.3374f, 0.3433f, 0.3434f, 0.3384f, 0.3191f,
      0.2789f, 0.2229f, 0.1637f, 0.1125f, 0.0737f, 0.0467f, 0.0290f, 0.0177f, 0.0108f, 0.0065f });
  CheckGPairOverGridPoints(obj.get(), 16.0f, 200.0f, "extreme",
    { -8.0000f, -4.8004f, -2.8805f, -1.7284f, -1.0371f, -0.6168f, -0.3140f, -0.0121f, 0.2841f,
      0.5261f, 0.6989f, 0.8132f, 0.8857f, 0.9306f, 0.9581f, 0.9747f, 0.9848f, 0.9909f, 0.9945f,
      0.9967f },
    { 8.0000f, 4.8004f, 2.8805f, 1.7284f, 1.0380f, 0.6567f, 0.5727f, 0.6033f, 0.5384f, 0.4051f,
      0.2757f, 0.1776f, 0.1110f, 0.0682f, 0.0415f, 0.0251f, 0.0151f, 0.0091f, 0.0055f, 0.0033f });
}

}  // namespace common
}  // namespace xgboost
