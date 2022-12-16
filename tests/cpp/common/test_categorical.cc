/*!
 * Copyright 2021-2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>
#include <xgboost/learner.h>

#include <limits>

#include "../../../src/common/categorical.h"
#include "../helpers.h"

namespace xgboost {
namespace common {
TEST(Categorical, Decision) {
  // inf
  float a = std::numeric_limits<float>::infinity();

  ASSERT_TRUE(common::InvalidCat(a));
  std::vector<uint32_t> cats(256, 0);
  ASSERT_TRUE(Decision(cats, a));

  // larger than size
  a = 256;
  ASSERT_TRUE(Decision(cats, a));

  // negative
  a = -1;
  ASSERT_TRUE(Decision(cats, a));

  CatBitField bits{cats};
  bits.Set(0);
  a = -0.5;
  ASSERT_TRUE(Decision(cats, a));

  // round toward 0
  a = 0.5;
  ASSERT_FALSE(Decision(cats, a));

  // valid
  a = 13;
  bits.Set(a);
  ASSERT_FALSE(Decision(bits.Bits(), a));
}

/**
 * Test for running inference with input category greater than the one stored in tree.
 */
TEST(Categorical, MinimalSet) {
  std::size_t constexpr kRows = 256, kCols = 1, kCat = 3;
  std::vector<FeatureType> types{FeatureType::kCategorical};
  auto Xy =
      RandomDataGenerator{kRows, kCols, 0.0}.Type(types).MaxCategory(kCat).GenerateDMatrix(true);

  std::unique_ptr<Learner> learner{Learner::Create({Xy})};
  learner->SetParam("max_depth", "1");
  learner->SetParam("tree_method", "hist");
  learner->Configure();
  learner->UpdateOneIter(0, Xy);

  Json model{Object{}};
  learner->SaveModel(&model);
  auto tree = model["learner"]["gradient_booster"]["model"]["trees"][0];
  ASSERT_GE(get<I32Array const>(tree["categories"]).size(), 1);
  auto v = get<I32Array const>(tree["categories"])[0];

  HostDeviceVector<float> predt;
  {
    std::vector<float> data{static_cast<float>(kCat),
                            static_cast<float>(kCat + 1), 32.0f, 33.0f, 34.0f};
    auto test = GetDMatrixFromData(data, data.size(), kCols);
    learner->Predict(test, false, &predt, 0, 0, false, /*pred_leaf=*/true);
    ASSERT_EQ(predt.Size(), data.size());
    auto const& h_predt = predt.ConstHostSpan();
    for (auto v : h_predt) {
      ASSERT_EQ(v, 1);  // left child of root node
    }
  }

  {
    std::unique_ptr<Learner> learner{Learner::Create({Xy})};
    learner->LoadModel(model);
    std::vector<float> data = {static_cast<float>(v)};
    auto test = GetDMatrixFromData(data, data.size(), kCols);
    learner->Predict(test, false, &predt, 0, 0, false, /*pred_leaf=*/true);
    auto const& h_predt = predt.ConstHostSpan();
    for (auto v : h_predt) {
      ASSERT_EQ(v, 2);  // right child of root node
    }
  }
}
}  // namespace common
}  // namespace xgboost
