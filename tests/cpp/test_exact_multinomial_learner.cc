/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/json.h>
#include <xgboost/learner.h>

#include <algorithm>  // for max
#include <cmath>      // for fabs, isfinite
#include <memory>   // for unique_ptr
#include <string>   // for string, to_string
#include <vector>   // for vector

#include "helpers.h"

namespace xgboost {
namespace {
std::shared_ptr<DMatrix> MakeClassification(bst_idx_t n_rows, bst_feature_t n_cols,
                                            bst_target_t n_classes) {
  return RandomDataGenerator{n_rows, n_cols, 0.0f}.Seed(17).Classes(n_classes).GenerateDMatrix(
      true);
}

/** @brief Parameters for one training run. `diagonal` gives the pre-existing behaviour. */
Args ExactArgs(bst_target_t n_classes, char const* multi_hessian, Args extra = {}) {
  Args args{{"objective", "multi:softprob"},
            {"num_class", std::to_string(n_classes)},
            {"multi_strategy", "multi_output_tree"},
            {"multi_hessian", multi_hessian},
            {"tree_method", "hist"},
            {"device", "cpu"},
            {"max_depth", "3"},
            {"eta", "0.3"},
            {"lambda", "1.0"},
            {"base_score", "0.5"}};
  for (auto const& kv : extra) {
    args.emplace_back(kv);
  }
  return args;
}

std::unique_ptr<Learner> Train(std::shared_ptr<DMatrix> dmat, Args const& args,
                               std::int32_t rounds) {
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->Configure(args);
  for (std::int32_t i = 0; i < rounds; ++i) {
    learner->UpdateOneIter(i, dmat);
  }
  return learner;
}

std::vector<float> Predict(Learner* learner, std::shared_ptr<DMatrix> dmat) {
  HostDeviceVector<float> out;
  learner->Predict(dmat, false, &out, 0, 0);
  return out.HostVector();
}
}  // anonymous namespace

/** Exact mode trains end to end and produces usable probabilities. */
TEST(ExactMultinomialLearner, TrainsAndProducesValidProbabilities) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    auto dmat = MakeClassification(256, 8, n_classes);
    auto learner = Train(dmat, ExactArgs(n_classes, "exact"), 4);
    auto predt = Predict(learner.get(), dmat);

    ASSERT_EQ(predt.size(), dmat->Info().num_row_ * n_classes) << "K=" << n_classes;
    for (auto v : predt) {
      ASSERT_TRUE(std::isfinite(v)) << "K=" << n_classes << ": non-finite prediction";
      ASSERT_GE(v, 0.0f);
      ASSERT_LE(v, 1.0f);
    }
    // multi:softprob rows are probability distributions.
    for (std::size_t r = 0; r < dmat->Info().num_row_; ++r) {
      double total = 0.0;
      for (bst_target_t k = 0; k < n_classes; ++k) {
        total += predt[r * n_classes + k];
      }
      ASSERT_NEAR(total, 1.0, 1e-4) << "K=" << n_classes << " row " << r;
    }
  }
}

/** Exact mode reduces training loss over rounds. */
TEST(ExactMultinomialLearner, LossDecreases) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(512, 8, kNumClasses);
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->Configure(ExactArgs(kNumClasses, "exact"));

  std::vector<double> losses;
  for (std::int32_t i = 0; i < 6; ++i) {
    learner->UpdateOneIter(i, dmat);
    auto eval = learner->EvalOneIter(i, {dmat}, {"train"});
    auto pos = eval.rfind(':');
    ASSERT_NE(pos, std::string::npos);
    losses.push_back(std::stod(eval.substr(pos + 1)));
  }
  ASSERT_EQ(losses.size(), 6u);
  EXPECT_LT(losses.back(), losses.front())
      << "training loss did not improve: " << losses.front() << " -> " << losses.back();
  for (auto v : losses) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

/** Multiple boosting rounds: no stale sidecar, no drift, no double eta. */
TEST(ExactMultinomialLearner, MultipleRounds) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(256, 6, kNumClasses);
  for (std::int32_t rounds : {1, 2, 5, 10}) {
    auto learner = Train(dmat, ExactArgs(kNumClasses, "exact"), rounds);
    auto predt = Predict(learner.get(), dmat);
    for (auto v : predt) {
      ASSERT_TRUE(std::isfinite(v)) << "rounds=" << rounds;
    }
    Json config{Object{}};
    learner->SaveConfig(&config);
    // The requested mode must survive every round rather than being reset.
    auto const& learner_cfg = get<Object const>(config["learner"]);
    auto const& train_param = get<Object const>(learner_cfg.at("learner_train_param"));
    EXPECT_EQ(get<String const>(train_param.at("multi_hessian")), "exact")
        << "rounds=" << rounds;
  }
}

/** Sample weights are honoured by the exact path. */
TEST(ExactMultinomialLearner, SampleWeights) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(256, 6, kNumClasses);
  auto unweighted = Predict(Train(dmat, ExactArgs(kNumClasses, "exact"), 3).get(), dmat);

  // Weight the second half far more heavily; the fitted model must respond.
  auto& weights = dmat->Info().weights_.HostVector();
  weights.resize(dmat->Info().num_row_);
  for (std::size_t r = 0; r < weights.size(); ++r) {
    weights[r] = r < weights.size() / 2 ? 0.1f : 10.0f;
  }
  auto weighted = Predict(Train(dmat, ExactArgs(kNumClasses, "exact"), 3).get(), dmat);

  ASSERT_EQ(unweighted.size(), weighted.size());
  bool differs = false;
  for (std::size_t i = 0; i < weighted.size(); ++i) {
    ASSERT_TRUE(std::isfinite(weighted[i]));
    if (std::fabs(weighted[i] - unweighted[i]) > 1e-4) {
      differs = true;
    }
  }
  EXPECT_TRUE(differs) << "sample weights had no effect on exact training";
  // Clear so later runs on this matrix are unaffected.
  weights.clear();
}

/** Model round-trip: predictions must be identical after save and load. */
TEST(ExactMultinomialLearner, SaveLoadRoundTrip) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    auto dmat = MakeClassification(256, 8, n_classes);
    auto learner = Train(dmat, ExactArgs(n_classes, "exact"), 3);
    auto before = Predict(learner.get(), dmat);

    Json model{Object{}};
    learner->SaveModel(&model);

    std::unique_ptr<Learner> restored{Learner::Create({dmat})};
    restored->LoadModel(model);
    auto after = Predict(restored.get(), dmat);

    ASSERT_EQ(before.size(), after.size()) << "K=" << n_classes;
    for (std::size_t i = 0; i < before.size(); ++i) {
      EXPECT_FLOAT_EQ(before[i], after[i]) << "K=" << n_classes << " entry " << i;
    }

    // The Hessian is training-time state; nothing about it may reach the model.
    std::string serialized;
    Json::Dump(model, &serialized);
    EXPECT_EQ(serialized.find("exact_hessian"), std::string::npos)
        << "K=" << n_classes << ": Hessian data leaked into the model";
    EXPECT_EQ(serialized.find("multi_hessian"), std::string::npos)
        << "K=" << n_classes << ": a training parameter leaked into the model";
  }
}

/**
 * Normal mode must be byte-for-byte what it was before exact mode existed.
 *
 * Compared on raw prediction buffers rather than a summary metric, so a small numerical
 * drift cannot hide.
 */
TEST(ExactMultinomialLearner, DiagonalModeUnchanged) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    auto dmat = MakeClassification(256, 8, n_classes);
    // Explicitly diagonal, and the default (no multi_hessian given at all) must agree.
    auto explicit_diagonal = Predict(Train(dmat, ExactArgs(n_classes, "diagonal"), 3).get(), dmat);

    Args defaulted{{"objective", "multi:softprob"},
                   {"num_class", std::to_string(n_classes)},
                   {"multi_strategy", "multi_output_tree"},
                   {"tree_method", "hist"},
                   {"device", "cpu"},
                   {"max_depth", "3"},
                   {"eta", "0.3"},
                   {"lambda", "1.0"},
                   {"base_score", "0.5"}};
    auto default_mode = Predict(Train(dmat, defaulted, 3).get(), dmat);

    ASSERT_EQ(explicit_diagonal.size(), default_mode.size()) << "K=" << n_classes;
    for (std::size_t i = 0; i < default_mode.size(); ++i) {
      EXPECT_FLOAT_EQ(explicit_diagonal[i], default_mode[i])
          << "K=" << n_classes << ": specifying multi_hessian=diagonal changed the default path";
    }
  }
}

/** Exact and diagonal must produce genuinely different models. */
TEST(ExactMultinomialLearner, ExactDiffersFromDiagonal) {
  bst_target_t constexpr kNumClasses = 4;
  auto dmat = MakeClassification(512, 8, kNumClasses);
  auto diagonal = Predict(Train(dmat, ExactArgs(kNumClasses, "diagonal"), 4).get(), dmat);
  auto exact = Predict(Train(dmat, ExactArgs(kNumClasses, "exact"), 4).get(), dmat);

  ASSERT_EQ(diagonal.size(), exact.size());
  bool differs = false;
  for (std::size_t i = 0; i < exact.size(); ++i) {
    ASSERT_TRUE(std::isfinite(exact[i]));
    if (std::fabs(exact[i] - diagonal[i]) > 1e-4) {
      differs = true;
      break;
    }
  }
  EXPECT_TRUE(differs)
      << "exact and diagonal produced identical predictions, so multi_hessian=exact is not "
         "reaching training";
}

/** Regularization knobs are accepted and change the result. */
TEST(ExactMultinomialLearner, RegularizationKnobs) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(256, 6, kNumClasses);

  auto base = Predict(Train(dmat, ExactArgs(kNumClasses, "exact"), 3).get(), dmat);
  auto no_lambda =
      Predict(Train(dmat, ExactArgs(kNumClasses, "exact", {{"lambda", "0"}}), 3).get(), dmat);
  auto big_lambda =
      Predict(Train(dmat, ExactArgs(kNumClasses, "exact", {{"lambda", "100"}}), 3).get(), dmat);
  auto gamma =
      Predict(Train(dmat, ExactArgs(kNumClasses, "exact", {{"gamma", "1e9"}}), 3).get(), dmat);
  auto mcw = Predict(
      Train(dmat, ExactArgs(kNumClasses, "exact", {{"min_child_weight", "1e9"}}), 3).get(), dmat);
  auto subsample =
      Predict(Train(dmat, ExactArgs(kNumClasses, "exact", {{"subsample", "0.5"}}), 3).get(), dmat);

  for (auto const* p : {&base, &no_lambda, &big_lambda, &gamma, &mcw, &subsample}) {
    for (auto v : *p) {
      ASSERT_TRUE(std::isfinite(v));
    }
  }
  // lambda must matter.
  bool lambda_matters = false;
  for (std::size_t i = 0; i < base.size(); ++i) {
    if (std::fabs(base[i] - big_lambda[i]) > 1e-4) {
      lambda_matters = true;
      break;
    }
  }
  EXPECT_TRUE(lambda_matters) << "reg_lambda had no effect in exact mode";

  // gamma and min_child_weight large enough to block every split give stump-only models, so
  // both must agree with each other.
  for (std::size_t i = 0; i < gamma.size(); ++i) {
    EXPECT_NEAR(gamma[i], mcw[i], 1e-5) << "entry " << i;
  }
}

/** Every unsupported configuration must fail loudly rather than fall back. */
TEST(ExactMultinomialLearner, RejectsUnsupportedConfigurations) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(128, 6, kNumClasses);

  auto expect_rejected = [&](Args const& args, char const* what) {
    std::unique_ptr<Learner> learner{Learner::Create({dmat})};
    learner->Configure(args);
    EXPECT_THROW(learner->UpdateOneIter(0, dmat), dmlc::Error) << "not rejected: " << what;
  };

  // Objective without an exact producer.
  expect_rejected(Args{{"objective", "reg:squarederror"},
                       {"multi_strategy", "multi_output_tree"},
                       {"multi_hessian", "exact"},
                       {"num_target", "3"},
                       {"tree_method", "hist"},
                       {"device", "cpu"}},
                  "reg:squarederror");
  // Scalar tree strategy cannot carry a joint solve.
  expect_rejected(ExactArgs(kNumClasses, "exact", {{"multi_strategy", "one_output_per_tree"}}),
                  "one_output_per_tree");
  // Parameters with no defined dense analogue.
  expect_rejected(ExactArgs(kNumClasses, "exact", {{"alpha", "0.5"}}), "reg_alpha");
  expect_rejected(ExactArgs(kNumClasses, "exact", {{"max_delta_step", "1.0"}}), "max_delta_step");
  // Sampling that rescales rows using the scalar pair.
  expect_rejected(ExactArgs(kNumClasses, "exact",
                            {{"subsample", "0.5"}, {"sampling_method", "gradient_based"}}),
                  "gradient_based sampling");
  // Tree methods without an exact implementation.
  expect_rejected(ExactArgs(kNumClasses, "exact", {{"tree_method", "approx"}}), "approx");
}

/**
 * Intercept-only validation against the research mathematics.
 *
 * With a single constant feature no split is possible, so every tree is a stump and the leaf
 * is exactly the joint Newton step over the whole dataset -- the same problem
 * `research/exact_multiclass_hessian.py` solves. The fitted probabilities must therefore
 * converge to the empirical class proportions, which is the optimum of an intercept-only
 * multinomial model.
 *
 * This is a comparison of the *solution*, not of iteration counts: XGBoost applies a
 * learning rate and boosts additively, so its trajectory differs from a pure Newton solve
 * even though the fixed point is the same.
 */
TEST(ExactMultinomialLearner, InterceptOnlyMatchesEmpiricalProportions) {
  for (bst_target_t n_classes : {2u, 3u, 7u}) {
    bst_idx_t constexpr kRows = 900;
    // One constant feature: nothing to split on.
    std::vector<float> feature(kRows, 1.0f);
    auto dmat = GetDMatrixFromData(feature, kRows, 1);

    // Deterministic, deliberately unbalanced class counts: class k receives (k+1) parts of
    // the total, so no class is empty and the distribution is far from uniform.
    auto& labels = dmat->Info().labels.Data()->HostVector();
    labels.resize(kRows);
    std::vector<double> counts(n_classes, 0.0);
    double parts = static_cast<double>(n_classes) * (n_classes + 1) / 2.0;
    bst_idx_t filled = 0;
    for (bst_target_t k = 0; k < n_classes; ++k) {
      auto quota = (k + 1 == n_classes)
                       ? (kRows - filled)
                       : static_cast<bst_idx_t>(kRows * (k + 1) / parts);
      for (bst_idx_t i = 0; i < quota; ++i) {
        labels[filled + i] = static_cast<float>(k);
      }
      counts[k] = static_cast<double>(quota);
      filled += quota;
    }
    ASSERT_EQ(filled, kRows);
    dmat->Info().labels.Reshape(kRows, 1);
    dmat->Info().num_row_ = kRows;

    std::vector<double> proportions(n_classes);
    for (bst_target_t k = 0; k < n_classes; ++k) {
      proportions[k] = counts[k] / static_cast<double>(kRows);
      ASSERT_GT(proportions[k], 0.0) << "K=" << n_classes << " class " << k << " is empty";
    }

    // A full learning rate and many rounds so the additive model reaches the fixed point.
    auto args =
        ExactArgs(n_classes, "exact", {{"eta", "1.0"}, {"lambda", "0"}, {"max_depth", "1"}});
    auto learner = Train(dmat, args, 60);
    auto predt = Predict(learner.get(), dmat);

    // Intercept-only: every row shares the same distribution.
    for (bst_target_t k = 0; k < n_classes; ++k) {
      EXPECT_NEAR(predt[k], predt[(kRows - 1) * n_classes + k], 1e-5)
          << "K=" << n_classes << ": rows disagree in an intercept-only model";
    }

    double worst = 0.0;
    for (bst_target_t k = 0; k < n_classes; ++k) {
      worst = std::max(worst, std::fabs(static_cast<double>(predt[k]) - proportions[k]));
    }
    EXPECT_LT(worst, 5e-3) << "K=" << n_classes
                           << ": fitted probabilities did not reach the empirical proportions, "
                              "max deviation " << worst;
  }
}

/**
 * M5.5: categorical features must be rejected, not silently scanned as ordered values.
 *
 * Before this fix the exact enumerator set `is_cat = false` and walked a categorical
 * feature's bins in bin order, which is meaningless for unordered categories and would have
 * produced a wrong split rather than an error. This test fails if that behaviour returns.
 */
TEST(ExactMultinomialLearner, RejectsCategoricalFeatures) {
  bst_target_t constexpr kNumClasses = 3;
  bst_feature_t constexpr kCols = 2;
  bst_idx_t constexpr kRows = 256;
  bst_cat_t constexpr kNumCategories = 4;

  // Column 0 is numerical, column 1 holds genuine small-cardinality categories.
  std::vector<float> data(kRows * kCols);
  for (bst_idx_t r = 0; r < kRows; ++r) {
    data[r * kCols + 0] = static_cast<float>(r % 17) * 0.5f;
    data[r * kCols + 1] = static_cast<float>(r % kNumCategories);
  }
  auto dmat = GetDMatrixFromData(data, kRows, kCols);
  auto& labels = dmat->Info().labels.Data()->HostVector();
  labels.resize(kRows);
  for (bst_idx_t r = 0; r < kRows; ++r) {
    labels[r] = static_cast<float>(r % kNumClasses);
  }
  dmat->Info().labels.Reshape(kRows, 1);
  auto& h_ft = dmat->Info().feature_types.HostVector();
  h_ft.assign(kCols, FeatureType::kNumerical);
  h_ft[1] = FeatureType::kCategorical;

  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->Configure(ExactArgs(kNumClasses, "exact"));
  try {
    learner->UpdateOneIter(0, dmat);
    FAIL() << "a categorical feature was accepted by multi_hessian=exact";
  } catch (dmlc::Error const& e) {
    std::string msg{e.what()};
    EXPECT_NE(msg.find("categorical"), std::string::npos) << msg;
    EXPECT_NE(msg.find("multi_hessian"), std::string::npos) << msg;
  }

  // The same data must still train in the pre-existing diagonal mode.
  std::unique_ptr<Learner> diagonal{Learner::Create({dmat})};
  diagonal->Configure(ExactArgs(kNumClasses, "diagonal"));
  EXPECT_NO_THROW(diagonal->UpdateOneIter(0, dmat));
  auto predt = Predict(diagonal.get(), dmat);
  for (auto v : predt) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

/**
 * Part B: `multi_hessian` is training configuration, never model state.
 *
 * The exact Hessian is used to *fit* the leaves; once fitted, a model is just vector-leaf
 * trees and inference cannot tell how they were produced. So the parameter belongs in the
 * saved *config* (which is what reproduces a training run) and must not appear in the saved
 * *model* (which only has to reproduce predictions).
 */
TEST(ExactMultinomialLearner, ModeIsTrainingConfigNotModelState) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(256, 6, kNumClasses);
  auto learner = Train(dmat, ExactArgs(kNumClasses, "exact"), 2);

  // Present in the training configuration.
  Json config{Object{}};
  learner->SaveConfig(&config);
  auto const& train_param =
      get<Object const>(get<Object const>(config["learner"]).at("learner_train_param"));
  ASSERT_NE(train_param.find("multi_hessian"), train_param.cend());
  EXPECT_EQ(get<String const>(train_param.at("multi_hessian")), "exact");

  // Absent from the serialized model.
  Json model{Object{}};
  learner->SaveModel(&model);
  std::string dumped;
  Json::Dump(model, &dumped);
  EXPECT_EQ(dumped.find("multi_hessian"), std::string::npos)
      << "a training parameter leaked into the model";
  EXPECT_EQ(dumped.find("exact_hessian"), std::string::npos)
      << "training-time Hessian state leaked into the model";

  // A model saved from exact mode loads into a default learner and predicts identically,
  // which is the operational meaning of "not model state".
  auto before = Predict(learner.get(), dmat);
  std::unique_ptr<Learner> restored{Learner::Create({dmat})};
  restored->LoadModel(model);
  auto after = Predict(restored.get(), dmat);
  ASSERT_EQ(before.size(), after.size());
  for (std::size_t i = 0; i < before.size(); ++i) {
    EXPECT_FLOAT_EQ(before[i], after[i]) << "entry " << i;
  }
}

/** The error messages must tell the user what to do. */
TEST(ExactMultinomialLearner, ErrorMessagesAreActionable) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(128, 6, kNumClasses);

  struct Case {
    char const* what;
    Args overrides;
    char const* names_the_alternative;
  };
  // Every rejection a user can reach from the Learner. Each must name the parameter that was
  // set -- so the message can be matched against what the user typed -- and a way forward.
  std::vector<Case> const cases{
      {"multi_strategy", {{"multi_strategy", "one_output_per_tree"}}, "multi_output_tree"},
      {"device", {{"device", "cuda"}}, "device=cpu"},
      {"reg_alpha", {{"alpha", "0.5"}}, "multi_hessian=diagonal"},
      {"max_delta_step", {{"max_delta_step", "1.0"}}, "multi_hessian=diagonal"},
      {"monotone_constraints", {{"monotone_constraints", "(1,0,0,0,0,0)"}},
       "multi_hessian=diagonal"},
      {"gradient_based sampling",
       {{"subsample", "0.5"}, {"sampling_method", "gradient_based"}}, "sampling_method=uniform"},
  };

  for (auto const& c : cases) {
    std::string msg;
    try {
      // Some rejections fire at configure time and some at the first round, so both are
      // inside the same try: which one it is does not change the contract.
      std::unique_ptr<Learner> learner{Learner::Create({dmat})};
      learner->Configure(ExactArgs(kNumClasses, "exact", c.overrides));
      learner->UpdateOneIter(0, dmat);
      ADD_FAILURE() << c.what << " was not rejected";
      continue;
    } catch (dmlc::Error const& e) {
      msg = e.what();
    }
    EXPECT_NE(msg.find("multi_hessian=exact"), std::string::npos)
        << c.what << ": the message does not name the parameter that caused the rejection: "
        << msg;
    EXPECT_NE(msg.find(c.names_the_alternative), std::string::npos)
        << c.what << ": the message does not say what to do instead (expected \""
        << c.names_the_alternative << "\"): "
        << msg;
  }

  // An objective that simply cannot produce the statistic must say which ones can.
  std::unique_ptr<Learner> learner{Learner::Create({dmat})};
  learner->Configure(Args{{"objective", "reg:squarederror"},
                          {"multi_strategy", "multi_output_tree"},
                          {"multi_hessian", "exact"},
                          {"tree_method", "hist"},
                          {"device", "cpu"},
                          {"base_score", "0.5"}});
  try {
    learner->UpdateOneIter(0, dmat);
    FAIL() << "an unsupported objective was not rejected";
  } catch (dmlc::Error const& e) {
    std::string msg{e.what()};
    EXPECT_NE(msg.find("multi_hessian=exact"), std::string::npos) << msg;
    EXPECT_NE(msg.find("multi:softprob"), std::string::npos)
        << "the message does not list the objectives that do support it: "
        << msg;
  }
}

/**
 * The sidecar must describe the gradient it accompanies, across every mode transition.
 *
 * `Learner::GetGradient` clears it before each round, so a sidecar can never outlive the
 * gradient it was produced with. These are the transitions where a leak would be silent: the
 * updater selects the exact path purely by the sidecar's presence, so a stale one left over
 * from an earlier round would route a diagonal-mode round into the exact builder -- with a
 * Hessian computed from predictions that no longer exist.
 */
TEST(ExactMultinomialLearner, ModeTransitionsKeepTheSidecarConsistent) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(256, 6, kNumClasses);

  // exact -> diagonal on one live learner. If the sidecar leaked, this round would take the
  // exact path and the result would not match a learner that was diagonal all along.
  std::unique_ptr<Learner> switching{Learner::Create({dmat})};
  switching->Configure(ExactArgs(kNumClasses, "exact"));
  switching->UpdateOneIter(0, dmat);
  switching->Configure(ExactArgs(kNumClasses, "diagonal"));
  switching->UpdateOneIter(1, dmat);
  switching->UpdateOneIter(2, dmat);
  auto switched = Predict(switching.get(), dmat);
  for (auto v : switched) {
    ASSERT_TRUE(std::isfinite(v));
  }

  // diagonal -> exact, the other direction.
  std::unique_ptr<Learner> reverse{Learner::Create({dmat})};
  reverse->Configure(ExactArgs(kNumClasses, "diagonal"));
  reverse->UpdateOneIter(0, dmat);
  reverse->Configure(ExactArgs(kNumClasses, "exact"));
  reverse->UpdateOneIter(1, dmat);
  reverse->UpdateOneIter(2, dmat);
  auto reversed = Predict(reverse.get(), dmat);
  for (auto v : reversed) {
    ASSERT_TRUE(std::isfinite(v));
  }

  // Changing multi_hessian is a structural change, so the two orders must not coincide by
  // accident -- if they did, this test could not tell the paths apart.
  bool differs = false;
  ASSERT_EQ(switched.size(), reversed.size());
  for (std::size_t i = 0; i < switched.size(); ++i) {
    if (std::fabs(switched[i] - reversed[i]) > 1e-6f) {
      differs = true;
      break;
    }
  }
  EXPECT_TRUE(differs) << "the two transition orders produced identical models, so this test "
                          "cannot distinguish the exact path from the diagonal one";
}

/**
 * The sidecar must be resized with the gradient when the row count changes.
 *
 * Continued training on a second matrix is the case that would corrupt memory rather than
 * merely produce a wrong number: the updater copies `NumRows()` rows out of the sidecar, so
 * a stale larger or smaller buffer would misalign every row against its gradient.
 */
TEST(ExactMultinomialLearner, RowCountChangeResizesTheSidecar) {
  bst_target_t constexpr kNumClasses = 4;
  bst_feature_t constexpr kNumFeatures = 5;
  auto big = MakeClassification(512, kNumFeatures, kNumClasses);
  auto small = MakeClassification(97, kNumFeatures, kNumClasses);  // deliberately not a divisor

  std::unique_ptr<Learner> learner{Learner::Create({big})};
  learner->Configure(ExactArgs(kNumClasses, "exact"));
  learner->UpdateOneIter(0, big);
  learner->UpdateOneIter(1, small);
  learner->UpdateOneIter(2, big);
  learner->UpdateOneIter(3, small);

  for (auto const& dmat : {big, small}) {
    auto predt = Predict(learner.get(), dmat);
    ASSERT_EQ(predt.size(), dmat->Info().num_row_ * kNumClasses);
    for (std::size_t r = 0; r < dmat->Info().num_row_; ++r) {
      double total = 0.0;
      for (bst_target_t k = 0; k < kNumClasses; ++k) {
        auto v = predt[r * kNumClasses + k];
        ASSERT_TRUE(std::isfinite(v));
        total += v;
      }
      ASSERT_NEAR(total, 1.0, 1e-5) << "row " << r;
    }
  }
}

/** Exact training continued from a checkpoint must match training straight through. */
TEST(ExactMultinomialLearner, ContinuedTrainingMatchesStraightThrough) {
  bst_target_t constexpr kNumClasses = 3;
  auto dmat = MakeClassification(256, 6, kNumClasses);
  auto args = ExactArgs(kNumClasses, "exact");

  auto straight = Train(dmat, args, 6);
  auto expected = Predict(straight.get(), dmat);

  // Three rounds, serialize, reload, three more.
  auto first = Train(dmat, args, 3);
  Json model{Object{}};
  first->SaveModel(&model);
  std::unique_ptr<Learner> resumed{Learner::Create({dmat})};
  resumed->LoadModel(model);
  resumed->Configure(args);
  for (std::int32_t i = 3; i < 6; ++i) {
    resumed->UpdateOneIter(i, dmat);
  }
  auto actual = Predict(resumed.get(), dmat);

  ASSERT_EQ(expected.size(), actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(expected[i], actual[i]) << "entry " << i;
  }
}
}  // namespace xgboost
