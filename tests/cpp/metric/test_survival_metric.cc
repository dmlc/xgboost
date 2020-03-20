/*!
 * Copyright (c) by Contributors 2020
 */
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

#include "xgboost/metric.h"
#include "xgboost/logging.h"
#include "../helpers.h"
#include "../../../src/common/survival_util.h"

namespace xgboost {
namespace common {

/**
 * Reference values obtained from
 * https://github.com/avinashbarnwal/GSOC-2019/blob/master/AFT/R/combined_assignment.R
 **/

TEST(Metric, AFTNegLogLik) {
  auto lparam = CreateEmptyGenericParam(-1);  // currently AFT metric is CPU only

  /**
   * Test aggregate output from the AFT metric over a small test data set.
   * This is unlike AFTLoss.* tests, which verify metric values over individual data points.
   **/
  MetaInfo info;
  info.num_row_ = 4;
  info.labels_lower_bound_.HostVector()
    = { 100.0f, -std::numeric_limits<bst_float>::infinity(), 60.0f, 16.0f };
  info.labels_upper_bound_.HostVector()
    = { 100.0f, 20.0f, std::numeric_limits<bst_float>::infinity(), 200.0f };
  info.weights_.HostVector() = std::vector<bst_float>();
  HostDeviceVector<bst_float> preds(4, std::log(64));

  struct TestCase {
    std::string dist_type;
    bst_float reference_value;
  };
  for (const auto& test_case : std::vector<TestCase>{ {"normal", 2.1508f}, {"logistic", 2.1804f},
                                                      {"extreme", 2.0706f} }) {
    std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &lparam));
    metric->Configure({ {"aft_loss_distribution", test_case.dist_type},
                        {"aft_loss_distribution_scale", "1.0"} });
    EXPECT_NEAR(metric->Eval(preds, info, false), test_case.reference_value, 1e-4);
  }
}

// Test configuration of AFT metric
TEST(AFTNegLogLikMetric, Configuration) {
  auto lparam = CreateEmptyGenericParam(-1);  // currently AFT metric is CPU only
  std::unique_ptr<Metric> metric(Metric::Create("aft-nloglik", &lparam));
  metric->Configure({{"aft_loss_distribution", "normal"}, {"aft_loss_distribution_scale", "10"}});

  // Configuration round-trip test
  Json j_obj{ Object() };
  metric->SaveConfig(&j_obj);
  auto aft_param_json = j_obj["aft_loss_param"];
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution"]), "normal");
  EXPECT_EQ(get<String>(aft_param_json["aft_loss_distribution_scale"]), "10");
}

/**
 * AFTLoss.* tests verify metric values over individual data points.
 **/

// Generate prediction value ranging from 2**1 to 2**15, using grid points in log scale
// Then check prediction against the reference values
static inline void CheckLossOverGridPoints(
                      double true_label_lower_bound,
                      double true_label_upper_bound,
                      ProbabilityDistributionType dist_type,
                      const std::vector<double>& reference_values) {
  const int num_point = 20;
  const double log_y_low = 1.0;
  const double log_y_high = 15.0;
  std::unique_ptr<AFTLoss> loss(new AFTLoss(dist_type));
  CHECK_EQ(num_point, reference_values.size());
  for (int i = 0; i < num_point; ++i) {
    const double y_pred
      = std::pow(2.0, i * (log_y_high - log_y_low) / (num_point - 1) + log_y_low);
    const double loss_val
      = loss->Loss(true_label_lower_bound, true_label_upper_bound, std::log(y_pred), 1.0);
    EXPECT_NEAR(loss_val, reference_values[i], 1e-4);
  }
}

TEST(AFTLoss, Uncensored) {
  // Given label 100, compute the AFT loss for various prediction values
  const double true_label_lower_bound = 100.0;
  const double true_label_upper_bound = true_label_lower_bound;

  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kNormal,
    { 13.1761, 11.3085, 9.7017, 8.3558, 7.2708, 6.4466, 5.8833, 5.5808, 5.5392, 5.7585, 6.2386,
      6.9795, 7.9813, 9.2440, 10.7675, 12.5519, 14.5971, 16.9032, 19.4702, 22.2980 });
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kLogistic,
    { 8.5568, 8.0720, 7.6038, 7.1620, 6.7612, 6.4211, 6.1659, 6.0197, 5.9990, 6.1064, 6.3293,
      6.6450, 7.0289, 7.4594, 7.9205, 8.4008, 8.8930, 9.3926, 9.8966, 10.4033 });
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kExtreme,
    { 27.6310, 27.6310, 19.7177, 13.0281, 9.2183, 7.1365, 6.0916, 5.6688, 5.6195, 5.7941, 6.1031,
      6.4929, 6.9310, 7.3981, 7.8827, 8.3778, 8.8791, 9.3842, 9.8916, 10.40033 });
}

TEST(AFTLoss, LeftCensored) {
  // Given label (-inf, 20], compute the AFT loss for various prediction values
  const double true_label_lower_bound = -std::numeric_limits<double>::infinity();
  const double true_label_upper_bound = 20.0;

  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kNormal,
    { 0.0107, 0.0373, 0.1054, 0.2492, 0.5068, 0.9141, 1.5003, 2.2869, 3.2897, 4.5196, 5.9846,
      7.6902, 9.6405, 11.8385, 14.2867, 16.9867, 19.9399, 23.1475, 26.6103, 27.6310 });
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kLogistic,
    { 0.0953, 0.1541, 0.2451, 0.3804, 0.5717, 0.8266, 1.1449, 1.5195, 1.9387, 2.3902, 2.8636,
      3.3512, 3.8479, 4.3500, 4.8556, 5.3632, 5.8721, 6.3817, 6.8918, 7.4021 });
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kExtreme,
    { 0.0000, 0.0025, 0.0277, 0.1225, 0.3195, 0.6150, 0.9862, 1.4094, 1.8662, 2.3441, 2.8349,
      3.3337, 3.8372, 4.3436, 4.8517, 5.3609, 5.8707, 6.3808, 6.8912, 7.4018 });
}

TEST(AFTLoss, RightCensored) {
  // Given label [60, +inf), compute the AFT loss for various prediction values
  const double true_label_lower_bound = 60.0;
  const double true_label_upper_bound = std::numeric_limits<double>::infinity();

  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kNormal,
    { 8.0000, 6.2537, 4.7487, 3.4798, 2.4396, 1.6177, 0.9993, 0.5638, 0.2834, 0.1232, 0.0450,
      0.0134, 0.0032, 0.0006, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000 });
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kLogistic,
    { 3.4340, 2.9445, 2.4683, 2.0125, 1.5871, 1.2041, 0.8756, 0.6099, 0.4083, 0.2643, 0.1668,
      0.1034, 0.0633, 0.0385, 0.0233, 0.0140, 0.0084, 0.0051, 0.0030, 0.0018 });
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kExtreme,
    { 27.6310, 18.0015, 10.8018, 6.4817, 3.8893, 2.3338, 1.4004, 0.8403, 0.5042, 0.3026, 0.1816,
      0.1089, 0.0654, 0.0392, 0.0235, 0.0141, 0.0085, 0.0051, 0.0031, 0.0018 });
}

TEST(AFTLoss, IntervalCensored) {
  // Given label [16, 200], compute the AFT loss for various prediction values
  const double true_label_lower_bound = 16.0;
  const double true_label_upper_bound = 200.0;
  
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kNormal,
    { 3.9746, 2.8415, 1.9319, 1.2342, 0.7335, 0.4121, 0.2536, 0.2470, 0.3919, 0.6982, 1.1825,
      1.8622, 2.7526, 3.8656, 5.2102, 6.7928, 8.6183, 10.6901, 13.0108, 15.5826 });
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kLogistic,
    { 2.2906, 1.8578, 1.4667, 1.1324, 0.8692, 0.6882, 0.5948, 0.5909, 0.6764, 0.8499, 1.1061,
      1.4348, 1.8215, 2.2511, 2.7104, 3.1891, 3.6802, 4.1790, 4.6825, 5.1888 });
  CheckLossOverGridPoints(true_label_lower_bound, true_label_upper_bound,
    ProbabilityDistributionType::kExtreme,
    { 8.0000, 4.8004, 2.8805, 1.7284, 1.0372, 0.6231, 0.3872, 0.3031, 0.3740, 0.5839, 0.8995,
      1.2878, 1.7231, 2.1878, 2.6707, 3.1647, 3.6653, 4.1699, 4.6770, 5.1856 });
}

}  // namespace common
}  // namespace xgboost
