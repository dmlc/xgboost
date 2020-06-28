
/*!
 * Copyright 2017 XGBoost contributors
 */
#include <thrust/device_vector.h>
#include <xgboost/base.h>
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/quantile.h"
#include "../helpers.h"
#include "gtest/gtest.h"

TEST(SumReduce, Test) {
  thrust::device_vector<float> data(100, 1.0f);
  auto sum = dh::SumReduction(data.data().get(), data.size());
  ASSERT_NEAR(sum, 100.0f, 1e-5);
}

void TestAtomicSizeT() {
  size_t constexpr kThreads = 235;
  dh::device_vector<size_t> out(1, 0);
  auto d_out = dh::ToSpan(out);
  dh::LaunchN(0, kThreads, [=]__device__(size_t idx){
      atomicAdd(&d_out[0], static_cast<size_t>(1));
  });
  ASSERT_EQ(out[0], kThreads);
}

TEST(AtomicAdd, SizeT) {
  TestAtomicSizeT();
}

TEST(SegmentedUnique, Basic) {
  std::vector<float> values{0.1f, 0.2f, 0.3f, 0.62448811531066895f, 0.62448811531066895f, 0.4f};
  std::vector<size_t> segments{0, 3, 6};

  thrust::device_vector<float> d_values(values);
  thrust::device_vector<xgboost::bst_feature_t> d_segments{segments};

  thrust::device_vector<xgboost::bst_feature_t> d_segs_out(d_segments.size());
  thrust::device_vector<float> d_vals_out(d_values.size());

  size_t n_uniques = dh::SegmentedUnique(
      d_segments.data().get(), d_segments.data().get() + d_segments.size(),
      d_values.data().get(), d_values.data().get() + d_values.size(),
      d_segs_out.data().get(), d_vals_out.data().get(),
      thrust::equal_to<float>{});
  CHECK_EQ(n_uniques, 5);

  std::vector<float> values_sol{0.1f, 0.2f, 0.3f, 0.62448811531066895f, 0.4f};
  for (auto i = 0 ; i < values_sol.size(); i ++) {
    ASSERT_EQ(d_vals_out[i], values_sol[i]);
  }

  std::vector<xgboost::bst_feature_t> segments_sol{0, 3, 5};
  for (size_t i = 0; i < d_segments.size(); ++i) {
    ASSERT_EQ(segments_sol[i], d_segs_out[i]);
  }

  d_segments[1] = 4;
  d_segments[2] = 6;
  n_uniques = dh::SegmentedUnique(
      d_segments.data().get(), d_segments.data().get() + d_segments.size(),
      d_values.data().get(), d_values.data().get() + d_values.size(),
      d_segs_out.data().get(), d_vals_out.data().get(),
      thrust::equal_to<float>{});
  ASSERT_EQ(n_uniques, values.size());
  for (auto i = 0 ; i < values.size(); i ++) {
    ASSERT_EQ(d_vals_out[i], values[i]);
  }
}

namespace {
using SketchEntry = xgboost::common::WQSummary<float, float>::Entry;
struct SketchUnique {
  bool __device__ operator()(SketchEntry const& a, SketchEntry const& b) const {
    return a.value - b.value == 0;
  }
};
struct IsSorted {
  bool __device__ operator()(SketchEntry const& a, SketchEntry const& b) const {
    return a.value < b.value;
  }
};
}  // namespace

namespace xgboost {
namespace common {

void TestSegmentedUniqueRegression(std::vector<SketchEntry> values, size_t n_duplicated) {
  std::vector<bst_feature_t> segments{0, static_cast<bst_feature_t>(values.size())};

  thrust::device_vector<SketchEntry> d_values(values);
  thrust::device_vector<bst_feature_t> d_segments(segments);
  thrust::device_vector<bst_feature_t> d_segments_out(segments.size());

  size_t n_uniques = dh::SegmentedUnique(
      d_segments.data().get(), d_segments.data().get() + d_segments.size(), d_values.data().get(),
      d_values.data().get() + d_values.size(), d_segments_out.data().get(), d_values.data().get(),
      SketchUnique{});
  ASSERT_EQ(n_uniques, values.size() - n_duplicated);
  ASSERT_TRUE(thrust::is_sorted(thrust::device, d_values.begin(),
                                d_values.begin() + n_uniques, IsSorted{}));
  ASSERT_EQ(segments.at(0), d_segments_out[0]);
  ASSERT_EQ(segments.at(1), d_segments_out[1] + n_duplicated);
}


TEST(SegmentedUnique, Regression) {
  {
    std::vector<SketchEntry> values{{3149, 3150, 1, 0.62392902374267578},
                                    {3151, 3152, 1, 0.62418866157531738},
                                    {3152, 3153, 1, 0.62419462203979492},
                                    {3153, 3154, 1, 0.62431186437606812},
                                    {3154, 3155, 1, 0.6244881153106689453125},
                                    {3155, 3156, 1, 0.6244881153106689453125},
                                    {3155, 3156, 1, 0.6244881153106689453125},
                                    {3155, 3156, 1, 0.6244881153106689453125},
                                    {3157, 3158, 1, 0.62552797794342041},
                                    {3158, 3159, 1, 0.6256556510925293},
                                    {3159, 3160, 1, 0.62571090459823608},
                                    {3160, 3161, 1, 0.62577134370803833}};
    TestSegmentedUniqueRegression(values, 3);
  }
  {
    std::vector<SketchEntry> values{{3149, 3150, 1, 0.62392902374267578},
                                    {3151, 3152, 1, 0.62418866157531738},
                                    {3152, 3153, 1, 0.62419462203979492},
                                    {3153, 3154, 1, 0.62431186437606812},
                                    {3154, 3155, 1, 0.6244881153106689453125},
                                    {3157, 3158, 1, 0.62552797794342041},
                                    {3158, 3159, 1, 0.6256556510925293},
                                    {3159, 3160, 1, 0.62571090459823608},
                                    {3160, 3161, 1, 0.62577134370803833}};
    TestSegmentedUniqueRegression(values, 0);
  }
  {
    std::vector<SketchEntry> values;
    TestSegmentedUniqueRegression(values, 0);
  }
}
}  // namespace common
}  // namespace xgboost