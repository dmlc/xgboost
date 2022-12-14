/*!
 * Copyright 2022 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "../../../src/common/stats.cuh"
#include "../../../src/common/stats.h"
#include "xgboost/base.h"
#include "xgboost/context.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace common {
namespace {
class StatsGPU : public ::testing::Test {
 private:
  linalg::Tensor<float, 1> arr_{
      {1.f, 2.f, 3.f, 4.f, 5.f,
       2.f, 4.f, 5.f, 3.f, 1.f},
      {10}, 0};
  linalg::Tensor<size_t, 1> indptr_{{0, 5, 10}, {3}, 0};
  HostDeviceVector<float> resutls_;
  using TestSet = std::vector<std::pair<float, float>>;
  Context ctx_;

  void Check(float expected) {
    auto const& h_results = resutls_.HostVector();
    ASSERT_EQ(h_results.size(), indptr_.Size() - 1);
    ASSERT_EQ(h_results.front(), expected);
    EXPECT_EQ(h_results.back(), expected);
  }

 public:
  void SetUp() override { ctx_.gpu_id = 0; }
  void Weighted() {
    auto d_arr = arr_.View(0);
    auto d_key = indptr_.View(0);

    auto key_it = dh::MakeTransformIterator<size_t>(thrust::make_counting_iterator(0ul),
                                                    [=] __device__(size_t i) { return d_key(i); });
    auto val_it = dh::MakeTransformIterator<float>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) { return d_arr(i); });
    linalg::Tensor<float, 1> weights{{10}, 0};
    linalg::ElementWiseTransformDevice(weights.View(0),
                                       [=] XGBOOST_DEVICE(size_t, float) { return 1.0; });
    auto w_it = weights.Data()->ConstDevicePointer();
    for (auto const& pair : TestSet{{0.0f, 1.0f}, {0.5f, 3.0f}, {1.0f, 5.0f}}) {
      SegmentedWeightedQuantile(&ctx_, pair.first, key_it, key_it + indptr_.Size(), val_it,
                                val_it + arr_.Size(), w_it, w_it + weights.Size(), &resutls_);
      this->Check(pair.second);
    }
  }

  void NonWeighted() {
    auto d_arr = arr_.View(0);
    auto d_key = indptr_.View(0);

    auto key_it = dh::MakeTransformIterator<size_t>(thrust::make_counting_iterator(0ul),
                                                    [=] __device__(size_t i) { return d_key(i); });
    auto val_it = dh::MakeTransformIterator<float>(
        thrust::make_counting_iterator(0ul), [=] XGBOOST_DEVICE(size_t i) { return d_arr(i); });

    for (auto const& pair : TestSet{{0.0f, 1.0f}, {0.5f, 3.0f}, {1.0f, 5.0f}}) {
      SegmentedQuantile(&ctx_, pair.first, key_it, key_it + indptr_.Size(), val_it,
                        val_it + arr_.Size(), &resutls_);
      this->Check(pair.second);
    }
  }
};
}  // anonymous namespace

TEST_F(StatsGPU, Quantile) { this->NonWeighted(); }
TEST_F(StatsGPU, WeightedQuantile) { this->Weighted(); }
}  // namespace common
}  // namespace xgboost
