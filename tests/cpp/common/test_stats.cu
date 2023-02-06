/**
 * Copyright 2022-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <cstddef>                            // std::size_t
#include <utility>                            // std::pair
#include <vector>                             // std::vector

#include "../../../src/common/linalg_op.cuh"  // ElementWiseTransformDevice
#include "../../../src/common/stats.cuh"
#include "xgboost/base.h"                     // XGBOOST_DEVICE
#include "xgboost/context.h"                  // Context
#include "xgboost/host_device_vector.h"       // HostDeviceVector
#include "xgboost/linalg.h"                   // Tensor

namespace xgboost {
namespace common {
namespace {
class StatsGPU : public ::testing::Test {
 private:
  linalg::Tensor<float, 1> arr_{{1.f, 2.f, 3.f, 4.f, 5.f, 2.f, 4.f, 5.f, 3.f, 1.f}, {10}, 0};
  linalg::Tensor<std::size_t, 1> indptr_{{0, 5, 10}, {3}, 0};
  HostDeviceVector<float> results_;
  using TestSet = std::vector<std::pair<float, float>>;
  Context ctx_;

  void Check(float expected) {
    auto const& h_results = results_.HostVector();
    ASSERT_EQ(h_results.size(), indptr_.Size() - 1);
    ASSERT_EQ(h_results.front(), expected);
    ASSERT_EQ(h_results.back(), expected);
  }

 public:
  void SetUp() override { ctx_.gpu_id = 0; }

  void WeightedMulti() {
    // data for one segment
    std::vector<float> seg{1.f, 2.f, 3.f, 4.f, 5.f};
    auto seg_size = seg.size();

    // 3 segments
    std::vector<float> data;
    data.insert(data.cend(), seg.begin(), seg.end());
    data.insert(data.cend(), seg.begin(), seg.end());
    data.insert(data.cend(), seg.begin(), seg.end());
    linalg::Tensor<float, 1> arr{data.cbegin(), data.cend(), {data.size()}, 0};
    auto d_arr = arr.View(0);

    auto key_it = dh::MakeTransformIterator<std::size_t>(
        thrust::make_counting_iterator(0ul),
        [=] XGBOOST_DEVICE(std::size_t i) { return i * seg_size; });
    auto val_it =
        dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                         [=] XGBOOST_DEVICE(std::size_t i) { return d_arr(i); });

    // one alpha for each segment
    HostDeviceVector<float> alphas{0.0f, 0.5f, 1.0f};
    alphas.SetDevice(0);
    auto d_alphas = alphas.ConstDeviceSpan();
    auto w_it = thrust::make_constant_iterator(0.1f);
    SegmentedWeightedQuantile(&ctx_, d_alphas.data(), key_it, key_it + d_alphas.size() + 1, val_it,
                              val_it + d_arr.Size(), w_it, w_it + d_arr.Size(), &results_);

    auto const& h_results = results_.HostVector();
    ASSERT_EQ(1.0f, h_results[0]);
    ASSERT_EQ(3.0f, h_results[1]);
    ASSERT_EQ(5.0f, h_results[2]);
  }

  void Weighted() {
    auto d_arr = arr_.View(0);
    auto d_key = indptr_.View(0);

    auto key_it = dh::MakeTransformIterator<std::size_t>(
        thrust::make_counting_iterator(0ul),
        [=] XGBOOST_DEVICE(std::size_t i) { return d_key(i); });
    auto val_it =
        dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                         [=] XGBOOST_DEVICE(std::size_t i) { return d_arr(i); });
    linalg::Tensor<float, 1> weights{{10}, 0};
    linalg::ElementWiseTransformDevice(weights.View(0),
                                       [=] XGBOOST_DEVICE(std::size_t, float) { return 1.0; });
    auto w_it = weights.Data()->ConstDevicePointer();
    for (auto const& pair : TestSet{{0.0f, 1.0f}, {0.5f, 3.0f}, {1.0f, 5.0f}}) {
      SegmentedWeightedQuantile(&ctx_, pair.first, key_it, key_it + indptr_.Size(), val_it,
                                val_it + arr_.Size(), w_it, w_it + weights.Size(), &results_);
      this->Check(pair.second);
    }
  }

  void NonWeightedMulti() {
    // data for one segment
    std::vector<float> seg{20.f, 15.f, 50.f, 40.f, 35.f};
    auto seg_size = seg.size();

    // 3 segments
    std::vector<float> data;
    data.insert(data.cend(), seg.begin(), seg.end());
    data.insert(data.cend(), seg.begin(), seg.end());
    data.insert(data.cend(), seg.begin(), seg.end());
    linalg::Tensor<float, 1> arr{data.cbegin(), data.cend(), {data.size()}, 0};
    auto d_arr = arr.View(0);

    auto key_it = dh::MakeTransformIterator<std::size_t>(
        thrust::make_counting_iterator(0ul),
        [=] XGBOOST_DEVICE(std::size_t i) { return i * seg_size; });
    auto val_it =
        dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                         [=] XGBOOST_DEVICE(std::size_t i) { return d_arr(i); });

    // one alpha for each segment
    HostDeviceVector<float> alphas{0.1f, 0.2f, 0.4f};
    alphas.SetDevice(0);
    auto d_alphas = alphas.ConstDeviceSpan();
    SegmentedQuantile(&ctx_, d_alphas.data(), key_it, key_it + d_alphas.size() + 1, val_it,
                      val_it + d_arr.Size(), &results_);

    auto const& h_results = results_.HostVector();
    EXPECT_EQ(15.0f, h_results[0]);
    EXPECT_EQ(16.0f, h_results[1]);
    ASSERT_EQ(26.0f, h_results[2]);
  }

  void NonWeighted() {
    auto d_arr = arr_.View(0);
    auto d_key = indptr_.View(0);

    auto key_it = dh::MakeTransformIterator<std::size_t>(
        thrust::make_counting_iterator(0ul), [=] __device__(std::size_t i) { return d_key(i); });
    auto val_it =
        dh::MakeTransformIterator<float>(thrust::make_counting_iterator(0ul),
                                         [=] XGBOOST_DEVICE(std::size_t i) { return d_arr(i); });

    for (auto const& pair : TestSet{{0.0f, 1.0f}, {0.5f, 3.0f}, {1.0f, 5.0f}}) {
      SegmentedQuantile(&ctx_, pair.first, key_it, key_it + indptr_.Size(), val_it,
                        val_it + arr_.Size(), &results_);
      this->Check(pair.second);
    }
  }
};
}  // anonymous namespace

TEST_F(StatsGPU, Quantile) {
  this->NonWeighted();
  this->NonWeightedMulti();
}

TEST_F(StatsGPU, WeightedQuantile) {
  this->Weighted();
  this->WeightedMulti();
}
}  // namespace common
}  // namespace xgboost
