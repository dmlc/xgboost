/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>
#include <xgboost/linalg.h>  // Tensor,Vector

#include <algorithm>  // for min
#include <thread>     // for thread

#include "../../../src/common/linalg_op.h"  // for begin, end
#include "../../../src/common/stats.h"
#include "../../../src/common/transform_iterator.h"  // common::MakeIndexTransformIter
#include "../collective/test_worker.h"
#include "../helpers.h"

namespace xgboost::common {
TEST(Stats, Quantile) {
  Context ctx;
  {
    linalg::Tensor<float, 1> arr({20.f, 0.f, 15.f, 50.f, 40.f, 0.f, 35.f}, {7}, DeviceOrd::CPU());
    std::vector<size_t> index{0, 2, 3, 4, 6};
    auto h_arr = arr.HostView();
    auto beg = MakeIndexTransformIter([&](size_t i) { return h_arr(index[i]); });
    auto end = beg + index.size();
    auto q = Quantile(&ctx, 0.40f, beg, end);
    ASSERT_EQ(q, 26.0);

    q = Quantile(&ctx, 0.20f, beg, end);
    ASSERT_EQ(q, 16.0);

    q = Quantile(&ctx, 0.10f, beg, end);
    ASSERT_EQ(q, 15.0);
  }

  {
    std::vector<float> vec{1., 2., 3., 4., 5.};
    auto beg = MakeIndexTransformIter([&](size_t i) { return vec[i]; });
    auto end = beg + vec.size();
    auto q = Quantile(&ctx, 0.5f, beg, end);
    ASSERT_EQ(q, 3.);
  }
}

TEST(Stats, WeightedQuantile) {
  Context ctx;
  linalg::Tensor<float, 1> arr({1.f, 2.f, 3.f, 4.f, 5.f}, {5}, DeviceOrd::CPU());
  linalg::Tensor<float, 1> weight({1.f, 1.f, 1.f, 1.f, 1.f}, {5}, DeviceOrd::CPU());

  auto h_arr = arr.HostView();
  auto h_weight = weight.HostView();

  auto beg = MakeIndexTransformIter([&](size_t i) { return h_arr(i); });
  auto end = beg + arr.Size();
  auto w = MakeIndexTransformIter([&](size_t i) { return h_weight(i); });

  auto q = WeightedQuantile(&ctx, 0.50f, beg, end, w);
  ASSERT_EQ(q, 3);

  q = WeightedQuantile(&ctx, 0.0, beg, end, w);
  ASSERT_EQ(q, 1);

  q = WeightedQuantile(&ctx, 1.0, beg, end, w);
  ASSERT_EQ(q, 5);
}

TEST(Stats, Median) {
  Context ctx;

  {
    linalg::Tensor<float, 2> values{{.0f, .0f, 1.f, 2.f}, {4}, DeviceOrd::CPU()};
    HostDeviceVector<float> weights;
    linalg::Tensor<float, 1> out;
    Median(&ctx, values, weights, &out);
    auto m = out(0);
    ASSERT_EQ(m, .5f);

#if defined(XGBOOST_USE_CUDA)
    ctx = ctx.MakeCUDA(0);
    ASSERT_FALSE(ctx.IsCPU());
    Median(&ctx, values, weights, &out);
    m = out(0);
    ASSERT_EQ(m, .5f);
#endif  // defined(XGBOOST_USE_CUDA)
  }

  {
    ctx = ctx.MakeCPU();
    // 4x2 matrix
    linalg::Tensor<float, 2> values{{0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 2.f, 2.f}, {4, 2}, ctx.Device()};
    HostDeviceVector<float> weights;
    linalg::Tensor<float, 1> out;
    Median(&ctx, values, weights, &out);
    ASSERT_EQ(out(0), .5f);
    ASSERT_EQ(out(1), .5f);

#if defined(XGBOOST_USE_CUDA)
    ctx = ctx.MakeCUDA(0);
    Median(&ctx, values, weights, &out);
    ASSERT_EQ(out(0), .5f);
    ASSERT_EQ(out(1), .5f);
#endif  // defined(XGBOOST_USE_CUDA)
  }
}

namespace {
void TestMean(Context const* ctx) {
  std::size_t n{128};
  linalg::Vector<float> data({n}, ctx->Device());
  auto h_v = data.HostView().Values();
  std::iota(h_v.begin(), h_v.end(), .0f);

  auto nf = static_cast<float>(n);
  float mean = nf * (nf - 1) / 2 / n;

  linalg::Vector<float> res{{1}, ctx->Device()};
  Mean(ctx, data, &res);
  auto h_res = res.HostView();
  ASSERT_EQ(h_res.Size(), 1);
  ASSERT_EQ(mean, h_res(0));
}
}  // anonymous namespace

TEST(Stats, Mean) {
  Context ctx;
  TestMean(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(Stats, GpuMean) {
  auto ctx = MakeCUDACtx(0);
  TestMean(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)

namespace {
void TestSampleMean(Context const* ctx) {
  std::size_t m{32}, n{16};
  linalg::Matrix<float> data({m, n}, ctx->Device());
  auto h_data = data.HostView();
  std::iota(linalg::begin(h_data), linalg::end(h_data), .0f);
  linalg::Vector<float> mean;
  SampleMean(ctx, false, data, &mean);
  ASSERT_FLOAT_EQ(mean(0), 248.0f);
  for (std::size_t i = 1; i < mean.Size(); ++i) {
    ASSERT_EQ(mean(i), mean(i - 1) + 1.0f);
  }

  auto device = ctx->Device();
  std::int32_t n_workers =
      device.IsCPU() ? std::min(4u, std::thread::hardware_concurrency()) : curt::AllVisibleGPUs();
#if !defined(XGBOOST_USE_NCCL)
  if (device.IsCUDA()) {
    return;
  }
#endif  // !defined(XGBOOST_USE_NCCL)
  collective::TestDistributedGlobal(n_workers, [m, n, device, n_workers] {
    auto rank = collective::GetRank();
    Context ctx = device.IsCUDA() ? MakeCUDACtx(DistGpuIdx()) : Context{};
    collective::GetWorkerLocalThreads(collective::GetWorldSize(), &ctx);
    linalg::Matrix<float> data({m, n}, ctx.Device());
    auto h_data = data.HostView();
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        h_data(i, j) = i + (m * rank) + j;
      }
    }
    linalg::Vector<float> mean;
    SampleMean(&ctx, false, data, &mean);
    ASSERT_EQ(mean.Size(), n);
    double total = n_workers * m;
    for (std::size_t i = 0; i < n; ++i) {
      ASSERT_EQ(mean(i), (i + total - 1.0 + i) * total / 2.0 / total);
    }
  });
}

void TestWeightedSampleMean(Context const* ctx) {
  std::size_t m{32}, n{16};
  {
    auto data = linalg::Constant(ctx, 1.0f, m, n);
    HostDeviceVector<float> w{m, 0.0f, ctx->Device()};
    auto h_w = w.HostSpan();
    std::iota(h_w.data(), h_w.data() + h_w.size(), 1.0f);
    linalg::Vector<float> mean;
    WeightedSampleMean(ctx, false, data, w, &mean);
    for (auto v : mean.HostView()) {
      ASSERT_FLOAT_EQ(v, 1.0f);
    }
  }
  {
    linalg::Matrix<float> data({m, n}, ctx->Device());
    auto h_data = data.HostView();
    std::iota(linalg::begin(h_data), linalg::end(h_data), .0f);
    HostDeviceVector<float> w{m, 1.0f, ctx->Device()};
    linalg::Vector<float> mean;
    WeightedSampleMean(ctx, false, data, w, &mean);
    ASSERT_FLOAT_EQ(mean(0), 248.0f);
    for (std::size_t i = 1; i < mean.Size(); ++i) {
      ASSERT_EQ(mean(i), mean(i - 1) + 1.0f);
    }
  }

  auto device = ctx->Device();
  std::int32_t n_workers =
      device.IsCPU() ? std::min(4u, std::thread::hardware_concurrency()) : curt::AllVisibleGPUs();
#if !defined(XGBOOST_USE_NCCL)
  if (device.IsCUDA()) {
    return;
  }
#endif  // !defined(XGBOOST_USE_NCCL)
  collective::TestDistributedGlobal(n_workers, [m, n, device, n_workers] {
    auto rank = collective::GetRank();
    Context ctx = device.IsCUDA() ? MakeCUDACtx(DistGpuIdx()) : Context{};
    collective::GetWorkerLocalThreads(collective::GetWorldSize(), &ctx);
    linalg::Matrix<float> data({m, n}, ctx.Device());
    auto h_data = data.HostView();
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        h_data(i, j) = i + (m * rank) + j;
      }
    }
    HostDeviceVector<float> w{m, 1.0f, ctx.Device()};
    linalg::Vector<float> mean;
    WeightedSampleMean(&ctx, false, data, w, &mean);
    ASSERT_EQ(mean.Size(), n);
    double total = n_workers * m;
    for (std::size_t i = 0; i < n; ++i) {
      ASSERT_EQ(mean(i), (i + total - 1.0 + i) * total / 2.0 / total);
    }
  });
}
}  // namespace

TEST(Stats, SampleMean) {
  Context ctx;
  TestSampleMean(&ctx);
}


TEST(Stats, WeightedSampleMean) {
  Context ctx;
  TestWeightedSampleMean(&ctx);
}

#if defined(XGBOOST_USE_CUDA)
TEST(Stats, GpuSampleMean) {
  auto ctx = MakeCUDACtx(0);
  TestSampleMean(&ctx);
}

TEST(Stats, GpuWeightedSampleMean) {
  auto ctx = MakeCUDACtx(0);
  TestWeightedSampleMean(&ctx);
}
#endif  // defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::common
