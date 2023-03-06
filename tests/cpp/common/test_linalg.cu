/**
 * Copyright 2021-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/common/linalg_op.cuh"
#include "xgboost/context.h"
#include "xgboost/linalg.h"

namespace xgboost::linalg {
namespace {
void TestElementWiseKernel() {
  Tensor<float, 3> l{{2, 3, 4}, 0};
  {
    /**
     * Non-contiguous
     */
    // GPU view
    auto t = l.View(0).Slice(linalg::All(), 1, linalg::All());
    ASSERT_FALSE(t.CContiguous());
    ElementWiseTransformDevice(t, [] __device__(size_t i, float) { return i; });
    // CPU view
    t = l.View(Context::kCpuId).Slice(linalg::All(), 1, linalg::All());
    size_t k = 0;
    for (size_t i = 0; i < l.Shape(0); ++i) {
      for (size_t j = 0; j < l.Shape(2); ++j) {
        ASSERT_EQ(k++, t(i, j));
      }
    }

    t = l.View(0).Slice(linalg::All(), 1, linalg::All());
    ElementWiseKernelDevice(t, [] XGBOOST_DEVICE(size_t i, float v) { SPAN_CHECK(v == i); });
  }

  {
    /**
     * Contiguous
     */
    auto t = l.View(0);
    ElementWiseTransformDevice(t, [] XGBOOST_DEVICE(size_t i, float) { return i; });
    ASSERT_TRUE(t.CContiguous());
    // CPU view
    t = l.View(Context::kCpuId);

    size_t ind = 0;
    for (size_t i = 0; i < l.Shape(0); ++i) {
      for (size_t j = 0; j < l.Shape(1); ++j) {
        for (size_t k = 0; k < l.Shape(2); ++k) {
          ASSERT_EQ(ind++, t(i, j, k));
        }
      }
    }
  }
}

void TestSlice() {
  Context ctx;
  ctx.gpu_id = 1;
  thrust::device_vector<double> data(2 * 3 * 4);
  auto t = MakeTensorView(&ctx, dh::ToSpan(data), 2, 3, 4);
  dh::LaunchN(1, [=] __device__(size_t) {
    auto s = t.Slice(linalg::All(), linalg::Range(0, 3), linalg::Range(0, 4));
    auto all = t.Slice(linalg::All(), linalg::All(), linalg::All());
    static_assert(decltype(s)::kDimension == 3);
    for (size_t i = 0; i < s.Shape(0); ++i) {
      for (size_t j = 0; j < s.Shape(1); ++j) {
        for (size_t k = 0; k < s.Shape(2); ++k) {
          SPAN_CHECK(s(i, j, k) == all(i, j, k));
        }
      }
    }
  });
}
}  // anonymous namespace

TEST(Linalg, GPUElementWise) { TestElementWiseKernel(); }

TEST(Linalg, GPUTensorView) { TestSlice(); }
}  // namespace xgboost::linalg
