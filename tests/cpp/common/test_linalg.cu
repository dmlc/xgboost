/*!
 * Copyright 2021 by XGBoost Contributors
 */
#include <gtest/gtest.h>

#include "../../../src/common/linalg_op.cuh"
#include "xgboost/generic_parameters.h"
#include "xgboost/linalg.h"

namespace xgboost {
namespace linalg {
namespace {
void TestElementWiseKernel() {
  Tensor<float, 3> l{{2, 3, 4}, 0};
  {
    /**
     * Non-contiguous
     */
    // GPU view
    auto t = l.View(0).Slice(linalg::All(), 1, linalg::All());
    ASSERT_FALSE(t.Contiguous());
    ElementWiseKernelDevice(t, [] __device__(size_t i, float) { return i; });
    // CPU view
    t = l.View(GenericParameter::kCpuId).Slice(linalg::All(), 1, linalg::All());
    size_t k = 0;
    for (size_t i = 0; i < l.Shape(0); ++i) {
      for (size_t j = 0; j < l.Shape(2); ++j) {
        ASSERT_EQ(k++, t(i, j));
      }
    }

    t = l.View(0).Slice(linalg::All(), 1, linalg::All());
    ElementWiseKernelDevice(t, [] __device__(size_t i, float v) {
      SPAN_CHECK(v == i);
      return v;
    });
  }

  {
    /**
     * Contiguous
     */
    auto t = l.View(0);
    ElementWiseKernelDevice(t, [] __device__(size_t i, float) { return i; });
    ASSERT_TRUE(t.Contiguous());
    // CPU view
    t = l.View(GenericParameter::kCpuId);

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
}  // anonymous namespace
TEST(Linalg, GPUElementWise) { TestElementWiseKernel(); }
}  // namespace linalg
}  // namespace xgboost
