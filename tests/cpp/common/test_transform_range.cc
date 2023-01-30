/**
 * Copyright 2018-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/span.h>
#include <xgboost/host_device_vector.h>

#include <vector>

#include "../../../src/common/transform.h"
#include "../helpers.h"

#if defined(__CUDACC__)

#define TRANSFORM_GPU 0

#else

#define TRANSFORM_GPU -1

#endif

namespace xgboost {
namespace common {

template <typename T>
struct TestTransformRange {
  void XGBOOST_DEVICE operator()(size_t _idx,
                                 Span<bst_float> _out, Span<const bst_float> _in) {
    _out[_idx] = _in[_idx];
  }
};

TEST(Transform, DeclareUnifiedTest(Basic)) {
  const size_t size {256};
  std::vector<bst_float> h_in(size);
  std::vector<bst_float> h_out(size);
  std::iota(h_in.begin(), h_in.end(), 0);
  std::vector<bst_float> h_sol(size);
  std::iota(h_sol.begin(), h_sol.end(), 0);

  const HostDeviceVector<bst_float> in_vec{h_in, TRANSFORM_GPU};
  HostDeviceVector<bst_float> out_vec{h_out, TRANSFORM_GPU};
  out_vec.Fill(0);

  Transform<>::Init(TestTransformRange<bst_float>{},
                    Range{0, static_cast<Range::DifferenceType>(size)}, AllThreadsForTest(),
                    TRANSFORM_GPU)
      .Eval(&out_vec, &in_vec);
  std::vector<bst_float> res = out_vec.HostVector();

  ASSERT_TRUE(std::equal(h_sol.begin(), h_sol.end(), res.begin()));
}

#if !defined(__CUDACC__)
TEST(TransformDeathTest, Exception) {
  size_t const kSize {16};
  std::vector<bst_float> h_in(kSize);
  const HostDeviceVector<bst_float> in_vec{h_in, -1};
  EXPECT_DEATH(
      {
        Transform<>::Init([](size_t idx, common::Span<float const> _in) { _in[idx + 1]; },
                          Range(0, static_cast<Range::DifferenceType>(kSize)), AllThreadsForTest(),
                          -1)
            .Eval(&in_vec);
      },
      "");
}
#endif

} // namespace common
} // namespace xgboost
