/**
 * Copyright 2018-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#include <xgboost/host_device_vector.h>
#include <xgboost/span.h>

#include <numeric>  // for iota
#include <vector>

#include "../../../src/common/transform.h"
#include "../helpers.h"

namespace xgboost::common {
namespace {
constexpr DeviceOrd TransformDevice() {
#if defined(__CUDACC__)
  return DeviceOrd::CUDA(0);
#else
  return DeviceOrd::CPU();
#endif
}
}  // namespace

template <typename T>
struct TestTransformRange {
  template <class kBoolConst>
  void XGBOOST_DEVICE operator()(std::size_t _idx, kBoolConst has_fp64_support, Span<float> _out, Span<const float> _in) {
    _out[_idx] = _in[_idx];
  }
};

TEST(Transform, DeclareUnifiedTest(Basic)) {
  const size_t size{256};
  std::vector<float> h_in(size);
  std::vector<float> h_out(size);
  std::iota(h_in.begin(), h_in.end(), 0);
  std::vector<float> h_sol(size);
  std::iota(h_sol.begin(), h_sol.end(), 0);

  auto device = TransformDevice();
  HostDeviceVector<float> const in_vec{h_in, device};
  HostDeviceVector<float> out_vec{h_out, device};
  out_vec.Fill(0);

  Transform<>::Init(TestTransformRange<float>{},
                    Range{0, static_cast<Range::DifferenceType>(size)}, AllThreadsForTest(),
                    TransformDevice())
      .Eval(&out_vec, &in_vec);
  std::vector<float> res = out_vec.HostVector();

  ASSERT_TRUE(std::equal(h_sol.begin(), h_sol.end(), res.begin()));
}

#if !defined(__CUDACC__)
TEST(TransformDeathTest, Exception) {
  size_t const kSize{16};
  std::vector<float> h_in(kSize);
  const HostDeviceVector<float> in_vec{h_in, DeviceOrd::CPU()};
  EXPECT_DEATH(
      {
        Transform<>::Init([](size_t idx, auto has_fp64_support, common::Span<float const> _in) { _in[idx + 1]; },
                          Range(0, static_cast<Range::DifferenceType>(kSize)), AllThreadsForTest(),
                          DeviceOrd::CPU())
            .Eval(&in_vec);
      },
      "");
}
#endif
}  // namespace xgboost::common
