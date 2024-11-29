/**
 * Copyright 2018-2024, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/base.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/host_device_vector.h>
#pragma GCC diagnostic pop
#include <xgboost/span.h>

#include <numeric>  // for iota
#include <vector>

#include "../../../src/common/transform.h"
#include "../helpers.h"

namespace xgboost::common {

template <typename T>
struct TestTransformRange {
  void operator()(std::size_t _idx, Span<float> _out, Span<const float> _in) {
    _out[_idx] = _in[_idx];
  }
};

TEST(SyclTransform, DeclareUnifiedTest(Basic)) {
  const size_t size{256};
  std::vector<float> h_in(size);
  std::vector<float> h_out(size);
  std::iota(h_in.begin(), h_in.end(), 0);
  std::vector<float> h_sol(size);
  std::iota(h_sol.begin(), h_sol.end(), 0);

  auto device =  DeviceOrd::SyclDefault();
  HostDeviceVector<float> const in_vec{h_in, device};
  HostDeviceVector<float> out_vec{h_out, device};
  out_vec.Fill(0);

  Transform<>::Init(TestTransformRange<float>{},
                    Range{0, static_cast<Range::DifferenceType>(size)}, 1,
                    device)
      .Eval(&out_vec, &in_vec);
  std::vector<float> res = out_vec.HostVector();

  ASSERT_TRUE(std::equal(h_sol.begin(), h_sol.end(), res.begin()));
}
}  // namespace xgboost::common
