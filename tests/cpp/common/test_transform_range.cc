#include <xgboost/base.h>
#include <gtest/gtest.h>
#include <vector>

#include "../../../src/common/host_device_vector.h"
#include "../../../src/common/transform.h"
#include "../../../src/common/span.h"
#include "../helpers.h"

#if defined(__CUDACC__)

#define TRANSFORM_GPU_RANGE GPUSet::Range(0, 1)
#define TRANSFORM_GPU_DIST GPUDistribution::Block(GPUSet::Range(0, 1))

#else

#define TRANSFORM_GPU_RANGE GPUSet::Empty()
#define TRANSFORM_GPU_DIST GPUDistribution::Block(GPUSet::Empty())

#endif

template <typename Iter>
void InitializeRange(Iter _begin, Iter _end) {
  float j = 0;
  for (Iter i = _begin; i != _end; ++i, ++j) {
    *i = j;
  }
}

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
  InitializeRange(h_in.begin(), h_in.end());
  std::vector<bst_float> h_sol(size);
  InitializeRange(h_sol.begin(), h_sol.end());

  const HostDeviceVector<bst_float> in_vec{h_in, TRANSFORM_GPU_DIST};
  HostDeviceVector<bst_float> out_vec{h_out, TRANSFORM_GPU_DIST};
  out_vec.Fill(0);

  Transform<>::Init(TestTransformRange<bst_float>{}, Range{0, size}, TRANSFORM_GPU_RANGE)
      .Eval(&out_vec, &in_vec);
  std::vector<bst_float> res = out_vec.HostVector();

  ASSERT_TRUE(std::equal(h_sol.begin(), h_sol.end(), res.begin()));
}

} // namespace common
} // namespace xgboost
