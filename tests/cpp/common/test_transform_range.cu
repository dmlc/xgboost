#include "../../../src/common/host_device_vector.h"
#include "../../../src/common/span.h"
#include "../../../src/common/transform.h"
#include "../helpers.h"
#include "test_common.h"
#include <xgboost/base.h>

#include <gtest/gtest.h>
#include <vector>

#if defined(XGBOOST_USE_CUDA)
#define TRANSFORM_GPU_RANGE GPUSet::Range(0, 1)
#else
#define TRANSFORM_GPU_RANGE GPUSet::Empty()
#endif

namespace xgboost {
namespace common {

struct TestTransformRange {
  void XGBOOST_DEVICE operator()(Span<bst_float> _out, Span<bst_float> _in0,
                                 Span<bst_float> _in1) {
    for (auto out_it = _out.begin(), in_it0 = _in0.begin(),
              in_it1 = _in1.begin();
         out_it != _out.end(); ++out_it, ++in_it0, ++in_it1) {
      *out_it = *in_it0 + *in_it1;
    }
  }

  void XGBOOST_DEVICE operator()(unsigned int *_flag, Span<bst_float> _out,
                                 Span<bst_float> _in0, Span<bst_float> _in1) {
    for (auto out_it = _out.begin(), in_it0 = _in0.begin(),
              in_it1 = _in1.begin();
         out_it != _out.end(); ++out_it, ++in_it0, ++in_it1) {
      *out_it = *in_it0 + *in_it1;
    }
    _flag[0] = 1;
  }

  void XGBOOST_DEVICE operator()(unsigned int *_flag, Span<bst_float> _empty) {
    *_flag = _empty.size() == 0;
  }
};

TEST(TransRange, Basic) {
  std::vector<bst_float> h_in0(16);
  std::vector<bst_float> h_in1(16);
  InitializeRange(h_in0.begin(), h_in0.end());
  InitializeRange(h_in1.begin(), h_in1.end());

  std::vector<bst_float> h_out(16);

  HostDeviceVector<bst_float> in_vec0{h_in0, TRANSFORM_GPU_RANGE};
  HostDeviceVector<bst_float> in_vec1{h_in1, TRANSFORM_GPU_RANGE};

  HostDeviceVector<bst_float> out_vec{h_out, TRANSFORM_GPU_RANGE};

  Range range_divisible{1, 16, 4};

  // divisible
  SegTransform(TestTransformRange{}, range_divisible, TRANSFORM_GPU_RANGE,
               &out_vec, &in_vec0, &in_vec1);

  std::vector<bst_float> sol(16);

  for (size_t i = 0; i < h_out.size(); ++i) {
    sol[i] = h_in0[i] + h_in1[i];
  }
  ASSERT_TRUE(IsNear(sol.cbegin(), sol.cend(), out_vec.HostVector().cbegin()));

  // With flags
  std::vector<unsigned int> h_flags{0};
  HostDeviceVector<unsigned int> flags_vec{h_flags, TRANSFORM_GPU_RANGE};
  SegTransform(TestTransformRange{}, range_divisible, &flags_vec,
               TRANSFORM_GPU_RANGE, &out_vec, &in_vec0, &in_vec1);
  ASSERT_EQ(flags_vec.HostVector().at(0), 1);

  // Not divisible
  Range range_undivisible{1, 16, 3};
  EXPECT_ANY_THROW(SegTransform(TestTransformRange{}, range_undivisible,
                                TRANSFORM_GPU_RANGE, &out_vec, &in_vec0,
                                &in_vec1););
}

TEST(TransRange, Empty) {
  std::vector<unsigned int> h_flags{0};
  HostDeviceVector<unsigned int> flags_vec{h_flags, TRANSFORM_GPU_RANGE};

  Range range_divisible{1, 16, 4};

  std::vector<bst_float> h_empty{};
  HostDeviceVector<bst_float> empty_vec{h_empty, TRANSFORM_GPU_RANGE};

  SegTransform(TestTransformRange{}, range_divisible, &flags_vec,
               TRANSFORM_GPU_RANGE, &empty_vec);
  ASSERT_EQ(flags_vec.HostVector().at(0), 1);
}

} // namespace common
} // namespace xgboost
