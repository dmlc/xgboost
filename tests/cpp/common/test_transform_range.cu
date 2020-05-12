// This converts all tests from CPU to GPU.
#include "test_transform_range.cc"

#if defined(XGBOOST_USE_NCCL)
namespace xgboost {
namespace common {

TEST(Transform, MGPU_SpecifiedGpuId) {  // NOLINT
  if (AllVisibleGPUs() < 2) {
    LOG(WARNING) << "Not testing in multi-gpu environment.";
    return;
  }
  // Use 1 GPU, Numbering of GPU starts from 1
  auto device = 1;
  const size_t size {256};
  std::vector<bst_float> h_in(size);
  std::vector<bst_float> h_out(size);
  InitializeRange(h_in.begin(), h_in.end());
  std::vector<bst_float> h_sol(size);
  InitializeRange(h_sol.begin(), h_sol.end());

  const HostDeviceVector<bst_float> in_vec {h_in, device};
  HostDeviceVector<bst_float> out_vec {h_out, device};

  ASSERT_NO_THROW(
      Transform<>::Init(TestTransformRange<bst_float>{}, Range{0, size}, device)
      .Eval(&out_vec, &in_vec));
  std::vector<bst_float> res = out_vec.HostVector();
  ASSERT_TRUE(std::equal(h_sol.begin(), h_sol.end(), res.begin()));
}

}  // namespace common
}  // namespace xgboost
#endif