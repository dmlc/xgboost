// This converts all tests from CPU to GPU.
#include "test_transform_range.cc"

#if defined(XGBOOST_USE_NCCL)
namespace xgboost {
namespace common {

// Test here is multi gpu specific
TEST(Transform, MGPU_Basic) {
  auto devices = GPUSet::AllVisible();
  CHECK_GT(devices.Size(), 1);
  const size_t size {256};
  std::vector<bst_float> h_in(size);
  std::vector<bst_float> h_out(size);
  InitializeRange(h_in.begin(), h_in.end());
  std::vector<bst_float> h_sol(size);
  InitializeRange(h_sol.begin(), h_sol.end());

  const HostDeviceVector<bst_float> in_vec {h_in,
        GPUDistribution::Block(GPUSet::Empty())};
  HostDeviceVector<bst_float> out_vec {h_out,
        GPUDistribution::Block(GPUSet::Empty())};
  out_vec.Fill(0);

  in_vec.Reshard(GPUDistribution::Granular(devices, 8));
  out_vec.Reshard(GPUDistribution::Block(devices));

  // Granularity is different, resharding will throw.
  EXPECT_ANY_THROW(
      Transform<>::Init(TestTransformRange<bst_float>{}, Range{0, size}, devices)
      .Eval(&out_vec, &in_vec));


  Transform<>::Init(TestTransformRange<bst_float>{}, Range{0, size},
                    devices, false).Eval(&out_vec, &in_vec);
  std::vector<bst_float> res = out_vec.HostVector();

  ASSERT_TRUE(std::equal(h_sol.begin(), h_sol.end(), res.begin()));
}

}  // namespace xgboost
}  // namespace common
#endif