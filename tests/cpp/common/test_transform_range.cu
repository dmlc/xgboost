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

// Test for multi-classes setting.
template <typename T>
struct TestTransformRangeGranular {
  const size_t granularity = 8;

  TestTransformRangeGranular(const size_t granular) : granularity{granular} {}
  void XGBOOST_DEVICE operator()(size_t _idx,
                                 Span<bst_float> _out, Span<const bst_float> _in) {
    auto in_sub = _in.subspan(_idx * granularity, granularity);
    auto out_sub = _out.subspan(_idx * granularity, granularity);
    for (size_t i = 0; i < granularity; ++i) {
      out_sub[i] = in_sub[i];
    }
  }
};

TEST(Transform, MGPU_Granularity) {
  GPUSet devices = GPUSet::All(0, -1);

  const size_t size {8990};
  const size_t granularity = 10;

  GPUDistribution distribution =
      GPUDistribution::Granular(devices, granularity);

  std::vector<bst_float> h_in(size);
  std::vector<bst_float> h_out(size);
  InitializeRange(h_in.begin(), h_in.end());
  std::vector<bst_float> h_sol(size);
  InitializeRange(h_sol.begin(), h_sol.end());

  const HostDeviceVector<bst_float> in_vec {h_in, distribution};
  HostDeviceVector<bst_float> out_vec {h_out, distribution};

  ASSERT_NO_THROW(
      Transform<>::Init(
          TestTransformRangeGranular<bst_float>{granularity},
          Range{0, size / granularity},
          distribution)
      .Eval(&out_vec, &in_vec));
  std::vector<bst_float> res = out_vec.HostVector();

  ASSERT_TRUE(std::equal(h_sol.begin(), h_sol.end(), res.begin()));
}

TEST(Transform, MGPU_SpecifiedGpuId) {
  // Use 1 GPU, Numbering of GPU starts from 1
  auto devices = GPUSet::All(1, 1);
  const size_t size {256};
  std::vector<bst_float> h_in(size);
  std::vector<bst_float> h_out(size);
  InitializeRange(h_in.begin(), h_in.end());
  std::vector<bst_float> h_sol(size);
  InitializeRange(h_sol.begin(), h_sol.end());

  const HostDeviceVector<bst_float> in_vec {h_in,
        GPUDistribution::Block(devices)};
  HostDeviceVector<bst_float> out_vec {h_out,
        GPUDistribution::Block(devices)};

  ASSERT_NO_THROW(
      Transform<>::Init(TestTransformRange<bst_float>{}, Range{0, size}, devices)
      .Eval(&out_vec, &in_vec));
  std::vector<bst_float> res = out_vec.HostVector();
  ASSERT_TRUE(std::equal(h_sol.begin(), h_sol.end(), res.begin()));
}

}  // namespace xgboost
}  // namespace common
#endif