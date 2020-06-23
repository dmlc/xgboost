#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include "../../../src/common/segmented_uniques.cuh"

TEST(SegmentedUnique, Basic) {
  std::vector<float> values{0.1f, 0.2f, 0.3f, 0.62448811531066895f, 0.62448811531066895f, 0.4f};
  std::vector<size_t> segments{0, 3, 6};

  thrust::device_vector<float> d_values(values);
  thrust::device_vector<size_t> d_segments{segments};

  thrust::device_vector<size_t> d_segs_out(d_segments.size());
  thrust::device_vector<float> d_vals_out(d_values.size());

  size_t n_uniques = SegmentedUnique(
      d_segments.data().get(), d_segments.data().get() + d_segments.size(),
      d_values.data().get(), d_values.data().get() + d_values.size(),
      d_segments.data().get(), d_vals_out.data().get(),
      thrust::equal_to<float>{});
  CHECK_EQ(n_uniques, 5);

  std::vector<float> values_sol{0.1f, 0.2f, 0.3f, 0.62448811531066895f, 0.4f};
  for (auto i = 0 ; i < values_sol.size(); i ++) {
    ASSERT_EQ(d_vals_out[i], values_sol[i]);
  }

  std::vector<size_t> segments_sol{0, 3, 5};
  for (size_t i = 0; i < d_segments.size(); ++i) {
    ASSERT_EQ(segments_sol[i], d_segments[i]);
  }

  d_segments[1] = 4;
  d_segments[2] = 6;
  n_uniques = SegmentedUnique(
      d_segments.data().get(), d_segments.data().get() + d_segments.size(),
      d_values.data().get(), d_values.data().get() + d_values.size(),
      d_segments.data().get(), d_vals_out.data().get(),
      thrust::equal_to<float>{});
  ASSERT_EQ(n_uniques, values.size());
  for (auto i = 0 ; i < values.size(); i ++) {
    ASSERT_EQ(d_vals_out[i], values[i]);
  }
}