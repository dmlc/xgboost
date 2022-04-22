/*!
 * Copyright 2018-2022 by XGBoost contributors
 */

#include "../../../src/objective/adaptive.cuh"
#include "test_regression_obj.cc"

namespace xgboost {
namespace obj {
void TestFillMissingLeaf() {
  std::vector<bst_node_t> missing{1, 3};
  Context ctx;

  HostDeviceVector<bst_node_t> node_idx = {2, 4, 5};
  HostDeviceVector<size_t> node_ptr = {0, 4, 8, 16};
  node_idx.SetDevice(0);
  node_ptr.SetDevice(0);

  detail::FillMissingLeaf(missing, &node_idx, &node_ptr);

  auto const& h_nidx = node_idx.HostVector();
  auto const& h_nptr = node_ptr.HostVector();

  ASSERT_EQ(h_nidx[0], missing[0]);
  ASSERT_EQ(h_nidx[2], missing[1]);
  ASSERT_EQ(h_nidx[1], 2);
  ASSERT_EQ(h_nidx[3], 4);
  ASSERT_EQ(h_nidx[4], 5);

  ASSERT_EQ(h_nptr[0], 0);
  ASSERT_EQ(h_nptr[1], 0);  // empty
  ASSERT_EQ(h_nptr[2], 4);
  ASSERT_EQ(h_nptr[3], 4);  // empty
  ASSERT_EQ(h_nptr[4], 8);
  ASSERT_EQ(h_nptr[5], 16);
}

TEST(Adaptive, MissingLeaf) { TestFillMissingLeaf(); }
}  // namespace obj
}  // namespace xgboost
