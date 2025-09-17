/**
 * Copyright 2020-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "../../../plugin/sycl/common/row_set.h"
#include "../../../plugin/sycl/device_manager.h"
#include "../helpers.h"

namespace xgboost::sycl::common {
TEST(SyclRowSetCollection, AddSplits) {
  const size_t num_rows = 16;

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(DeviceOrd::SyclDefault());

  RowSetCollection row_set_collection;

  auto& row_indices = row_set_collection.Data();
  row_indices.Resize(qu, num_rows);
  size_t* p_row_indices = row_indices.Data();

  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(num_rows),
                       [p_row_indices](::sycl::item<1> pid) {
      const size_t idx = pid.get_id(0);
      p_row_indices[idx] = idx;
    });
  }).wait_and_throw();
  row_set_collection.Init();

  CHECK_EQ(row_set_collection.Size(), 1);
  {
    size_t nid_test = 0;
    auto& elem = row_set_collection[nid_test];
    CHECK_EQ(elem.begin, row_indices.Begin());
    CHECK_EQ(elem.end, row_indices.End());
    CHECK_EQ(elem.node_id , 0);
  }

  size_t nid = 0;
  size_t nid_left = 1;
  size_t nid_right = 2;
  size_t n_left = 4;
  size_t n_right = num_rows - n_left;
  row_set_collection.AddSplit(nid, nid_left, nid_right, n_left, n_right);
  CHECK_EQ(row_set_collection.Size(), 3);

  {
    size_t nid_test = 0;
    auto& elem = row_set_collection[nid_test];
    CHECK_EQ(elem.begin, nullptr);
    CHECK_EQ(elem.end, nullptr);
    CHECK_EQ(elem.node_id , -1);
  }

  {
    size_t nid_test = 1;
    auto& elem = row_set_collection[nid_test];
    CHECK_EQ(elem.begin, row_indices.Begin());
    CHECK_EQ(elem.end, row_indices.Begin() + n_left);
    CHECK_EQ(elem.node_id , nid_test);
  }

  {
    size_t nid_test = 2;
    auto& elem = row_set_collection[nid_test];
    CHECK_EQ(elem.begin, row_indices.Begin() + n_left);
    CHECK_EQ(elem.end, row_indices.End());
    CHECK_EQ(elem.node_id , nid_test);
  }

}
}  // namespace xgboost::sycl::common
