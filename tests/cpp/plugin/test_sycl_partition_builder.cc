/**
 * Copyright 2020-2024 by XGBoost contributors
 */
#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "../../../plugin/sycl/common/partition_builder.h"
#include "../../../plugin/sycl/device_manager.h"
#include "../helpers.h"

namespace xgboost::sycl::common {

TEST(SyclPartitionBuilder, BasicTest) {
  constexpr size_t kNodes = 5;
  // Number of rows for each node
  std::vector<size_t> rows = { 5, 5, 10, 1, 2 };

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(DeviceOrd::SyclDefault());
  PartitionBuilder builder;
  builder.Init(&qu, kNodes, [&](size_t i) {
    return rows[i];
  });

  // We test here only the basics, thus syntetic partition builder is adopted
  // Number of rows to go left for each node.
  std::vector<size_t> rows_for_left_node = { 2, 0, 7, 1, 2 };

  size_t first_row_id = 0;
  for(size_t nid = 0; nid < kNodes; ++nid) {
    size_t n_rows_nodes = rows[nid];

    auto rid_buff = builder.GetData(nid);
    size_t rid_buff_size = rid_buff.size();
    auto* rid_buff_ptr = rid_buff.data();

    size_t n_left  = rows_for_left_node[nid];
    size_t n_right = rows[nid] - n_left;

    qu.submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<>(::sycl::range<1>(n_left), [=](::sycl::id<1> pid) {
        int row_id = first_row_id + pid[0];
        rid_buff_ptr[pid[0]] = row_id;
      });
    });
    qu.wait();
    first_row_id += n_left;

    // We are storing indexes for the right side in the tail of the array to save some memory
    qu.submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<>(::sycl::range<1>(n_right), [=](::sycl::id<1> pid) {
        int row_id = first_row_id + pid[0];
        rid_buff_ptr[rid_buff_size - pid[0] - 1] = row_id;
      });
    });
    qu.wait();
    first_row_id += n_right;

    builder.SetNLeftElems(nid, n_left);
    builder.SetNRightElems(nid, n_right);
  }

  ::sycl::event event;
  std::vector<size_t> v(*std::max_element(rows.begin(), rows.end()));
  size_t row_id = 0;
  for(size_t nid = 0; nid < kNodes; ++nid) {
    builder.MergeToArray(nid, v.data(), event);
    qu.wait();

    // Check that row_id for left side are correct
    for(size_t j = 0; j < rows_for_left_node[nid]; ++j) {
       ASSERT_EQ(v[j], row_id++);
    }

    // Check that row_id for right side are correct
    for(size_t j = 0; j < rows[nid] - rows_for_left_node[nid]; ++j) {
      ASSERT_EQ(v[rows[nid] - j - 1], row_id++);
    }

    // Check that number of left/right rows are correct
    size_t n_left  = builder.GetNLeftElems(nid);
    size_t n_right = builder.GetNRightElems(nid);
    ASSERT_EQ(n_left, rows_for_left_node[nid]);
    ASSERT_EQ(n_right, (rows[nid] - rows_for_left_node[nid]));
  }
}

}  // namespace xgboost::common
