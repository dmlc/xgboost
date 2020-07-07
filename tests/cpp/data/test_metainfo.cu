/*! Copyright 2019 by Contributors */

#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <thrust/device_vector.h>
#include "../../../src/common/device_helpers.cuh"

namespace xgboost {

template <typename T>
std::string PrepareData(std::string typestr, thrust::device_vector<T>* out, const size_t kRows=16) {
  out->resize(kRows);
  auto& d_data = *out;

  for (size_t i = 0; i < d_data.size(); ++i) {
    d_data[i] = i * 2.0;
  }

  Json column { Object() };

  std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
  column["shape"] = Array(j_shape);
  column["strides"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(sizeof(T))))});
  column["version"] = Integer(static_cast<Integer::Int>(1));
  column["typestr"] = String(typestr);

  auto p_d_data = d_data.data().get();
  std::vector<Json> j_data {
        Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
        Json(Boolean(false))};
  column["data"] = j_data;
  Json array(std::vector<Json>{column});

  std::string str;
  Json::Dump(array, &str);

  return str;
}

TEST(MetaInfo, FromInterface) {
  cudaSetDevice(0);
  thrust::device_vector<float> d_data;

  std::string str = PrepareData<float>("<f4", &d_data);

  MetaInfo info;
  info.SetInfo("label", str.c_str());

  auto const& h_label = info.labels_.HostVector();
  for (size_t i = 0; i < d_data.size(); ++i) {
    ASSERT_EQ(h_label[i], d_data[i]);
  }

  info.SetInfo("weight", str.c_str());
  auto const& h_weight = info.weights_.HostVector();
  for (size_t i = 0; i < d_data.size(); ++i) {
    ASSERT_EQ(h_weight[i], d_data[i]);
  }

  info.SetInfo("base_margin", str.c_str());
  auto const& h_base_margin = info.base_margin_.HostVector();
  for (size_t i = 0; i < d_data.size(); ++i) {
    ASSERT_EQ(h_base_margin[i], d_data[i]);
  }

  thrust::device_vector<int> d_group_data;
  std::string group_str = PrepareData<int>("<i4", &d_group_data, 4);
  d_group_data[0] = 4;
  d_group_data[1] = 3;
  d_group_data[2] = 2;
  d_group_data[3] = 1;
  info.SetInfo("group", group_str.c_str());
  std::vector<bst_group_t> expected_group_ptr = {0, 4, 7, 9, 10};
  EXPECT_EQ(info.group_ptr_, expected_group_ptr);
}

TEST(MetaInfo, Group) {
  cudaSetDevice(0);
  MetaInfo info;

  thrust::device_vector<uint32_t> d_uint;
  std::string uint_str = PrepareData<uint32_t>("<u4", &d_uint);
  info.SetInfo("group", uint_str.c_str());
  auto& h_group = info.group_ptr_;
  ASSERT_EQ(h_group.size(), d_uint.size() + 1);
  for (size_t i = 1; i < h_group.size(); ++i) {
    ASSERT_EQ(h_group[i], d_uint[i - 1] + h_group[i - 1]) << "i: " << i;
  }

  thrust::device_vector<int64_t> d_int64;
  std::string int_str = PrepareData<int64_t>("<i8", &d_int64);
  info = MetaInfo();
  info.SetInfo("group", int_str.c_str());
  h_group = info.group_ptr_;
  ASSERT_EQ(h_group.size(), d_uint.size() + 1);
  for (size_t i = 1; i < h_group.size(); ++i) {
    ASSERT_EQ(h_group[i], d_uint[i - 1] + h_group[i - 1]) << "i: " << i;
  }

  // Incorrect type
  thrust::device_vector<float> d_float;
  std::string float_str = PrepareData<float>("<f4", &d_float);
  info = MetaInfo();
  EXPECT_ANY_THROW(info.SetInfo("group", float_str.c_str()));
}

TEST(MetaInfo, DeviceExtend) {
  dh::safe_cuda(cudaSetDevice(0));
  size_t const kRows = 100;
  MetaInfo lhs, rhs;

  thrust::device_vector<float> d_data;
  std::string str = PrepareData<float>("<f4", &d_data, kRows);
  lhs.SetInfo("label", str.c_str());
  rhs.SetInfo("label", str.c_str());
  ASSERT_FALSE(rhs.labels_.HostCanRead());
  lhs.num_row_ = kRows;
  rhs.num_row_ = kRows;

  lhs.Extend(rhs, true);
  ASSERT_EQ(lhs.num_row_, kRows * 2);
  ASSERT_FALSE(lhs.labels_.HostCanRead());

  ASSERT_FALSE(lhs.labels_.HostCanRead());
  ASSERT_FALSE(rhs.labels_.HostCanRead());
}
}  // namespace xgboost
