/*! Copyright 2019 by Contributors */

#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <thrust/device_vector.h>
#include "../../../src/common/device_helpers.cuh"

namespace xgboost {

template <typename T>
std::string PrepareData(std::string typestr, thrust::device_vector<T>* out) {
  constexpr size_t kRows = 16;
  out->resize(kRows);
  auto& d_data = *out;

  for (size_t i = 0; i < d_data.size(); ++i) {
    d_data[i] = i * 2.0;
  }

  Json column { Object() };

  std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
  column["shape"] = Array(j_shape);
  column["strides"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(4)))});
  column["version"] = Integer(static_cast<Integer::Int>(1));
  column["typestr"] = String(typestr);

  auto p_d_data = dh::Raw(d_data);
  std::vector<Json> j_data {
        Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
        Json(Boolean(false))};
  column["data"] = j_data;
  Json array(std::vector<Json>{column});

  std::stringstream ss;
  Json::Dump(array, &ss);
  std::string str = ss.str();

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

  EXPECT_ANY_THROW({info.SetInfo("group", str.c_str());});
}

TEST(MetaInfo, Group) {
  cudaSetDevice(0);
  thrust::device_vector<uint32_t> d_data;
  std::string str = PrepareData<uint32_t>("<u4", &d_data);

  MetaInfo info;

  info.SetInfo("group", str.c_str());
  auto const& h_group = info.group_ptr_;
  ASSERT_EQ(h_group.size(), d_data.size() + 1);
  for (size_t i = 1; i < h_group.size(); ++i) {
    ASSERT_EQ(h_group[i], d_data[i-1] + h_group[i-1]) << "i: " << i;
  }
}
}  // namespace xgboost