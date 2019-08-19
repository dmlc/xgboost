/*! Copyright 2019 by Contributors */

#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <thrust/device_vector.h>
#include "../../../src/common/device_helpers.cuh"

namespace xgboost {
TEST(MetaInfo, FromInterface) {
  cudaSetDevice(0);
  constexpr size_t kRows = 16;

  thrust::device_vector<float> d_data(kRows);
  for (size_t i = 0; i < d_data.size(); ++i) {
    d_data[i] = i * 2.0;
  }

  Json column { Object() };

  std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
  column["shape"] = Array(j_shape);
  column["strides"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(4)))});
  column["version"] = Integer(static_cast<Integer::Int>(1));
  column["typestr"] = String("<f4");

  auto p_d_data = dh::Raw(d_data);
  std::vector<Json> j_data {
        Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
        Json(Boolean(false))};
  column["data"] = j_data;

  std::stringstream ss;
  Json::Dump(column, &ss);
  std::string str = ss.str();

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
}
}  // namespace xgboost