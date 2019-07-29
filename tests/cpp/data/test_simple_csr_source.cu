// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <thrust/device_vector.h>

#include <memory>
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/data/simple_csr_source.h"

namespace xgboost {

TEST(SimpleCSRSource, FromColumnar) {
  constexpr size_t kRows = 16;
  Json column { Object() };
  std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
  column["shape"] = Array(j_shape);
  column["strides"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(4)))});

  thrust::device_vector<float> d_data(kRows);
  for (size_t i = 0; i < d_data.size(); ++i) {
    d_data[i] = i * 2.0;
  }

  auto p_d_data = dh::Raw(d_data);

  std::vector<Json> j_data {
        Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
        Json(Boolean(false))};
  column["data"] = j_data;

  column["version"] = Integer(static_cast<Integer::Int>(1));
  column["typestr"] = String("<f4");

  std::stringstream ss;
  Json::Dump(column, &ss);
  LOG(INFO) << ss.str();

  std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
  source->CopyFrom({column});
}

}  // namespace xgboost