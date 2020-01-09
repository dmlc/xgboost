// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <thrust/device_vector.h>

#include <memory>
#include "../../../src/common/bitfield.h"
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/data/simple_csr_source.h"
#include "../../../src/data/columnar.h"

namespace xgboost {

template <typename T>
Json GenerateDenseColumn(std::string const& typestr, size_t kRows,
                         thrust::device_vector<T>* out_d_data) {
  auto& d_data = *out_d_data;
  d_data.resize(kRows);
  Json column { Object() };
  std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
  column["shape"] = Array(j_shape);
  column["strides"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(sizeof(T))))});

  d_data.resize(kRows);
  thrust::sequence(thrust::device, d_data.begin(), d_data.end(), 0.0f, 2.0f);

  auto p_d_data = dh::Raw(d_data);

  std::vector<Json> j_data {
    Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
        Json(Boolean(false))};
  column["data"] = j_data;

  column["version"] = Integer(static_cast<Integer::Int>(1));
  column["typestr"] = String(typestr);
  return column;
}

template <typename T>
Json GenerateSparseColumn(std::string const& typestr, size_t kRows,
                         thrust::device_vector<T>* out_d_data) {
  auto& d_data = *out_d_data;
  Json column { Object() };
  std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
  column["shape"] = Array(j_shape);
  column["strides"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(sizeof(T))))});

  d_data.resize(kRows);
  for (size_t i = 0; i < d_data.size(); ++i) {
    d_data[i] = i * 2.0;
  }

  auto p_d_data = dh::Raw(d_data);

  std::vector<Json> j_data {
    Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
        Json(Boolean(false))};
  column["data"] = j_data;

  column["version"] = Integer(static_cast<Integer::Int>(1));
  column["typestr"] = String(typestr);
  return column;
}
}  // namespace xgboost
