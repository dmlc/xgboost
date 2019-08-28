// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <xgboost/json.h>
#include <thrust/device_vector.h>

#include <memory>
#include "../../../src/common/bitfield.h"
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/data/simple_csr_source.h"

namespace xgboost {

TEST(SimpleCSRSource, FromColumnarDense) {
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
  Json column_arr {Array{std::vector<Json>{column}}};

  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();

  // no missing value
  {
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    source->CopyFrom(str.c_str(), false);

    auto const& data = source->page_.data.HostVector();
    auto const& offset = source->page_.offset.HostVector();
    for (size_t i = 0; i < kRows; ++i) {
      auto e = data[i];
      ASSERT_NEAR(e.fvalue, i * 2.0, kRtEps);
      ASSERT_EQ(e.index, 0);  // feature 0
    }
    ASSERT_EQ(offset.back(), 16);
    for (size_t i = 0; i < kRows + 1; ++i) {
      ASSERT_EQ(offset[i], i);
    }
  }

  // with missing value specified
  {
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    source->CopyFrom(str.c_str(), true, 4.0);

    auto const& data = source->page_.data.HostVector();
    auto const& offset = source->page_.offset.HostVector();
    ASSERT_EQ(data.size(), 15);
    ASSERT_NEAR(data[2].fvalue, 6.0, kRtEps);
    ASSERT_EQ(offset.back(), 15);
    for (size_t i = 3; i < kRows + 1; ++i) {
      ASSERT_EQ(offset[i], i - 1);
    }
  }

  {
    // no missing value, but has NaN
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    d_data[3] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(std::isnan(d_data[3]));  // removes 6.0
    source->CopyFrom(str.c_str(), false);

    auto const& data = source->page_.data.HostVector();
    auto const& offset = source->page_.offset.HostVector();
    ASSERT_EQ(data.size(), 15);
    ASSERT_NEAR(data[3].fvalue, 8.0, kRtEps);
  }
}

TEST(SimpleCSRSource, FromColumnarWithEmptyRows) {
  constexpr size_t kRows = 102;
  constexpr size_t kCols = 24;

  std::vector<Json> v_columns (kCols);
  std::vector<dh::device_vector<float>> columns_data(kCols);
  std::vector<dh::device_vector<unsigned char>> column_bitfields(kCols);

  unsigned char constexpr kUCOne = 1;

  for (size_t i = 0; i < kCols; ++i) {
    auto& col = v_columns[i];
    col = Object();
    auto& data = columns_data[i];
    data.resize(kRows);
    thrust::sequence(data.begin(), data.end(), 0);
    dh::safe_cuda(cudaDeviceSynchronize());
    dh::safe_cuda(cudaGetLastError());

    ASSERT_EQ(data.size(), kRows);

    auto p_d_data = raw_pointer_cast(data.data());
    std::vector<Json> j_data {
      Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
          Json(Boolean(false))};
    col["data"] = j_data;
    std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
    col["shape"] = Array(j_shape);
    col["version"] = Integer(static_cast<Integer::Int>(1));
    col["typestr"] = String("<f4");

    // Construct the mask object.
    col["mask"] = Object();
    auto& j_mask = col["mask"];
    j_mask["version"] = Integer(static_cast<Integer::Int>(1));
    auto& mask_storage = column_bitfields[i];
    mask_storage.resize(16);  // 16 bytes

    mask_storage[0] = ~(kUCOne << 2);  // 3^th row is missing
    mask_storage[1] = ~(kUCOne << 3);  // 12^th row is missing
    size_t last_ind = 12;
    mask_storage[last_ind] = ~(kUCOne << 5);
    std::set<size_t> missing_row_index {0, 1, last_ind};

    for (size_t i = 0; i < mask_storage.size(); ++i) {
      if (missing_row_index.find(i) == missing_row_index.cend()) {
        // all other rows are valid
        mask_storage[i] = ~0;
      }
    }

    j_mask["data"] = std::vector<Json>{
      Json(Integer(reinterpret_cast<Integer::Int>(mask_storage.data().get()))),
      Json(Boolean(false))};
    j_mask["shape"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(16)))});
    j_mask["typestr"] = String("|i1");
  }

  Json column_arr {Array(v_columns)};
  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();
  std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
  source->CopyFrom(str.c_str(), false);

  auto const& data = source->page_.data.HostVector();
  auto const& offset = source->page_.offset.HostVector();

  ASSERT_EQ(offset.size(), kRows + 1);
  for (size_t i = 1; i < offset.size(); ++i) {
    for (size_t j = offset[i-1]; j < offset[i]; ++j) {
      ASSERT_EQ(data[j].index, j % kCols);
      ASSERT_NEAR(data[j].fvalue, i - 1, kRtEps);
    }
  }
}

TEST(SimpleCSRSource, FromColumnarSparse) {
  constexpr size_t kRows = 32;
  constexpr size_t kCols = 2;
  unsigned char constexpr kUCOne = 1;

  std::vector<dh::device_vector<float>> columns_data(kCols);
  std::vector<dh::device_vector<unsigned char>> column_bitfields(kCols);

  {
    // column 0
    auto& mask = column_bitfields[0];
    mask.resize(8);

    for (size_t j = 0; j < mask.size(); ++j) {
      mask[j] = ~0;
    }
    // the 2^th entry of first column is invalid
    // [0 0 0 0 0 1 0 0]
    mask[0] = ~(kUCOne << 2);
  }
  {
    // column 1
    auto& mask = column_bitfields[1];
    mask.resize(8);

    for (size_t j = 0; j < mask.size(); ++j) {
      mask[j] = ~0;
    }
    // the 19^th entry of second column is invalid
    // [~0~], [~0~], [0 0 0 0 1 0 0 0]
    mask[2] = ~(kUCOne << 3);
  }

  for (size_t c = 0; c < kCols; ++c) {
    columns_data[c].resize(kRows);
    thrust::sequence(columns_data[c].begin(), columns_data[c].end(), 0);
  }

  std::vector<Json> j_columns(kCols);

  for (size_t c = 0; c < kCols; ++c) {
    auto& column = j_columns[c];
    column = Object();
    column["version"] = Integer(static_cast<Integer::Int>(1));
    column["typestr"] = String("<f4");
    auto p_d_data = raw_pointer_cast(columns_data[c].data());
    std::vector<Json> j_data {
      Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
          Json(Boolean(false))};
    column["data"] = j_data;
    std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
    column["shape"] = Array(j_shape);
    column["version"] = Integer(static_cast<Integer::Int>(1));
    column["typestr"] = String("<f4");

    column["mask"] = Object();
    auto& j_mask = column["mask"];
    j_mask["version"] = Integer(static_cast<Integer::Int>(1));
    j_mask["data"] = std::vector<Json>{
      Json(Integer(reinterpret_cast<Integer::Int>(column_bitfields[c].data().get()))),
      Json(Boolean(false))};
    j_mask["shape"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(8)))});
    j_mask["typestr"] = String("|i1");
  }

  Json column_arr {Array(j_columns)};

  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();

  {
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    source->CopyFrom(str.c_str(), false);

    auto const& data = source->page_.data.HostVector();
    auto const& offset = source->page_.offset.HostVector();

    ASSERT_EQ(offset.size(), kRows + 1);
    ASSERT_EQ(data[4].index, 1);
    ASSERT_EQ(data[4].fvalue, 2);
    ASSERT_EQ(data[37].index, 0);
    ASSERT_EQ(data[37].fvalue, 19);
  }

  {
    // with missing value
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    source->CopyFrom(str.c_str(), true, /*missing=*/2.0);

    auto const& data = source->page_.data.HostVector();
    ASSERT_NE(data[4].fvalue, 2.0);
  }

  {
    // no missing value, but has NaN
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    columns_data[0][4] = std::numeric_limits<float>::quiet_NaN();  // 0^th column 4^th row
    ASSERT_TRUE(std::isnan(columns_data[0][4]));
    source->CopyFrom(str.c_str(), false);

    auto const& data = source->page_.data.HostVector();
    auto const& offset = source->page_.offset.HostVector();
    // Two invalid entries and one NaN, in CSC
    // 0^th column: 0, 1, 4, 5, 6, ..., kRows
    // 1^th column: 0, 1, 2, 3, ..., 19, 21, ..., kRows
    // Turning it into CSR:
    // | 0, 0 | 1, 1 | 2 | 3, 3 | 4 | ...
    ASSERT_EQ(data.size(), kRows * kCols - 3);
    ASSERT_EQ(data[4].index, 1);  // from 1^th column
    ASSERT_EQ(data[5].fvalue, 3.0);
    ASSERT_EQ(data[7].index, 1);  // from 1^th column
    ASSERT_EQ(data[7].fvalue, 4.0);

    ASSERT_EQ(data[offset[2]].fvalue, 2.0);
    ASSERT_EQ(data[offset[4]].fvalue, 4.0);
  }

  {
    // with NaN as missing value
    // NaN is already set up by above test
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    source->CopyFrom(str.c_str(), true,
                     /*missing=*/std::numeric_limits<float>::quiet_NaN());

    auto const& data = source->page_.data.HostVector();
    ASSERT_EQ(data.size(), kRows * kCols - 1);
    ASSERT_EQ(data[8].fvalue, 4.0);
  }
}

}  // namespace xgboost