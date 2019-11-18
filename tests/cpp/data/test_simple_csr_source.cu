// Copyright (c) 2019 by Contributors
#include <gtest/gtest.h>
#include <xgboost/data.h>
#include <thrust/device_vector.h>

#include <memory>
#include "../../../src/common/bitfield.h"
#include "../../../src/common/device_helpers.cuh"
#include "../../../src/common/json_experimental.h"
#include "../../../src/common/json_reader_experimental.h"
#include "../../../src/common/json_writer_experimental.h"

#include "../../../src/data/simple_csr_source.h"
#include "../../../src/data/columnar.h"

namespace xgboost {

void TestInvalidArrayInterface(experimental::Document* p_doc) {
  constexpr size_t kRows {16};
  auto& column_doc = *p_doc;

  experimental::Json& column = column_doc.GetValue();

  auto j_shape = column.CreateMember("shape");
  j_shape.SetArray();
  j_shape.CreateArrayElem().SetInteger(kRows);
  j_shape.EndArray();

  // missing version
  try {
    ArrayInterfaceHandler::ExtractArray<float>(column);
  } catch (dmlc::Error const& e) {
    ASSERT_NE(std::string{e.what()}.find("version"), std::string::npos);
  }
  column.CreateMember("version").SetInteger(1);
  try {
    ArrayInterfaceHandler::ExtractArray<float>(column);
  } catch (dmlc::Error const& e) {
    ASSERT_EQ(std::string{e.what()}.find("version"), std::string::npos);
  }

  // missing typestr
  EXPECT_THROW(ArrayInterfaceHandler::ExtractArray<float>(column), dmlc::Error);
  column.CreateMember("typestr").SetString("<f4");

  // missing data
  EXPECT_THROW(ArrayInterfaceHandler::ExtractArray<float>(column), dmlc::Error);
  auto j_data = column.CreateMember("data");
  j_data.SetArray(2);
  j_data.GetArrayElem(0).SetInteger(reinterpret_cast<int64_t>(nullptr));
  j_data.GetArrayElem(1).SetFalse();
  j_data.EndArray();

  // nullptr is not valid
  EXPECT_THROW(ArrayInterfaceHandler::ExtractArray<float>(column), dmlc::Error);
  thrust::device_vector<float> d_data(kRows);
  j_data.GetArrayElem(0).SetInteger(reinterpret_cast<int64_t>(d_data.data().get()));
  EXPECT_NO_THROW(ArrayInterfaceHandler::ExtractArray<float>(column));

  // Create mask with wrong shape.
  auto j_mask = column.CreateMember("mask");
  j_mask.SetObject();

  auto j_mask_shape = j_mask.CreateMember("shape");
  j_mask_shape.SetArray();
  j_mask_shape.CreateArrayElem().SetInteger(kRows - 1);
  j_mask_shape.EndArray();

  auto j_mask_data = j_mask.CreateMember("data");
  j_mask_data.SetArray();
  ASSERT_TRUE(j_mask_data.IsArray());
  j_mask_data.CreateArrayElem().SetInteger(reinterpret_cast<int64_t>(d_data.data().get()));
  ASSERT_TRUE(j_mask_data.IsArray());
  j_mask_data.EndArray();

  j_mask.CreateMember("version").SetInteger(1);
  j_mask.CreateMember("typestr") = "<i1";
  j_mask.EndObject();

  EXPECT_THROW(ArrayInterfaceHandler::ExtractArray<float>(column), dmlc::Error);
}

TEST(ArrayInterfaceHandler, Error) {
  experimental::Document column_doc;
  TestInvalidArrayInterface(&column_doc);
}

template <typename T>
void GenerateDenseColumn(experimental::Json* columns_arr,
                         std::string const& typestr, size_t kRows,
                         thrust::device_vector<T>* out_d_data) {
  auto& d_data = *out_d_data;
  auto column = columns_arr->CreateArrayElem();
  column.SetObject();
  auto j_shape = column.CreateMember("shape");
  j_shape.SetArray();
  j_shape.CreateArrayElem().SetInteger(kRows);

  auto j_strides = column.CreateMember("strides");
  j_strides.SetArray();
  j_strides.CreateArrayElem() = static_cast<int64_t>(sizeof(T));

  d_data.resize(kRows);
  for (size_t i = 0; i < d_data.size(); ++i) {
    d_data[i] = i * 2.0;
  }

  auto p_d_data = dh::Raw(d_data);

  auto j_data = column.CreateMember("data");
  j_data.SetArray(2);
  j_data.GetArrayElem(0).SetInteger(reinterpret_cast<int64_t>(p_d_data));
  j_data.GetArrayElem(1).SetFalse();

  column.CreateMember("version") = static_cast<int64_t>(1);
  column.CreateMember("typestr") = typestr;
}

void TestDenseColumn(std::unique_ptr<data::SimpleCSRSource> const& source,
                     size_t n_rows, size_t n_cols) {
  auto const& data = source->page_.data.HostVector();
  auto const& offset = source->page_.offset.HostVector();

  for (size_t i = 0; i < n_rows; i++) {
    auto const idx = i * n_cols;
    auto const e_0 = data.at(idx);
    ASSERT_NEAR(e_0.fvalue, i * 2.0, kRtEps) << "idx: " << idx;
    ASSERT_EQ(e_0.index, 0);  // feature 0

    auto e_1 = data.at(idx+1);
    ASSERT_NEAR(e_1.fvalue, i * 2.0, kRtEps);
    ASSERT_EQ(e_1.index, 1);  // feature 1
  }
  ASSERT_EQ(offset.back(), n_rows * n_cols);
  for (size_t i = 0; i < n_rows + 1; ++i) {
    ASSERT_EQ(offset[i], i * n_cols);
  }
  ASSERT_EQ(source->info.num_row_, n_rows);
  ASSERT_EQ(source->info.num_col_, n_cols);
}

TEST(SimpleCSRSource, FromColumnarDense) {
  constexpr size_t kRows {16};
  constexpr size_t kCols {2};
  experimental::Document column_arr(experimental::ValueKind::kArray);

  thrust::device_vector<float> d_data_0(kRows);
  thrust::device_vector<int32_t> d_data_1(kRows);
  GenerateDenseColumn<float>(&(column_arr.GetValue()), "<f4", kRows, &d_data_0);
  GenerateDenseColumn<int32_t>(&(column_arr.GetValue()), "<i4", kRows, &d_data_1);

  std::string str = column_arr.Dump<experimental::JsonWriter>();

  // no missing value
  {
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    source->CopyFrom(str.c_str(), false);
    TestDenseColumn(source, kRows, kCols);
  }

  // with missing value specified
  {
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    source->CopyFrom(str.c_str(), true, 4.0);

    auto const& data = source->page_.data.HostVector();
    auto const& offset = source->page_.offset.HostVector();
    ASSERT_EQ(data.size(), kRows * kCols - 2);
    ASSERT_NEAR(data[4].fvalue, 6.0, kRtEps);  // kCols * 2
    ASSERT_EQ(offset.back(), 30);
    for (size_t i = 3; i < kRows + 1; ++i) {
      ASSERT_EQ(offset[i], (i - 1) * 2);
    }
    ASSERT_EQ(source->info.num_row_, kRows);
    ASSERT_EQ(source->info.num_col_, kCols);
  }

  {
    // no missing value, but has NaN
    std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
    d_data_0[3] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(std::isnan(d_data_0[3]));  // removes 6.0
    source->CopyFrom(str.c_str(), false);

    auto const& data = source->page_.data.HostVector();
    auto const& offset = source->page_.offset.HostVector();
    ASSERT_EQ(data.size(), kRows * kCols - 1);
    ASSERT_NEAR(data[7].fvalue, 8.0, kRtEps);
    ASSERT_EQ(source->info.num_row_, kRows);
    ASSERT_EQ(source->info.num_col_, kCols);
  }
}

TEST(SimpleCSRSource, FromColumnarWithEmptyRows) {
  constexpr size_t kRows = 102;
  constexpr size_t kCols = 24;

  experimental::Document column_arr(experimental::ValueKind::kArray);
  std::vector<dh::device_vector<float>> columns_data(kCols);
  std::vector<dh::device_vector<RBitField8::value_type>> column_bitfields(kCols);

  RBitField8::value_type constexpr kUCOne = 1;

  for (size_t i = 0; i < kCols; ++i) {
    auto col = column_arr.GetValue().CreateArrayElem();
    col.SetObject();
    auto& data = columns_data[i];
    data.resize(kRows);
    thrust::sequence(data.begin(), data.end(), 0);
    dh::safe_cuda(cudaDeviceSynchronize());
    dh::safe_cuda(cudaGetLastError());

    ASSERT_EQ(data.size(), kRows);

    auto p_d_data = raw_pointer_cast(data.data());
    auto j_data = col.CreateMember("data");
    j_data.SetArray();
    j_data.CreateArrayElem().SetInteger(reinterpret_cast<int64_t>(p_d_data));
    j_data.CreateArrayElem().SetFalse();

    auto j_shape = col.CreateMember("shape");
    j_shape.SetArray();
    j_shape.CreateArrayElem().SetInteger((static_cast<int64_t>(kRows)));

    col.CreateMember("version") = static_cast<int64_t>(1);
    col.CreateMember("typestr") = "<f4";

    // Construct the mask object.
    auto j_mask = col.CreateMember("mask");
    j_mask.SetObject();
    j_mask.CreateMember("version") = static_cast<int64_t>(1);
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

    auto j_mask_data = j_mask.CreateMember("data");
    j_mask_data.SetArray();
    j_mask_data.CreateArrayElem() = reinterpret_cast<int64_t>(mask_storage.data().get());
    j_mask_data.CreateArrayElem().SetFalse();

    auto j_mask_shape = j_mask.CreateMember("shape");
    j_mask_shape.SetArray();
    j_mask_shape.CreateArrayElem() = static_cast<int64_t>(kRows);

    j_mask.CreateMember("typestr") = "|i1";
  }

  std::string str = column_arr.Dump<experimental::JsonWriter>();
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
  ASSERT_EQ(source->info.num_row_, kRows);
}

TEST(SimpleCSRSource, FromColumnarSparse) {
  constexpr size_t kRows = 32;
  constexpr size_t kCols = 2;
  RBitField8::value_type constexpr kUCOne = 1;

  std::vector<dh::device_vector<float>> columns_data(kCols);
  std::vector<dh::device_vector<RBitField8::value_type>> column_bitfields(kCols);

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

  experimental::Document column_arr(experimental::ValueKind::kArray);
  auto& j_columns = column_arr.GetValue();

  for (size_t c = 0; c < kCols; ++c) {
    auto column = j_columns.CreateArrayElem();
    column.SetObject();
    column.CreateMember("version") = static_cast<int64_t>(1);
    column.CreateMember("typestr") = "<f4";
    auto p_d_data = raw_pointer_cast(columns_data[c].data());
    auto j_data = column.CreateMember("data");
    j_data.SetArray();
    j_data.CreateArrayElem() = reinterpret_cast<int64_t>(p_d_data);
    j_data.CreateArrayElem().SetFalse();

    auto j_shape = column.CreateMember("shape");
    j_shape.SetArray().CreateArrayElem().SetInteger(static_cast<int64_t>(kRows));

    column.CreateMember("version") = static_cast<int64_t>(1);
    column.CreateMember("typestr") = "<f4";

    // mask
    auto j_mask = column.CreateMember("mask");
    j_mask.SetObject();
    j_mask.CreateMember("version") = static_cast<int64_t>(1);
    auto j_mask_data = j_mask.CreateMember("data");
    j_mask_data.SetArray();
    j_mask_data.CreateArrayElem() =
        reinterpret_cast<int64_t>(column_bitfields[c].data().get());
    j_mask_data.CreateArrayElem().SetFalse();

    auto j_mask_shape = j_mask.CreateMember("shape");
    j_mask_shape.SetArray();
    j_mask_shape.CreateArrayElem() = static_cast<int64_t>(kRows);
    j_mask.CreateMember("typestr") = "|i1";
  }

  std::string str = column_arr.Dump<experimental::JsonWriter>();

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

TEST(SimpleCSRSource, Types) {
  // Test with different types of different size
  constexpr size_t kRows {16};
  constexpr size_t kCols {2};

  thrust::device_vector<double> d_data_0(kRows);
  thrust::device_vector<uint32_t> d_data_1(kRows);

  experimental::Document columns(experimental::ValueKind::kArray);
  GenerateDenseColumn<double>(&(columns.GetValue()), "<f8", kRows, &d_data_0);
  GenerateDenseColumn<uint32_t>(&(columns.GetValue()), "<u4", kRows, &d_data_1);

  std::string str = columns.Dump<experimental::JsonWriter>();

  std::unique_ptr<data::SimpleCSRSource> source (new data::SimpleCSRSource());
  source->CopyFrom(str.c_str(), false);
  TestDenseColumn(source, kRows, kCols);
}
}  // namespace xgboost