// Copyright by Contributors
#include <dmlc/filesystem.h>
#include <xgboost/data.h>
#include "../../../src/data/simple_dmatrix.h"

#include <thrust/sequence.h>
#include "../../../src/data/device_adapter.cuh"
#include "../helpers.h"
#include "test_array_interface.h"
#include "../../../src/data/array_interface.h"

using namespace xgboost;  // NOLINT

TEST(SimpleDMatrix, FromColumnarDenseBasic) {
  constexpr size_t kRows{16};
  std::vector<Json> columns;
  thrust::device_vector<double> d_data_0(kRows);
  thrust::device_vector<uint32_t> d_data_1(kRows);

  columns.emplace_back(GenerateDenseColumn<double>("<f8", kRows, &d_data_0));
  columns.emplace_back(GenerateDenseColumn<uint32_t>("<u4", kRows, &d_data_1));

  Json column_arr{columns};

  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();

  data::CudfAdapter adapter(str);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           -1);
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 16);
  EXPECT_EQ(dmat.Info().num_nonzero_, 32);
}

void TestDenseColumn(DMatrix* dmat, size_t n_rows, size_t n_cols) {
  for (auto& batch : dmat->GetBatches<SparsePage>()) {
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = batch[i];
      for (auto j = 0ull; j < inst.size(); j++) {
        EXPECT_EQ(inst[j].fvalue, i * 2);
        EXPECT_EQ(inst[j].index, j);
      }
    }
  }
  ASSERT_EQ(dmat->Info().num_row_, n_rows);
  ASSERT_EQ(dmat->Info().num_col_, n_cols);
}

TEST(SimpleDMatrix, FromColumnarDense) {
  constexpr size_t kRows{16};
  constexpr size_t kCols{2};
  std::vector<Json> columns;
  thrust::device_vector<float> d_data_0(kRows);
  thrust::device_vector<int32_t> d_data_1(kRows);
  columns.emplace_back(GenerateDenseColumn<float>("<f4", kRows, &d_data_0));
  columns.emplace_back(GenerateDenseColumn<int32_t>("<i4", kRows, &d_data_1));

  Json column_arr{columns};

  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();

  // no missing value
  {
    data::CudfAdapter adapter(str);
    data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                             -1);
    TestDenseColumn(&dmat, kRows, kCols);
  }

  // with missing value specified
  {
    data::CudfAdapter adapter(str);
    data::SimpleDMatrix dmat(&adapter, 4.0, -1);

    ASSERT_EQ(dmat.Info().num_row_, kRows);
    ASSERT_EQ(dmat.Info().num_col_, kCols);
    ASSERT_EQ(dmat.Info().num_nonzero_, kCols * kRows - 2);
  }

  {
    // no missing value, but has NaN
    d_data_0[3] = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(std::isnan(d_data_0[3]));  // removes 6.0
    data::CudfAdapter adapter(str);
    data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                             -1);
    ASSERT_EQ(dmat.Info().num_nonzero_, kRows * kCols - 1);
    ASSERT_EQ(dmat.Info().num_row_, kRows);
    ASSERT_EQ(dmat.Info().num_col_, kCols);
  }
}

TEST(SimpleDMatrix, FromColumnarWithEmptyRows) {
  constexpr size_t kRows = 102;
  constexpr size_t kCols = 24;

  std::vector<Json> v_columns(kCols);
  std::vector<dh::device_vector<float>> columns_data(kCols);
  std::vector<dh::device_vector<RBitField8::value_type>> column_bitfields(
      kCols);

  RBitField8::value_type constexpr kUCOne = 1;

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
    std::vector<Json> j_data{
        Json(Integer(reinterpret_cast<Integer::Int>(p_d_data))),
        Json(Boolean(false))};
    col["data"] = j_data;
    std::vector<Json> j_shape{Json(Integer(static_cast<Integer::Int>(kRows)))};
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
    std::set<size_t> missing_row_index{0, 1, last_ind};

    for (size_t j = 0; j < mask_storage.size(); ++j) {
      if (missing_row_index.find(j) == missing_row_index.cend()) {
        // all other rows are valid
        mask_storage[j] = ~0;
      }
    }

    j_mask["data"] = std::vector<Json>{
        Json(
            Integer(reinterpret_cast<Integer::Int>(mask_storage.data().get()))),
        Json(Boolean(false))};
    j_mask["shape"] = Array(
        std::vector<Json>{Json(Integer(static_cast<Integer::Int>(kRows)))});
    j_mask["typestr"] = String("|i1");
  }

  Json column_arr{Array(v_columns)};
  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();
  data::CudfAdapter adapter(str);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           -1);

  for (auto& batch : dmat.GetBatches<SparsePage>()) {
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = batch[i];
      for (auto j = 0ull; j < inst.size(); j++) {
        EXPECT_EQ(inst[j].fvalue, i);
        EXPECT_EQ(inst[j].index, j);
      }
    }
  }
  ASSERT_EQ(dmat.Info().num_nonzero_, (kRows - 3) * kCols);
  ASSERT_EQ(dmat.Info().num_row_, kRows);
  ASSERT_EQ(dmat.Info().num_col_, kCols);
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
    j_mask["shape"] = Array(std::vector<Json>{Json(Integer(static_cast<Integer::Int>(kRows)))});
    j_mask["typestr"] = String("|i1");
  }

  Json column_arr {Array(j_columns)};

  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();

  {
    data::CudfAdapter adapter(str);
    data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(), -1);

    ASSERT_EQ(dmat.Info().num_row_, kRows);
    ASSERT_EQ(dmat.Info().num_nonzero_, (kRows*kCols)-2);
  }

  {
    data::CudfAdapter adapter(str);
    data::SimpleDMatrix dmat(&adapter, 2.0, -1);
    for (auto& batch : dmat.GetBatches<SparsePage>()) {
      for (auto i = 0ull; i < batch.Size(); i++) {
        auto inst = batch[i];
        for (auto e : inst) {
          ASSERT_NE(e.fvalue, 2.0);
        }
      }
    }
  }

  {
    // no missing value, but has NaN
    data::CudfAdapter adapter(str);
    columns_data[0][4] = std::numeric_limits<float>::quiet_NaN();  // 0^th column 4^th row
    data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                             -1);
    ASSERT_TRUE(std::isnan(columns_data[0][4]));

    // Two invalid entries and one NaN, in CSC
    // 0^th column: 0, 1, 4, 5, 6, ..., kRows
    // 1^th column: 0, 1, 2, 3, ..., 19, 21, ..., kRows
    ASSERT_EQ(dmat.Info().num_nonzero_, kRows * kCols - 3);
  }
}


TEST(SimpleDMatrix, FromColumnarSparseBasic) {
  constexpr size_t kRows{16};
  std::vector<Json> columns;
  thrust::device_vector<double> d_data_0(kRows);
  thrust::device_vector<uint32_t> d_data_1(kRows);

  columns.emplace_back(GenerateSparseColumn<double>("<f8", kRows, &d_data_0));
  columns.emplace_back(GenerateSparseColumn<uint32_t>("<u4", kRows, &d_data_1));

  Json column_arr{columns};

  std::stringstream ss;
  Json::Dump(column_arr, &ss);
  std::string str = ss.str();

  data::CudfAdapter adapter(str);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(),
                           -1);
  EXPECT_EQ(dmat.Info().num_col_, 2);
  EXPECT_EQ(dmat.Info().num_row_, 16);
  EXPECT_EQ(dmat.Info().num_nonzero_, 32);

  for (auto& batch : dmat.GetBatches<SparsePage>()) {
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = batch[i];
      for (auto j = 0ull; j < inst.size(); j++) {
        EXPECT_EQ(inst[j].fvalue, i * 2);
        EXPECT_EQ(inst[j].index, j);
      }
    }
  }
}


TEST(SimpleDMatrix, FromCupy){
  int rows = 50;
  int cols = 10;
  thrust::device_vector< float> data(rows*cols);
  auto json_array_interface = Generate2dArrayInterface(rows, cols, "<f4", &data);
  std::stringstream ss;
  Json::Dump(json_array_interface, &ss);
  std::string str = ss.str();
  data::CupyAdapter adapter(str);
  data::SimpleDMatrix dmat(&adapter, -1, 1);
  EXPECT_EQ(dmat.Info().num_col_, cols);
  EXPECT_EQ(dmat.Info().num_row_, rows);
  EXPECT_EQ(dmat.Info().num_nonzero_, rows*cols);

  for (auto& batch : dmat.GetBatches<SparsePage>()) {
    for (auto i = 0ull; i < batch.Size(); i++) {
      auto inst = batch[i];
      for (auto j = 0ull; j < inst.size(); j++) {
        EXPECT_EQ(inst[j].fvalue, i * cols + j);
        EXPECT_EQ(inst[j].index, j);
      }
    }
  }
}

TEST(SimpleDMatrix, FromCupySparse){
  int rows = 2;
  int cols = 2;
  thrust::device_vector< float> data(rows*cols);
  auto json_array_interface = Generate2dArrayInterface(rows, cols, "<f4", &data);
  data[1] = std::numeric_limits<float>::quiet_NaN();
  data[2] = std::numeric_limits<float>::quiet_NaN();
  std::stringstream ss;
  Json::Dump(json_array_interface, &ss);
  std::string str = ss.str();
  data::CupyAdapter adapter(str);
  data::SimpleDMatrix dmat(&adapter, -1, 1);
  EXPECT_EQ(dmat.Info().num_col_, cols);
  EXPECT_EQ(dmat.Info().num_row_, rows);
  EXPECT_EQ(dmat.Info().num_nonzero_, rows * cols - 2);
  auto& batch = *dmat.GetBatches<SparsePage>().begin();
  auto inst0 = batch[0];
  auto inst1 = batch[1];
  EXPECT_EQ(batch[0].size(), 1);
  EXPECT_EQ(batch[1].size(), 1);
  EXPECT_EQ(batch[0][0].fvalue, 0.0f);
  EXPECT_EQ(batch[0][0].index, 0);
  EXPECT_EQ(batch[1][0].fvalue, 3.0f);
  EXPECT_EQ(batch[1][0].index, 1);
}
