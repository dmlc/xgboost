// Copyright by Contributors 2019
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include "../../../src/c_api/c_api.cu"

using namespace xgboost;

TEST(c_api, XGDMatrixCreateFromCUDF) {
  size_t num_columns = 2;
  int num_rows = 5;
  std::vector<cudf_interchange_column> cudf(num_columns);
  std::vector<cudf_interchange_column *> column_ptrs(num_columns);
  for (auto i = 0ull; i < cudf.size(); i++) {
    column_ptrs.at(i) = &cudf.at(i);
  }
  thrust::device_vector<int8_t> col0(num_rows, 3);
  cudf[0].size = num_rows;
  cudf[0].data = col0.data().get();
  cudf[0].dtype = cudf_interchange_dtype::INT8;
  cudf[0].null_count = 0;
  cudf[0].valid = nullptr;

  thrust::device_vector<double> col1(num_rows, 5.0);
  cudf[1].size = num_rows;
  cudf[1].data = col1.data().get();
  cudf[1].dtype = cudf_interchange_dtype::FLOAT64;
  cudf[1].null_count = 2;
  std::vector<cudf_interchange_valid_type> h_valid = {
      255 ^ 3};  // First two elements missing
  thrust::device_vector<cudf_interchange_valid_type> valid = h_valid;
  cudf[1].valid = valid.data().get();

  DMatrixHandle handle;
  xgboost::XGDMatrixCreateFromCUDF(
      reinterpret_cast<void **>(column_ptrs.data()), num_columns, &handle);
  std::shared_ptr<xgboost::DMatrix> *dmat =
      static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  xgboost::MetaInfo &info = (*dmat)->Info();
  ASSERT_EQ(info.num_col_, num_columns);
  ASSERT_EQ(info.num_row_, num_rows);
  ASSERT_EQ(info.num_nonzero_, 8);

  auto &batch = *(*dmat)->GetRowBatches().begin();
  ASSERT_EQ(batch[0][0].fvalue, 3.0f);
  // First two rows have missing values in the second column
  ASSERT_EQ(batch[0].size(), 1);
  ASSERT_EQ(batch[1].size(), 1);
  ASSERT_EQ(batch[2].size(), 2);

  ASSERT_EQ(batch[3][1].fvalue, 5.0f);
  ASSERT_EQ(batch[4][0].fvalue, 3.0f);
}

TEST(c_api, XGDMatrixSetCUDFInfo) {
  size_t num_columns = 1;
  int num_rows = 5;
  std::vector<cudf_interchange_column> cudf(num_columns);
  std::vector<cudf_interchange_column *> column_ptrs(num_columns);
  for (auto i = 0ull; i < cudf.size(); i++) {
    column_ptrs.at(i) = &cudf.at(i);
  }
  thrust::device_vector<float> h_col0(num_rows, 3);
  thrust::device_vector<float> col0 = h_col0;
  cudf[0].size = num_rows;
  cudf[0].data = col0.data().get();
  cudf[0].dtype = cudf_interchange_dtype::FLOAT32;
  cudf[0].null_count = 0;
  cudf[0].valid = nullptr;

  DMatrixHandle handle;
  // create a dmatrix
  xgboost::XGDMatrixCreateFromCUDF(
      reinterpret_cast<void **>(column_ptrs.data()), num_columns, &handle);
  // set labels
  xgboost::XGDMatrixSetCUDFInfo(handle, "label",
                                reinterpret_cast<void **>(column_ptrs.data()),
                                num_columns);
  // set weights
  xgboost::XGDMatrixSetCUDFInfo(handle, "weight",
                                reinterpret_cast<void **>(column_ptrs.data()),
                                num_columns);
  std::shared_ptr<xgboost::DMatrix> *dmat =
      static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  xgboost::MetaInfo &info = (*dmat)->Info();
  ASSERT_EQ(info.labels_.HostVector(), h_col0);
  ASSERT_EQ(info.weights_.HostVector(), h_col0);
}
