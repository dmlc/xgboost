// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
#include <xgboost/data.h>
#include <xgboost/data.h>

TEST(c_api, XGDMatrixCreateFromMatDT) {
  std::vector<int> col0 = {0, -1, 3};
  std::vector<float> col1 = {-4.0f, 2.0f, 0.0f};
  const char *col0_type = "int32";
  const char *col1_type = "float32";
  std::vector<void *> data = {col0.data(), col1.data()};
  std::vector<const char *> types = {col0_type, col1_type};
  DMatrixHandle handle;
  XGDMatrixCreateFromDT(data.data(), types.data(), 3, 2, &handle,
                        0);
  std::shared_ptr<xgboost::DMatrix> *dmat =
      static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  xgboost::MetaInfo &info = (*dmat)->Info();
  ASSERT_EQ(info.num_col_, 2);
  ASSERT_EQ(info.num_row_, 3);
  ASSERT_EQ(info.num_nonzero_, 6);

  for (const auto &batch : (*dmat)->GetRowBatches()) {
    ASSERT_EQ(batch[0][0].fvalue, 0.0f);
    ASSERT_EQ(batch[0][1].fvalue, -4.0f);
    ASSERT_EQ(batch[2][0].fvalue, 3.0f);
    ASSERT_EQ(batch[2][1].fvalue, 0.0f);
  }

  delete dmat;
}

TEST(c_api, XGDMatrixCreateFromMat_omp) {
  std::vector<int> num_rows = {100, 11374, 15000};
  for (auto row : num_rows) {
    int num_cols = 50;
    int num_missing = 5;
    DMatrixHandle handle;
    std::vector<float> data(num_cols * row, 1.5);
    for (int i = 0; i < num_missing; i++) {
      data[i] = std::numeric_limits<float>::quiet_NaN();
    }

    XGDMatrixCreateFromMat_omp(data.data(), row, num_cols,
                               std::numeric_limits<float>::quiet_NaN(), &handle,
                               0);

    std::shared_ptr<xgboost::DMatrix> *dmat =
        static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
    xgboost::MetaInfo &info = (*dmat)->Info();
    ASSERT_EQ(info.num_col_, num_cols);
    ASSERT_EQ(info.num_row_, row);
    ASSERT_EQ(info.num_nonzero_, num_cols * row - num_missing);

    for (const auto &batch : (*dmat)->GetRowBatches()) {
      for (int i = 0; i < batch.Size(); i++) {
        auto inst = batch[i];
        for (int j = 0; i < inst.size(); i++) {
          ASSERT_EQ(inst[j].fvalue, 1.5);
        }
      }
    }
    delete dmat;
  }
}

TEST(c_api, XGDMatrixCreateFromDataSource) {
  DataSourceHandle handle;
  XGDataSourceCreate(0, 0, &handle);
  // test multiple data appending
  const std::vector<size_t>& indptr = {0, 2, 3, 6};
  const std::vector<unsigned>& indices = {1, 3, 0, 2, 4, 3};
  const std::vector<float>& data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  XGDataSourceAppendData(indptr.data(), indices.data(), data.data(),
                         indptr.size(),
                         handle);
  const std::vector<size_t>& indptr1 = {0, 3, 4};
  const std::vector<unsigned>& indices1 = {0, 2, 9, 0};
  const std::vector<float>& data1 = {-1.0f, -2.0f, -3.0f, -4.0f};
  XGDataSourceAppendData(indptr1.data(), indices1.data(), data1.data(),
                         indptr1.size(),
                         handle);
  auto *ds = static_cast<std::unique_ptr<xgboost::DataSource> *>(handle);
  xgboost::MetaInfo &ds_info = (*ds)->info;
  ASSERT_EQ(ds_info.num_col_, 10);
  ASSERT_EQ(ds_info.num_row_, 5);
  ASSERT_EQ(ds_info.num_nonzero_, 10);

  DMatrixHandle mat_handle;
  XGDMatrixCreateFromDataSource(handle, &mat_handle);
  auto *dmat = static_cast<std::shared_ptr<xgboost::DMatrix> *>(mat_handle);
  xgboost::MetaInfo &info = (*dmat)->Info();
  ASSERT_EQ(info.num_col_, 10);
  ASSERT_EQ(info.num_row_, 5);
  ASSERT_EQ(info.num_nonzero_, 10);
  for (const auto &batch : (*dmat)->GetRowBatches()) {
    ASSERT_EQ(batch[0][0].index, 1);
    ASSERT_EQ(batch[0][1].index, 3);
    ASSERT_EQ(batch[1][0].index, 0);
    ASSERT_EQ(batch[2][0].index, 2);
    ASSERT_EQ(batch[2][1].index, 4);
    ASSERT_EQ(batch[2][2].index, 3);
    ASSERT_EQ(batch[3][0].index, 0);
    ASSERT_EQ(batch[3][1].index, 2);
    ASSERT_EQ(batch[3][2].index, 9);
    ASSERT_EQ(batch[4][0].index, 0);
    ASSERT_EQ(batch[0][0].fvalue, 0.0f);
    ASSERT_EQ(batch[0][1].fvalue, 1.0f);
    ASSERT_EQ(batch[1][0].fvalue, 2.0f);
    ASSERT_EQ(batch[2][0].fvalue, 3.0f);
    ASSERT_EQ(batch[2][1].fvalue, 4.0f);
    ASSERT_EQ(batch[2][2].fvalue, 5.0f);
    ASSERT_EQ(batch[3][0].fvalue, -1.0f);
    ASSERT_EQ(batch[3][1].fvalue, -2.0f);
    ASSERT_EQ(batch[3][2].fvalue, -3.0f);
    ASSERT_EQ(batch[4][0].fvalue, -4.0f);
  }

  delete ds;
  delete dmat;
}

