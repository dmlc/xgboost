// Copyright by Contributors
#include <gtest/gtest.h>
#include <xgboost/c_api.h>
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
  std::shared_ptr<xgboost::DMatrix> dmat =
      *static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  xgboost::MetaInfo &info = dmat->Info();
  ASSERT_EQ(info.num_col_, 2);
  ASSERT_EQ(info.num_row_, 3);
  ASSERT_EQ(info.num_nonzero_, 6);

  auto iter = dmat->RowIterator();
  iter->BeforeFirst();
  while (iter->Next()) {
    auto batch = iter->Value();
    ASSERT_EQ(batch[0][0].fvalue, 0.0f);
    ASSERT_EQ(batch[0][1].fvalue, -4.0f);
    ASSERT_EQ(batch[2][0].fvalue, 3.0f);
    ASSERT_EQ(batch[2][1].fvalue, 0.0f);
  }
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

    std::shared_ptr<xgboost::DMatrix> dmat =
        *static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
    xgboost::MetaInfo &info = dmat->Info();
    ASSERT_EQ(info.num_col_, num_cols);
    ASSERT_EQ(info.num_row_, row);
    ASSERT_EQ(info.num_nonzero_, num_cols * row - num_missing);

    auto iter = dmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      auto batch = iter->Value();
      for (int i = 0; i < batch.Size(); i++) {
        auto inst = batch[i];
        for (int j = 0; i < inst.length; i++) {
          ASSERT_EQ(inst[j].fvalue, 1.5);
        }
      }
    }
  }
}