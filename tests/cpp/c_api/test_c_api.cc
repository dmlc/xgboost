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
  std::shared_ptr<xgboost::DMatrix> *dmat =
      static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);
  xgboost::MetaInfo &info = (*dmat)->Info();
  ASSERT_EQ(info.num_col_, 2);
  ASSERT_EQ(info.num_row_, 3);
  ASSERT_EQ(info.num_nonzero_, 6);

  for (const auto &batch : (*dmat)->GetBatches<xgboost::SparsePage>()) {
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

    for (const auto &batch : (*dmat)->GetBatches<xgboost::SparsePage>()) {
      for (size_t i = 0; i < batch.Size(); i++) {
        auto inst = batch[i];
        for (auto e : inst) {
          ASSERT_EQ(e.fvalue, 1.5);
        }
      }
    }
    delete dmat;
  }
}

TEST(c_api, XGDMatrixCopyDataToCSR) {
  auto origin_thread_num = omp_get_num_threads();
  size_t constexpr kThreads { 16 };
  omp_set_num_threads(kThreads);

  std::vector<size_t> row_ptr = {0, 2, 5, 8, 10, 11, 15};
  std::vector<unsigned> indices = {0, 2, 1, 3, 4, 0, 1, 2, 2, 4, 4, 1, 2, 3, 4};
  std::vector<float> data (row_ptr.back());
  for (int i = 0; i < row_ptr.back(); ++i) {
    data[i] = static_cast<float>(i);
  }
  DMatrixHandle handle;
  XGDMatrixCreateFromCSREx(row_ptr.data(), indices.data(), data.data(),
                           7, 15, 5, &handle);
  std::shared_ptr<xgboost::DMatrix> *dmat =
      static_cast<std::shared_ptr<xgboost::DMatrix> *>(handle);

  std::vector<size_t> row_ptr_((*dmat)->Info().num_row_ + 1);
  std::vector<unsigned> indices_((*dmat)->Info().num_nonzero_);
  std::vector<float> data_((*dmat)->Info().num_nonzero_);
  auto* row_ptr_impl = row_ptr_.data();
  auto* indices_impl = indices_.data();
  auto* data_impl = data_.data();
  XGDMatrixCopyDataToCSR(handle, &row_ptr_impl, &indices_impl, &data_impl);

  for (int i = 0; i < row_ptr.size(); ++i) {
    ASSERT_EQ(row_ptr[i], row_ptr_[i]);
  }
  for (size_t i = 0; i < row_ptr.back(); ++i) {
    ASSERT_EQ(indices[i], indices_[i]);
    ASSERT_EQ(data[i], data_[i]);
  }

  delete dmat;
  omp_set_num_threads(origin_thread_num);
}
