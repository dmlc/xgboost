// Copyright (c) 2019-2021 by XGBoost Contributors
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>
#include <xgboost/data.h>
#include "../../../src/data/adapter.h"
#include "../../../src/data/simple_dmatrix.h"
#include "../../../src/common/timer.h"
#include "../helpers.h"

#include "xgboost/base.h"
#include "xgboost/c_api.h"

namespace xgboost {
TEST(Adapter, CSRAdapter) {
  int n = 2;
  std::vector<float> data = {1, 2, 3, 4, 5};
  std::vector<unsigned> feature_idx = {0, 1, 0, 1, 1};
  std::vector<size_t> row_ptr = {0, 2, 4, 5};
  data::CSRAdapter adapter(row_ptr.data(), feature_idx.data(), data.data(),
                           row_ptr.size() - 1, data.size(), n);
  adapter.Next();
  auto & batch = adapter.Value();
  auto line0 = batch.GetLine(0);
  EXPECT_EQ(line0.GetElement(0).value, 1);
  EXPECT_EQ(line0.GetElement(1).value, 2);

  auto line1 = batch.GetLine(1);
  EXPECT_EQ(line1.GetElement(0).value, 3);
  EXPECT_EQ(line1.GetElement(1).value, 4);

  auto line2 = batch.GetLine(2);
  EXPECT_EQ(line2.GetElement(0).value, 5);
  EXPECT_EQ(line2.GetElement(0).row_idx, 2);
  EXPECT_EQ(line2.GetElement(0).column_idx, 1);
}

TEST(Adapter, CSRArrayAdapter) {
  HostDeviceVector<bst_row_t> indptr;
  HostDeviceVector<float> values;
  HostDeviceVector<bst_feature_t> indices;
  size_t n_features = 100, n_samples = 10;
  RandomDataGenerator{n_samples, n_features, 0.5}.GenerateCSR(&values, &indptr, &indices);
  using linalg::MakeVec;
  auto indptr_arr = ArrayInterfaceStr(MakeVec(indptr.HostPointer(), indptr.Size()));
  auto values_arr = ArrayInterfaceStr(MakeVec(values.HostPointer(), values.Size()));
  auto indices_arr = ArrayInterfaceStr(MakeVec(indices.HostPointer(), indices.Size()));
  auto adapter = data::CSRArrayAdapter(
      StringView{indptr_arr.c_str(), indptr_arr.size()},
      StringView{values_arr.c_str(), values_arr.size()},
      StringView{indices_arr.c_str(), indices_arr.size()}, n_features);
  auto batch = adapter.Value();
  ASSERT_EQ(batch.NumRows(), n_samples);
  ASSERT_EQ(batch.NumCols(), n_features);

  ASSERT_EQ(adapter.NumRows(), n_samples);
  ASSERT_EQ(adapter.NumColumns(), n_features);
}

TEST(Adapter, CSCAdapterColsMoreThanRows) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<unsigned> row_idx = {0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<size_t> col_ptr = {0, 2, 4, 6, 8};
  // Infer row count
  data::CSCAdapter adapter(col_ptr.data(), row_idx.data(), data.data(), 4, 0);
  data::SimpleDMatrix dmat(&adapter, std::numeric_limits<float>::quiet_NaN(), -1);
  EXPECT_EQ(dmat.Info().num_col_, 4);
  EXPECT_EQ(dmat.Info().num_row_, 2);
  EXPECT_EQ(dmat.Info().num_nonzero_, 8);

  auto &batch = *dmat.GetBatches<SparsePage>().begin();
  auto page = batch.GetView();
  auto inst = page[0];
  EXPECT_EQ(inst[0].fvalue, 1);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 3);
  EXPECT_EQ(inst[1].index, 1);
  EXPECT_EQ(inst[2].fvalue, 5);
  EXPECT_EQ(inst[2].index, 2);
  EXPECT_EQ(inst[3].fvalue, 7);
  EXPECT_EQ(inst[3].index, 3);

  inst = page[1];
  EXPECT_EQ(inst[0].fvalue, 2);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 4);
  EXPECT_EQ(inst[1].index, 1);
  EXPECT_EQ(inst[2].fvalue, 6);
  EXPECT_EQ(inst[2].index, 2);
  EXPECT_EQ(inst[3].fvalue, 8);
  EXPECT_EQ(inst[3].index, 3);
}

// A mock for JVM data iterator.
class CSRIterForTest {
  std::vector<float> data_ {1, 2, 3, 4, 5};
  std::vector<std::remove_pointer<decltype(std::declval<XGBoostBatchCSR>().index)>::type>
      feature_idx_ {0, 1, 0, 1, 1};
  std::vector<std::remove_pointer<decltype(std::declval<XGBoostBatchCSR>().offset)>::type>
      row_ptr_ {0, 2, 4, 5, 5};
  size_t iter_ {0};

 public:
  size_t static constexpr kRows { 4 };  // Test for the last row being empty
  size_t static constexpr kCols { 13 };  // Test for having some missing columns

  XGBoostBatchCSR Next() {
    for (auto& v : data_) {
      v += iter_;
    }
    XGBoostBatchCSR batch;
    batch.columns = 2;
    batch.offset = dmlc::BeginPtr(row_ptr_);
    batch.index = dmlc::BeginPtr(feature_idx_);
    batch.value = dmlc::BeginPtr(data_);
    batch.size = kRows;

    batch.label = nullptr;
    batch.weight = nullptr;

    iter_++;

    return batch;
  }
  size_t Iter() const { return iter_; }
};

size_t constexpr CSRIterForTest::kCols;

int CSRSetDataNextForTest(DataIterHandle data_handle,
                          XGBCallbackSetData *set_function,
                          DataHolderHandle set_function_handle) {
  size_t constexpr kIters { 2 };
  auto iter = static_cast<CSRIterForTest *>(data_handle);
  if (iter->Iter() < kIters) {
    auto batch = iter->Next();
    batch.columns = CSRIterForTest::kCols;
    set_function(set_function_handle, batch);
    return 1;
  } else {
    return 0;  // stoping condition
  }
}

TEST(Adapter, IteratorAdapter) {
  CSRIterForTest iter;
  data::IteratorAdapter<DataIterHandle, XGBCallbackDataIterNext,
                        XGBoostBatchCSR> adapter{&iter, CSRSetDataNextForTest};
  constexpr size_t kRows { 8 };

  std::unique_ptr<DMatrix> data {
    DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)
  };
  ASSERT_EQ(data->Info().num_col_, CSRIterForTest::kCols);
  ASSERT_EQ(data->Info().num_row_, kRows);
  int num_batch = 0;
  for (auto const& batch : data->GetBatches<SparsePage>()) {
    ASSERT_EQ(batch.offset.HostVector(), std::vector<bst_row_t>({0, 2, 4, 5, 5, 7, 9, 10, 10}));
    ++num_batch;
  }
  ASSERT_EQ(num_batch, 1);
}
}  // namespace xgboost
