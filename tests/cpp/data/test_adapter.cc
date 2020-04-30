// Copyright (c) 2019 by Contributors
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
  EXPECT_EQ(line1 .GetElement(0).value, 3);
  EXPECT_EQ(line1 .GetElement(1).value, 4);
  auto line2 = batch.GetLine(2);
  EXPECT_EQ(line2 .GetElement(0).value, 5);
  EXPECT_EQ(line2 .GetElement(0).row_idx, 2);
  EXPECT_EQ(line2 .GetElement(0).column_idx, 1);
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
  auto inst = batch[0];
  EXPECT_EQ(inst[0].fvalue, 1);
  EXPECT_EQ(inst[0].index, 0);
  EXPECT_EQ(inst[1].fvalue, 3);
  EXPECT_EQ(inst[1].index, 1);
  EXPECT_EQ(inst[2].fvalue, 5);
  EXPECT_EQ(inst[2].index, 2);
  EXPECT_EQ(inst[3].fvalue, 7);
  EXPECT_EQ(inst[3].index, 3);

  inst = batch[1];
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
      row_ptr_ {0, 2, 4, 5};
  size_t iter_ {0};

 public:
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
    batch.size = 3;

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

TEST(Adapter, IteratorAdaper) {
  CSRIterForTest iter;
  data::IteratorAdapter adapter{&iter, CSRSetDataNextForTest};
  constexpr size_t kRows { 6 };

  std::unique_ptr<DMatrix> data {
    DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)
  };
  ASSERT_EQ(data->Info().num_col_, CSRIterForTest::kCols);
  ASSERT_EQ(data->Info().num_row_, kRows);
}

class CudaArrayIterForTest {
  std::vector< std::string > arrays_;
  std::vector< HostDeviceVector<float> > storages_;
  size_t it_ { 0 };
  DataHolderHandle* adapter_;

 public:
  static constexpr size_t kRows = 128;
  static constexpr size_t kCols = 64;
  static constexpr size_t kParts = 4;

  explicit CudaArrayIterForTest(DataHolderHandle* adapter) : adapter_{adapter} {
    size_t rows_per_partition = kRows / kParts;
    auto rng = RandomDataGenerator{rows_per_partition, kCols, 0}.Device(0);
    storages_.resize(kParts);
    for (size_t i = 0; i < kParts; ++i) {
      arrays_.emplace_back(rng.GenerateArrayInterface(&storages_[i]));
    }
  }

  int Next(CudaArrayInterfaceCallBackSetData set) {
    if (it_ == kParts) {
      return false;
    } else {
      auto interface = arrays_[it_];
      set(*adapter_, interface.c_str());
      it_++;
      return true;
    }
  }

  void Reset() { it_ = 0; }
};

void Reset(DataIterHandle iter) {
  static_cast<CudaArrayIterForTest*>(iter)->Reset();
}

int CudaArrayIterNextForTest(DataIterHandle iter,
                             CudaArrayInterfaceCallBackSetData set) {
  std::string interface;
  auto code = static_cast<CudaArrayIterForTest*>(iter)->Next(set);
  return code;
}

TEST(Adapter, CudaArrayCallbackAdapter) {
  DataHolderHandle holder;
  CudaArrayIterForTest iter{&holder};

  data::CudaArrayInterfaceCallbackAdapter adapter(
      Reset, CudaArrayIterNextForTest, &iter, 0, &holder);
}

}  // namespace xgboost
