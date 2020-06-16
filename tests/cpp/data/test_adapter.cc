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
  EXPECT_EQ(line1.GetElement(0).value, 3);
  EXPECT_EQ(line1.GetElement(1).value, 4);

  auto line2 = batch.GetLine(2);
  EXPECT_EQ(line2.GetElement(0).value, 5);
  EXPECT_EQ(line2.GetElement(0).row_idx, 2);
  EXPECT_EQ(line2.GetElement(0).column_idx, 1);
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

#if defined(XGBOOST_BUILD_ARROW_SUPPORT)
struct ArrowTestInputBuilder {
  arrow::ChunkedArrayVector columns{};
  std::vector<std::vector<float>> cols_in_float{};
  std::vector<float> raveled{};
  int num_rows{}, num_cols{};

  template<typename Type>
  void AddColumn(
          const std::vector<std::vector<typename Type::c_type>>& chunks) {
    arrow::ArrayVector arrvec{};
    std::vector<float> col;
    auto f = [&](const std::vector<typename Type::c_type>& chunk) {
      arrow::NumericBuilder<Type> builder;
      builder.Resize(chunk.size());
      ASSERT_TRUE(builder.AppendValues(chunk).ok());
      std::shared_ptr<arrow::Array> arr;
      ASSERT_TRUE(builder.Finish(&arr).ok());
      arrvec.push_back(arr);
      col.insert(col.end(), chunk.cbegin(), chunk.cend());
    };
    std::for_each(chunks.begin(), chunks.end(), f);
    columns.push_back(std::make_shared<arrow::ChunkedArray>(arrvec));
    cols_in_float.push_back(std::move(col));
  }

  template<typename Type>
  void AddColumn(
          const std::vector<std::pair<
                                      std::vector<typename Type::c_type>,
                                      std::vector<bool>
                                     >>& chunks) {
    arrow::ArrayVector arrvec{};
    std::vector<float> col;
    auto f = [&](const std::pair<
                                  std::vector<typename Type::c_type>,
                                  std::vector<bool>
                                >& chunk) {
      arrow::NumericBuilder<Type> builder;
      builder.Resize(chunk.first.size());
      ASSERT_TRUE(builder.AppendValues(chunk.first, chunk.second).ok());
      std::shared_ptr<arrow::Array> arr;
      ASSERT_TRUE(builder.Finish(&arr).ok());
      arrvec.push_back(arr);
      col.insert(col.end(), chunk.first.cbegin(), chunk.first.cend());
    };
    std::for_each(chunks.begin(), chunks.end(), f);
    columns.push_back(std::make_shared<arrow::ChunkedArray>(arrvec));
    cols_in_float.push_back(std::move(col));
  }

  void Finish() {
    num_cols = columns.size();
    num_rows = num_cols > 0 ? columns[0]->length() : 0;
    for (int i = 0; i < num_rows; ++i) {
      for (int j = 0; j < num_cols; ++j) {
        raveled.emplace_back(cols_in_float[j][i]);
      }
    }
  }
};

TEST(Adapter, ArrowAdapterCreateDMatCorrectSize) {
  // add 3 single-chunk columns
  std::vector<uint16_t> col1{1, 2, 3, 4, 5};
  std::vector<float> col2{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<double> col3{1.2, 2.1, 3.2, 4.3, 5.4};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({col1});
  input_builder.AddColumn<arrow::FloatType>({col2});
  input_builder.AddColumn<arrow::DoubleType>({col3});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);
  EXPECT_EQ(dmat->Info().num_col_, 3);
  EXPECT_EQ(dmat->Info().num_row_, 5);
}

TEST(Adapter, ArrowAdapterCreateDMatCorrectValues) {
  // add 3 single-chunk columns
  std::vector<uint16_t> col1{1, 2, 3, 4, 5};
  std::vector<float> col2{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<double> col3{1.2, 2.1, 3.2, 4.3, 5.4};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({col1});
  input_builder.AddColumn<arrow::FloatType>({col2});
  input_builder.AddColumn<arrow::DoubleType>({col3});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterChunkedColumnsT1) {
  // add 3 2-chunk columns
  std::vector<uint16_t> col1_1{1, 2, 3, 4, 5};
  std::vector<uint16_t> col1_2{6, 7, 8};
  std::vector<float> col2_1{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<float> col2_2{6.4f, 5.3f, 4.2f};
  std::vector<double> col3_1{1.2, 2.1, 3.2, 4.3, 5.4};
  std::vector<double> col3_2{0.00123, 0.0456, 0.789};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({col1_1, col1_2});
  input_builder.AddColumn<arrow::FloatType>({col2_1, col2_2});
  input_builder.AddColumn<arrow::DoubleType>({col3_1, col3_2});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterChunkedColumnsT2) {
  // add 3 2-chunk columns
  std::vector<uint16_t> col1_1{1, 2, 3, 4, 5};
  std::vector<uint16_t> col1_2{6, 7, 8};
  std::vector<float> col2_1{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<float> col2_2{6.4f, 5.3f, 4.2f};
  std::vector<double> col3_1{1.2, 2.1, 3.2, 4.3, 5.4};
  std::vector<double> col3_2{0.00123, 0.0456, 0.789};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({col1_1, col1_2});
  input_builder.AddColumn<arrow::FloatType>({col2_2, col2_1});
  input_builder.AddColumn<arrow::DoubleType>({col3_1, col3_2});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterChunkedColumnsT3) {
  // add 3 chunked columns of different number of chunks
  std::vector<uint16_t> col1_1{1, 2, 3, 4, 5};
  std::vector<uint16_t> col1_2{6, 7, 8};
  std::vector<float> col2_1{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<float> col2_2{6.4f, 5.3f, 4.2f};
  std::vector<double> col3_1{1.2, 2.1, 3.2, 4.3, 5.4, 0.00123, 0.0456, 0.789};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({col1_1, col1_2});
  input_builder.AddColumn<arrow::FloatType>({col2_2, col2_1});
  input_builder.AddColumn<arrow::DoubleType>({col3_1});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterChunkedColumnsT4) {
  // add 3 chunked columns of different number of chunks
  std::vector<uint16_t> col1_1{1, 2, 3, 4, 5};
  std::vector<uint16_t> col1_2{6, 7, 8};
  std::vector<float> col2_1{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<float> col2_2{6.4f, 5.3f, 4.2f};
  std::vector<double> col3_1{1.2, 2.1, 3.2, 4.3, 5.4, 0.00123, 0.0456, 0.789};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({col1_1, col1_2});
  input_builder.AddColumn<arrow::FloatType>({col2_2, col2_1});
  input_builder.AddColumn<arrow::DoubleType>({col3_1});
  input_builder.AddColumn<arrow::Int32Type>({{1001}, {1002}, {1003}, {1004},
                                            {1005}, {1006}, {1007}, {1008}});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64()),
        arrow::field("col4", arrow::int32())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterInt32Columns) {
  std::default_random_engine e;
  std::uniform_int_distribution<int32_t> d(-100, 100);
  std::vector<int32_t> col1(7);
  std::vector<int32_t> col2(7);
  std::vector<int32_t> col3(7);
  std::generate(col1.begin(), col1.end(), [&]{ return d(e); });
  std::generate(col2.begin(), col2.end(), [&]{ return d(e); });
  std::generate(col3.begin(), col3.end(), [&]{ return d(e); });

  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::Int32Type>({col1});
  input_builder.AddColumn<arrow::Int32Type>({col2});
  input_builder.AddColumn<arrow::Int32Type>({col3});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::int32()),
        arrow::field("col2", arrow::int32()),
        arrow::field("col3", arrow::int32())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterInt64Columns) {
  std::default_random_engine e;
  std::uniform_int_distribution<int64_t> d(-100, 100);
  std::vector<int64_t> col1(7);
  std::vector<int64_t> col2(7);
  std::vector<int64_t> col3(7);
  std::generate(col1.begin(), col1.end(), [&]{ return d(e); });
  std::generate(col2.begin(), col2.end(), [&]{ return d(e); });
  std::generate(col3.begin(), col3.end(), [&]{ return d(e); });

  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::Int64Type>({col1});
  input_builder.AddColumn<arrow::Int64Type>({col2});
  input_builder.AddColumn<arrow::Int64Type>({col3});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::int64()),
        arrow::field("col2", arrow::int64()),
        arrow::field("col3", arrow::int64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterFloatColumns) {
  std::default_random_engine e;
  std::uniform_real_distribution<float> d(-100, 100);
  std::vector<float> col1(7);
  std::vector<float> col2(7);
  std::vector<float> col3(7);
  std::generate(col1.begin(), col1.end(), [&]{ return d(e); });
  std::generate(col2.begin(), col2.end(), [&]{ return d(e); });
  std::generate(col3.begin(), col3.end(), [&]{ return d(e); });

  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::FloatType>({col1});
  input_builder.AddColumn<arrow::FloatType>({col2});
  input_builder.AddColumn<arrow::FloatType>({col3});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::float32()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float32())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterDoubleColumns) {
  std::default_random_engine e;
  std::uniform_real_distribution<double> d(-100, 100);
  std::vector<double> col1(7);
  std::vector<double> col2(7);
  std::vector<double> col3(7);
  std::generate(col1.begin(), col1.end(), [&]{ return d(e); });
  std::generate(col2.begin(), col2.end(), [&]{ return d(e); });
  std::generate(col3.begin(), col3.end(), [&]{ return d(e); });

  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::DoubleType>({col1});
  input_builder.AddColumn<arrow::DoubleType>({col2});
  input_builder.AddColumn<arrow::DoubleType>({col3});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::float64()),
        arrow::field("col2", arrow::float64()),
        arrow::field("col3", arrow::float64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterCreateDMatWithThreads) {
  int nthread = 4;
  // add 3 chunked columns of different number of chunks
  std::vector<uint16_t> col1_1{1, 2, 3, 4, 5};
  std::vector<uint16_t> col1_2{6, 7, 8};
  std::vector<float> col2_1{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<float> col2_2{6.4f, 5.3f, 4.2f};
  std::vector<double> col3_1{1.2, 2.1, 3.2, 4.3, 5.4, 0.00123, 0.0456, 0.789};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({col1_1, col1_2});
  input_builder.AddColumn<arrow::FloatType>({col2_2, col2_1});
  input_builder.AddColumn<arrow::DoubleType>({col3_1});
  input_builder.AddColumn<arrow::Int32Type>({{1001}, {1002}, {1003}, {1004},
                                            {1005}, {1006}, {1007}, {1008}});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64()),
        arrow::field("col4", arrow::int32())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      nthread)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
}

TEST(Adapter, ArrowAdapterCreateDMatWithLabels) {
  // add 3 single-chunk columns
  std::vector<uint16_t> col1{1, 2, 3, 4, 5};
  std::vector<float> col2{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<double> col3{1.2, 2.1, 3.2, 4.3, 5.4};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({col1});
  input_builder.AddColumn<arrow::FloatType>({col2});
  input_builder.AddColumn<arrow::DoubleType>({col3});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);
  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  // labels
  std::default_random_engine e;
  std::uniform_int_distribution<uint8_t> d(0, 1);
  std::vector<uint8_t> labels(5);
  std::generate(labels.begin(), labels.end(), [&]{ return d(e); });
  arrow::NumericBuilder<arrow::UInt8Type> builder;
  builder.Resize(labels.size());
  ASSERT_TRUE(builder.AppendValues(labels).ok());
  std::shared_ptr<arrow::Array> arr;
  ASSERT_TRUE(builder.Finish(&arr).ok());
  std::shared_ptr<arrow::ChunkedArray> label_col = 
      std::make_shared<arrow::ChunkedArray>(arr);

  data::ArrowAdapter adapter(rb, label_col,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k++]);
      }
    }
  }
  
  const std::vector<float>& dmat_labels = dmat->Info().labels_.HostVector();
  for (size_t i = 0; i < dmat_labels.size(); ++i) {
    EXPECT_FLOAT_EQ(dmat_labels[i], static_cast<float>(labels[i]));
  }
}

TEST(Adapter, ArrowAdapterCreateDMatHandleMissing) {
  // add 3 single-chunk columns
  std::vector<uint16_t> col1{1, 2, 3, 4, 5};
  std::vector<float> col2{5.4f, 4.3f, 3.2f, 2.1f, 1.5f};
  std::vector<double> col3{1.2, 2.1, 3.2, 4.3, 5.4};
  /*
   * true   true    false
   * true   true    true
   * true   false   true
   * false  true    true
   * false  true    true
   */
  std::vector<bool> col1_valid{true, true, true, false, false};
  std::vector<bool> col2_valid{true, true, false, true, true};
  std::vector<bool> col3_valid{false, true, true, true, true};
  ArrowTestInputBuilder input_builder;
  input_builder.AddColumn<arrow::UInt16Type>({std::make_pair(col1, col1_valid)});
  input_builder.AddColumn<arrow::FloatType>({std::make_pair(col2, col2_valid)});
  input_builder.AddColumn<arrow::DoubleType>({std::make_pair(col3, col3_valid)});
  input_builder.Finish();
  // create a table with theses columns
  std::vector<std::shared_ptr<arrow::Field>> schema_vec{
        arrow::field("col1", arrow::uint16()),
        arrow::field("col2", arrow::float32()),
        arrow::field("col3", arrow::float64())};
  std::shared_ptr<arrow::Schema> schema{
        std::make_shared<arrow::Schema>(schema_vec)};
  std::shared_ptr<arrow::Table> table =
          arrow::Table::Make(schema, input_builder.columns);

  arrow::TableBatchReader treader{*table};
  data::RecordBatches rb;
  treader.ReadAll(&rb);
  data::ArrowAdapter adapter(rb, nullptr,
                             table->num_rows(), table->num_columns());
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      999.0f,   // replace missing values with 999.0f
                      1)};
  ASSERT_NE(dmat, nullptr);

  size_t k{};
  for (const auto& batch : dmat->GetBatches<xgboost::SparsePage>()) {
    for (size_t i = 0; i < batch.Size(); ++i) { // NOLINT(*)
      xgboost::SparsePage::Inst const inst = batch[i];
      for (auto const& entry : inst) {
        // the 2nd, 7th, 9th, and 12th elements are missing values
        if (k == 2 || k == 7 || k == 9 || k == 12) {
          EXPECT_FLOAT_EQ(entry.fvalue, 999.0f);
        } else {
          EXPECT_FLOAT_EQ(entry.fvalue, input_builder.raveled[k]);
        }
        ++k;
      }
    }
  }
}

TEST(Adapter, ArrowAdapterEmptyTable) {
  data::RecordBatches rb;
  data::ArrowAdapter adapter(rb, nullptr, 0, 0);
  std::shared_ptr<DMatrix> dmat{
      DMatrix::Create(&adapter,
                      std::numeric_limits<float>::quiet_NaN(),
                      1)};
  ASSERT_NE(dmat, nullptr);
  EXPECT_EQ(dmat->Info().num_col_, 0);
  EXPECT_EQ(dmat->Info().num_row_, 0);
}

#endif

}  // namespace xgboost
