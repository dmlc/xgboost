/**
 * Copyright 2020-2023 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include "../helpers.h"
#include "../../../src/data/array_interface.h"
#include "dmlc/logging.h"
#include "xgboost/json.h"

namespace xgboost {
TEST(ArrayInterface, Initialize) {
  size_t constexpr kRows = 10, kCols = 10;
  HostDeviceVector<float> storage;
  auto array = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  auto arr_interface = ArrayInterface<2>(StringView{array});
  ASSERT_EQ(arr_interface.Shape(0), kRows);
  ASSERT_EQ(arr_interface.Shape(1), kCols);
  ASSERT_EQ(arr_interface.data, storage.ConstHostPointer());
  ASSERT_EQ(arr_interface.ElementSize(), 4);
  ASSERT_EQ(arr_interface.type, ArrayInterfaceHandler::kF4);

  HostDeviceVector<size_t> u64_storage(storage.Size());
  std::string u64_arr_str{ArrayInterfaceStr(linalg::TensorView<size_t const, 2>{
      u64_storage.ConstHostSpan(), {kRows, kCols}, Context::kCpuId})};
  std::copy(storage.ConstHostVector().cbegin(), storage.ConstHostVector().cend(),
            u64_storage.HostSpan().begin());
  auto u64_arr = ArrayInterface<2>{u64_arr_str};
  ASSERT_EQ(u64_arr.ElementSize(), 8);
  ASSERT_EQ(u64_arr.type, ArrayInterfaceHandler::kU8);
}

TEST(ArrayInterface, Error) {
  constexpr size_t kRows = 16, kCols = 10;
  Json column { Object() };
  std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
  column["shape"] = Array(j_shape);
  std::vector<Json> j_data{Json(Integer(reinterpret_cast<Integer::Int>(nullptr))),
                           Json(Boolean(false))};

  auto const& column_obj = get<Object>(column);
  std::string typestr{"<f4"};
  size_t n = kRows * kCols;

  // missing version
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj, n), dmlc::Error);
  column["version"] = 3;
  // missing data
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj, n),
               dmlc::Error);
  // null data
  column["data"] = Null{};
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj, n),
               dmlc::Error);
  column["data"] = j_data;
  // missing typestr
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj, n),
               dmlc::Error);
  column["typestr"] = String("<f4");
  // nullptr is not valid
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj, n),
               dmlc::Error);

  HostDeviceVector<float> storage;
  auto array = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  j_data = {
      Json(Integer(reinterpret_cast<Integer::Int>(storage.ConstHostPointer()))),
      Json(Boolean(false))};
  column["data"] = j_data;
  EXPECT_NO_THROW(ArrayInterfaceHandler::ExtractData(column_obj, n));
  // null data in mask
  column["mask"] = Object{};
  column["mask"]["data"] = Null{};
  common::Span<RBitField8::value_type> s_mask;
  EXPECT_THROW(ArrayInterfaceHandler::ExtractMask(column_obj, &s_mask), dmlc::Error);

  get<Object>(column).erase("mask");
  // misaligned.
  j_data = {Json(Integer(reinterpret_cast<Integer::Int>(
                reinterpret_cast<char const*>(storage.ConstHostPointer()) + 1))),
            Json(Boolean(false))};
  column["data"] = j_data;
  EXPECT_THROW({ ArrayInterface<1> arr{column}; }, dmlc::Error);
}

TEST(ArrayInterface, GetElement) {
  size_t kRows = 4, kCols = 2;
  HostDeviceVector<float> storage;
  auto intefrace_str = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  ArrayInterface<2> array_interface{intefrace_str};

  auto const& h_storage = storage.ConstHostVector();
  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols; ++j) {
      float v0 = array_interface(i, j);
      float v1 = h_storage.at(i * kCols + j);
      ASSERT_EQ(v0, v1);
    }
  }
}

TEST(ArrayInterface, TrivialDim) {
  size_t kRows{1000}, kCols = 1;
  HostDeviceVector<float> storage;
  auto interface_str = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  {
    ArrayInterface<1> arr_i{interface_str};
    ASSERT_EQ(arr_i.n, kRows);
    ASSERT_EQ(arr_i.Shape(0), kRows);
  }

  std::swap(kRows, kCols);
  interface_str = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  {
    ArrayInterface<1> arr_i{interface_str};
    ASSERT_EQ(arr_i.n, kCols);
    ASSERT_EQ(arr_i.Shape(0), kCols);
  }
}

TEST(ArrayInterface, ToDType) {
  static_assert(ToDType<float>::kType == ArrayInterfaceHandler::kF4);
  static_assert(ToDType<double>::kType == ArrayInterfaceHandler::kF8);

  static_assert(ToDType<uint32_t>::kType == ArrayInterfaceHandler::kU4);
  static_assert(ToDType<uint64_t>::kType == ArrayInterfaceHandler::kU8);

  static_assert(ToDType<int32_t>::kType == ArrayInterfaceHandler::kI4);
  static_assert(ToDType<int64_t>::kType == ArrayInterfaceHandler::kI8);
}
}  // namespace xgboost
