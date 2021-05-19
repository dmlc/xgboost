/*!
 * Copyright 2020-2021 by XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/host_device_vector.h>
#include "../helpers.h"
#include "../../../src/data/array_interface.h"

namespace xgboost {
TEST(ArrayInterface, Initialize) {
  size_t constexpr kRows = 10, kCols = 10;
  HostDeviceVector<float> storage;
  auto array = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  auto arr_interface = ArrayInterface(array);
  ASSERT_EQ(arr_interface.num_rows, kRows);
  ASSERT_EQ(arr_interface.num_cols, kCols);
  ASSERT_EQ(arr_interface.data, storage.ConstHostPointer());
  ASSERT_EQ(arr_interface.ElementSize(), 4);
  ASSERT_EQ(arr_interface.type, ArrayInterface::kF4);

  HostDeviceVector<size_t> u64_storage(storage.Size());
  std::string u64_arr_str;
  Json::Dump(GetArrayInterface(&u64_storage, kRows, kCols), &u64_arr_str);
  std::copy(storage.ConstHostVector().cbegin(), storage.ConstHostVector().cend(),
            u64_storage.HostSpan().begin());
  auto u64_arr = ArrayInterface{u64_arr_str};
  ASSERT_EQ(u64_arr.ElementSize(), 8);
  ASSERT_EQ(u64_arr.type, ArrayInterface::kU8);
}

TEST(ArrayInterface, Error) {
  constexpr size_t kRows = 16, kCols = 10;
  Json column { Object() };
  std::vector<Json> j_shape {Json(Integer(static_cast<Integer::Int>(kRows)))};
  column["shape"] = Array(j_shape);
  std::vector<Json> j_data {
    Json(Integer(reinterpret_cast<Integer::Int>(nullptr))),
        Json(Boolean(false))};

  auto const& column_obj = get<Object>(column);
  std::pair<size_t, size_t> shape{kRows, kCols};
  std::string typestr{"<f4"};

  // missing version
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj,
                                                  StringView{typestr}, shape),
               dmlc::Error);
  column["version"] = Integer(static_cast<Integer::Int>(1));
  // missing data
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj,
                                                  StringView{typestr}, shape),
               dmlc::Error);
  column["data"] = j_data;
  // missing typestr
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj,
                                                  StringView{typestr}, shape),
               dmlc::Error);
  column["typestr"] = String("<f4");
  // nullptr is not valid
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData(column_obj,
                                                  StringView{typestr}, shape),
               dmlc::Error);

  HostDeviceVector<float> storage;
  auto array = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  j_data = {
      Json(Integer(reinterpret_cast<Integer::Int>(storage.ConstHostPointer()))),
      Json(Boolean(false))};
  column["data"] = j_data;
  EXPECT_NO_THROW(ArrayInterfaceHandler::ExtractData(
      column_obj, StringView{typestr}, shape));
}

TEST(ArrayInterface, GetElement) {
  size_t kRows = 4, kCols = 2;
  HostDeviceVector<float> storage;
  auto intefrace_str = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  ArrayInterface array_interface{intefrace_str};

  auto const& h_storage = storage.ConstHostVector();
  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols; ++j) {
      float v0 = array_interface.GetElement(i, j);
      float v1 = h_storage.at(i * kCols + j);
      ASSERT_EQ(v0, v1);
    }
  }
}
}  // namespace xgboost
