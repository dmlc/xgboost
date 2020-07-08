/*!
 * Copyright 2020 by XGBoost Contributors
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
  // missing version
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData<float>(column_obj), dmlc::Error);
  column["version"] = Integer(static_cast<Integer::Int>(1));
  // missing data
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData<float>(column_obj), dmlc::Error);
  column["data"] = j_data;
  // missing typestr
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData<float>(column_obj), dmlc::Error);
  column["typestr"] = String("<f4");
  // nullptr is not valid
  EXPECT_THROW(ArrayInterfaceHandler::ExtractData<float>(column_obj), dmlc::Error);

  HostDeviceVector<float> storage;
  auto array = RandomDataGenerator{kRows, kCols, 0}.GenerateArrayInterface(&storage);
  j_data = {
      Json(Integer(reinterpret_cast<Integer::Int>(storage.ConstHostPointer()))),
      Json(Boolean(false))};
  column["data"] = j_data;
  EXPECT_NO_THROW(ArrayInterfaceHandler::ExtractData<float>(column_obj));
}

}  // namespace xgboost
