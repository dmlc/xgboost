/**
 * Copyright 2025-2026, XGBoost contributors
 */

#include "test_cat_container.h"

#include <gtest/gtest.h>

#include <cmath>    // for
#include <cstdint>  // for uint16_t
#include <ios>      // for openmode
#include <limits>   // for numeric_limits
#include <utility>  // for move
#include <vector>   // for vector

#include "../../../src/data/adapter.h"  // for ColumnarAdapter
#include "../encoder/df_mock.h"
#include "../helpers.h"  // for GMockThrow

namespace xgboost {
using DfTest = enc::cpu_impl::DfTest;

auto eq_check = [](common::Span<bst_cat_t const> sorted_idx, std::vector<bst_cat_t> const& sol) {
  ASSERT_EQ(sorted_idx, common::Span{sol});
};

TEST(CatContainer, Str) {
  Context ctx;
  TestCatContainerStr<DfTest>(&ctx, eq_check);
}

TEST(CatContainer, Mixed) {
  Context ctx;
  TestCatContainerMixed<DfTest>(&ctx, eq_check);
}

template <typename T, typename JsonArray>
void TestUnsignedSerialization(std::vector<T> const& values, std::int64_t type) {
  std::vector<enc::HostCatIndexView> columns{common::Span<T const>{values.data(), values.size()}};
  std::vector<std::int32_t> segments{0, static_cast<std::int32_t>(values.size())};
  auto view = enc::HostColumnsView{common::Span{columns}, common::Span{segments}, segments.back()};
  CatContainer cats{view, false};

  Json saved{Object{}};
  cats.Save(&saved);
  auto const& enc = get<Array const>(saved["enc"]);
  ASSERT_EQ(enc.size(), 1);
  EXPECT_EQ(get<Integer const>(enc.front()["type"]), type);
  EXPECT_TRUE(IsA<JsonArray>(enc.front()["values"]));

  for (auto mode : {std::ios::out, std::ios::binary}) {
    std::vector<char> buffer;
    Json::Dump(saved, &buffer, mode);
    auto parsed = Json::Load(StringView{buffer.data(), buffer.size()}, mode);

    CatContainer loaded;
    loaded.Load(parsed);
    auto loaded_view = loaded.HostView();
    ASSERT_EQ(loaded_view.columns.size(), 1);
    auto actual = std::get<common::Span<T const>>(loaded_view.columns.front());
    ASSERT_EQ(actual.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(actual[i], values[i]);
    }
  }
}

TEST(CatContainer, UnsignedSerialization) {
  TestUnsignedSerialization<std::uint16_t, U16Array>({1, std::numeric_limits<std::uint16_t>::max()},
                                                     12);
  TestUnsignedSerialization<std::uint32_t, U32Array>({1, std::numeric_limits<std::uint32_t>::max()},
                                                     14);
  constexpr auto kI64Max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  TestUnsignedSerialization<std::uint64_t, U64Array>({1, kI64Max}, 16);
}

TEST(CatContainer, RejectInvalidValues) {
  // Uint64
  {
    std::vector<std::uint64_t> values{
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1};
    std::vector<enc::HostCatIndexView> columns{
        common::Span<std::uint64_t const>{values.data(), values.size()}};
    std::vector<std::int32_t> segments{0, 1};
    auto view = enc::HostColumnsView{common::Span{columns}, common::Span{segments}, 1};
    CatContainer cats{view, false};

    Json saved;
    EXPECT_THAT([&] { cats.Save(&saved); }, GMockThrow("signed 64-bit range"));
  }

  // Floating points
  {
    Json column{Object{}};
    column["type"] = static_cast<std::int64_t>(Value::ValueKind::kF32Array);
    column["values"] = F32Array{1};

    Json in{Object{}};
    in["enc"] = Array(std::vector<Json>{std::move(column)});

    CatContainer cats;
    EXPECT_THAT([&] { cats.Load(in); }, GMockThrow("floating point dtype"));
  }
}

TEST(CatContainer, AccessorKeepsMissing) {
  // Re-coding map for one feature: code 0 -> 2, 1 -> 0, 2 -> 1.
  std::vector<std::int32_t> offsets{0, 3}, mapping{2, 0, 1};
  CatAccessor acc{
      enc::MappingView{common::Span<std::int32_t const>{offsets.data(), offsets.size()},
                       common::Span<std::int32_t const>{mapping.data(), mapping.size()}}};
  ASSERT_EQ(acc(0.0f, 0), 2.0f);
  ASSERT_EQ(acc(2.0f, 0), 1.0f);
  // A missing value must not be re-coded into a category.
  ASSERT_TRUE(std::isnan(acc(std::numeric_limits<float>::quiet_NaN(), 0)));
}

TEST(CatContainer, RecodeDMatrixKeepsMissing) {
  // Reference (training) categories {10, 20, 30}.
  std::vector<std::int32_t> ref_values{10, 20, 30};
  std::vector<enc::HostCatIndexView> ref_columns{
      common::Span<std::int32_t const>{ref_values.data(), ref_values.size()}};
  std::vector<std::int32_t> ref_segments{0, 3};
  CatContainer ref_cats{
      enc::HostColumnsView{common::Span{ref_columns}, common::Span{ref_segments}, 3}, false};

  // New data with the categories in another order and a missing value (NaN code, as pandas sends it).
  HostDeviceVector<std::int32_t> names{30, 10, 20};
  HostDeviceVector<float> codes{1.0f, std::numeric_limits<float>::quiet_NaN(),
                                0.0f};  // 10, NaN, 30
  Json column{Array{std::vector<Json>{GetArrayInterface(&names, names.Size(), 1),
                                      GetArrayInterface(&codes, codes.Size(), 1)}}};
  Json df{Object{}};
  df["ref_categories"] = Integer{reinterpret_cast<std::int64_t>(&ref_cats)};
  df["columns"] = Array{std::vector<Json>{column}};

  data::ColumnarAdapter adapter{Json::Dump(df)};
  std::unique_ptr<DMatrix> p_fmat{
      DMatrix::Create(&adapter, std::numeric_limits<float>::quiet_NaN(), 1)};
  for (auto const& page : p_fmat->GetBatches<SparsePage>()) {
    auto h_page = page.GetView();
    ASSERT_EQ(h_page[0].size(), 1);
    ASSERT_EQ(h_page[0][0].fvalue, 0.0f);  // 10 -> training code 0
    ASSERT_EQ(h_page[1].size(), 0);        // missing stays missing
    ASSERT_EQ(h_page[2].size(), 1);
    ASSERT_EQ(h_page[2][0].fvalue, 2.0f);  // 30 -> training code 2
  }
}
}  // namespace xgboost
