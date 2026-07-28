/**
 * Copyright 2025-2026, XGBoost contributors
 */

#include "test_cat_container.h"

#include <gtest/gtest.h>

#include <cstdint>  // for uint16_t
#include <ios>      // for openmode
#include <limits>   // for numeric_limits
#include <utility>  // for move
#include <vector>   // for vector

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
}  // namespace xgboost
