/**
 * Copyright 2024-2025, XGBoost contributors
 */
#pragma once
#include <gtest/gtest.h>

#include <cstdint>  // for int32_t, int8_t
#include <numeric>  // for partial_sum
#include <string>   // for string
#include <utility>  // for forward
#include <variant>  // for visit
#include <vector>   // for vector

#include "../../../src/data/cat_container.h"  // for ColumnType, CatStrArray
#include "../../../src/encoder/ordinal.h"     // for CatStrArrayView
#include "../../../src/encoder/types.h"       // for Overloaded

namespace enc {
template <typename... Strs>
auto MakeStrArrayImpl(Strs&&... strs) {
  std::vector<std::string> names{strs...};
  std::vector<std::int8_t> values;
  std::vector<std::int32_t> offsets{0};

  for (const auto& name : names) {
    for (char c : name) {
      values.push_back(c);
    }
    offsets.push_back(name.size());
  }
  std::partial_sum(offsets.cbegin(), offsets.cend(), offsets.begin());
  return xgboost::cpu_impl::CatStrArray{offsets, values};
}
}  // namespace enc

namespace enc::cpu_impl {
using ColumnType = xgboost::cpu_impl::ColumnType;
class DfTest {
 private:
  std::vector<ColumnType> columns_;
  std::vector<enc::HostCatIndexView> columns_v_;
  std::vector<std::int32_t> segments_;

  std::vector<std::int32_t> mapping_;

  template <typename Head>
  static auto MakeImpl(std::vector<ColumnType>* p_out, std::vector<std::int32_t>* p_sizes,
                       Head&& col) {
    p_out->emplace_back(col);
    p_sizes->push_back(col.size());
    p_sizes->insert(p_sizes->begin(), 0);
    std::partial_sum(p_sizes->cbegin(), p_sizes->cend(), p_sizes->begin());
  }

  template <typename Head, typename... Col>
  static void MakeImpl(std::vector<ColumnType>* p_out, std::vector<std::int32_t>* p_sizes,
                       Head&& col, Col&&... columns) {
    p_out->emplace_back(col);
    p_sizes->push_back(col.size());

    MakeImpl(p_out, p_sizes, columns...);
  }

 public:
  template <typename... Col>
  static DfTest Make(Col&&... columns) {
    DfTest df;
    MakeImpl(&df.columns_, &df.segments_, std::forward<Col>(columns)...);
    for (std::size_t i = 0; i < df.columns_.size(); ++i) {
      auto const& col = df.columns_[i];
      std::visit(Overloaded{[&](xgboost::cpu_impl::CatStrArray const& str) {
                              df.columns_v_.emplace_back(enc::CatStrArrayView(str));
                            },
                            [&](auto&& args) {
                              df.columns_v_.emplace_back(Span{args});
                            }},
                 col);
    }
    auto check = [&] {
      // the macro needs to return void.
      ASSERT_EQ(df.columns_v_.size(), sizeof...(columns));
    };
    check();
    df.mapping_.resize(df.segments_.back());
    return df;
  }

  template <typename... Strs>
  static auto MakeStrs(Strs&&... strs) {
    return MakeStrArrayImpl(std::forward<Strs>(strs)...);
  }

  template <typename... Ints>
  static auto MakeInts(Ints&&... names) {
    return std::vector<std::int32_t>{names...};
  }

  auto View() const { return enc::HostColumnsView{Span{columns_v_}, segments_, segments_.back()}; }

  auto Segment() const { return Span{segments_}; }
  auto MappingView() { return Span{mapping_}; }
  auto const& Mapping() { return mapping_; }
};
}  // namespace enc::cpu_impl
