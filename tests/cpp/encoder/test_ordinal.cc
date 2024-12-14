/**
 * Copyright 2024, XGBoost contributors
 */
#include "test_ordinal.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>  // for partial_sum
#include <vector>

#include "../../../src/encoder/ordinal.h"

namespace enc {
namespace {
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

class OrdRecoderTest {
 public:
  void Recode(HostColumnsView orig_enc, HostColumnsView new_enc, Span<std::int32_t> mapping) {
    std::vector<std::int32_t> sorted_idx(orig_enc.n_total_cats);
    SortNames(DftHostPolicy{}, orig_enc, sorted_idx);
    ::enc::Recode(DftHostPolicy{}, orig_enc, sorted_idx, new_enc, mapping);
  }
};
}  // namespace

TEST(CategoricalEncoder, Str) { TestOrdinalEncoderStrs<OrdRecoderTest, DfTest>(); }

TEST(CategoricalEncoder, Int) { TestOrdinalEncoderInts<OrdRecoderTest, DfTest>(); }

TEST(CategoricalEncoder, Mixed) { TestOrdinalEncoderMixed<OrdRecoderTest, DfTest>(); }

TEST(CategoricalEncoder, Empty) { TestOrdinalEncoderEmpty<OrdRecoderTest, DfTest>(); }
}  // namespace enc
