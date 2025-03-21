/**
 * Copyright 2025, XGBoost contributors
 */
#include "test_ordinal.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>  // for int32_t
#include <sstream>  // for stringstream
#include <vector>   // for vector

#include "../../../src/encoder/ordinal.h"
#include "df_mock.h"  // for DfTest

namespace enc {
namespace {
using DfTest = cpu_impl::DfTest;

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

TEST(CategoricalEncoder, Print) {
  auto df = DfTest::Make(DfTest::MakeInts(0, 1), DfTest::MakeStrs("cbd", "bbd", "dbd", "ab"),
                         DfTest::MakeInts(2, 3));
  std::stringstream ss;
  ss << df.View();
  auto str = ss.str();
  auto sol = R"(f0: [0, 1]
f1: [cbd, bbd, dbd, ab]
f2: [2, 3]
)";
  ASSERT_EQ(sol, str);
}
}  // namespace enc
