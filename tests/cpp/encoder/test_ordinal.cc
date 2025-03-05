/**
 * Copyright 2025, XGBoost contributors
 */
#include "test_ordinal.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

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
}  // namespace enc
