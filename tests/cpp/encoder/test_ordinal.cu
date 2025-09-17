/**
 * Copyright 2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include "../../src/encoder/ordinal.cuh"
#include "df_mock.cuh"
#include "test_ordinal.h"

namespace enc::cuda_impl {
namespace {
class OrdRecoderTest {
 public:
  void Recode(DeviceColumnsView orig_enc, DeviceColumnsView new_enc, Span<std::int32_t> mapping) {
    auto policy = DftDevicePolicy{};
    thrust::device_vector<std::int32_t> ref_sorted_idx(orig_enc.n_total_cats);
    SortNames(policy, orig_enc, dh::ToSpan(ref_sorted_idx));
    auto d_sorted_idx = dh::ToSpan(ref_sorted_idx);
    ::enc::Recode(policy, orig_enc, d_sorted_idx, new_enc, mapping);
  }
};
}  // namespace

TEST(CategoricalEncoder, StrGpu) { TestOrdinalEncoderStrs<OrdRecoderTest, DfTest>(); }

TEST(CategoricalEncoder, IntGpu) { TestOrdinalEncoderInts<OrdRecoderTest, DfTest>(); }

TEST(CategoricalEncoder, MixedGpu) { TestOrdinalEncoderMixed<OrdRecoderTest, DfTest>(); }

TEST(CategoricalEncoder, EmptyGpu) { TestOrdinalEncoderEmpty<OrdRecoderTest, DfTest>(); }
}  // namespace enc::cuda_impl
