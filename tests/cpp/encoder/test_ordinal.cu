/**
 * Copyright 2025, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include "../../src/data/cat_container.cuh"  // for CatIndexTypes
#include "../../src/encoder/ordinal.cuh"
#include "test_ordinal.h"

namespace enc::cuda_impl {
namespace {
using CatIndexTypes = ::xgboost::cuda_impl::CatIndexTypes;
using ColumnType = enc::cpu_impl::TupToVarT<CatIndexTypes>;

class DfTest {
 public:
  template <typename T>
  using Vector = dh::device_vector<T>;

 private:
  std::vector<ColumnType> columns_;
  dh::device_vector<enc::DeviceCatIndexView> columns_v_;
  dh::device_vector<std::int32_t> segments_;
  std::vector<std::int32_t> h_segments_;

  dh::device_vector<std::int32_t> mapping_;

  template <typename Head>
  static void MakeImpl(std::vector<ColumnType>* p_out, dh::device_vector<std::int32_t>* p_sizes,
                       Head&& col) {
    p_sizes->push_back(col.size());
    p_out->emplace_back(std::forward<Head>(col));

    p_sizes->insert(p_sizes->begin(), 0);
    thrust::inclusive_scan(p_sizes->cbegin(), p_sizes->cend(), p_sizes->begin());
  }

  template <typename Head, typename... Col>
  static void MakeImpl(std::vector<ColumnType>* p_out, dh::device_vector<std::int32_t>* p_sizes,
                       Head&& col, Col&&... columns) {
    p_sizes->push_back(col.size());
    p_out->emplace_back(std::forward<Head>(col));
    MakeImpl(p_out, p_sizes, std::forward<Col>(columns)...);
  }

 public:
  template <typename... Col>
  static DfTest Make(Col&&... columns) {
    DfTest df;
    MakeImpl(&df.columns_, &df.segments_, std::forward<Col>(columns)...);
    for (std::size_t i = 0; i < df.columns_.size(); ++i) {
      auto const& col = df.columns_[i];
      std::visit(Overloaded{[&](xgboost::cuda_impl::CatStrArray const& str) {
                              df.columns_v_.push_back(enc::CatStrArrayView(str));
                            },
                            [&](auto&& args) {
                              df.columns_v_.push_back(dh::ToSpan(args));
                            }},
                 col);
    }
    CHECK_EQ(df.columns_v_.size(), sizeof...(columns));
    df.h_segments_.resize(df.segments_.size());
    thrust::copy_n(df.segments_.cbegin(), df.segments_.size(), df.h_segments_.begin());
    df.mapping_.resize(df.h_segments_.back());
    return df;
  }

  template <typename... Strs>
  static auto MakeStrs(Strs&&... strs) {
    auto array = MakeStrArrayImpl(std::forward<Strs>(strs)...);
    return xgboost::cuda_impl::CatStrArray{array.offsets, array.values};
  }

  template <typename... Ints>
  static auto MakeInts(Ints&&... names) {
    return dh::device_vector<std::int32_t>{names...};
  }

  auto View() const {
    return enc::DeviceColumnsView{dh::ToSpan(this->columns_v_), dh::ToSpan(segments_),
                                  h_segments_.back()};
  }
  auto Segment() const { return Span{h_segments_}; }

  auto MappingView() { return dh::ToSpan(mapping_); }
  auto const& Mapping() { return mapping_; }
};

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
