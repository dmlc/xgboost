/**
 * Copyright 2023, XGBoost contributors
 */
#include <gtest/gtest.h>
#include <thrust/device_vector.h>  // for device_vector
#include <thrust/execution_policy.h>
#include <thrust/iterator/reverse_iterator.h>  // for make_reverse_iterator
#include <thrust/sequence.h>                   // for sequence
#include <xgboost/base.h>                      // for bst_feature_t
#include <xgboost/data.h>                      // for Entry
#include <xgboost/host_device_vector.h>        // for HostDeviceVector

#include <bitset>     // for bitset
#include <cinttypes>  // for uint32_t
#include <cstddef>    // for size_t
#include <random>  // for default_random_engine, uniform_int_distribution, uniform_real_distribution
#include <sstream>  // for stringstream
#include <tuple>    // for tuple
#include <vector>   // for vector

#include "../../../src/cub_sort/device/device_radix_sort.cuh"

namespace cub_argsort {
using xgboost::Entry;
using xgboost::HostDeviceVector;

namespace {
void TestBitCast() {
  Entry e;
  e.index = 3;
  e.fvalue = -512.0f;

  static_assert(sizeof(Entry) == sizeof(std::uint64_t));
  std::uint64_t bits;
  std::memcpy(&bits, &e, sizeof(e));

  std::bitset<64> set{bits};

  std::uint32_t* ptr = reinterpret_cast<std::uint32_t*>(&bits);
  std::bitset<32> lhs{ptr[0]};
  std::bitset<32> rhs{ptr[1]};
  // The first 32-bit segment contains the feature index
  ASSERT_EQ(lhs, std::bitset<32>{e.index});

  std::swap(ptr[0], ptr[1]);
  set = bits;
  // after swap, the second segment contains the feature index
  ASSERT_EQ(std::bitset<32>{ptr[1]}, std::bitset<32>{e.index});

  bits = EntryTrait::LOWEST_KEY;
  auto pptr = reinterpret_cast<std::uint32_t const*>(&bits);
  ASSERT_EQ(pptr[0], []() { return ::cub::NumericTraits<float>::LOWEST_KEY; }());
  ASSERT_EQ(pptr[1], []() { return ::cub::NumericTraits<xgboost::bst_feature_t>::LOWEST_KEY; }());

  bits = EntryTrait::MAX_KEY;
  pptr = reinterpret_cast<std::uint32_t const*>(&bits);
  ASSERT_EQ(pptr[0], []() { return ::cub::NumericTraits<float>::MAX_KEY; }());
  ASSERT_EQ(pptr[1], []() { return ::cub::NumericTraits<xgboost::bst_feature_t>::MAX_KEY; }());
}

enum TestType { kf32, kf64, ki32 };

class RadixArgSortNumeric : public ::testing::TestWithParam<std::tuple<TestType, std::size_t>> {
 public:
  template <typename T>
  void TestArgSort(std::size_t n) {
    HostDeviceVector<T> data;
    data.SetDevice(0);
    data.Resize(n, 0.0f);
    auto d_data = data.DeviceSpan();
    auto beg = thrust::make_reverse_iterator(d_data.data() + d_data.size());
    thrust::sequence(thrust::device, beg, beg + d_data.size(), -static_cast<std::int64_t>(n / 2.0));
    auto const& h_in = data.ConstHostSpan();

    HostDeviceVector<std::uint32_t> idx_out(data.Size(), 0u);
    idx_out.SetDevice(0);
    auto d_idx_out = idx_out.DeviceSpan();

    std::size_t bytes{0};
    DeviceRadixSort<cub::ShiftDigitExtractor<T>>::Argsort(nullptr, bytes, d_data.data(),
                                                          d_idx_out.data(), d_data.size());
    thrust::device_vector<xgboost::common::byte> temp(bytes);
    DeviceRadixSort<cub::ShiftDigitExtractor<T>>::Argsort(temp.data().get(), bytes, d_data.data(),
                                                          d_idx_out.data(), d_data.size());
    ASSERT_GT(bytes, n * sizeof(std::uint32_t));

    auto const& h_idx_out = idx_out.ConstHostSpan();
    for (std::size_t i = 1; i < h_idx_out.size(); ++i) {
      ASSERT_EQ(h_idx_out[i] + 1, h_idx_out[i - 1]);
      ASSERT_EQ(h_in[h_idx_out[i]], h_in[h_idx_out[i - 1]] + 1);
    }
  }

  template <typename T>
  void TestSameValue(std::size_t n) {
    HostDeviceVector<T> data(n, static_cast<T>(1.0), 0);

    auto d_data = data.ConstDeviceSpan();
    HostDeviceVector<std::uint32_t> idx_out(n);
    idx_out.SetDevice(0);
    auto d_idx_out = idx_out.DeviceSpan();

    std::size_t bytes{0};
    DeviceRadixSort<EntryExtractor>::Argsort(nullptr, bytes, d_data.data(), d_idx_out.data(),
                                             d_data.size());
    thrust::device_vector<xgboost::common::byte> temp(bytes);
    DeviceRadixSort<EntryExtractor>::Argsort(temp.data().get(), bytes, d_data.data(),
                                             d_idx_out.data(), d_data.size());

    auto const& h_idx = idx_out.ConstHostVector();
    std::vector<std::uint32_t> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    ASSERT_EQ(h_idx, expected);
  }
};

class RadixArgSortEntry : public ::testing::TestWithParam<std::size_t> {
 public:
  void TestCustomExtractor(std::size_t n) {
    HostDeviceVector<Entry> data(n, Entry{0, 0}, 0);

    auto& h_data = data.HostVector();

    std::default_random_engine rng;
    rng.seed(1);

    std::uniform_int_distribution<xgboost::bst_feature_t> fdist(0, 27);
    std::uniform_real_distribution<float> vdist(-8.0f, 8.0f);

    for (auto it = h_data.rbegin(); it != h_data.rend(); ++it) {
      auto d = std::distance(h_data.rbegin(), it);
      it->fvalue = vdist(rng);
      it->index = fdist(rng);
    }

    HostDeviceVector<std::uint32_t> out_idx(n, 0u, 0);

    auto d_data = data.ConstDeviceSpan();
    auto d_idx_out = out_idx.DeviceSpan();
    std::size_t bytes{0};

    DeviceRadixSort<EntryExtractor>::Argsort(nullptr, bytes, d_data.data(), d_idx_out.data(),
                                             d_data.size());
    thrust::device_vector<xgboost::common::byte> temp(bytes);
    DeviceRadixSort<EntryExtractor>::Argsort(temp.data().get(), bytes, d_data.data(),
                                             d_idx_out.data(), d_data.size());

    auto const& h_idx = out_idx.ConstHostVector();

    for (std::size_t i = 1; i < h_idx.size(); ++i) {
      ASSERT_GE(h_data[h_idx[i]].index, h_data[h_idx[i - 1]].index);
      if (h_data[h_idx[i]].index == h_data[h_idx[i - 1]].index) {
        // within the same feature, value should be increasing.
        ASSERT_GE(h_data[h_idx[i]].fvalue, h_data[h_idx[i - 1]].fvalue);
      }
    }
  }
};
}  // namespace

TEST_P(RadixArgSortNumeric, Basic) {
  auto [t, n] = GetParam();
  switch (t) {
    case kf32: {
      TestArgSort<float>(n);
      break;
    }
    case kf64: {
      TestArgSort<double>(n);
      break;
    }
    case ki32: {
      TestArgSort<std::int32_t>(n);
      break;
    }
  };
}

TEST_P(RadixArgSortNumeric, SameValue) {
  auto [t, n] = GetParam();
  switch (t) {
    case kf32: {
      TestSameValue<float>(n);
      break;
    }
    case kf64: {
      TestSameValue<double>(n);
      break;
    }
    case ki32: {
      TestSameValue<std::int32_t>(n);
      break;
    }
  };
}

INSTANTIATE_TEST_SUITE_P(RadixArgSort, RadixArgSortNumeric,
                         testing::Values(std::tuple{kf32, 128}, std::tuple{kf64, 128},
                                         std::tuple{ki32, 128}, std::tuple{kf32, 8192},
                                         std::tuple{kf64, 8192}, std::tuple{ki32, 8192}),
                         ([](::testing::TestParamInfo<RadixArgSortNumeric::ParamType> const& info) {
                           auto [t, n] = info.param;
                           std::stringstream ss;
                           ss << static_cast<std::int32_t>(t) << "_" << n;
                           return ss.str();
                         }));

TEST_P(RadixArgSortEntry, Basic) {
  std::size_t n = GetParam();
  TestCustomExtractor(n);
}

INSTANTIATE_TEST_SUITE_P(RadixArgSort, RadixArgSortEntry, testing::Values(128, 8192),
                         ([](::testing::TestParamInfo<RadixArgSortEntry::ParamType> const& info) {
                           auto n = info.param;
                           std::stringstream ss;
                           ss << n;
                           return ss.str();
                         }));

TEST(RadixArgSort, BitCast) { TestBitCast(); }
}  // namespace cub_argsort
