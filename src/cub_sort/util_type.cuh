#pragma once
#include <thrust/swap.h>  // for swap
#include <xgboost/data.h>

#include <cub/util_type.cuh>

namespace cub_argsort {
struct EntryTrait : public ::cub::BaseTraits<::cub::NOT_A_NUMBER, false, false, unsigned long long,
                                             xgboost::Entry> {
  using Entry = xgboost::Entry;
  using UnsignedBits = unsigned long long;

  static constexpr ::cub::Category CATEGORY = ::cub::NOT_A_NUMBER;
  // The calculation for bst_feature_t is not necessary as it's unsigned integer, only
  // performed here for clarify.
  static constexpr UnsignedBits LOWEST_KEY =
      (UnsignedBits{::cub::NumericTraits<decltype(Entry::index)>::LOWEST_KEY}
       << sizeof(decltype(Entry::index)) * 8) ^
      UnsignedBits { ::cub::NumericTraits<decltype(Entry::fvalue)>::LOWEST_KEY };

  static constexpr UnsignedBits MAX_KEY =
      (UnsignedBits{::cub::NumericTraits<decltype(Entry::index)>::MAX_KEY}
       << sizeof(decltype(Entry::index)) * 8) ^
      UnsignedBits { ::cub::NumericTraits<decltype(Entry::fvalue)>::MAX_KEY };

  static __device__ __forceinline__ UnsignedBits TwiddleIn(UnsignedBits key) {
    using F32T = ::cub::NumericTraits<decltype(Entry::fvalue)>;

    auto ptr = reinterpret_cast<std::uint32_t*>(&key);
    // Make index the most significant element
    // after swap, 0^th is favlue, 1^th is index
    thrust::swap(ptr[0], ptr[1]);

    auto& fv_key = ptr[0];
    fv_key = F32T::TwiddleIn(fv_key);

    return key;
  };

  static __device__ __forceinline__ UnsignedBits TwiddleOut(UnsignedBits key) {
    using F32T = ::cub::NumericTraits<decltype(Entry::fvalue)>;

    auto ptr = reinterpret_cast<std::uint32_t*>(&key);
    // after swap, 0^th is index, 1^th is fvalue
    thrust::swap(ptr[0], ptr[1]);
    auto& fv_key = ptr[1];
    fv_key = F32T::TwiddleOut(fv_key);

    return key;
  }
};

template <typename T>
struct MyTraits : public std::conditional_t<std::is_same_v<T, xgboost::Entry>, EntryTrait,
                                            ::cub::NumericTraits<T>> {};
}  // namespace cub_argsort
