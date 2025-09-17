/**
 * Copyright 2019-2023, XGBoost contributors
 */
#include <gtest/gtest.h>
#include "../../../src/common/bitfield.h"

namespace xgboost {

TEST(BitField, Check) {
  {
    std::vector<LBitField64::value_type> storage(4, 0);
    storage[2] = 2;
    auto bits = LBitField64({storage.data(),
                static_cast<typename common::Span<LBitField64::value_type>::index_type>(
                    storage.size())});
    size_t true_bit = 190;
    for (size_t i = true_bit + 1; i < bits.Capacity(); ++i) {
      ASSERT_FALSE(bits.Check(i));
    }
    ASSERT_TRUE(bits.Check(true_bit));
    for (size_t i = 0; i < true_bit; ++i) {
      ASSERT_FALSE(bits.Check(i));
    }
  }

  {
    std::vector<RBitField8::value_type> storage(4, 0);
    storage[2] = 1 << 3;
    auto bits = RBitField8({storage.data(),
                static_cast<typename common::Span<RBitField8::value_type>::index_type>(
                    storage.size())});
    size_t true_bit = 19;
    for (size_t i = 0; i < true_bit; ++i) {
      ASSERT_FALSE(bits.Check(i));
    }
    ASSERT_TRUE(bits.Check(true_bit));
    for (size_t i = true_bit + 1; i < bits.Capacity(); ++i) {
      ASSERT_FALSE(bits.Check(i));
    }
  }

  {
    // regression test for correct index type.
    std::vector<RBitField8::value_type> storage(33, 0);
    storage[32] = static_cast<uint8_t>(1);
    auto bits = RBitField8({storage.data(), storage.size()});
    ASSERT_TRUE(bits.Check(256));
  }
}

template <typename BitFieldT, typename VT = typename BitFieldT::value_type>
void TestBitFieldSet(typename BitFieldT::value_type res, size_t index, size_t true_bit) {
  using IndexT = typename common::Span<VT>::index_type;
  std::vector<VT> storage(4, 0);
  auto bits = BitFieldT({storage.data(), static_cast<IndexT>(storage.size())});

  bits.Set(true_bit);

  for (size_t i = 0; i < true_bit; ++i) {
    ASSERT_FALSE(bits.Check(i));
  }

  ASSERT_TRUE(bits.Check(true_bit));

  for (size_t i = true_bit + 1; i < storage.size() * BitFieldT::kValueSize; ++i) {
    ASSERT_FALSE(bits.Check(i));
  }
  ASSERT_EQ(storage[index], res);
}

TEST(BitField, Set) {
  {
    TestBitFieldSet<LBitField64>(2, 2, 190);
  }
  {
    TestBitFieldSet<RBitField8>(1 << 3, 2, 19);
  }
}

template <typename BitFieldT, typename VT = typename BitFieldT::value_type>
void TestBitFieldClear(size_t clear_bit) {
  using IndexT = typename common::Span<VT>::index_type;
  std::vector<VT> storage(4, 0);
  auto bits = BitFieldT({storage.data(), static_cast<IndexT>(storage.size())});

  bits.Set(clear_bit);
  bits.Clear(clear_bit);

  ASSERT_FALSE(bits.Check(clear_bit));
}

TEST(BitField, Clear) {
  {
    TestBitFieldClear<LBitField64>(190);
  }
  {
    TestBitFieldClear<RBitField8>(19);
  }
}

TEST(BitField, CTZ) {
  {
    auto cnt = TrailingZeroBits(0);
    ASSERT_EQ(cnt, sizeof(std::uint32_t) * 8);
  }
  {
    auto cnt = TrailingZeroBits(0b00011100);
    ASSERT_EQ(cnt, 2);
    cnt = detail::TrailingZeroBitsImpl(0b00011100);
    ASSERT_EQ(cnt, 2);
  }
  {
    auto cnt = TrailingZeroBits(0b00011101);
    ASSERT_EQ(cnt, 0);
    cnt = detail::TrailingZeroBitsImpl(0b00011101);
    ASSERT_EQ(cnt, 0);
  }
  {
    auto cnt = TrailingZeroBits(0b1000000000000000);
    ASSERT_EQ(cnt, 15);
    cnt = detail::TrailingZeroBitsImpl(0b1000000000000000);
    ASSERT_EQ(cnt, 15);
  }
}
}  // namespace xgboost
