/**
 * Copyright 2026, XGBoost Contributors
 */
#include <gtest/gtest.h>
#include <xgboost/philox_engine.h>

#include <cstddef>  // for size_t
#include <cstdint>  // for uint32_t, uint64_t
#include <random>   // for philox_engine
#include <vector>   // for vector

namespace xgboost {
namespace {
// The sequence has to be the one C++26 specifies, since the point of writing this out
// ourselves is to be able to drop it for `std::philox_engine` later without changing a
// single number. [rand.predef] pins the two engines down with these values.
TEST(PhiloxEngine, RequiredBehavior) {
  {
    Philox4x32 rng;
    std::uint32_t v = 0;
    for (std::size_t i = 0; i < 10000; ++i) {
      v = rng();
    }
    ASSERT_EQ(v, 1955073260u);
  }
  {
    Philox4x64 rng;
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < 10000; ++i) {
      v = rng();
    }
    ASSERT_EQ(v, 3409172418970261260ull);
  }
}

// Philox predates the standard, and the standard adopted the generator unchanged. Seeding
// with 0 leaves both the key and the counter all zero, which is the first published test
// vector for philox4x32-10 in Random123, the reference implementation.
TEST(PhiloxEngine, KnownAnswer) {
  Philox4x32 rng{0};
  std::vector<std::uint32_t> const expected{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8};
  for (auto e : expected) {
    ASSERT_EQ(rng(), e);
  }
}

TEST(PhiloxEngine, Constants) {
  ASSERT_EQ(Philox4x32::min(), 0u);
  ASSERT_EQ(Philox4x32::max(), 4294967295u);
  ASSERT_EQ(Philox4x64::min(), 0ull);
  ASSERT_EQ(Philox4x64::max(), 18446744073709551615ull);

  ASSERT_EQ(Philox4x32::word_size, 32ul);
  ASSERT_EQ(Philox4x32::word_count, 4ul);
  ASSERT_EQ(Philox4x32::round_count, 10ul);
  ASSERT_EQ(Philox4x32::default_seed, 20111115u);

  // Even-indexed template constants are the multipliers, odd-indexed the Weyl constants.
  ASSERT_EQ(Philox4x32::multipliers[0], 0xCD9E8D57u);
  ASSERT_EQ(Philox4x32::multipliers[1], 0xD2511F53u);
  ASSERT_EQ(Philox4x32::round_consts[0], 0x9E3779B9u);
  ASSERT_EQ(Philox4x32::round_consts[1], 0xBB67AE85u);
}

// `discard` is the reason for a counter-based engine here: restoring a serialized state
// skips forward by however many draws the model took while training, and that has to cost
// the same whether it is ten draws or ten billion. It still has to land exactly where the
// draws would have.
TEST(PhiloxEngine, DiscardMatchesDraws) {
  // Around and across the four-word block boundary.
  for (std::uint64_t z :
       {0ull, 1ull, 2ull, 3ull, 4ull, 5ull, 7ull, 8ull, 9ull, 63ull, 64ull, 1000ull}) {
    Philox4x32 drawn{1994};
    for (std::uint64_t i = 0; i < z; ++i) {
      static_cast<void>(drawn());
    }
    Philox4x32 skipped{1994};
    skipped.discard(z);
    ASSERT_EQ(drawn, skipped) << "z: " << z;
    for (std::size_t i = 0; i < 8; ++i) {
      ASSERT_EQ(drawn(), skipped()) << "z: " << z;
    }
  }
}

TEST(PhiloxEngine, DiscardIsAdditive) {
  // Far enough that replaying the draws would not be an option.
  std::uint64_t constexpr kHalf = 1ull << 39;
  Philox4x32 once{5};
  once.discard(kHalf * 2);
  Philox4x32 twice{5};
  twice.discard(kHalf);
  twice.discard(kHalf);
  ASSERT_EQ(once, twice);
  for (std::size_t i = 0; i < 8; ++i) {
    ASSERT_EQ(once(), twice());
  }
}

// The counter is an n * w bit integer, so advancing it has to carry between words. Only
// the 64-bit words exercise the wrap of a whole word, and they take 2^64 blocks to get
// there, which is only reachable at all because `discard` does not replay anything.
TEST(PhiloxEngine, CounterCarry) {
  Philox4x64 rng{1994};
  // 2^64 - 4 draws is 2^62 - 1 blocks, the most a single call can cover.
  for (std::size_t i = 0; i < 4; ++i) {
    rng.discard(~0ull - 3);
  }
  // Four blocks short of 2^64. These take the low word over.
  rng.discard(16);

  Philox4x64 from_zero{1994};
  bool differs = false;
  for (std::size_t i = 0; i < 4; ++i) {
    if (rng() != from_zero()) {
      differs = true;
    }
  }
  // Losing the carry would leave the counter back at zero, repeating the first block.
  ASSERT_TRUE(differs);
}

TEST(PhiloxEngine, Seed) {
  Philox4x32 rng{7};
  std::vector<std::uint32_t> first;
  for (std::size_t i = 0; i < 16; ++i) {
    first.push_back(rng());
  }
  // Re-seeding restarts the sequence, whatever the engine was doing before.
  rng.discard(1234);
  rng.seed(7);
  for (auto v : first) {
    ASSERT_EQ(rng(), v);
  }

  ASSERT_NE(Philox4x32{7}, Philox4x32{8});
  ASSERT_EQ(Philox4x32{}, Philox4x32{Philox4x32::default_seed});
}

#if defined(__cpp_lib_philox_engine)
// Once a standard library ships one, ours has to agree with it. When they all do, this
// implementation can go and `RandomEngine` can name `std::philox4x32` directly.
TEST(PhiloxEngine, MatchesStd) {
  Philox4x32 ours{1994};
  std::philox4x32 theirs{1994};
  for (std::size_t i = 0; i < 4096; ++i) {
    ASSERT_EQ(ours(), theirs());
  }
  ours.discard(100000);
  theirs.discard(100000);
  ASSERT_EQ(ours(), theirs());

  Philox4x64 ours64{1994};
  std::philox4x64 theirs64{1994};
  for (std::size_t i = 0; i < 4096; ++i) {
    ASSERT_EQ(ours64(), theirs64());
  }
}
#endif  // defined(__cpp_lib_philox_engine)
}  // namespace
}  // namespace xgboost
