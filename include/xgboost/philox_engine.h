/**
 * Copyright 2026, XGBoost Contributors
 *
 * @brief A counter-based random number engine, matching C++26's `std::philox_engine`.
 */
#ifndef XGBOOST_PHILOX_ENGINE_H_
#define XGBOOST_PHILOX_ENGINE_H_

#include <array>        // for array
#include <cstddef>      // for size_t
#include <cstdint>      // for uint32_t, uint64_t
#include <limits>       // for numeric_limits
#include <type_traits>  // for is_unsigned_v

namespace xgboost {
namespace detail {
/**
 * @brief The full 2w-bit product of two w-bit words, as a pair of 64-bit halves.
 *
 * Written out by hand from 32-bit partial products. `__uint128_t` is not available on
 * MSVC, and `_umul128` is not available off x86-64, but this is portable everywhere and
 * compiles down to a single multiply instruction on the platforms that have one.
 */
inline void UMul64(std::uint64_t a, std::uint64_t b, std::uint64_t* p_hi, std::uint64_t* p_lo) {
  constexpr std::uint64_t kLoMask = 0xffffffffULL;
  std::uint64_t const a_lo = a & kLoMask, a_hi = a >> 32;
  std::uint64_t const b_lo = b & kLoMask, b_hi = b >> 32;

  std::uint64_t const ll = a_lo * b_lo;
  std::uint64_t const lh = a_lo * b_hi;
  std::uint64_t const hl = a_hi * b_lo;
  std::uint64_t const hh = a_hi * b_hi;

  std::uint64_t const cross = (ll >> 32) + (hl & kLoMask) + lh;
  *p_hi = hh + (cross >> 32) + (hl >> 32);
  *p_lo = (cross << 32) | (ll & kLoMask);
}
}  // namespace detail

/**
 * @brief A counter-based random number engine of the Philox family.
 *
 * The interface and the generated sequence both follow `std::philox_engine`, specified in
 * [rand.eng.philox] for C++26. We carry our own copy because no standard library we build
 * against ships one yet; when they do, this can become an alias for `std::philox_engine`
 * without changing a single number. @ref Philox4x32 and @ref Philox4x64 correspond to
 * `std::philox4x32` and `std::philox4x64`.
 *
 * Unlike the Mersenne twister, the state here is a key and a counter rather than a large
 * buffer that each draw shuffles forward. The output is a pure function of the two, which
 * buys two properties that matter to us:
 *
 * - @ref discard is O(1). Skipping ahead is addition on the counter, not replay. That is
 *   what lets a serialized state be restored in constant time. See
 *   @ref SerializableRandomEngine.
 * - The state is a handful of integers with a layout we define, so it can be written down
 *   and read back on any platform. The textual form of `std::mt19937` cannot: libstdc++,
 *   libc++ and the MSVC STL each spell it differently.
 *   See https://github.com/dmlc/xgboost/issues/12459 .
 *
 * The device-side code already draws from the same family through libcu++
 * (`common::cuda_impl::DefaultRng` is `cuda::std::philox4x64`). Host and device still run
 * separate engines over separate streams, so the draws do not line up; only the algorithm
 * is now shared.
 *
 * @tparam UIntType The unsigned result type.
 * @tparam w        Word size in bits.
 * @tparam n        Number of words in the counter and in the output block.
 * @tparam r        Number of rounds.
 * @tparam consts   The n constants, alternating multiplier and round constant.
 */
template <typename UIntType, std::size_t w, std::size_t n, std::size_t r, UIntType... consts>
class PhiloxEngine {
  static_assert(std::is_unsigned_v<UIntType>, "UIntType must be an unsigned integer type.");
  static_assert(sizeof...(consts) == n, "Expecting n constants.");
  static_assert(n == 2 || n == 4, "Only 2 and 4 words are specified.");
  static_assert(r > 0, "At least one round is required.");
  static_assert(w > 0 && w <= std::numeric_limits<UIntType>::digits, "Invalid word size.");
  static_assert(w <= 64, "Words wider than 64 bits are not supported.");

 public:
  using result_type = UIntType;  // NOLINT

  static constexpr std::size_t word_size = w;             // NOLINT
  static constexpr std::size_t word_count = n;            // NOLINT
  static constexpr std::size_t round_count = r;           // NOLINT
  static constexpr result_type default_seed = 20111115u;  // NOLINT

 private:
  static constexpr std::array<result_type, n> kConsts{consts...};

  static constexpr std::array<result_type, n / 2> SelectConsts(std::size_t offset) {
    std::array<result_type, n / 2> out{};
    for (std::size_t k = 0; k < n / 2; ++k) {
      out[k] = kConsts[2 * k + offset];
    }
    return out;
  }

 public:
  /** @brief The multipliers, the even-indexed template constants. */
  static constexpr std::array<result_type, n / 2> multipliers = SelectConsts(0);  // NOLINT
  /** @brief The Weyl round constants, the odd-indexed template constants. */
  static constexpr std::array<result_type, n / 2> round_consts = SelectConsts(1);  // NOLINT

  [[nodiscard]] static constexpr result_type min() { return 0; }  // NOLINT
  [[nodiscard]] static constexpr result_type max() {              // NOLINT
    if constexpr (w == std::numeric_limits<result_type>::digits) {
      return std::numeric_limits<result_type>::max();
    } else {
      return static_cast<result_type>((static_cast<result_type>(1) << w) - 1);
    }
  }

 private:
  // The counter, incremented once per generated block.
  std::array<result_type, n> x_{};
  // The key, derived from the seed.
  std::array<result_type, n / 2> k_{};
  // The current block of outputs.
  std::array<result_type, n> y_{};
  // Index of the next output within the block.
  std::size_t i_{n - 1};

  static constexpr std::uint64_t kMax64 = static_cast<std::uint64_t>(max());

  [[nodiscard]] static result_type MulHi(result_type a, result_type b) {
    std::uint64_t hi = 0, lo = 0;
    detail::UMul64(static_cast<std::uint64_t>(a), static_cast<std::uint64_t>(b), &hi, &lo);
    if constexpr (w == 64) {
      return static_cast<result_type>(hi);
    } else {
      return static_cast<result_type>(((hi << (64 - w)) | (lo >> w)) & kMax64);
    }
  }
  [[nodiscard]] static result_type MulLo(result_type a, result_type b) {
    auto product = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    return static_cast<result_type>(product & kMax64);
  }
  /**
   * @brief The word permutation f_n. Identity for n == 2, (2, 1, 0, 3) for n == 4.
   */
  [[nodiscard]] static constexpr std::size_t Permute(std::size_t j) {
    if constexpr (n == 2) {
      return j;
    } else {
      return j == 0 ? 2 : (j == 2 ? 0 : j);
    }
  }

  /** @brief Fill the output block from the current key and counter. */
  void GenerateBlock() {
    auto state = this->x_;
    for (std::size_t q = 0; q < r; ++q) {
      std::array<result_type, n> v{};
      for (std::size_t j = 0; j < n; ++j) {
        v[j] = state[Permute(j)];
      }
      for (std::size_t k = 0; k < n / 2; ++k) {
        // The key for this round, bumped by a Weyl increment each round.
        auto key = static_cast<result_type>(
            (static_cast<std::uint64_t>(this->k_[k]) +
             static_cast<std::uint64_t>(q) * static_cast<std::uint64_t>(round_consts[k])) &
            kMax64);
        state[2 * k] =
            static_cast<result_type>(MulHi(v[2 * k], multipliers[k]) ^ key ^ v[2 * k + 1]);
        state[2 * k + 1] = MulLo(v[2 * k], multipliers[k]);
      }
    }
    this->y_ = state;
  }

  /** @brief Add @p z to the counter, which is an n * w bit unsigned integer. */
  void AdvanceCounter(std::uint64_t z) {
    for (std::size_t j = 0; j < n && z != 0; ++j) {
      auto const x = static_cast<std::uint64_t>(this->x_[j]);
      std::uint64_t const sum = x + (z & kMax64);
      this->x_[j] = static_cast<result_type>(sum & kMax64);
      if constexpr (w >= 64) {
        // A word this wide has already wrapped modulo 2^64 by the time we get here, so the
        // carry shows up as a sum that came out below the addend. There is no value of
        // `sum` above `kMax64` to compare against.
        z = sum < x ? 1 : 0;
      } else {
        z = (z >> w) + (sum > kMax64 ? 1 : 0);
      }
    }
  }

 public:
  PhiloxEngine() : PhiloxEngine{default_seed} {}
  explicit PhiloxEngine(result_type value) { this->seed(value); }

  /**
   * @brief Restart the sequence from @p value.
   *
   * Sets the first word of the key to @p value, zeros the rest of the key and the whole
   * counter, and positions the engine at the end of a block so that the next draw
   * generates a fresh one.
   */
  void seed(result_type value = default_seed) {  // NOLINT
    this->x_.fill(0);
    this->k_.fill(0);
    this->y_.fill(0);
    this->k_[0] = static_cast<result_type>(static_cast<std::uint64_t>(value) & kMax64);
    this->i_ = n - 1;
  }

  result_type operator()() {
    this->i_++;
    if (this->i_ == n) {
      this->GenerateBlock();
      this->AdvanceCounter(1);
      this->i_ = 0;
    }
    return this->y_[this->i_];
  }

  /**
   * @brief Skip @p z draws.
   *
   * Equivalent to calling @ref operator() @p z times, but the cost does not depend on
   * @p z: at most one block is generated no matter how far ahead we jump.
   */
  void discard(std::uint64_t z) {  // NOLINT
    std::uint64_t const pos = static_cast<std::uint64_t>(this->i_) + z;
    std::uint64_t const n_blocks = pos / n;
    if (n_blocks > 0) {
      // The block we land in was generated from the counter as it stood `n_blocks - 1`
      // steps on, and the counter is left one step past that.
      this->AdvanceCounter(n_blocks - 1);
      this->GenerateBlock();
      this->AdvanceCounter(1);
    }
    this->i_ = static_cast<std::size_t>(pos % n);
  }

  [[nodiscard]] bool operator==(PhiloxEngine const& that) const {
    return this->x_ == that.x_ && this->k_ == that.k_ && this->y_ == that.y_ && this->i_ == that.i_;
  }
  [[nodiscard]] bool operator!=(PhiloxEngine const& that) const { return !(*this == that); }
};

/** @brief Equivalent to `std::philox4x32`. */
using Philox4x32 =
    PhiloxEngine<std::uint32_t, 32, 4, 10, 0xCD9E8D57, 0x9E3779B9, 0xD2511F53, 0xBB67AE85>;
/** @brief Equivalent to `std::philox4x64`. */
using Philox4x64 =
    PhiloxEngine<std::uint64_t, 64, 4, 10, 0xCA5A826395121157ULL, 0x9E3779B97F4A7C15ULL,
                 0xD2E7470EE14C6C93ULL, 0xBB67AE8584CAA73BULL>;
}  // namespace xgboost
#endif  // XGBOOST_PHILOX_ENGINE_H_
