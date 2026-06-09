/**
 * Copyright 2026, XGBoost Contributors
 */
#pragma once

#include <cstddef>  // for size_t
#include <cstdint>  // for uint8_t, uint64_t
#include <variant>  // for visit

#include "../encoder/ordinal.h"  // for HostColumnsView
#include "../encoder/types.h"    // for Overloaded

namespace xgboost::data {
// Primary stream is standard FNV-1a 64-bit (offset basis + prime); secondary stream
// uses CityHash/FarmHash mixing constants for an independent fold. Agreement on both
// components after Allreduce kMin+kMax indicates dictionary parity.
inline constexpr std::uint64_t kHashSeedPrimary = 0xcbf29ce484222325ULL;
inline constexpr std::uint64_t kHashSeedSecondary = 0x9ae16a3b2f90404fULL;
inline constexpr std::uint64_t kHashPrimePrimary = 0x100000001b3ULL;
inline constexpr std::uint64_t kHashPrimeSecondary = 0xc2b2ae3d27d4eb4fULL;
static_assert(kHashPrimePrimary != kHashPrimeSecondary,
              "secondary prime must differ from primary to keep streams independent");
static_assert(kHashSeedPrimary != kHashSeedSecondary,
              "secondary seed must differ from primary to keep streams independent");

/** @brief Dual 64-bit content digest used for cross-worker dictionary parity. */
struct CatContentDigest {
  std::uint64_t primary;
  std::uint64_t secondary;

  friend bool operator==(CatContentDigest const& a, CatContentDigest const& b) {
    return a.primary == b.primary && a.secondary == b.secondary;
  }
  friend bool operator!=(CatContentDigest const& a, CatContentDigest const& b) {
    return !(a == b);
  }
};

/**
 * @brief Fold a host CatContainer view's bytes into the dual-stream content digest.
 *
 * @param view Host columns view; offsets and values bytes are folded per column.
 * @return Dual-component digest seeded with `n_total_cats` and `columns.size()`.
 */
[[nodiscard]] inline CatContentDigest HashCatHostContent(enc::HostColumnsView const& view) {
  // seed with n_total_cats + columns.size() so empty-bytes workers diverge on shape
  std::uint64_t h1 = kHashSeedPrimary ^ static_cast<std::uint64_t>(view.n_total_cats);
  std::uint64_t h2 = kHashSeedSecondary ^ static_cast<std::uint64_t>(view.columns.size());
  auto fold = [&](void const* p, std::size_t n) {
    auto const* b = static_cast<std::uint8_t const*>(p);
    for (std::size_t i = 0; i < n; ++i) {
      h1 = (h1 ^ b[i]) * kHashPrimePrimary;
      h2 = (h2 ^ b[i]) * kHashPrimeSecondary;
    }
  };
  for (auto const& col : view.columns) {
    std::visit(enc::Overloaded{[&](enc::CatStrArrayView const& s) {
                                 fold(s.offsets.data(), s.offsets.size_bytes());
                                 fold(s.values.data(), s.values.size_bytes());
                               },
                               [&](auto const& idx) {
                                 fold(idx.data(), idx.size_bytes());
                               }},
               col);
  }
  return {h1, h2};
}
}  // namespace xgboost::data
