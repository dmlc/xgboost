/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "topo.h"

#include "../common/bitfield.h"  // for TrailingZeroBits, RBitField32
namespace xgboost::collective {
std::int32_t ParentRank(std::int32_t rank, std::int32_t depth) {
  std::uint32_t mask{std::uint32_t{0} - 1};  // Oxff...
  RBitField32 maskbits{common::Span<std::uint32_t>{&mask, 1}};
  RBitField32 rankbits{common::Span<std::uint32_t>{reinterpret_cast<std::uint32_t*>(&rank), 1}};
  // prepare for counting trailing zeros.
  for (std::int32_t i = 0; i < depth + 1; ++i) {
    if (rankbits.Check(i)) {
      maskbits.Set(i);
    } else {
      maskbits.Clear(i);
    }
  }

  CHECK_NE(mask, 0);
  auto k = TrailingZeroBits(mask);
  auto parent = rank - (1 << k);
  return parent;
}
}  // namespace xgboost::collective
