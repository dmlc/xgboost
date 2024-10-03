/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int32_t

namespace xgboost::collective {
inline std::int32_t BootstrapNext(std::int32_t r, std::int32_t world) {
  auto nrank = (r + world + 1) % world;
  return nrank;
}

inline std::int32_t BootstrapPrev(std::int32_t r, std::int32_t world) {
  auto nrank = (r + world - 1) % world;
  return nrank;
}

std::int32_t ParentRank(std::int32_t rank, std::int32_t depth);
}  // namespace xgboost::collective
