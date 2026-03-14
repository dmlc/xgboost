/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#pragma once

#include <cstdint>  // for int32_t
#include <set>      // for set
#include <vector>   // for vector

#include "../common/bitfield.h"  // for TrailingZeroBits

namespace xgboost::collective {

// Indexing into the ring
inline std::int32_t BootstrapNext(std::int32_t r, std::int32_t world) {
  auto nrank = (r + world + 1) % world;
  return nrank;
}

inline std::int32_t BootstrapPrev(std::int32_t r, std::int32_t world) {
  auto nrank = (r + world - 1) % world;
  return nrank;
}

/**
 * @brief Helpers for the binomial tree rooted at rank 0.
 *
 * References
 * - https://people.mpi-inf.mpg.de/~mehlhorn/ftp/NewToolbox/collective.pdf
 * - https://en.wikipedia.org/wiki/Broadcast_(parallel_pattern)
 */
namespace binomial_tree {
inline std::int32_t ParentLevel(std::int32_t rank) {
  CHECK_GT(rank, 0);
  return static_cast<std::int32_t>(TrailingZeroBits(static_cast<std::uint32_t>(rank)));
}

inline std::int32_t Parent(std::int32_t rank) {
  return rank - (std::int32_t{1} << ParentLevel(rank));
}

inline std::int32_t Child(std::int32_t rank, std::int32_t level) {
  return rank + (std::int32_t{1} << level);
}

inline bool HasChild(std::int32_t rank, std::int32_t level, std::int32_t world) {
  return rank % (std::int32_t{1} << (level + 1)) == 0 && Child(rank, level) < world;
}

inline std::int32_t Depth(std::int32_t world) {
  if (world <= 1) return -1;
  std::int32_t depth = 0;
  while ((std::int32_t{1} << (depth + 1)) < world) {
    ++depth;
  }
  return depth;
}
}  // namespace binomial_tree

/**
 * @brief Compute the sparse peer set for a given rank: ring neighbors union binomial tree
 *        neighbors (rooted at rank 0).
 */
inline std::vector<std::int32_t> SparsePeers(std::int32_t rank, std::int32_t world) {
  if (world <= 1) {
    return {};
  }
  std::set<std::int32_t> peers;

  peers.insert(BootstrapNext(rank, world));
  peers.insert(BootstrapPrev(rank, world));

  // Connect tree parents and children
  if (rank > 0) {
    peers.insert(binomial_tree::Parent(rank));
  }

  for (std::int32_t level = 0; (std::int32_t{1} << level) < world; ++level) {
    if (binomial_tree::HasChild(rank, level, world)) {
      peers.insert(binomial_tree::Child(rank, level));
    }
  }

  peers.erase(rank);
  return {peers.begin(), peers.end()};
}

}  // namespace xgboost::collective
