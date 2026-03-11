/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#include "broadcast.h"

#include <cmath>    // for ceil, log2
#include <cstdint>  // for int32_t, int8_t
#include <utility>  // for move
#include <vector>   // for vector

#include "../common/bitfield.h"         // for TrailingZeroBits, RBitField32
#include "comm.h"                       // for Comm
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective::cpu_impl {
namespace {
// Binomial tree broadcast using a fixed tree rooted at rank 0.
Result BroadcastTree(Comm const& comm, common::Span<std::int8_t> data) {
  auto rank = comm.Rank();
  auto world = comm.World();
  std::int32_t depth = std::ceil(std::log2(static_cast<double>(world))) - 1;

  if (rank != 0) {
    auto k = TrailingZeroBits(static_cast<std::uint32_t>(rank));
    auto parent = rank - (1 << k);
    auto rc = Success() << [&] { return comm.Chan(parent)->RecvAll(data); }
                        << [&] { return comm.Chan(parent)->Block(); };
    if (!rc.OK()) {
      return Fail("broadcast failed.", std::move(rc));
    }
  }

  for (std::int32_t i = depth; i >= 0; --i) {
    if (rank % (1 << (i + 1)) == 0 && rank + (1 << i) < world) {
      auto child = rank + (1 << i);
      auto rc = comm.Chan(child)->SendAll(data);
      if (!rc.OK()) {
        return rc;
      }
    }
  }

  return comm.Block();
}

// Compute the path from `src` to rank 0 through the binomial tree (excluding 0).
std::vector<std::int32_t> TreePathToRoot(std::int32_t src) {
  std::vector<std::int32_t> path;
  auto cursor = src;
  while (cursor > 0) {
    path.push_back(cursor);
    auto k = TrailingZeroBits(static_cast<std::uint32_t>(cursor));
    cursor -= (1 << k);
  }
  return path;
}

// Relay data from `root` up to rank 0 through the binomial tree.
// Only nodes on the path from root to 0 participate; all others skip.
Result RelayToRoot(Comm const& comm, common::Span<std::int8_t> data, std::int32_t root) {
  auto rank = comm.Rank();
  auto path = TreePathToRoot(root);

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(path.size()); ++i) {
    auto node = path[i];
    auto k = TrailingZeroBits(static_cast<std::uint32_t>(node));
    auto parent = node - (1 << k);

    if (rank == node) {
      auto rc = Success() << [&] { return comm.Chan(parent)->SendAll(data); }
                          << [&] { return comm.Chan(parent)->Block(); };
      if (!rc.OK()) {
        return Fail("Relay broadcast: failed to send from " + std::to_string(node),
                     std::move(rc));
      }
    } else if (rank == parent) {
      auto rc = Success() << [&] { return comm.Chan(node)->RecvAll(data); }
                          << [&] { return comm.Chan(node)->Block(); };
      if (!rc.OK()) {
        return Fail("Relay broadcast: failed to recv at " + std::to_string(parent),
                     std::move(rc));
      }
    }
  }
  return Success();
}
}  // namespace

Result Broadcast(Comm const& comm, common::Span<std::int8_t> data, std::int32_t root) {
  if (comm.World() <= 1) {
    return Success();
  }

  if (root == 0) {
    return BroadcastTree(comm, data);
  }

  // For non-zero root, relay data up to rank 0 through the tree, then broadcast.
  auto rc = RelayToRoot(comm, data, root);
  if (!rc.OK()) {
    return Fail("Relay broadcast failed.", std::move(rc));
  }
  return BroadcastTree(comm, data);
}
}  // namespace xgboost::collective::cpu_impl
