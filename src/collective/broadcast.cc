/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#include "broadcast.h"

#include <cstdint>  // for int32_t, int8_t
#include <utility>  // for move
#include <vector>   // for vector

#include "comm.h"                       // for Comm, binomial_tree
#include "topo.h"                       // for Parent, Child
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective::cpu_impl {
namespace {
// Binomial tree broadcast using a fixed tree rooted at rank 0.
Result BroadcastTree(Comm const& comm, common::Span<std::int8_t> data) {
  auto rank = comm.Rank();
  auto world = comm.World();

  if (rank != 0) {
    auto parent = binomial_tree::Parent(rank);
    auto rc = Success() << [&] {
      return comm.Chan(parent)->RecvAll(data);
    } << [&] {
      return comm.Chan(parent)->Block();
    };
    if (!rc.OK()) {
      return Fail("broadcast failed.", std::move(rc));
    }
  }

  for (std::int32_t level = binomial_tree::Depth(world); level >= 0; --level) {
    if (binomial_tree::HasChild(rank, level, world)) {
      auto child = binomial_tree::Child(rank, level);
      auto rc = comm.Chan(child)->SendAll(data);
      if (!rc.OK()) {
        return rc;
      }
    }
  }

  return comm.Block();
}

// Compute the path from `src` to rank 0 through the binomial tree (excluding 0).
std::vector<std::int32_t> TreePathToRoot(std::int32_t node) {
  std::vector<std::int32_t> path;
  auto cursor = node;
  while (cursor > 0) {
    path.push_back(cursor);
    cursor = binomial_tree::Parent(cursor);
  }
  return path;
}

// Relay data from `node` up to rank 0 through the binomial tree.
// Only nodes on the path from `node` to 0 participate; all others skip.
Result RelayToRoot(Comm const& comm, common::Span<std::int8_t> data, std::int32_t node) {
  auto rank = comm.Rank();
  auto path = TreePathToRoot(node);

  for (auto node : path) {
    CHECK_GT(node, 0);
    auto parent = binomial_tree::Parent(node);

    if (rank == node) {
      auto rc = Success() << [&] {
        return comm.Chan(parent)->SendAll(data);
      } << [&] {
        return comm.Chan(parent)->Block();
      };
      if (!rc.OK()) {
        return Fail("Relay broadcast: failed to send from " + std::to_string(node), std::move(rc));
      }
    } else if (rank == parent) {
      auto rc = Success() << [&] {
        return comm.Chan(node)->RecvAll(data);
      } << [&] {
        return comm.Chan(node)->Block();
      };
      if (!rc.OK()) {
        return Fail("Relay broadcast: failed to recv at " + std::to_string(parent), std::move(rc));
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
  CHECK(!data.empty());

  if (root == 0) {
    return BroadcastTree(comm, data);
  }

  // For non-zero root, relay data up to rank 0 through the tree, then broadcast.
  return Success() << [&] {
    return RelayToRoot(comm, data, root);
  } << [&] {
    return BroadcastTree(comm, data);
  };
}
}  // namespace xgboost::collective::cpu_impl
