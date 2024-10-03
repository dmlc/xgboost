/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "broadcast.h"

#include <cmath>    // for ceil, log2
#include <cstdint>  // for int32_t, int8_t
#include <utility>  // for move


#include "comm.h"                       // for Comm
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span
#include "topo.h"                       // for ParentRank

namespace xgboost::collective::cpu_impl {
Result Broadcast(Comm const& comm, common::Span<std::int8_t> data, std::int32_t root) {
  // Binomial tree broadcast
  // * Wiki
  // https://en.wikipedia.org/wiki/Broadcast_(parallel_pattern)#Binomial_Tree_Broadcast
  // * Impl
  // https://people.mpi-inf.mpg.de/~mehlhorn/ftp/NewToolbox/collective.pdf

  auto rank = comm.Rank();
  auto world = comm.World();

  // Send data to the root to preserve the topology. Alternative is to shift the rank, but
  // it requires a all-to-all connection.
  //
  // Most of the use of broadcasting in XGBoost are short messages, this should be
  // fine. Otherwise, we can implement a linear pipeline broadcast.
  if (root != 0) {
    auto rc = Success() << [&] {
      return (rank == 0) ? comm.Chan(root)->RecvAll(data) : Success();
    } << [&] {
      return (rank == root) ? comm.Chan(0)->SendAll(data) : Success();
    } << [&] {
      return comm.Block();
    };
    if (!rc.OK()) {
      return Fail("Broadcast failed to send data to root.", std::move(rc));
    }
    root = 0;
  }

  std::int32_t depth = std::ceil(std::log2(static_cast<double>(world))) - 1;

  if (rank != 0) {  // not root
    auto parent = ParentRank(rank, depth);
    auto rc = Success() << [&] {
      return comm.Chan(parent)->RecvAll(data);
    } << [&] {
      return comm.Chan(parent)->Block();
    };
    if (!rc.OK()) {
      return Fail("Broadcast failed to send data to parent.", std::move(rc));
    }
  }

  for (std::int32_t i = depth; i >= 0; --i) {
    CHECK_GE((i + 1), 0);  // weird clang-tidy error that i might be negative
    if (rank % (1 << (i + 1)) == 0 && rank + (1 << i) < world) {
      auto peer = rank + (1 << i);
      CHECK_NE(peer, root);
      auto rc = comm.Chan(peer)->SendAll(data);
      if (!rc.OK()) {
        return Fail("Failed to seed to " + std::to_string(peer), std::move(rc));
      }
    }
  }

  return comm.Block();
}
}  // namespace xgboost::collective::cpu_impl
