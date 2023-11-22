/**
 * Copyright 2023, XGBoost Contributors
 */
#include "broadcast.h"

#include <cmath>    // for ceil, log2
#include <cstdint>  // for int32_t, int8_t
#include <utility>  // for move

#include "../common/bitfield.h"         // for TrailingZeroBits, RBitField32
#include "comm.h"                       // for Comm
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective::cpu_impl {
namespace {
std::int32_t ShiftedParentRank(std::int32_t shifted_rank, std::int32_t depth) {
  std::uint32_t mask{std::uint32_t{0} - 1};  // Oxff...
  RBitField32 maskbits{common::Span<std::uint32_t>{&mask, 1}};
  RBitField32 rankbits{
      common::Span<std::uint32_t>{reinterpret_cast<std::uint32_t*>(&shifted_rank), 1}};
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
  auto shifted_parent = shifted_rank - (1 << k);
  return shifted_parent;
}

// Shift the root node to rank 0
std::int32_t ShiftLeft(std::int32_t rank, std::int32_t world, std::int32_t root) {
  auto shifted_rank = (rank + world - root) % world;
  return shifted_rank;
}
// shift back to the original rank
std::int32_t ShiftRight(std::int32_t rank, std::int32_t world, std::int32_t root) {
  auto orig = (rank + root) % world;
  return orig;
}
}  // namespace

Result Broadcast(Comm const& comm, common::Span<std::int8_t> data, std::int32_t root) {
  // Binomial tree broadcast
  // * Wiki
  // https://en.wikipedia.org/wiki/Broadcast_(parallel_pattern)#Binomial_Tree_Broadcast
  // * Impl
  // https://people.mpi-inf.mpg.de/~mehlhorn/ftp/NewToolbox/collective.pdf

  auto rank = comm.Rank();
  auto world = comm.World();

  // shift root to rank 0
  auto shifted_rank = ShiftLeft(rank, world, root);
  std::int32_t depth = std::ceil(std::log2(static_cast<double>(world))) - 1;

  if (shifted_rank != 0) {  // not root
    auto parent = ShiftRight(ShiftedParentRank(shifted_rank, depth), world, root);
    auto rc = Success() << [&] { return comm.Chan(parent)->RecvAll(data); }
                        << [&] { return comm.Chan(parent)->Block(); };
    if (!rc.OK()) {
      return Fail("broadcast failed.", std::move(rc));
    }
  }

  for (std::int32_t i = depth; i >= 0; --i) {
    CHECK_GE((i + 1), 0);  // weird clang-tidy error that i might be negative
    if (shifted_rank % (1 << (i + 1)) == 0 && shifted_rank + (1 << i) < world) {
      auto sft_peer = shifted_rank + (1 << i);
      auto peer = ShiftRight(sft_peer, world, root);
      CHECK_NE(peer, root);
      auto rc = comm.Chan(peer)->SendAll(data);
      if (!rc.OK()) {
        return rc;
      }
    }
  }

  return comm.Block();
}
}  // namespace xgboost::collective::cpu_impl
