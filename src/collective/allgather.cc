/**
 * Copyright 2023, XGBoost Contributors
 */
#include "allgather.h"

#include <algorithm>  // for min, copy_n, fill_n
#include <cstddef>    // for size_t
#include <cstdint>    // for int8_t, int32_t, int64_t
#include <memory>     // for shared_ptr

#include "broadcast.h"
#include "comm.h"                       // for Comm, Channel
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
namespace cpu_impl {
Result RingAllgather(Comm const& comm, common::Span<std::int8_t> data, std::size_t segment_size,
                     std::int32_t worker_off, std::shared_ptr<Channel> prev_ch,
                     std::shared_ptr<Channel> next_ch) {
  auto world = comm.World();
  auto rank = comm.Rank();
  CHECK_LT(worker_off, world);
  if (world == 1) {
    return Success();
  }

  for (std::int32_t r = 0; r < world; ++r) {
    auto send_rank = (rank + world - r + worker_off) % world;
    auto send_off = send_rank * segment_size;
    send_off = std::min(send_off, data.size_bytes());
    auto send_seg = data.subspan(send_off, std::min(segment_size, data.size_bytes() - send_off));
    next_ch->SendAll(send_seg.data(), send_seg.size_bytes());

    auto recv_rank = (rank + world - r - 1 + worker_off) % world;
    auto recv_off = recv_rank * segment_size;
    recv_off = std::min(recv_off, data.size_bytes());
    auto recv_seg = data.subspan(recv_off, std::min(segment_size, data.size_bytes() - recv_off));
    prev_ch->RecvAll(recv_seg.data(), recv_seg.size_bytes());
    auto rc = prev_ch->Block();
    if (!rc.OK()) {
      return rc;
    }
  }

  return Success();
}

Result BroadcastAllgatherV(Comm const& comm, common::Span<std::int64_t const> sizes,
                           common::Span<std::int8_t> recv) {
  std::size_t offset = 0;
  for (std::int32_t r = 0; r < comm.World(); ++r) {
    auto as_bytes = sizes[r];
    auto rc = Broadcast(comm, recv.subspan(offset, as_bytes), r);
    if (!rc.OK()) {
      return rc;
    }
    offset += as_bytes;
  }
  return Success();
}
}  // namespace cpu_impl

namespace detail {
[[nodiscard]] Result RingAllgatherV(Comm const& comm, common::Span<std::int64_t const> sizes,
                                    common::Span<std::int64_t const> offset,
                                    common::Span<std::int8_t> erased_result) {
  auto world = comm.World();
  if (world == 1) {
    return Success();
  }
  auto rank = comm.Rank();

  auto prev = BootstrapPrev(rank, comm.World());
  auto next = BootstrapNext(rank, comm.World());

  auto prev_ch = comm.Chan(prev);
  auto next_ch = comm.Chan(next);

  for (std::int32_t r = 0; r < world; ++r) {
    auto send_rank = (rank + world - r) % world;
    auto send_off = offset[send_rank];
    auto send_size = sizes[send_rank];
    auto send_seg = erased_result.subspan(send_off, send_size);
    next_ch->SendAll(send_seg);

    auto recv_rank = (rank + world - r - 1) % world;
    auto recv_off = offset[recv_rank];
    auto recv_size = sizes[recv_rank];
    auto recv_seg = erased_result.subspan(recv_off, recv_size);
    prev_ch->RecvAll(recv_seg.data(), recv_seg.size_bytes());

    auto rc = prev_ch->Block();
    if (!rc.OK()) {
      return rc;
    }
  }
  return comm.Block();
}
}  // namespace detail
}  // namespace xgboost::collective
