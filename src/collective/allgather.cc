/**
 * Copyright 2023, XGBoost Contributors
 */
#include "allgather.h"

#include <algorithm>  // for min
#include <cstddef>    // for size_t
#include <cstdint>    // for int8_t
#include <memory>     // for shared_ptr

#include "comm.h"          // for Comm, Channel
#include "xgboost/span.h"  // for Span

namespace xgboost::collective::cpu_impl {
Result RingAllgather(Comm const& comm, common::Span<std::int8_t> data, std::size_t segment_size,
                     std::int32_t worker_off, std::shared_ptr<Channel> prev_ch,
                     std::shared_ptr<Channel> next_ch) {
  auto world = comm.World();
  auto rank = comm.Rank();
  CHECK_LT(worker_off, world);

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
}  // namespace xgboost::collective::cpu_impl
