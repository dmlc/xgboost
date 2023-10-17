/**
 * Copyright 2023, XGBoost Contributors
 */
#include "allreduce.h"

#include <algorithm>  // for min
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, int8_t
#include <vector>     // for vector

#include "../data/array_interface.h"    // for Type, DispatchDType
#include "allgather.h"                  // for RingAllgather
#include "comm.h"                       // for Comm
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective::cpu_impl {
template <typename T>
Result RingScatterReduceTyped(Comm const& comm, common::Span<std::int8_t> data,
                              std::size_t n_bytes_in_seg, Func const& op) {
  auto rank = comm.Rank();
  auto world = comm.World();

  auto dst_rank = BootstrapNext(rank, world);
  auto src_rank = BootstrapPrev(rank, world);
  auto next_ch = comm.Chan(dst_rank);
  auto prev_ch = comm.Chan(src_rank);

  std::vector<std::int8_t> buffer(n_bytes_in_seg, 0);
  auto s_buf = common::Span{buffer.data(), buffer.size()};

  for (std::int32_t r = 0; r < world - 1; ++r) {
    // send to ring next
    auto send_off = ((rank + world - r) % world) * n_bytes_in_seg;
    send_off = std::min(send_off, data.size_bytes());
    auto seg_nbytes = std::min(data.size_bytes() - send_off, n_bytes_in_seg);
    auto send_seg = data.subspan(send_off, seg_nbytes);

    next_ch->SendAll(send_seg);

    // receive from ring prev
    auto recv_off = ((rank + world - r - 1) % world) * n_bytes_in_seg;
    recv_off = std::min(recv_off, data.size_bytes());
    seg_nbytes = std::min(data.size_bytes() - recv_off, n_bytes_in_seg);
    CHECK_EQ(seg_nbytes % sizeof(T), 0);
    auto recv_seg = data.subspan(recv_off, seg_nbytes);
    auto seg = s_buf.subspan(0, recv_seg.size());

    prev_ch->RecvAll(seg);
    auto rc = prev_ch->Block();
    if (!rc.OK()) {
      return rc;
    }

    // accumulate to recv_seg
    CHECK_EQ(seg.size(), recv_seg.size());
    op(seg, recv_seg);
  }

  return Success();
}

Result RingAllreduce(Comm const& comm, common::Span<std::int8_t> data, Func const& op,
                     ArrayInterfaceHandler::Type type) {
  return DispatchDType(type, [&](auto t) {
    using T = decltype(t);
    // Divide the data into segments according to the number of workers.
    auto n_bytes_elem = sizeof(T);
    CHECK_EQ(data.size_bytes() % n_bytes_elem, 0);
    auto n = data.size_bytes() / n_bytes_elem;
    auto world = comm.World();
    auto n_bytes_in_seg = common::DivRoundUp(n, world) * sizeof(T);
    auto rc = RingScatterReduceTyped<T>(comm, data, n_bytes_in_seg, op);
    if (!rc.OK()) {
      return rc;
    }

    auto prev = BootstrapPrev(comm.Rank(), comm.World());
    auto next = BootstrapNext(comm.Rank(), comm.World());
    auto prev_ch = comm.Chan(prev);
    auto next_ch = comm.Chan(next);

    rc = RingAllgather(comm, data, n_bytes_in_seg, 1, prev_ch, next_ch);
    if (!rc.OK()) {
      return rc;
    }
    return comm.Block();
  });
}
}  // namespace xgboost::collective::cpu_impl
