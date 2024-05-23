/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "allreduce.h"

#include <algorithm>  // for min
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, int8_t
#include <utility>    // for move
#include <vector>     // for vector

#include "../data/array_interface.h"    // for Type, DispatchDType
#include "allgather.h"                  // for RingAllgather
#include "comm.h"                       // for Comm
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective::cpu_impl {
namespace {
template <typename T>
Result RingAllreduceSmall(Comm const& comm, common::Span<std::int8_t> data, Func const& op) {
  auto rank = comm.Rank();
  auto world = comm.World();

  auto next_ch = comm.Chan(BootstrapNext(rank, world));
  auto prev_ch = comm.Chan(BootstrapPrev(rank, world));

  std::vector<std::int8_t> buffer(data.size_bytes() * world, 0);
  auto s_buffer = common::Span{buffer.data(), buffer.size()};

  auto offset = data.size_bytes() * rank;
  auto self = s_buffer.subspan(offset, data.size_bytes());
  std::copy_n(data.data(), data.size_bytes(), self.data());

  auto typed = common::RestoreType<T>(s_buffer);
  auto rc = RingAllgather(comm, typed);

  if (!rc.OK()) {
    return Fail("Ring allreduce small failed.", std::move(rc));
  }
  auto first = s_buffer.subspan(0, data.size_bytes());
  CHECK_EQ(first.size(), data.size());

  for (std::int32_t r = 1; r < world; ++r) {
    auto offset = data.size_bytes() * r;
    auto buf = s_buffer.subspan(offset, data.size_bytes());
    op(buf, first);
  }
  std::copy_n(first.data(), first.size(), data.data());

  return Success();
}
}  // namespace

template <typename T>
// note that n_bytes_in_seg is calculated with round-down.
Result RingScatterReduceTyped(Comm const& comm, common::Span<std::int8_t> data,
                              std::size_t n_bytes_in_seg, Func const& op) {
  auto rank = comm.Rank();
  auto world = comm.World();

  auto dst_rank = BootstrapNext(rank, world);
  auto src_rank = BootstrapPrev(rank, world);
  auto next_ch = comm.Chan(dst_rank);
  auto prev_ch = comm.Chan(src_rank);

  std::vector<std::int8_t> buffer(data.size_bytes() - (world - 1) * n_bytes_in_seg, -1);
  auto s_buf = common::Span{buffer.data(), buffer.size()};

  for (std::int32_t r = 0; r < world - 1; ++r) {
    common::Span<std::int8_t> seg, recv_seg;
    auto rc = Success() << [&] {
      // send to ring next
      auto send_rank = (rank + world - r) % world;
      auto send_off = send_rank * n_bytes_in_seg;

      bool is_last_segment = send_rank == (world - 1);

      auto seg_nbytes = is_last_segment ? data.size_bytes() - send_off : n_bytes_in_seg;
      CHECK_EQ(seg_nbytes % sizeof(T), 0);

      auto send_seg = data.subspan(send_off, seg_nbytes);
      return next_ch->SendAll(send_seg);
    } << [&] {
      // receive from ring prev
      auto recv_rank = (rank + world - r - 1) % world;
      auto recv_off = recv_rank * n_bytes_in_seg;

      bool is_last_segment = recv_rank == (world - 1);

      auto seg_nbytes = is_last_segment ? (data.size_bytes() - recv_off) : n_bytes_in_seg;
      CHECK_EQ(seg_nbytes % sizeof(T), 0);

      recv_seg = data.subspan(recv_off, seg_nbytes);
      seg = s_buf.subspan(0, recv_seg.size());
      return prev_ch->RecvAll(seg);
    } << [&] {
      return comm.Block();
    };
    if (!rc.OK()) {
      return Fail("Ring scatter reduce failed, current iteration:" + std::to_string(r),
                  std::move(rc));
    }

    // accumulate to recv_seg
    CHECK_EQ(seg.size(), recv_seg.size());
    op(seg, recv_seg);
  }

  return Success();
}

Result RingAllreduce(Comm const& comm, common::Span<std::int8_t> data, Func const& op,
                     ArrayInterfaceHandler::Type type) {
  if (comm.World() == 1) {
    return Success();
  }
  if (data.size_bytes() == 0) {
    return Success();
  }
  return DispatchDType(type, [&](auto t) {
    using T = decltype(t);
    // Divide the data into segments according to the number of workers.
    auto n_bytes_elem = sizeof(T);
    CHECK_EQ(data.size_bytes() % n_bytes_elem, 0);
    auto n = data.size_bytes() / n_bytes_elem;
    auto world = comm.World();
    if (n < static_cast<decltype(n)>(world)) {
      return RingAllreduceSmall<T>(comm, data, op);
    }

    auto n_bytes_in_seg = (n / world) * sizeof(T);
    auto rc = RingScatterReduceTyped<T>(comm, data, n_bytes_in_seg, op);
    if (!rc.OK()) {
      return Fail("Ring Allreduce failed.", std::move(rc));
    }

    auto prev = BootstrapPrev(comm.Rank(), comm.World());
    auto next = BootstrapNext(comm.Rank(), comm.World());
    auto prev_ch = comm.Chan(prev);
    auto next_ch = comm.Chan(next);

    return std::move(rc) << [&] {
      return RingAllgather(comm, data, n_bytes_in_seg, 1, prev_ch, next_ch);
    } << [&] {
      return comm.Block();
    };
  });
}
}  // namespace xgboost::collective::cpu_impl
