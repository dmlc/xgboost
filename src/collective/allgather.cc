/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "allgather.h"

#include <algorithm>  // for min, copy_n, fill_n
#include <cstddef>    // for size_t
#include <cstdint>    // for int8_t, int32_t, int64_t
#include <memory>     // for shared_ptr
#include <utility>    // for move

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
    auto rc = Success() << [&] {
      auto send_rank = (rank + world - r + worker_off) % world;
      auto send_off = send_rank * segment_size;
      bool is_last_segment = send_rank == (world - 1);
      auto send_nbytes = is_last_segment ? (data.size_bytes() - send_off) : segment_size;
      auto send_seg = data.subspan(send_off, send_nbytes);
      CHECK_NE(send_seg.size(), 0);
      return next_ch->SendAll(send_seg.data(), send_seg.size_bytes());
    } << [&] {
      auto recv_rank = (rank + world - r - 1 + worker_off) % world;
      auto recv_off = recv_rank * segment_size;
      bool is_last_segment = recv_rank == (world - 1);
      auto recv_nbytes = is_last_segment ? (data.size_bytes() - recv_off) : segment_size;
      auto recv_seg = data.subspan(recv_off, recv_nbytes);
      CHECK_NE(recv_seg.size(), 0);
      return prev_ch->RecvAll(recv_seg.data(), recv_seg.size_bytes());
    } << [&] {
      return comm.Block();
    };
    if (!rc.OK()) {
      return Fail("Ring allgather failed, current iteration:" + std::to_string(r), std::move(rc));
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
      return Fail("Broadcast AllgatherV failed, current iteration:" + std::to_string(r),
                  std::move(rc));
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
    auto rc = Success() << [&] {
      auto send_rank = (rank + world - r) % world;
      auto send_off = offset[send_rank];
      auto send_size = sizes[send_rank];
      auto send_seg = erased_result.subspan(send_off, send_size);
      return next_ch->SendAll(send_seg);
    } << [&] {
      auto recv_rank = (rank + world - r - 1) % world;
      auto recv_off = offset[recv_rank];
      auto recv_size = sizes[recv_rank];
      auto recv_seg = erased_result.subspan(recv_off, recv_size);
      return prev_ch->RecvAll(recv_seg.data(), recv_seg.size_bytes());
    } << [&] {
      return prev_ch->Block();
    };
    if (!rc.OK()) {
      return Fail("Ring AllgatherV failed, current iterataion:" + std::to_string(r), std::move(rc));
    }
  }
  return comm.Block();
}
}  // namespace detail

[[nodiscard]] std::vector<std::vector<char>> VectorAllgatherV(
    Context const* ctx, CommGroup const& comm, std::vector<std::vector<char>> const& input) {
  auto n_inputs = input.size();
  std::vector<std::int64_t> sizes(n_inputs);
  std::transform(input.cbegin(), input.cend(), sizes.begin(),
                 [](auto const& vec) { return vec.size(); });

  std::vector<std::int64_t> recv_segments(comm.World() + 1, 0);

  HostDeviceVector<std::int8_t> recv;
  auto rc =
      AllgatherV(ctx, comm, linalg::MakeVec(sizes.data(), sizes.size()), &recv_segments, &recv);
  SafeColl(rc);

  auto global_sizes = common::RestoreType<std::int64_t const>(recv.ConstHostSpan());
  std::vector<std::int64_t> offset(global_sizes.size() + 1);
  offset[0] = 0;
  for (std::size_t i = 1; i < offset.size(); i++) {
    offset[i] = offset[i - 1] + global_sizes[i - 1];
  }

  std::vector<char> collected;
  for (auto const& vec : input) {
    collected.insert(collected.end(), vec.cbegin(), vec.cend());
  }
  rc = AllgatherV(ctx, comm, linalg::MakeVec(collected.data(), collected.size()), &recv_segments,
                  &recv);
  SafeColl(rc);
  auto out = common::RestoreType<char const>(recv.ConstHostSpan());

  std::vector<std::vector<char>> result;
  for (std::size_t i = 1; i < offset.size(); ++i) {
    std::vector<char> local(out.cbegin() + offset[i - 1], out.cbegin() + offset[i]);
    result.emplace_back(std::move(local));
  }
  return result;
}

[[nodiscard]] std::vector<std::vector<char>> VectorAllgatherV(
    Context const* ctx, std::vector<std::vector<char>> const& input) {
  return VectorAllgatherV(ctx, *GlobalCommGroup(), input);
}
}  // namespace xgboost::collective
