/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t
#include <memory>   // for shared_ptr

#include "comm.h"          // for Comm, Channel
#include "xgboost/span.h"  // for Span

namespace xgboost::collective {
namespace cpu_impl {
/**
 * @param worker_off Segment offset. For example, if the rank 2 worker specifis worker_off
 *                   = 1, then it owns the third segment.
 */
[[nodiscard]] Result RingAllgather(Comm const& comm, common::Span<std::int8_t> data,
                                   std::size_t segment_size, std::int32_t worker_off,
                                   std::shared_ptr<Channel> prev_ch,
                                   std::shared_ptr<Channel> next_ch);
}  // namespace cpu_impl

template <typename T>
[[nodiscard]] Result RingAllgather(Comm const& comm, common::Span<T> data, std::size_t size) {
  auto n_total_bytes = data.size_bytes();
  auto n_bytes = sizeof(T) * size;
  auto erased =
      common::Span<std::int8_t>{reinterpret_cast<std::int8_t*>(data.data()), n_total_bytes};

  auto rank = comm.Rank();
  auto prev = BootstrapPrev(rank, comm.World());
  auto next = BootstrapNext(rank, comm.World());

  auto prev_ch = comm.Chan(prev);
  auto next_ch = comm.Chan(next);
  auto rc = cpu_impl::RingAllgather(comm, erased, n_bytes, 0, prev_ch, next_ch);
  if (!rc.OK()) {
    return rc;
  }
  return comm.Block();
}
}  // namespace xgboost::collective
