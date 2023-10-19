/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <memory>       // for shared_ptr
#include <numeric>      // for accumulate
#include <type_traits>  // for remove_cv_t
#include <vector>       // for vector

#include "comm.h"                       // for Comm, Channel, EraseType
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

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

[[nodiscard]] Result RingAllgatherV(Comm const& comm, common::Span<std::int64_t const> sizes,
                                    common::Span<std::int8_t const> data,
                                    common::Span<std::int64_t> offset,
                                    common::Span<std::int8_t> erased_result);
}  // namespace cpu_impl

template <typename T>
[[nodiscard]] Result RingAllgather(Comm const& comm, common::Span<T> data, std::size_t size) {
  auto n_bytes = sizeof(T) * size;
  auto erased = EraseType(data);

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

template <typename T>
[[nodiscard]] Result RingAllgatherV(Comm const& comm, common::Span<T> data,
                                    std::vector<std::remove_cv_t<T>>* p_out) {
  auto world = comm.World();
  auto rank = comm.Rank();

  std::vector<std::int64_t> sizes(world, 0);
  sizes[rank] = data.size_bytes();
  auto rc = RingAllgather(comm, common::Span{sizes.data(), sizes.size()}, 1);
  if (!rc.OK()) {
    return rc;
  }

  std::vector<T>& result = *p_out;
  auto n_total_bytes = std::accumulate(sizes.cbegin(), sizes.cend(), 0);
  result.resize(n_total_bytes / sizeof(T));
  auto h_result = common::Span{result.data(), result.size()};
  auto erased_result = EraseType(h_result);
  auto erased_data = EraseType(data);
  std::vector<std::int64_t> offset(world + 1);

  return cpu_impl::RingAllgatherV(comm, sizes, erased_data,
                                  common::Span{offset.data(), offset.size()}, erased_result);
}
}  // namespace xgboost::collective
