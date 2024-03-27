/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <memory>       // for shared_ptr
#include <numeric>      // for accumulate
#include <string>       // for string
#include <type_traits>  // for remove_cv_t
#include <vector>       // for vector

#include "../common/type.h"             // for EraseType
#include "comm.h"                       // for Comm, Channel
#include "comm_group.h"                 // for CommGroup
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/linalg.h"             // for MakeVec
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
namespace cpu_impl {
/**
 * @param worker_off Segment offset. For example, if the rank 2 worker specifies
 *                   worker_off = 1, then it owns the third segment (2 + 1).
 */
[[nodiscard]] Result RingAllgather(Comm const& comm, common::Span<std::int8_t> data,
                                   std::size_t segment_size, std::int32_t worker_off,
                                   std::shared_ptr<Channel> prev_ch,
                                   std::shared_ptr<Channel> next_ch);

/**
 * @brief Implement allgather-v using broadcast.
 *
 * https://arxiv.org/abs/1812.05964
 */
Result BroadcastAllgatherV(Comm const& comm, common::Span<std::int64_t const> sizes,
                           common::Span<std::int8_t> recv);
}  // namespace cpu_impl

namespace detail {
inline void AllgatherVOffset(common::Span<std::int64_t const> sizes,
                             common::Span<std::int64_t> offset) {
  // get worker offset
  std::fill_n(offset.data(), offset.size(), 0);
  std::partial_sum(sizes.cbegin(), sizes.cend(), offset.begin() + 1);
  CHECK_EQ(*offset.cbegin(), 0);
}

// An implementation that's used by both cpu and gpu
[[nodiscard]] Result RingAllgatherV(Comm const& comm, common::Span<std::int64_t const> sizes,
                                    common::Span<std::int64_t const> offset,
                                    common::Span<std::int8_t> erased_result);
}  // namespace detail

template <typename T>
[[nodiscard]] Result RingAllgather(Comm const& comm, common::Span<T> data) {
  // This function is also used for ring allreduce, hence we allow the last segment to be
  // larger due to round-down.
  auto n_bytes_per_segment = data.size_bytes() / comm.World();
  auto erased = common::EraseType(data);

  auto rank = comm.Rank();
  auto prev = BootstrapPrev(rank, comm.World());
  auto next = BootstrapNext(rank, comm.World());

  auto prev_ch = comm.Chan(prev);
  auto next_ch = comm.Chan(next);
  auto rc = cpu_impl::RingAllgather(comm, erased, n_bytes_per_segment, 0, prev_ch, next_ch);
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
  auto rc = RingAllgather(comm, common::Span{sizes.data(), sizes.size()});
  if (!rc.OK()) {
    return rc;
  }

  std::vector<T>& result = *p_out;
  auto n_total_bytes = std::accumulate(sizes.cbegin(), sizes.cend(), 0);
  result.resize(n_total_bytes / sizeof(T));
  auto h_result = common::Span{result.data(), result.size()};
  auto erased_result = common::EraseType(h_result);
  auto erased_data = common::EraseType(data);
  std::vector<std::int64_t> recv_segments(world + 1);
  auto s_segments = common::Span{recv_segments.data(), recv_segments.size()};

  // get worker offset
  detail::AllgatherVOffset(sizes, s_segments);
  // copy data
  auto current = erased_result.subspan(recv_segments[rank], data.size_bytes());
  std::copy_n(erased_data.data(), erased_data.size(), current.data());

  return detail::RingAllgatherV(comm, sizes, s_segments, erased_result);
}
}  // namespace xgboost::collective
