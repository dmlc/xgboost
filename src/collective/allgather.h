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
}  // namespace xgboost::collective
