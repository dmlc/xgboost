/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once

#include <cstdint>  // for int8_t, int64_t

#include "../common/device_helpers.cuh"  // for CUDAStream
#include "../common/threadpool.h"        // for ThreadPool
#include "../data/array_interface.h"     // for ArrayInterfaceHandler
#include "coll.h"                        // for Coll
#include "comm.h"                        // for Comm
#include "xgboost/span.h"                // for Span

namespace xgboost::collective {
class NCCLColl : public Coll {
  common::ThreadPool pool_;
  dh::CUDAStream stream_;

 public:
  NCCLColl();
  ~NCCLColl() override;

  [[nodiscard]] Result Allreduce(Comm const& comm, common::Span<std::int8_t> data,
                                 ArrayInterfaceHandler::Type type, Op op) override;
  [[nodiscard]] Result Broadcast(Comm const& comm, common::Span<std::int8_t> data,
                                 std::int32_t root) override;
  [[nodiscard]] Result Allgather(Comm const& comm, common::Span<std::int8_t> data) override;
  [[nodiscard]] Result AllgatherV(Comm const& comm, common::Span<std::int8_t const> data,
                                  common::Span<std::int64_t const> sizes,
                                  common::Span<std::int64_t> recv_segments,
                                  common::Span<std::int8_t> recv, AllgatherVAlgo algo) override;
};
}  // namespace xgboost::collective
