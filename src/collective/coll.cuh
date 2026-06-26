/**
 * Copyright 2023-2026, XGBoost Contributors
 */
#pragma once

#include <cstdint>  // for int8_t, int64_t

#include "../common/threadpool.h"     // for ThreadPool
#include "../data/array_interface.h"  // for ArrayInterfaceHandler
#include "coll.h"                     // for Coll
#include "comm.h"                     // for Comm
#include "xgboost/span.h"             // for Span

namespace xgboost::collective {
class NCCLColl : public Coll {
  common::ThreadPool pool_;

 public:
  NCCLColl();
  ~NCCLColl() override;

  [[nodiscard]] Result Allreduce(Context const* ctx, Comm const& comm,
                                 common::Span<std::int8_t> data,
                                 ArrayInterfaceHandler::Type type, Op op) override;
  [[nodiscard]] Result Broadcast(Context const* ctx, Comm const& comm,
                                 common::Span<std::int8_t> data, std::int32_t root) override;
  [[nodiscard]] Result Allgather(Context const* ctx, Comm const& comm,
                                 common::Span<std::int8_t> data) override;
  [[nodiscard]] Result AllgatherV(Context const* ctx, Comm const& comm,
                                  common::Span<std::int8_t const> data,
                                  common::Span<std::int64_t const> sizes,
                                  common::Span<std::int64_t> recv_segments,
                                  common::Span<std::int8_t> recv, AllgatherVAlgo algo) override;
};
}  // namespace xgboost::collective
