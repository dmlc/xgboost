/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int32_t, int8_t

#include "../common/type.h"
#include "comm.h"                       // for Comm, EraseType
#include "comm_group.h"                 // for CommGroup
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/context.h"            // for Context
#include "xgboost/linalg.h"             // for VectorView
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
namespace cpu_impl {
Result Broadcast(Comm const& comm, common::Span<std::int8_t> data, std::int32_t root);
}

/**
 * @brief binomial tree broadcast is used on CPU with the default implementation.
 */
template <typename T>
[[nodiscard]] Result Broadcast(Comm const& comm, common::Span<T> data, std::int32_t root) {
  auto n_total_bytes = data.size_bytes();
  auto erased =
      common::Span<std::int8_t>{reinterpret_cast<std::int8_t*>(data.data()), n_total_bytes};
  return cpu_impl::Broadcast(comm, erased, root);
}

template <typename T>
[[nodiscard]] Result Broadcast(Context const* ctx, CommGroup const& comm,
                               linalg::VectorView<T> data, std::int32_t root) {
  if (!comm.IsDistributed()) {
    return Success();
  }
  CHECK(data.Contiguous());
  auto erased = common::EraseType(data.Values());
  auto backend = comm.Backend(data.Device());
  return backend->Broadcast(comm.Ctx(ctx, data.Device()), erased, root);
}

template <typename T>
[[nodiscard]] Result Broadcast(Context const* ctx, linalg::VectorView<T> data, std::int32_t root) {
  return Broadcast(ctx, *GlobalCommGroup(), data, root);
}
}  // namespace xgboost::collective
