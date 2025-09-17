/**
 * Copyright 2023-2024, XGBoost contributors
 */
#include "../../src/collective/comm.h"  // for Comm, Coll
#include "federated_coll.h"             // for FederatedColl
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
class CUDAFederatedColl : public Coll {
  std::shared_ptr<FederatedColl> p_impl_;

 public:
  explicit CUDAFederatedColl(std::shared_ptr<FederatedColl> pimpl) : p_impl_{std::move(pimpl)} {}
  [[nodiscard]] Result Allreduce(Comm const &comm, common::Span<std::int8_t> data,
                                 ArrayInterfaceHandler::Type type, Op op) override;
  [[nodiscard]] Result Broadcast(Comm const &comm, common::Span<std::int8_t> data,
                                 std::int32_t root) override;
  [[nodiscard]] Result Allgather(Comm const &, common::Span<std::int8_t> data) override;
  [[nodiscard]] Result AllgatherV(Comm const &comm, common::Span<std::int8_t const> data,
                                  common::Span<std::int64_t const> sizes,
                                  common::Span<std::int64_t> recv_segments,
                                  common::Span<std::int8_t> recv, AllgatherVAlgo algo) override;
};
}  // namespace xgboost::collective
