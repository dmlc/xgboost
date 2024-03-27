/**
 * Copyright 2023, XGBoost Contributors
 */
#include <cstdint>  // for int8_t, int32_t
#include <memory>   // for dynamic_pointer_cast
#include <vector>   // for vector

#include "../../src/collective/comm.cuh"
#include "../../src/common/cuda_context.cuh"  // for CUDAContext
#include "../../src/data/array_interface.h"   // for ArrayInterfaceHandler::Type
#include "federated_coll.cuh"
#include "federated_comm.cuh"
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/span.h"               // for Span

namespace xgboost::collective {
Coll *FederatedColl::MakeCUDAVar() {
  return new CUDAFederatedColl{std::dynamic_pointer_cast<FederatedColl>(this->shared_from_this())};
}

[[nodiscard]] Result CUDAFederatedColl::Allreduce(Comm const &comm, common::Span<std::int8_t> data,
                                                  ArrayInterfaceHandler::Type type, Op op) {
  auto cufed = dynamic_cast<CUDAFederatedComm const *>(&comm);
  CHECK(cufed);

  std::vector<std::int8_t> h_data(data.size());

  return Success() << [&] {
    return GetCUDAResult(
        cudaMemcpy(h_data.data(), data.data(), data.size(), cudaMemcpyDeviceToHost));
  } << [&] {
    return p_impl_->Allreduce(comm, common::Span{h_data.data(), h_data.size()}, type, op);
  } << [&] {
    return GetCUDAResult(cudaMemcpyAsync(data.data(), h_data.data(), data.size(),
                                         cudaMemcpyHostToDevice, cufed->Stream()));
  };
}

[[nodiscard]] Result CUDAFederatedColl::Broadcast(Comm const &comm, common::Span<std::int8_t> data,
                                                  std::int32_t root) {
  auto cufed = dynamic_cast<CUDAFederatedComm const *>(&comm);
  CHECK(cufed);
  std::vector<std::int8_t> h_data(data.size());

  return Success() << [&] {
    return GetCUDAResult(
        cudaMemcpy(h_data.data(), data.data(), data.size(), cudaMemcpyDeviceToHost));
  } << [&] {
    return p_impl_->Broadcast(comm, common::Span{h_data.data(), h_data.size()}, root);
  } << [&] {
    return GetCUDAResult(cudaMemcpyAsync(data.data(), h_data.data(), data.size(),
                                         cudaMemcpyHostToDevice, cufed->Stream()));
  };
}

[[nodiscard]] Result CUDAFederatedColl::Allgather(Comm const &comm, common::Span<std::int8_t> data) {
  auto cufed = dynamic_cast<CUDAFederatedComm const *>(&comm);
  CHECK(cufed);
  std::vector<std::int8_t> h_data(data.size());

  return Success() << [&] {
    return GetCUDAResult(
        cudaMemcpy(h_data.data(), data.data(), data.size(), cudaMemcpyDeviceToHost));
  } << [&] {
    return p_impl_->Allgather(comm, common::Span{h_data.data(), h_data.size()});
  } << [&] {
    return GetCUDAResult(cudaMemcpyAsync(data.data(), h_data.data(), data.size(),
                                         cudaMemcpyHostToDevice, cufed->Stream()));
  };
}

[[nodiscard]] Result CUDAFederatedColl::AllgatherV(
    Comm const &comm, common::Span<std::int8_t const> data, common::Span<std::int64_t const> sizes,
    common::Span<std::int64_t> recv_segments, common::Span<std::int8_t> recv, AllgatherVAlgo algo) {
  auto cufed = dynamic_cast<CUDAFederatedComm const *>(&comm);
  CHECK(cufed);

  std::vector<std::int8_t> h_data(data.size());
  std::vector<std::int8_t> h_recv(recv.size());

  return Success() << [&] {
    return GetCUDAResult(
        cudaMemcpy(h_data.data(), data.data(), data.size(), cudaMemcpyDeviceToHost));
  } << [&] {
    return this->p_impl_->AllgatherV(comm, h_data, sizes, recv_segments, h_recv, algo);
  } << [&] {
    return GetCUDAResult(cudaMemcpyAsync(recv.data(), h_recv.data(), h_recv.size(),
                                         cudaMemcpyHostToDevice, cufed->Stream()));
  };
}
}  // namespace xgboost::collective
