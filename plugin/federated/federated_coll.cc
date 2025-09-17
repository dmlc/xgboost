/**
 * Copyright 2023, XGBoost contributors
 */
#include "federated_coll.h"

#include <federated.grpc.pb.h>
#include <federated.pb.h>

#include <algorithm>  // for copy_n

#include "../../src/collective/allgather.h"
#include "../../src/common/common.h"    // for AssertGPUSupport
#include "federated_comm.h"             // for FederatedComm
#include "xgboost/collective/result.h"  // for Result

namespace xgboost::collective {
namespace {
[[nodiscard]] Result GetGRPCResult(std::string const &name, grpc::Status const &status) {
  return Fail(name + " RPC failed. " + std::to_string(status.error_code()) + ": " +
              status.error_message());
}

[[nodiscard]] Result BroadcastImpl(Comm const &comm, std::uint64_t *sequence_number,
                                   common::Span<std::int8_t> data, std::int32_t root) {
  using namespace federated;  // NOLINT

  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();

  BroadcastRequest request;
  request.set_sequence_number((*sequence_number)++);
  request.set_rank(comm.Rank());
  if (comm.Rank() != root) {
    request.set_send_buffer(nullptr, 0);
  } else {
    request.set_send_buffer(data.data(), data.size());
  }
  request.set_root(root);

  BroadcastReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  grpc::Status status = stub->Broadcast(&context, request, &reply);
  if (!status.ok()) {
    return GetGRPCResult("Broadcast", status);
  }
  if (comm.Rank() != root) {
    auto const &r = reply.receive_buffer();
    std::copy_n(r.cbegin(), r.size(), data.data());
  }

  return Success();
}
}  // namespace

#if !defined(XGBOOST_USE_CUDA)
Coll *FederatedColl::MakeCUDAVar() {
  common::AssertGPUSupport();
  return nullptr;
}
#endif

[[nodiscard]] Result FederatedColl::Allreduce(Comm const &comm, common::Span<std::int8_t> data,
                                              ArrayInterfaceHandler::Type type, Op op) {
  using namespace federated;  // NOLINT
  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();

  AllreduceRequest request;
  request.set_sequence_number(sequence_number_++);
  request.set_rank(comm.Rank());
  request.set_send_buffer(data.data(), data.size());
  request.set_data_type(static_cast<::xgboost::collective::federated::DataType>(type));
  request.set_reduce_operation(static_cast<::xgboost::collective::federated::ReduceOperation>(op));

  AllreduceReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  grpc::Status status = stub->Allreduce(&context, request, &reply);
  if (!status.ok()) {
    return GetGRPCResult("Allreduce", status);
  }
  auto const &r = reply.receive_buffer();
  std::copy_n(r.cbegin(), r.size(), data.data());
  return Success();
}

[[nodiscard]] Result FederatedColl::Broadcast(Comm const &comm, common::Span<std::int8_t> data,
                                              std::int32_t root) {
  return BroadcastImpl(comm, &this->sequence_number_, data, root);
}

[[nodiscard]] Result FederatedColl::Allgather(Comm const &comm, common::Span<std::int8_t> data) {
  using namespace federated;  // NOLINT
  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();
  auto size = data.size_bytes() / comm.World();

  auto offset = comm.Rank() * size;
  auto segment = data.subspan(offset, size);

  AllgatherRequest request;
  request.set_sequence_number(sequence_number_++);
  request.set_rank(comm.Rank());
  request.set_send_buffer(segment.data(), segment.size());

  AllgatherReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  grpc::Status status = stub->Allgather(&context, request, &reply);

  if (!status.ok()) {
    return GetGRPCResult("Allgather", status);
  }
  auto const &r = reply.receive_buffer();
  std::copy_n(r.cbegin(), r.size(), data.begin());
  return Success();
}

[[nodiscard]] Result FederatedColl::AllgatherV(Comm const &comm,
                                               common::Span<std::int8_t const> data,
                                               common::Span<std::int64_t const>,
                                               common::Span<std::int64_t>,
                                               common::Span<std::int8_t> recv, AllgatherVAlgo) {
  using namespace federated;  // NOLINT

  auto fed = dynamic_cast<FederatedComm const *>(&comm);
  CHECK(fed);
  auto stub = fed->Handle();

  AllgatherVRequest request;
  request.set_sequence_number(sequence_number_++);
  request.set_rank(comm.Rank());
  request.set_send_buffer(data.data(), data.size());

  AllgatherVReply reply;
  grpc::ClientContext context;
  context.set_wait_for_ready(true);
  grpc::Status status = stub->AllgatherV(&context, request, &reply);
  if (!status.ok()) {
    return GetGRPCResult("AllgatherV", status);
  }
  std::string const &r = reply.receive_buffer();
  CHECK_EQ(r.size(), recv.size());
  std::copy_n(r.cbegin(), r.size(), recv.begin());
  return Success();
}
}  // namespace xgboost::collective
