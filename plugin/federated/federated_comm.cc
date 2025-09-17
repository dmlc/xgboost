/**
 * Copyright 2023-2024, XGBoost contributors
 */
#include "federated_comm.h"

#include <grpcpp/grpcpp.h>

#include <cstdint>  // for int32_t
#include <cstdlib>  // for getenv
#include <limits>   // for numeric_limits
#include <string>   // for string, stoi

#include "../../src/common/common.h"      // for Split
#include "../../src/common/io.h"          // for ReadAll
#include "../../src/common/json_utils.h"  // for OptionalArg
#include "xgboost/json.h"                 // for Json
#include "xgboost/logging.h"

namespace xgboost::collective {
void FederatedComm::Init(std::string const& host, std::int32_t port, std::int32_t world,
                         std::int32_t rank, std::string const& server_cert,
                         std::string const& client_key, std::string const& client_cert) {
  this->rank_ = rank;
  this->world_ = world;

  this->tracker_.host = host;
  this->tracker_.port = port;
  this->tracker_.rank = rank;

  CHECK_GE(world, 1) << "Invalid world size.";
  CHECK_GE(rank, 0) << "Invalid worker rank.";
  CHECK_LT(rank, world) << "Invalid worker rank.";

  auto certs = {server_cert, client_cert, client_cert};
  auto is_empty = [](auto const& s) { return s.empty(); };
  bool valid = std::all_of(certs.begin(), certs.end(), is_empty) ||
               std::none_of(certs.begin(), certs.end(), is_empty);
  CHECK(valid) << "Invalid arguments for certificates.";

  if (server_cert.empty()) {
    stub_ = [&] {
      grpc::ChannelArguments args;
      args.SetMaxReceiveMessageSize(std::numeric_limits<std::int32_t>::max());
      return federated::Federated::NewStub(grpc::CreateCustomChannel(
          host + ":" + std::to_string(port), grpc::InsecureChannelCredentials(), args));
    }();
  } else {
    stub_ = [&] {
      grpc::SslCredentialsOptions options;
      options.pem_root_certs = common::ReadAll(server_cert);
      options.pem_private_key = common::ReadAll(client_key);
      options.pem_cert_chain = common::ReadAll(client_cert);
      grpc::ChannelArguments args;
      args.SetMaxReceiveMessageSize(std::numeric_limits<std::int32_t>::max());
      auto channel = grpc::CreateCustomChannel(host + ":" + std::to_string(port),
                                               grpc::SslCredentials(options), args);
      channel->WaitForConnected(gpr_time_add(
          gpr_now(GPR_CLOCK_REALTIME), gpr_time_from_seconds(DefaultTimeoutSec(), GPR_TIMESPAN)));
      return federated::Federated::NewStub(channel);
    }();
  }
}

FederatedComm::FederatedComm(std::int32_t retry, std::chrono::seconds timeout, std::string task_id,
                             Json const& config) {
  /**
   * Topology
   */
  std::string server_address{};
  std::int32_t world_size{0};
  std::int32_t rank{-1};
  // Parse environment variables first.
  auto* value = std::getenv("FEDERATED_SERVER_ADDRESS");
  if (value != nullptr) {
    server_address = value;
  }
  value = std::getenv("FEDERATED_WORLD_SIZE");
  if (value != nullptr) {
    world_size = std::stoi(value);
  }
  value = std::getenv("FEDERATED_RANK");
  if (value != nullptr) {
    rank = std::stoi(value);
  }

  server_address = OptionalArg<String>(config, "federated_server_address", server_address);
  world_size =
      OptionalArg<Integer>(config, "federated_world_size", static_cast<Integer::Int>(world_size));
  rank = OptionalArg<Integer>(config, "federated_rank", static_cast<Integer::Int>(rank));

  auto parsed = common::Split(server_address, ':');
  CHECK_EQ(parsed.size(), 2) << "Invalid server address:" << server_address;

  CHECK(!server_address.empty()) << "Parameter `federated_server_address` is required.";

  /**
   * Basic config
   */
  this->retry_ = retry;
  this->timeout_ = timeout;
  this->task_id_ = task_id;

  /**
   * Certificates
   */
  std::string server_cert{};
  std::string client_key{};
  std::string client_cert{};
  value = getenv("FEDERATED_SERVER_CERT_PATH");
  if (value != nullptr) {
    server_cert = value;
  }
  value = getenv("FEDERATED_CLIENT_KEY_PATH");
  if (value != nullptr) {
    client_key = value;
  }
  value = getenv("FEDERATED_CLIENT_CERT_PATH");
  if (value != nullptr) {
    client_cert = value;
  }

  server_cert = OptionalArg<String>(config, "federated_server_cert_path", server_cert);
  client_key = OptionalArg<String>(config, "federated_client_key_path", client_key);
  client_cert = OptionalArg<String>(config, "federated_client_cert_path", client_cert);

  this->Init(parsed[0], std::stoi(parsed[1]), world_size, rank, server_cert, client_key,
             client_cert);
}

#if !defined(XGBOOST_USE_CUDA)
Comm* FederatedComm::MakeCUDAVar(Context const*, std::shared_ptr<Coll>) const {
  common::AssertGPUSupport();
  return nullptr;
}
#endif  //  !defined(XGBOOST_USE_CUDA)
}  // namespace xgboost::collective
