/**
 * Copyright 2022-2023, XGBoost contributors
 */
#include "federated_tracker.h"

#include <grpcpp/server_builder.h>  // for ServerBuilder

#include "../../src/common/io.h"          // for ReadAll
#include "../../src/common/json_utils.h"  // for RequiredArg
#include "../../src/common/timer.h"       // for ReadAll
#include "federated_server.h"             // for FederatedService

namespace xgboost::collective {
FederatedTracker::FederatedTracker(Json const& config) : Tracker{config} {
  auto is_secure = RequiredArg<Boolean const>(config, "federated_secure", __func__);
  if (is_secure) {
    server_key_path_ = RequiredArg<String const>(config, "server_key_path", __func__);
    server_cert_file_ = RequiredArg<String const>(config, "server_cert_file", __func__);
    client_cert_file_ = RequiredArg<String const>(config, "client_cert_path", __func__);
  }
}

std::future<collective::Result> FederatedTracker::Run() {
  return std::async([this]() {
    std::string const server_address = "0.0.0.0:" + std::to_string(this->port_);
    federated::FederatedService service{static_cast<std::int32_t>(this->n_workers_)};
    grpc::ServerBuilder builder;

    if (this->server_cert_file_.empty()) {
      builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
      if (this->port_ == 0) {
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), &port_);
      } else {
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
      }
      builder.RegisterService(&service);
      server_ = builder.BuildAndStart();
      LOG(CONSOLE) << "Insecure federated server listening on " << server_address << ", world size "
                   << this->n_workers_;
    } else {
      auto options = grpc::SslServerCredentialsOptions(
          GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY);
      options.pem_root_certs = xgboost::common::ReadAll(client_cert_file_);
      auto key = grpc::SslServerCredentialsOptions::PemKeyCertPair();
      key.private_key = xgboost::common::ReadAll(server_key_path_);
      key.cert_chain = xgboost::common::ReadAll(server_cert_file_);
      options.pem_key_cert_pairs.push_back(key);
      builder.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
      if (this->port_ == 0) {
        builder.AddListeningPort(server_address, grpc::SslServerCredentials(options), &port_);
      } else {
        builder.AddListeningPort(server_address, grpc::SslServerCredentials(options));
      }
      builder.RegisterService(&service);
      server_ = builder.BuildAndStart();
      LOG(CONSOLE) << "Federated server listening on " << server_address << ", world size "
                   << n_workers_;
    }

    try {
      server_->Wait();
    } catch (std::exception const& e) {
      return collective::Fail(std::string{e.what()});
    }
    return collective::Success();
  });
}

FederatedTracker::~FederatedTracker() = default;

Result FederatedTracker::Shutdown() {
  common::Timer timer;
  timer.Start();
  while (!server_) {
    auto ela = timer.ElapsedSeconds();
    if (ela > this->Timeout().count()) {
      return Fail("Timeout:" + std::to_string(this->Timeout().count()) + " seconds.");
    }
  }

  server_->Shutdown();
  return Success();
}
}  // namespace xgboost::collective
