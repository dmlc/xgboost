/**
 * Copyright 2022-2023, XGBoost contributors
 */
#pragma once
#include <federated.grpc.pb.h>  // for Server

#include <future>  // for future
#include <string>

#include "../../src/collective/tracker.h"  // for Tracker
#include "xgboost/collective/result.h"     // for Result
#include "xgboost/json.h"                  // for Json

namespace xgboost::collective {
class FederatedTracker : public collective::Tracker {
  std::unique_ptr<grpc::Server> server_;
  std::string server_key_path_;
  std::string server_cert_file_;
  std::string client_cert_file_;

 public:
  explicit FederatedTracker(Json const& config);
  ~FederatedTracker() override;
  std::future<collective::Result> Run() override;
  // federated tracker do not provide initialization parameters, users have to provide it
  // themseleves.
  [[nodiscard]] Json WorkerArgs() const override { return Json{Null{}}; }
  [[nodiscard]] Result Shutdown();
};
}  // namespace xgboost::collective
