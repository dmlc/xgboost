/*!
 * Copyright 2022 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>

#include "../../../src/collective/communicator_factory.h"

namespace xgboost {
namespace collective {

TEST(CommunicatorFactory, TypeFromEnv) {
  EXPECT_EQ(CommunicatorType::kUnknown, CommunicatorFactory::GetTypeFromEnv());

  dmlc::SetEnv<std::string>("XGBOOST_COMMUNICATOR", "rabit");
  EXPECT_EQ(CommunicatorType::kRabit, CommunicatorFactory::GetTypeFromEnv());

  dmlc::SetEnv<std::string>("XGBOOST_COMMUNICATOR", "MPI");
  EXPECT_EQ(CommunicatorType::kMPI, CommunicatorFactory::GetTypeFromEnv());

  dmlc::SetEnv<std::string>("XGBOOST_COMMUNICATOR", "Federated");
  EXPECT_EQ(CommunicatorType::kFederated, CommunicatorFactory::GetTypeFromEnv());

  dmlc::SetEnv<std::string>("XGBOOST_COMMUNICATOR", "foo");
  EXPECT_THROW(CommunicatorFactory::GetTypeFromEnv(), dmlc::Error);
}

TEST(CommunicatorFactory, TypeFromArgs) {
  Json config{JsonObject()};
  EXPECT_EQ(CommunicatorType::kUnknown, CommunicatorFactory::GetTypeFromConfig(config));

  config["xgboost_communicator"] = String("rabit");
  EXPECT_EQ(CommunicatorType::kRabit, CommunicatorFactory::GetTypeFromConfig(config));

  config["xgboost_communicator"] = String("MPI");
  EXPECT_EQ(CommunicatorType::kMPI, CommunicatorFactory::GetTypeFromConfig(config));

  config["xgboost_communicator"] = String("federated");
  EXPECT_EQ(CommunicatorType::kFederated, CommunicatorFactory::GetTypeFromConfig(config));

  config["xgboost_communicator"] = String("foo");
  EXPECT_THROW(CommunicatorFactory::GetTypeFromConfig(config), dmlc::Error);
}

TEST(CommunicatorFactory, TypeFromArgsUpperCase) {
  Json config{JsonObject()};
  EXPECT_EQ(CommunicatorType::kUnknown, CommunicatorFactory::GetTypeFromConfig(config));

  config["XGBOOST_COMMUNICATOR"] = String("rabit");
  EXPECT_EQ(CommunicatorType::kRabit, CommunicatorFactory::GetTypeFromConfig(config));

  config["XGBOOST_COMMUNICATOR"] = String("MPI");
  EXPECT_EQ(CommunicatorType::kMPI, CommunicatorFactory::GetTypeFromConfig(config));

  config["XGBOOST_COMMUNICATOR"] = String("federated");
  EXPECT_EQ(CommunicatorType::kFederated, CommunicatorFactory::GetTypeFromConfig(config));

  config["XGBOOST_COMMUNICATOR"] = String("foo");
  EXPECT_THROW(CommunicatorFactory::GetTypeFromConfig(config), dmlc::Error);
}

}  // namespace collective
}  // namespace xgboost
