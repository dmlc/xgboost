/*!
 * Copyright 2022 XGBoost contributors
 */
#include <dmlc/parameter.h>
#include <gtest/gtest.h>

#include "../../../src/collective/communicator.h"

namespace xgboost {
namespace collective {

TEST(CommunicatorFactory, TypeFromEnv) {
  EXPECT_EQ(CommunicatorType::kUnknown, Communicator::GetTypeFromEnv());

  dmlc::SetEnv<std::string>("XGBOOST_COMMUNICATOR", "foo");
  EXPECT_THROW(Communicator::GetTypeFromEnv(), dmlc::Error);

  dmlc::SetEnv<std::string>("XGBOOST_COMMUNICATOR", "rabit");
  EXPECT_EQ(CommunicatorType::kRabit, Communicator::GetTypeFromEnv());

  dmlc::SetEnv<std::string>("XGBOOST_COMMUNICATOR", "Federated");
  EXPECT_EQ(CommunicatorType::kFederated, Communicator::GetTypeFromEnv());

  dmlc::SetEnv<std::string>("XGBOOST_COMMUNICATOR", "In-Memory");
  EXPECT_EQ(CommunicatorType::kInMemory, Communicator::GetTypeFromEnv());
}

TEST(CommunicatorFactory, TypeFromArgs) {
  Json config{JsonObject()};
  EXPECT_EQ(CommunicatorType::kUnknown, Communicator::GetTypeFromConfig(config));

  config["xgboost_communicator"] = String("rabit");
  EXPECT_EQ(CommunicatorType::kRabit, Communicator::GetTypeFromConfig(config));

  config["xgboost_communicator"] = String("federated");
  EXPECT_EQ(CommunicatorType::kFederated, Communicator::GetTypeFromConfig(config));

  config["xgboost_communicator"] = String("in-memory");
  EXPECT_EQ(CommunicatorType::kInMemory, Communicator::GetTypeFromConfig(config));

  config["xgboost_communicator"] = String("foo");
  EXPECT_THROW(Communicator::GetTypeFromConfig(config), dmlc::Error);
}

TEST(CommunicatorFactory, TypeFromArgsUpperCase) {
  Json config{JsonObject()};
  EXPECT_EQ(CommunicatorType::kUnknown, Communicator::GetTypeFromConfig(config));

  config["XGBOOST_COMMUNICATOR"] = String("rabit");
  EXPECT_EQ(CommunicatorType::kRabit, Communicator::GetTypeFromConfig(config));

  config["XGBOOST_COMMUNICATOR"] = String("federated");
  EXPECT_EQ(CommunicatorType::kFederated, Communicator::GetTypeFromConfig(config));

  config["XGBOOST_COMMUNICATOR"] = String("in-memory");
  EXPECT_EQ(CommunicatorType::kInMemory, Communicator::GetTypeFromConfig(config));

  config["XGBOOST_COMMUNICATOR"] = String("foo");
  EXPECT_THROW(Communicator::GetTypeFromConfig(config), dmlc::Error);
}

}  // namespace collective
}  // namespace xgboost
