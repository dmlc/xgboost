/*!
 * Copyright 2022 XGBoost contributors
 */
#include <gtest/gtest.h>

#include "../../../src/collective/communicator_factory.h"

namespace xgboost {
namespace collective {

TEST(CommunicatorFactory, TypeFromEnv) {
  unsetenv(CommunicatorFactory::kCommunicatorKey);
  EXPECT_EQ(CommunicatorType::kUnknown, CommunicatorFactory::GetTypeFromEnv());

  setenv(CommunicatorFactory::kCommunicatorKey, "rabit", 1);
  EXPECT_EQ(CommunicatorType::kRabit, CommunicatorFactory::GetTypeFromEnv());

  setenv(CommunicatorFactory::kCommunicatorKey, "MPI", 1);
  EXPECT_EQ(CommunicatorType::kMPI, CommunicatorFactory::GetTypeFromEnv());

  setenv(CommunicatorFactory::kCommunicatorKey, "Federated", 1);
  EXPECT_EQ(CommunicatorType::kFederated, CommunicatorFactory::GetTypeFromEnv());

  setenv(CommunicatorFactory::kCommunicatorKey, "foo", 1);
  EXPECT_THROW(CommunicatorFactory::GetTypeFromEnv(), dmlc::Error);
}

TEST(CommunicatorFactory, TypeFromArgs) {
  char *args[1];
  args[0] = strdup("foo=bar");
  EXPECT_EQ(CommunicatorType::kUnknown, CommunicatorFactory::GetTypeFromArgs(1, args));

  args[0] = strdup("xgboost_communicator=rabit");
  EXPECT_EQ(CommunicatorType::kRabit, CommunicatorFactory::GetTypeFromArgs(1, args));

  args[0] = strdup("xgboost_communicator=MPI");
  EXPECT_EQ(CommunicatorType::kMPI, CommunicatorFactory::GetTypeFromArgs(1, args));

  args[0] = strdup("xgboost_communicator=Federated");
  EXPECT_EQ(CommunicatorType::kFederated, CommunicatorFactory::GetTypeFromArgs(1, args));

  args[0] = strdup("xgboost_communicator=foo");
  EXPECT_THROW(CommunicatorFactory::GetTypeFromArgs(1, args), dmlc::Error);
}

}  // namespace collective
}  // namespace xgboost
