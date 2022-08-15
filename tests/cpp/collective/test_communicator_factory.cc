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

  dmlc::SetEnv<std::string>(CommunicatorFactory::kCommunicatorKey, "rabit");
  EXPECT_EQ(CommunicatorType::kRabit, CommunicatorFactory::GetTypeFromEnv());

  dmlc::SetEnv<std::string>(CommunicatorFactory::kCommunicatorKey, "MPI");
  EXPECT_EQ(CommunicatorType::kMPI, CommunicatorFactory::GetTypeFromEnv());

  dmlc::SetEnv<std::string>(CommunicatorFactory::kCommunicatorKey, "Federated");
  EXPECT_EQ(CommunicatorType::kFederated, CommunicatorFactory::GetTypeFromEnv());

  dmlc::SetEnv<std::string>(CommunicatorFactory::kCommunicatorKey, "foo");
  EXPECT_THROW(CommunicatorFactory::GetTypeFromEnv(), dmlc::Error);
}

TEST(CommunicatorFactory, TypeFromArgs) {
  char *args0[] = {(char *)("foo=bar")};  // NOLINT(google-readability-casting)
  EXPECT_EQ(CommunicatorType::kUnknown, CommunicatorFactory::GetTypeFromArgs(1, args0));

  char *args1[] = {(char *)("xgboost_communicator=rabit")};  // NOLINT(google-readability-casting)
  EXPECT_EQ(CommunicatorType::kRabit, CommunicatorFactory::GetTypeFromArgs(1, args1));

  char *args2[] = {(char *)("xgboost_communicator=MPI")};  // NOLINT(google-readability-casting)
  EXPECT_EQ(CommunicatorType::kMPI, CommunicatorFactory::GetTypeFromArgs(1, args2));

  char *args3[] = {
      (char *)("xgboost_communicator=federated")};  // NOLINT(google-readability-casting)
  EXPECT_EQ(CommunicatorType::kFederated, CommunicatorFactory::GetTypeFromArgs(1, args3));

  char *args4[] = {(char *)("xgboost_communicator=foo")};  // NOLINT(google-readability-casting)
  EXPECT_THROW(CommunicatorFactory::GetTypeFromArgs(1, args4), dmlc::Error);
}

}  // namespace collective
}  // namespace xgboost
