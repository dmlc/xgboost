#define RABIT_CXXTESTDEFS_H
#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include "../../src/allreduce_mock.h"

TEST(allreduce_mock, mock_allreduce)
{
  rabit::engine::AllreduceMock m;

  std::string mock_str = "mock=0,0,0,0";
  char cmd[mock_str.size()+1];
  std::copy(mock_str.begin(), mock_str.end(), cmd);
  cmd[mock_str.size()] = '\0';

  char* argv[] = {cmd};
  m.Init(1, argv);
  m.rank = 0;
  EXPECT_EXIT(m.Allreduce(nullptr,0,0,nullptr,nullptr,nullptr), ::testing::ExitedWithCode(255), "");
}

TEST(allreduce_mock, mock_broadcast)
{
  rabit::engine::AllreduceMock m;
  std::string mock_str = "mock=0,1,2,0";
  char cmd[mock_str.size()+1];
  std::copy(mock_str.begin(), mock_str.end(), cmd);
  cmd[mock_str.size()] = '\0';
  char* argv[] = {cmd};
  m.Init(1, argv);
  m.rank = 0;
  m.version_number=1;
  m.seq_counter=2;
  EXPECT_EXIT(m.Broadcast(nullptr,0,0), ::testing::ExitedWithCode(255), "");
}
