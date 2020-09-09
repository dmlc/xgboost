#define RABIT_CXXTESTDEFS_H
#if !defined(_WIN32)
#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include "../../../rabit/src/allreduce_mock.h"

TEST(AllreduceMock, MockAllreduce)
{
  rabit::engine::AllreduceMock m;

  std::string mock_str = "mock=0,0,0,0";
  char cmd[mock_str.size()+1];
  std::copy(mock_str.begin(), mock_str.end(), cmd);
  cmd[mock_str.size()] = '\0';

  char* argv[] = {cmd};
  m.Init(1, argv);
  m.rank = 0;
  EXPECT_THROW(m.Allreduce(nullptr,0,0,nullptr,nullptr,nullptr), dmlc::Error);
}

TEST(AllreduceMock, MockBroadcast)
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
  EXPECT_THROW(m.Broadcast(nullptr,0,0), dmlc::Error);
}

TEST(AllreduceMock, MockGather)
{
  rabit::engine::AllreduceMock m;
  std::string mock_str = "mock=3,13,22,0";
  char cmd[mock_str.size()+1];
  std::copy(mock_str.begin(), mock_str.end(), cmd);
  cmd[mock_str.size()] = '\0';
  char* argv[] = {cmd};
  m.Init(1, argv);
  m.rank = 3;
  m.version_number=13;
  m.seq_counter=22;
  EXPECT_THROW({m.Allgather(nullptr,0,0,0,0);}, dmlc::Error);
}
#endif  // !defined(_WIN32)
