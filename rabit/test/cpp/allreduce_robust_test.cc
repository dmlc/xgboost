#define RABIT_CXXTESTDEFS_H
#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <iostream>
#include "../../src/allreduce_robust.h"

inline void mockerr(const char *fmt, ...) {EXPECT_STRCASEEQ(fmt, "[%d] exit due to time out %d s\n");}
inline void mockassert(bool val, const char *fmt, ...) {}
rabit::engine::AllreduceRobust::ReturnType err_type(rabit::engine::AllreduceRobust::ReturnTypeEnum::kSockError);
rabit::engine::AllreduceRobust::ReturnType succ_type(rabit::engine::AllreduceRobust::ReturnTypeEnum::kSuccess);

TEST(allreduce_robust, sync_error_timeout)
{
  rabit::engine::AllreduceRobust m;

  std::string rabit_timeout = "rabit_timeout=1";
  char cmd[rabit_timeout.size()+1];
  std::copy(rabit_timeout.begin(), rabit_timeout.end(), cmd);
  cmd[rabit_timeout.size()] = '\0';

  std::string rabit_timeout_sec = "rabit_timeout_sec=1";
  char cmd1[rabit_timeout_sec.size()+1];
  std::copy(rabit_timeout_sec.begin(), rabit_timeout_sec.end(), cmd1);
  cmd1[rabit_timeout_sec.size()] = '\0';

  char* argv[] = {cmd,cmd1};
  m.Init(2, argv);
  m.rank = 0;
  m.rabit_bootstrap_cache = 1;
  m._error = mockerr;
  m._assert = mockassert;
  EXPECT_EQ(m.CheckAndRecover(err_type), false);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  EXPECT_EQ(m.rabit_timeout_task.get(), false);
}

TEST(allreduce_robust, sync_error_reset)
{
  rabit::engine::AllreduceRobust m;

  std::string rabit_timeout = "rabit_timeout=1";
  char cmd[rabit_timeout.size()+1];
  std::copy(rabit_timeout.begin(), rabit_timeout.end(), cmd);
  cmd[rabit_timeout.size()] = '\0';

  std::string rabit_timeout_sec = "rabit_timeout_sec=1";
  char cmd1[rabit_timeout_sec.size()+1];
  std::copy(rabit_timeout_sec.begin(), rabit_timeout_sec.end(), cmd1);
  cmd1[rabit_timeout_sec.size()] = '\0';

  std::string rabit_debug = "rabit_debug=1";
  char cmd2[rabit_debug.size()+1];
  std::copy(rabit_debug.begin(), rabit_debug.end(), cmd2);
  cmd2[rabit_debug.size()] = '\0';

  char* argv[] = {cmd, cmd1,cmd2};
  m.Init(3, argv);
  m.rank = 0;
  m._assert = mockassert;
  EXPECT_EQ(m.CheckAndRecover(err_type), false);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(m.CheckAndRecover(succ_type), true);
  EXPECT_EQ(m.rabit_timeout_task.get(), true);
  m.Shutdown();
}

TEST(allreduce_robust, sync_success_error_timeout)
{
  rabit::engine::AllreduceRobust m;

  std::string rabit_timeout = "rabit_timeout=1";
  char cmd[rabit_timeout.size()+1];
  std::copy(rabit_timeout.begin(), rabit_timeout.end(), cmd);
  cmd[rabit_timeout.size()] = '\0';

  std::string rabit_timeout_sec = "rabit_timeout_sec=1";
  char cmd1[rabit_timeout_sec.size()+1];
  std::copy(rabit_timeout_sec.begin(), rabit_timeout_sec.end(), cmd1);
  cmd1[rabit_timeout_sec.size()] = '\0';

  std::string rabit_debug = "rabit_debug=1";
  char cmd2[rabit_debug.size()+1];
  std::copy(rabit_debug.begin(), rabit_debug.end(), cmd2);
  cmd2[rabit_debug.size()] = '\0';

  char* argv[] = {cmd, cmd1,cmd2};
  m.Init(3, argv);
  m.rank = 0;
  m.rabit_bootstrap_cache = 1;
  m._assert = mockassert;
  m._error = mockerr;
  EXPECT_EQ(m.CheckAndRecover(succ_type), true);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(m.CheckAndRecover(err_type), false);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  EXPECT_EQ(m.rabit_timeout_task.get(), false);
}

TEST(allreduce_robust, sync_success_error_success)
{
  rabit::engine::AllreduceRobust m;

  std::string rabit_timeout = "rabit_timeout=1";
  char cmd[rabit_timeout.size()+1];
  std::copy(rabit_timeout.begin(), rabit_timeout.end(), cmd);
  cmd[rabit_timeout.size()] = '\0';

  std::string rabit_timeout_sec = "rabit_timeout_sec=1";
  char cmd1[rabit_timeout_sec.size()+1];
  std::copy(rabit_timeout_sec.begin(), rabit_timeout_sec.end(), cmd1);
  cmd1[rabit_timeout_sec.size()] = '\0';

  std::string rabit_debug = "rabit_debug=1";
  char cmd2[rabit_debug.size()+1];
  std::copy(rabit_debug.begin(), rabit_debug.end(), cmd2);
  cmd2[rabit_debug.size()] = '\0';

  char* argv[] = {cmd, cmd1,cmd2};
  m.Init(3, argv);
  m.rank = 0;
  m.rabit_bootstrap_cache = 1;
  m._assert = mockassert;
  EXPECT_EQ(m.CheckAndRecover(succ_type), true);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_EQ(m.CheckAndRecover(err_type), false);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(m.CheckAndRecover(succ_type), true);
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  EXPECT_EQ(m.rabit_timeout_task.get(), true);
  m.Shutdown();
}

TEST(allreduce_robust, sync_error_no_reset_timeout)
{
  rabit::engine::AllreduceRobust m;

  std::string rabit_timeout = "rabit_timeout=1";
  char cmd[rabit_timeout.size()+1];
  std::copy(rabit_timeout.begin(), rabit_timeout.end(), cmd);
  cmd[rabit_timeout.size()] = '\0';

  std::string rabit_timeout_sec = "rabit_timeout_sec=1";
  char cmd1[rabit_timeout_sec.size()+1];
  std::copy(rabit_timeout_sec.begin(), rabit_timeout_sec.end(), cmd1);
  cmd1[rabit_timeout_sec.size()] = '\0';

  std::string rabit_debug = "rabit_debug=1";
  char cmd2[rabit_debug.size()+1];
  std::copy(rabit_debug.begin(), rabit_debug.end(), cmd2);
  cmd2[rabit_debug.size()] = '\0';

  char* argv[] = {cmd, cmd1,cmd2};
  m.Init(3, argv);
  m.rank = 0;
  m.rabit_bootstrap_cache = 1;
  m._assert = mockassert;
  m._error = mockerr;
  auto start = std::chrono::system_clock::now();

  EXPECT_EQ(m.CheckAndRecover(err_type), false);
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));

  EXPECT_EQ(m.CheckAndRecover(err_type), false);

  m.rabit_timeout_task.wait();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end-start;

  EXPECT_EQ(m.rabit_timeout_task.get(), false);
  // expect second error don't overwrite/reset timeout task
  EXPECT_LT(diff.count(), 2);
}

TEST(allreduce_robust, no_timeout_shut_down)
{
  rabit::engine::AllreduceRobust m;

  std::string rabit_timeout = "rabit_timeout=1";
  char cmd[rabit_timeout.size()+1];
  std::copy(rabit_timeout.begin(), rabit_timeout.end(), cmd);
  cmd[rabit_timeout.size()] = '\0';

  std::string rabit_timeout_sec = "rabit_timeout_sec=1";
  char cmd1[rabit_timeout_sec.size()+1];
  std::copy(rabit_timeout_sec.begin(), rabit_timeout_sec.end(), cmd1);
  cmd1[rabit_timeout_sec.size()] = '\0';

  std::string rabit_debug = "rabit_debug=1";
  char cmd2[rabit_debug.size()+1];
  std::copy(rabit_debug.begin(), rabit_debug.end(), cmd2);
  cmd2[rabit_debug.size()] = '\0';

  char* argv[] = {cmd, cmd1,cmd2};
  m.Init(3, argv);
  m.rank = 0;

  EXPECT_EQ(m.CheckAndRecover(succ_type), true);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  m.Shutdown();
}

TEST(allreduce_robust, shut_down_before_timeout)
{
  rabit::engine::AllreduceRobust m;

  std::string rabit_timeout = "rabit_timeout=1";
  char cmd[rabit_timeout.size()+1];
  std::copy(rabit_timeout.begin(), rabit_timeout.end(), cmd);
  cmd[rabit_timeout.size()] = '\0';

  std::string rabit_timeout_sec = "rabit_timeout_sec=1";
  char cmd1[rabit_timeout_sec.size()+1];
  std::copy(rabit_timeout_sec.begin(), rabit_timeout_sec.end(), cmd1);
  cmd1[rabit_timeout_sec.size()] = '\0';

  std::string rabit_debug = "rabit_debug=1";
  char cmd2[rabit_debug.size()+1];
  std::copy(rabit_debug.begin(), rabit_debug.end(), cmd2);
  cmd2[rabit_debug.size()] = '\0';

  char* argv[] = {cmd, cmd1,cmd2};
  m.Init(3, argv);
  m.rank = 0;
  rabit::engine::AllreduceRobust::LinkRecord a;
  m.err_link = &a;
  
  EXPECT_EQ(m.CheckAndRecover(err_type), false);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  m.Shutdown();
}