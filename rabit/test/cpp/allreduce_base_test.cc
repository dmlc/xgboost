#define RABIT_CXXTESTDEFS_H
#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include "../../src/allreduce_base.h"

TEST(allreduce_base, init_task)
{
  rabit::engine::AllreduceBase base;

  std::string rabit_task_id = "rabit_task_id=1";
  char cmd[rabit_task_id.size()+1];
  std::copy(rabit_task_id.begin(), rabit_task_id.end(), cmd);
  cmd[rabit_task_id.size()] = '\0';

  char* argv[] = {cmd};
  base.Init(1, argv);
  EXPECT_EQ(base.task_id, "1");
}

TEST(allreduce_base, init_with_cache_on)
{
  rabit::engine::AllreduceBase base;

  std::string rabit_task_id = "rabit_task_id=1";
  char cmd[rabit_task_id.size()+1];
  std::copy(rabit_task_id.begin(), rabit_task_id.end(), cmd);
  cmd[rabit_task_id.size()] = '\0';

  std::string rabit_bootstrap_cache = "rabit_bootstrap_cache=1";
  char cmd2[rabit_bootstrap_cache.size()+1];
  std::copy(rabit_bootstrap_cache.begin(), rabit_bootstrap_cache.end(), cmd2);
  cmd2[rabit_bootstrap_cache.size()] = '\0';

  std::string rabit_debug = "rabit_debug=1";
  char cmd3[rabit_debug.size()+1];
  std::copy(rabit_debug.begin(), rabit_debug.end(), cmd3);
  cmd3[rabit_debug.size()] = '\0';

  char* argv[] = {cmd, cmd2, cmd3};
  base.Init(3, argv);
  EXPECT_EQ(base.task_id, "1");
  EXPECT_EQ(base.rabit_bootstrap_cache, 1);
  EXPECT_EQ(base.rabit_debug, 1);
}

TEST(allreduce_base, init_with_ring_reduce)
{
  rabit::engine::AllreduceBase base;

  std::string rabit_task_id = "rabit_task_id=1";
  char cmd[rabit_task_id.size()+1];
  std::copy(rabit_task_id.begin(), rabit_task_id.end(), cmd);
  cmd[rabit_task_id.size()] = '\0';

  std::string rabit_reduce_ring_mincount = "rabit_reduce_ring_mincount=1";
  char cmd2[rabit_reduce_ring_mincount.size()+1];
  std::copy(rabit_reduce_ring_mincount.begin(), rabit_reduce_ring_mincount.end(), cmd2);
  cmd2[rabit_reduce_ring_mincount.size()] = '\0';

  char* argv[] = {cmd, cmd2};
  base.Init(2, argv);
  EXPECT_EQ(base.task_id, "1");
  EXPECT_EQ(base.reduce_ring_mincount, 1);
}
