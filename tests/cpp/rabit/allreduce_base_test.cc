#define RABIT_CXXTESTDEFS_H
#if !defined(_WIN32)
#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include "../../../rabit/src/allreduce_base.h"

TEST(AllreduceBase, InitTask)
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

TEST(AllreduceBase, InitWithRingReduce)
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
  EXPECT_EQ(base.reduce_ring_mincount, 1ul);
}
#endif  // !defined(_WIN32)
