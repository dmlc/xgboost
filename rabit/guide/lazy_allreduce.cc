/*!
 *  Copyright (c) 2014 by Contributors
 * \file basic.cc
 * \brief This is an example demonstrating what is Allreduce
 *
 * \author Tianqi Chen
 */
#include <rabit/rabit.h>

using namespace rabit;
const int N = 3;
int main(int argc, char *argv[]) {
  int a[N] = {0};
  rabit::Init(argc, argv);
  // lazy preparation function
  auto prepare = [&]() {
    printf("@node[%d] run prepare function\n", rabit::GetRank());
    for (int i = 0; i < N; ++i) {
      a[i] = rabit::GetRank() + i;
    }
  };
  printf("@node[%d] before-allreduce: a={%d, %d, %d}\n",
         rabit::GetRank(), a[0], a[1], a[2]);
  // allreduce take max of each elements in all processes
  Allreduce<op::Max>(&a[0], N, prepare);
  printf("@node[%d] after-allreduce-sum: a={%d, %d, %d}\n",
         rabit::GetRank(), a[0], a[1], a[2]);
  // rum second allreduce
  Allreduce<op::Sum>(&a[0], N);
  printf("@node[%d] after-allreduce-max: a={%d, %d, %d}\n",
         rabit::GetRank(), a[0], a[1], a[2]);
  rabit::Finalize();
  return 0;
}
