Tutorial of Rabit
=====
This is an tutorial of rabit, a Reliable Allreduce and Broadcast interface.
To run the examples locally, you will need to type ```make``` to build all the examples.

**List of Topics**
* [What is Allreduce](#what-is-allreduce)
* [Common Usecase of Allreduce](#common-use-case)

What is Allreduce
=====
The main method provided by rabit are Allreduce and Broadcast. Allreduce performs reduction across different computation nodes,
and returning the results to all the nodes. To understand the behavior of the function. Consider the following example in [basic.cc](basic.cc).
```c++
#include <rabit.h>
using namespace rabit;
const int N = 3;
int main(int argc, char *argv[]) {
  int a[N];
  rabit::Init(argc, argv);
  for (int i = 0; i < N; ++i) {
    a[i] = rabit::GetRank() + i;
  } 
  printf("@node[%d] before-allreduce: a={%d, %d, %d}\n",
         rabit::GetRank(), a[0], a[1], a[2]);
  // allreduce take max of each elements in all processes
  Allreduce<op::Max>(&a[0], N);
  printf("@node[%d] after-allreduce: a={%d, %d, %d}\n",
         rabit::GetRank(), a[0], a[1], a[2]);
  rabit::Finalize();
  return 0;
}
```
You can run the example using the rabit_demo.py script. The following commmand
start rabit program with two worker processes.
```bash
../tracker/rabit_demo.py -n 2 basic.rabit
```
This will start two process, one process with rank 0 and another rank 1, running the same code.
The ```rabit::GetRank()``` function return the rank of current process.

Before the call the allreduce, process 0 contains array ```a = {0, 1, 2}```, while process 1 have array 
```a = {1, 2, 3}```. After the call of Allreduce, the array contents in all processes are replaced by the
reduction result (in this case, the maximum value in each position across all the processes). So after the
Allreduce call, the result will become ```a={1, 2, 3}```.

You can also run example with different processes by setting -n to different values, to see the outcomming result.
Rabit provides different reduction operators, for example, you can change ```op::Max``` to ```op::Sum```, to change
the reduction method from maximum to summation.


Common Use Case
=====

