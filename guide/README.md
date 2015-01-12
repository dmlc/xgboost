Tutorial
=====
This is rabit's tutorial, a ***Reliable Allreduce and Broadcast Interface***.
To run the examples locally, you will need to build them with ```make```.

Please also refer to the [API Documentation](http://homes.cs.washington.edu/~tqchen/rabit/doc) for further details.


**List of Topics**
* [What is Allreduce](#what-is-allreduce)
* [Common Use Case](#common-use-case)
* [Structure of a Rabit Program](#structure-of-rabit-program)
* [Compile Programs with Rabit](#compile-programs-with-rabit)
* [Running Rabit Jobs](#running-rabit-jobs)
  - [Running Rabit on Hadoop](#running-rabit-on-hadoop)
  - [Running Rabit using MPI](#running-rabit-using-mpi)
  - [Customize Tracker Script](#customize-tracker-script)
* [Fault Tolerance](#fault-tolerance)

What is Allreduce
=====
The main methods provided by rabit are Allreduce and Broadcast. Allreduce performs reduction across different computation nodes,
and returns the result to every node. To understand the behavior of the function, consider the following example in [basic.cc](basic.cc).
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
You can run the example using the rabit_demo.py script. The following command
starts the rabit program with two worker processes.
```bash
../tracker/rabit_demo.py -n 2 basic.rabit
```
This will start two processes, one process with rank 0 and the other with rank 1, both processes run the same code.
The ```rabit::GetRank()``` function returns the rank of current process.

Before the call to Allreduce, process 0 contains the array ```a = {0, 1, 2}```, while process 1 has the array 
```a = {1, 2, 3}```. After the call to Allreduce, the array contents in all processes are replaced by the
reduction result (in this case, the maximum value in each position across all the processes). So, after the
Allreduce call, the result will become ```a = {1, 2, 3}```.
Rabit provides different reduction operators, for example,  if you change ```op::Max``` to ```op::Sum```,
the reduction operation will be a summation, and the result will become ```a = {1, 3, 5}```.
You can also run the example with different processes by setting -n to different values.

Broadcast is another method provided by rabit besides Allreduce. This function allows one node to broadcast its
local data to all other nodes. The following code in [broadcast.cc](broadcast.cc) broadcasts a string from
node 0 to all other nodes.
```c++
#include <rabit.h>
using namespace rabit;
const int N = 3;
int main(int argc, char *argv[]) {
  rabit::Init(argc, argv);
  std::string s;
  if (rabit::GetRank() == 0) s = "hello world";
  printf("@node[%d] before-broadcast: s=\"%s\"\n",
         rabit::GetRank(), s.c_str());
  // broadcast s from node 0 to all other nodes
  rabit::Broadcast(&s, 0);
  printf("@node[%d] after-broadcast: s=\"%s\"\n",
         rabit::GetRank(), s.c_str());
  rabit::Finalize();
  return 0;
}
```
The following command starts the program with three worker processes.
```bash
../tracker/rabit_demo.py -n 3 broadcast.rabit
```
Besides strings, rabit also allows to broadcast constant size array and vectors.

Common Use Case
=====
Many distributed machine learning algorithms involve splitting the data into different nodes,
computing statistics locally, and finally aggregating them. Such workflow is usually done repetitively through many iterations before the algorithm converges. Allreduce naturally meets the structure of such programs,
common use cases include:

* Aggregation of gradient values, which can be used in optimization methods such as L-BFGS.
* Aggregation of other statistics, which can be used in KMeans and Gaussian Mixture Models.
* Find the best split candidate and aggregation of split statistics, used for tree based models.

Rabit is a reliable and portable library for distributed machine learning programs, that allow programs to run reliably on different platforms.

Structure of a Rabit Program
=====
The following code illustrates the common structure of a rabit program. This is an abstract example,
you can also refer to [kmeans.cc](../toolkit/kmeans.cc) for an example implementation of kmeans algorithm.

```c++
#include <rabit.h>
int main(int argc, char *argv[]) {
  ...
  rabit::Init(argc, argv);
  // load the latest checked model
  int version = rabit::LoadCheckPoint(&model);
  // initialize the model if it is the first version
  if (version == 0) model.InitModel();
  // the version number marks the iteration to resume
  for (int iter = version; iter < max_iter; ++iter) {
    // model should be sufficient variable at this point
    ...
    // each iteration can contain multiple calls of allreduce/broadcast
    rabit::Allreduce(&data[0], n);
    ...
    // checkpoint model after one iteration finishes
    rabit::CheckPoint(&model);
  }
  rabit::Finalize();
  return 0;
}
```

Besides the common Allreduce and Broadcast functions, there are two additional functions: ```LoadCheckPoint```
and ```CheckPoint```. These two functions are used for fault-tolerance purposes. 
As mentioned before, traditional machine learning programs involve several iterations. In each iteration, we start with a model, make some calls
to Allreduce or Broadcast and update the model. The calling sequence in each iteration does not need to be the same.

* When the nodes start from the beginning (i.e. iteration 0), ```LoadCheckPoint``` returns 0, so we can initialize the model.
* ```CheckPoint``` saves the model after each iteration.
  - Efficiency Note: the model is only kept in local memory and no save to disk is performed when calling Checkpoint
* When a node goes down and restarts, ```LoadCheckPoint``` will recover the latest saved model, and 
* When a node goes down, the rest of the nodes will block in the call of Allreduce/Broadcast and wait for 
  the recovery of the failed node until it catches up. 

Please see the [Fault Tolerance](#fault-tolerance) section to understand the recovery procedure executed by rabit.

Compile Programs with Rabit
====
Rabit is a portable library, to use it, you only need to include the rabit header file.
* You will need to add the path to [../include](../include) to the header search path of the compiler
  - Solution 1: add ```-I/path/to/rabit/include``` to the compiler flag in gcc or clang
  - Solution 2: add the path to the environment variable CPLUS_INCLUDE_PATH
* You will need to add the path to [../lib](../lib) to the library search path of the compiler
  - Solution 1: add ```-L/path/to/rabit/lib``` to the linker flag
  - Solution 2: add the path to environment variable LIBRARY_PATH AND LD_LIBRARY_PATH
* Link against lib/rabit.a
  - Add ```-lrabit``` to the linker flag

The procedure above allows you to compile a program with rabit. The following two sections contain additional
options you can use to link against different backends other than the normal one.

#### Link against MPI Allreduce
You can link against ```rabit_mpi.a``` instead of using MPI Allreduce, however, the resulting program is backed by MPI and
is not fault tolerant anymore.
* Simply change the linker flag from ```-lrabit``` to ```-lrabit_mpi```
* The final linking needs to be done by mpi wrapper compiler ```mpicxx```

#### Link against Mock Test Rabit Library
If you want to use a mock to test the program in order to see the behavior of the code when some nodes go down, you can link against ```rabit_mock.a``` .
* Simply change the linker flag from ```-lrabit``` to ```-lrabit_mock```

The resulting rabit mock program can take in additional arguments in the following format
```
mock=rank,version,seq,ndeath
```

The four integers specify an event that will cause the program to ```commit suicide```(exit with -2)
* rank specifies the rank of the node to kill
* version specifies the version (iteration) of the model where you want the process to die
* seq specifies the sequence number of the Allreduce/Broadcast call since last checkpoint, where the process will be killed
* ndeath specifies how many times this node died already

For example, consider the following script in the test case
```bash
../tracker/rabit_demo.py -n 10 test_model_recover 10000\
                         mock=0,0,1,0 mock=1,1,1,0 mock=1,1,1,1
```
* The first mock will cause node 0 to exit when calling the second Allreduce/Broadcast (seq = 1) in iteration 0
* The second mock will cause node 1 to exit when calling the second Allreduce/Broadcast (seq = 1) in iteration 1
* The third mock will cause node 1 to exit again when calling second Allreduce/Broadcast (seq = 1) in iteration 1
  - Note that ndeath = 1 means this will happen only if node 1 died once, which is our case

Running Rabit Jobs
====
Rabit is a portable library that can run on multiple platforms. 

#### Running Rabit Locally
* You can use [../tracker/rabit_demo.py](../tracker/rabit_demo.py) to start n processes locally
* This script will restart the program when it exits with -2, so it can be used for [mock test](#link-against-mock-test-library)

#### Running Rabit on Hadoop
* You can use [../tracker/rabit_hadoop.py](../tracker/rabit_hadoop.py) to run rabit programs on hadoop
* This will start n rabit programs as mappers of MapReduce
* Each program can read its portion of data from stdin
* Yarn(Hadoop 2.0 or higher) is highly recommended, since Yarn allows specifying number of cpus and memory of each mapper:
  - This allows multi-threading programs in each node, which can be more efficient
  - An easy multi-threading solution could be to use OpenMP with rabit code

#### Running Rabit using MPI
* You can submit rabit programs to an MPI cluster using [../tracker/rabit_mpi.py](../tracker/rabit_mpi.py).
* If you linked your code against librabit_mpi.a, then you can directly use mpirun to submit the job

#### Customize Tracker Script
You can also modify the tracker script to allow rabit to run on other platforms. To do so, refer to existing
tracker scripts, such as [../tracker/rabit_hadoop.py](../tracker/rabit_hadoop.py) and [../tracker/rabit_mpi.py](../tracker/rabit_mpi.py) to get a sense of how it is done.

You will need to implement a platform dependent submission function with the following definition
```python
def fun_submit(nworkers, worker_args):
    """
      customized submit script, that submits nslave jobs,
      each must contain args as parameter
      note this can be a lambda closure
      Parameters
         nworkers number of worker processes to start
         worker_args tracker information which must be passed to the arguments 
              this usually includes the parameters of master_uri and port, etc.
    """
```
The submission function should start nworkers processes in the platform, and append worker_args to the end of the other arguments.
Then you can simply call ```tracker.submit``` with fun_submit to submit jobs to the target platform

Note that the current rabit tracker does not restart a worker when it dies, the restart of a node is done by the platform, otherwise we should write the fail-restart logic in the custom script.
* Fail-restart is usually provided by most platforms.
* For example, mapreduce will restart a mapper when it fails

Fault Tolerance
=====
This section introduces how fault tolerance works in rabit.
The following figure shows how rabit deals with failures.

![](http://homes.cs.washington.edu/~tqchen/rabit/fig/fault-tol.png)

The scenario is as follows:
* Node 1 fails between the first and second call of Allreduce after the second checkpoint
* The other nodes wait in the call of the second Allreduce in order to help node 1 to recover.
* When node 1 restarts, it will call ```LoadCheckPoint```, and get the latest checkpoint from one of the existing nodes.
* Then node 1 can start from the latest checkpoint and continue running.
* When node 1 calls the first Allreduce again, as the other nodes already know the result, node 1 can get it from one of them.
* When node 1 reaches the second Allreduce, the other nodes find out that node 1 has catched up and they can continue the program normally.

This fault tolerance model is based on a key property of Allreduce and Broadcast:
All the nodes get the same result when calling Allreduce/Broadcast. Because of this property, any node can record the history,
and when a node recovers, the result can be forwarded to it.

The checkpoint is introduced so that we can discard the history after checkpointing, this makes the iterative program more efficient. The strategy of rabit is different from the fail-restart strategy where all the nodes restart from the same checkpoint
when any of them fail. All the processes will block in the Allreduce call to help the recovery, and the checkpoint is only saved locally without
touching the disk. This makes rabit programs more reliable and efficient.

This is just a conceptual introduction to rabit's fault tolerance model. The actual implementation is more sophisticated,
and can deal with more complicated cases such as multiple nodes failure and node failure during recovery phase.

