Tutorial of Rabit
=====
This is an tutorial of rabit, a ***Reliable Allreduce and Broadcast interface***.
To run the examples locally, you will need to type ```make``` to build all the examples.

Please also refer to the [API Documentation](http://homes.cs.washington.edu/~tqchen/rabit/doc)


**List of Topics**
* [What is Allreduce](#what-is-allreduce)
* [Common Use Case](#common-use-case)
* [Structure of Rabit Program](#structure-of-rabit-program)
* [Compile Programs with Rabit](#compile-programs-with-rabit)
* [Running Rabit Jobs](#running-rabit-jobs)
  - [Running Rabit on Hadoop](#running-rabit-on-hadoop)
  - [Running Rabit using MPI](#running-rabit-using-mpi)
  - [Customize Tracker Script](#customize-tracker-script)
* [Fault Tolerance](#fault-tolerance)

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
Allreduce call, the result will become ```a = {1, 2, 3}```.
Rabit provides different reduction operators, for example,  you can change ```op::Max``` to ```op::Sum```,
then the reduction operation will become the summation, and the result will become ```a = {1, 3, 5}```.
You can also run example with different processes by setting -n to different values, to see the outcomming result.

Broadcast is another method provided by rabit besides Allreduce, this function allows one node to broadcast its
local data to all the other nodes. The following code in [broadcast.cc](broadcast.cc) broadcast a string from
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
You can run the program by the following command, using three workers.
```bash
../tracker/rabit_demo.py -n 3 broadcast.rabit
```
Besides string, rabit also allows broadcast of constant size array and vector.

Common Use Case
=====
Many distributed machine learning algorithm involves dividing the data into each node,
compute statistics locally and aggregates them together. Such process is usually done repeatively in 
many iterations before the algorithm converge. Allreduce naturally meets the need of such programs,
common use cases include:

* Aggregation of gradient values, which can be used in optimization methods such as L-BFGS.
* Aggregation of other statistics, which can be used in KMeans and Gaussian Mixture Model.
* Find the best split candidate and aggregation of split statistics, used for tree based models.

The main purpose of Rabit is to provide reliable and portable library for distributed machine learning programs.
So that the program can be run reliably on different types of platforms.

Structure of Rabit Program
=====
The following code illustrates the common structure of rabit program. This is an abstract example,
you can also refer to [kmeans.cc](../toolkit/kmeans.cc) for an example implementation of kmeans.

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

Besides the common Allreduce and Broadcast function, there are two additional functions: ```CheckPoint```
and ```CheckPoint```. These two functions are used for fault-tolerance purpose. 
Common machine learning programs involves several iterations. In each iteration, we start from a model, do some calls
to Allreduce or Broadcasts and update the model to a new one. The calling sequence in each iteration does not need to be the same.

* When the nodes start from beginning, LoadCheckPoint returns 0, and we can initialize the model.
* ```CheckPoint``` saves the model after each iteration.
  - Efficiency Note: the model is only kept in local memory and no save to disk is involved in Checkpoint
* When a node goes down and restarts, ```LoadCheckPoint``` will recover the latest saved model, and 
* When a node goes down, the rest of the node will block in the call of Allreduce/Broadcast and helps 
  the recovery of the failure nodes, util it catches up. 

Please also see the section of [fault tolerance procedure](#fault-tolerance) in rabit to understand the recovery procedure under going in rabit

Compile Programs with Rabit
====
Rabit is a portable library, to use it, you only need to include the rabit header file.
* You will need to add path to [../include](../include) to the header search path of compiler
  - Solution 1: add ```-I/path/to/rabit/include``` to the compiler flag in gcc or clang
  - Solution 2: add the path to enviroment variable CPLUS_INCLUDE_PATH
* You will need to add path to [../lib](../lib) to the library search path of compiler
  - Solution 1: add ```-L/path/to/rabit/lib``` to the linker flag
  - Solution 2: add the path to enviroment variable LIBRARY_PATH AND LD_LIBRARY_PATH
* Link against lib/rabit.a
  - Add ```-lrabit``` to linker flag

The procedure above allows you to compile a program with rabit. The following two sections are additional
advanced options you can take to link against different backend other than the normal one.

#### Link against MPI Allreduce
You can link against ```rabit_mpi.a``` instead to use MPI Allreduce, however, the resulting program is backed by MPI and
is not fault tolerant anymore.
* Simply change linker flag from ```-lrabit``` to ```-lrabit_mpi```
* The final linking needs to be done by mpi wrapper compiler ```mpicxx```

#### Link against Mock Test Rabit Library
If you want to mock test the program to see the behavior of the code when some nodes goes down. You can link against ```rabit_mock.a``` .
* Simply change linker flag from ```-lrabit``` to ```-lrabit_mock```

The resulting rabit program can take in additional arguments in format of 
```
mock=rank,version,seq,ndeath 
```

The four integers specifies an event that will cause the program to suicide(exit with -2)
* rank specifies the rank of the node
* version specifies the current version(iteration) of the model
* seq specifies the sequence number of Allreduce/Broadcast call since last checkpoint
* ndeath specifies how many times this node died already

For example, consider the  following script in the test case
```bash
../tracker/rabit_demo.py -n 10 test_model_recover 10000\
                         mock=0,0,1,0 mock=1,1,1,0 mock=1,1,1,1
```
* The first mock will cause node 0 to exit when calling second Allreduce/Broadcast (seq = 1) in iteration 0
* The second mock will cause node 1 to exit when calling second Allreduce/Broadcast (seq = 1) in iteration 1
* The second mock will cause node 0 to exit again when calling second Allreduce/Broadcast (seq = 1) in iteration 1
  - Note that ndeath = 1 means this will happen only if node 0 died once, which is our case

Running Rabit Jobs
====
Rabit is a portable library that can run on multiple platforms. 

#### Running Rabit Locally
* You can use [../tracker/rabit_demo.py](../tracker/rabit_demo.py) to start n process locally
* This script will restart the program when it exits with -2, so it can be used for [mock test](#link-against-mock-test-library)

#### Running Rabit on Hadoop
* You can use [../tracker/rabit_hadoop.py](../tracker/rabit_hadoop.py) to run rabit program on hadoop
* This will start n rabit program as mapper of MapReduce
* Each program can read its part of data from stdin
* Yarn is highly recommended, since Yarn allows specifying ncpu and memory of each mapper
  - This allows multi-threading programs in each node, which can be more efficient
  - A good possible practice is OpenMP-rabit hybrid code

#### Running Rabit on Yarn
* To Be modified from [../tracker/rabit_hadoop.py](../tracker/rabit_hadoop.py)

#### Running Rabit using MPI
* You can submit rabit programs to MPI cluster using [../tracker/rabit_mpi.py](../tracker/rabit_mpi.py).
* If you linked your code against librabit_mpi.a, then you can directly use mpirun to submit the job

#### Customize Tracker Script
You can also modify the tracker script to allow rabit run on other platforms. To do so, refer to the existing
tracker script such as [../tracker/rabit_hadoop.py](../tracker/rabit_hadoop.py) and [../tracker/rabit_mpi.py](../tracker/rabit_mpi.py)

You will need to implement a platform dependent submission function with the following definition
```python
def fun_submit(nslave, slave_args):
    """
      customized submit script, that submit nslave jobs,
      each must contain args as parameter
      note this can be a lambda closure
      Parameters
         nslave number of slave process to start up
         worker_args tracker information which must be passed to the arguments 
              this usually includes the parameters of master_uri and port etc.
    """
```
The submission function should start nslave process in the platform, and append slave_args to the end of other arguments.
Then we can simply call ```tracker.submit``` with fun_submit to submit jobs in the target platform

Note that the current rabit tracker do not restart a worker when it dies, the job of fail-restart thus lies on the platform itself or we should write
fail-restart logic in the customization script.
* Fail-restart is usually provided by most platforms.
* For example, mapreduce will restart a mapper when it fails

Fault Tolerance
=====
This section introduces the how fault tolerance works in rabit.
We can use the following figure to show the how rabit deals with failures.

![](http://homes.cs.washington.edu/~tqchen/rabit/fig/fault-tol.png)

The scenario is as follows:
* Node 1 fails between the first and second call of Allreduce after the latest checkpoint
* Other nodes stay in the call of second Allreduce to help node 1 to recover.
* When node 1 restarts, it will call ```LoadCheckPoint```, and get the latest checkpoint from one of the existing nodes.
* Then node 1 can start from the latest checkpoint and continue running.
* When node 1 call the first Allreduce again, because the other nodes already knows the result of allreduce, node 1 can get the result from one of the nodes.
* When node 1 reaches the second Allreduce, other nodes find out that node 1 has catched up and they can continue the program normally.

We can find that this fault tolerance model is based on the a key property of Allreduce and Broadcast:
All the nodes get the same result when calling Allreduce/Broadcast. Because of this property, we can have some node records the history,
and when a node recovers, the result can be forwarded to the recovering node.

The checkpoint is introduced so that we do not have to discard the history before the checkpoint, so that the iterative program can be more
efficient. The strategy of rabit is different from fail-restart strategy where all the nodes restarts from checkpoint
when any of the node fails. All the program only block in the Allreduce call to help the recovery, and the checkpoint is only saved locally without
touching the disk. This makes rabit program more reliable and efficient.

This is an conceptual introduction to the fault tolerant model of rabit. The actual implementation is more sophiscated,
and can deal with more complicated cases such as multiple nodes failure and node failure during recovery phase.

