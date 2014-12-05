## rabit: Reliable Allreduce and Broadcast Interface

rabit is a light weight library that provides a fault tolerant interface of Allreduce and Broadcast. It is designed to support easy implementations of distributed machine learning programs, many of which fall naturally under the Allreduce abstraction.

* See the [package interface file](src/rabit.h)

Features
====
* Portable library
  - Rabit is a library instead of a framework, a program only needs to link the library to run.
* Flexibility in programming
  - Programs can call rabit functions in any order, as opposed to frameworks where callbacks are offered and called by the framework, i.e. inversion of control principle.
  - Programs persist over all the iterations, unless they fail and recover.
* Fault tolerance 
  - Rabit programs can recover the model and results using synchronous function calls.
* MPI compatible
  - Code that uses the rabit interface also compiles with existing MPI compilers
  - Users can use MPI Allreduce with no code modification

Design Notes
====
* Rabit is designed for algorithms that replicate the same global model across nodes, while each node operates on a local partition of the data.
* The collection of global statistics is done using Allreduce

Design Goals
====
* rabit should run fast
* rabit should be light weight
* rabit should safely dig burrows to avoid disasters
