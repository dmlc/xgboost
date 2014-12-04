## rabit: Robust Allreduce and Broadcast Interface

rabit is a light weight library designed to provide fault tolerant interface of Allreduce and Broadcast. It is designed to support easy implementation of distributed machine learning programs, many of which sits naturally under Allreduce abstraction.

Contributors: https://github.com/tqchen/rabit/graphs/contributors

Design Note
====
* Rabit is designed for algorithms that replicate same global model across nodes, while each node operating on local parition of data.
* The global statistics collection is done by using Allreduce
* Currently, Rabit is not good at problems where model is distributed across nodes, other abstractions might suits the purpose (for example [parameter server](https://github.com/mli/parameter_server))

Design Goal
====
* rabit should run fast
* rabit is light weight
* rabit dig safe burrows to avoid disasters

Features
====
* Portable library
  - Rabit is a library instead of framework, program only need to link the library to run, without restricting to a single framework.
* Flexibility in programming
  - Programs call rabit functions, Allreduce, CheckPoint in any sequence, as opposed to defines limited functions and being called.
  - Program persist over all the iterations, unless it fails and recover
* Fault tolerance 
  - Rabit program can recover model and results of syncrhonization functions calls
* MPI compatible
  - Codes using rabit interface naturally compiles with existing MPI compiler
  - User can fall back to use MPI Allreduce if they like with no code modification

Persistence of Program
====
Many complicated Machine learning algorithm involves things like temporal memory allocation, result caching. It is good to have a persist program that runs over iterations and keeps the resources instead of re-allocate and re-compute the caching every time. Rabit allows the process to persist over all iterations.


