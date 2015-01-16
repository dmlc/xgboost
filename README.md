## rabit: Reliable Allreduce and Broadcast Interface

rabit is a light weight library that provides a fault tolerant interface of Allreduce and Broadcast. It is designed to support easy implementations of distributed machine learning programs, many of which fall naturally under the Allreduce abstraction.

* [Tutorial](guide)
* [API Documentation](http://homes.cs.washington.edu/~tqchen/rabit/doc)
* You can also directly read the [interface header](include/rabit.h)

Features
====
All these features comes from the facts about small rabbit:)
* Portable: rabit is light weight and runs everywhere
  - Rabit is a library instead of a framework, a program only needs to link the library to run
  - Rabit only replies on a mechanism to start program, which was provided by most framework
  - You can port rabit programs easily to many frameworks, including Hadoop, MPI without changing your code
* Scalable and Flexible: rabit runs fast
  * Rabit program use Allreduce to communicate, and do not suffer the cost between iterations of MapReduce abstraction.
  - Programs can call rabit functions in any order, as opposed to frameworks where callbacks are offered and called by the framework, i.e. inversion of control principle.
  - Programs persist over all the iterations, unless they fail and recover.
* Fault Tolerant: rabit dig burrows to avoid disasters
  - Rabit programs can recover the model and results using synchronous function calls.

Use Rabit
====
* Type make in the root folder will compile the rabit library in lib folder
* Add lib to the library path and include to the include path of compiler
