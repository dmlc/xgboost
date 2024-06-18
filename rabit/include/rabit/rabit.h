/*!
 *  Copyright (c) 2014 by Contributors
 * \file rabit.h
 * \brief This file defines rabit's Allreduce/Broadcast interface
 *   The rabit engine contains the actual implementation
 *   Code that only uses this header can also be compiled with MPI Allreduce (non fault-tolerant),
 *
 *   rabit.h and serializable.h is all what the user needs to use the rabit interface
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#ifndef RABIT_RABIT_H_  // NOLINT(*)
#define RABIT_RABIT_H_  // NOLINT(*)
#include <string>
#include <vector>
#include <functional>
// engine definition of rabit, defines internal implementation
// to use rabit interface, there is no need to read engine.h
// rabit.h and serializable.h are enough to use the interface
#include "./internal/engine.h"

/*! \brief rabit namespace */
namespace rabit {
/*!
 * \brief defines stream used in rabit
 * see definition of Stream in dmlc/io.h
 */
using Stream = dmlc::Stream;
/*!
 * \brief defines serializable objects used in rabit
 * see definition of Serializable in dmlc/io.h
 */
using Serializable = dmlc::Serializable;

/*!
 * \brief reduction operators namespace
 */
namespace op {
/*!
 * \class rabit::op::Max
 * \brief maximum reduction operator
 */
struct Max;
/*!
 * \class rabit::op::Min
 * \brief minimum reduction operator
 */
struct Min;
/*!
 * \class rabit::op::Sum
 * \brief sum reduction operator
 */
struct Sum;
/*!
 * \class rabit::op::BitAND
 * \brief bitwise AND reduction operator
 */
struct BitAND;
/*!
 * \class rabit::op::BitOR
 * \brief bitwise OR reduction operator
 */
struct BitOR;
/*!
 * \class rabit::op::BitXOR
 * \brief bitwise XOR reduction operator
 */
struct BitXOR;
}  // namespace op
/*!
 * \brief initializes rabit, call this once at the beginning of your program
 * \param argc number of arguments in argv
 * \param argv the array of input arguments
 * \return true if initialized successfully, otherwise false
 */
inline bool Init(int argc, char *argv[]);
/*!
 * \brief finalizes the rabit engine, call this function after you finished with all the jobs
 * \return true if finalized successfully, otherwise false
 */
inline bool Finalize();
/*! \brief gets rank of the current process
 * \return rank number of worker*/
inline int GetRank();
/*! \brief gets total number of processes
 * \return total world size*/
inline int GetWorldSize();
/*! \brief whether rabit env is in distributed mode
 * \return is distributed*/
inline bool IsDistributed();

/*! \brief gets processor's name
 * \return processor name*/
inline std::string GetProcessorName();
/*!
 * \brief prints the msg to the tracker,
 *    this function can be used to communicate progress information to
 *    the user who monitors the tracker
 * \param msg the message to be printed
 */
inline void TrackerPrint(const std::string &msg);

#ifndef RABIT_STRICT_CXX98_
/*!
 * \brief prints the msg to the tracker, this function may not be available
 *    in very strict c++98 compilers, though it usually is.
 *    this function can be used to communicate progress information to
 *    the user who monitors the tracker
 * \param fmt the format string
 */
inline void TrackerPrintf(const char *fmt, ...);
#endif  // RABIT_STRICT_CXX98_
/*!
 * \brief broadcasts a memory region to every node from the root
 *
 *     Example: int a = 1; Broadcast(&a, sizeof(a), root);
 * \param sendrecv_data the pointer to the send/receive buffer,
 * \param size the data size
 * \param root the process root
 */
inline void Broadcast(void *sendrecv_data, size_t size, int root);

/*!
 * \brief broadcasts an std::vector<DType> to every node from root
 * \param sendrecv_data the pointer to send/receive vector,
 *        for the receiver, the vector does not need to be pre-allocated
 * \param root the process root
 * \tparam DType the data type stored in the vector, has to be a simple data type
 *               that can be directly transmitted by sending the sizeof(DType)
 */
template<typename DType>
inline void Broadcast(std::vector<DType> *sendrecv_data, int root);
/*!
 * \brief broadcasts a std::string to every node from the root
 * \param sendrecv_data the pointer to the send/receive buffer,
 *        for the receiver, the vector does not need to be pre-allocated
 * \param _file caller file name used to generate unique cache key
 * \param _line caller line number used to generate unique cache key
 * \param _caller caller function name used to generate unique cache key
 * \param root the process root
 */
inline void Broadcast(std::string *sendrecv_data, int root);
/*!
 * \brief performs in-place Allreduce on sendrecvbuf
 *        this function is NOT thread-safe
 *
 * Example Usage: the following code does an Allreduce and outputs the sum as the result
 * \code{.cpp}
 * vector<int> data(10);
 * ...
 * Allreduce<op::Sum>(&data[0], data.size());
 * ...
 * \endcode
 *
 * \param sendrecvbuf buffer for both sending and receiving data
 * \param count number of elements to be reduced
 * \param prepare_fun Lazy preprocessing function, if it is not NULL, prepare_fun(prepare_arg)
 *                    will be called by the function before performing Allreduce in order to initialize the data in sendrecvbuf.
 *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
 * \param prepare_arg argument used to pass into the lazy preprocessing function
 * \tparam OP see namespace op, reduce operator
 * \tparam DType data type
 */
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count,
                      void (*prepare_fun)(void *) = nullptr,
                      void *prepare_arg = nullptr);

/*!
* \brief Allgather function, each node have a segment of data in the ring of sendrecvbuf,
*  the data provided by current node k is [slice_begin, slice_end),
*  the next node's segment must start with slice_end
*  after the call of Allgather, sendrecvbuf_ contains all the contents including all segments
*  use a ring based algorithm
*
* \param sendrecvbuf_ buffer for both sending and receiving data, it is a ring conceptually
* \param total_size total size of data to be gathered
* \param slice_begin beginning of the current slice
* \param slice_end end of the current slice
* \param size_prev_slice size of the previous slice i.e. slice of node (rank - 1) % world_size
*/
template<typename DType>
inline void Allgather(DType *sendrecvbuf_,
                  size_t total_size,
                  size_t slice_begin,
                  size_t slice_end,
                  size_t size_prev_slice);

// C++11 support for lambda prepare function
#if DMLC_USE_CXX11
/*!
 * \brief performs in-place Allreduce, on sendrecvbuf
 *        with a prepare function specified by a lambda function
 *
 * Example Usage:
 * \code{.cpp}
 * // the following code does an Allreduce and outputs the sum as the result
 * vector<int> data(10);
 * ...
 * Allreduce<op::Sum>(&data[0], data.size(), [&]() {
 *                     for (int i = 0; i < 10; ++i) {
 *                       data[i] = i;
 *                     }
 *                    });
 *     ...
 * \endcode
 * \param sendrecvbuf buffer for both sending and receiving data
 * \param count number of elements to be reduced
 * \param prepare_fun  Lazy lambda preprocessing function, prepare_fun() will be invoked
 *                     by the function before performing Allreduce in order to initialize the data in sendrecvbuf.
 *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
 * \tparam OP see namespace op, reduce operator
 * \tparam DType data type
 */
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count,
                      std::function<void()> prepare_fun);
#endif  // C++11

/*!
 * \brief deprecated, planned for removal after checkpoing from JVM package is removed.
 */
inline int LoadCheckPoint();
/*!
 * \brief deprecated, planned for removal after checkpoing from JVM package is removed.
 */
inline void CheckPoint();

/*!
 * \return version number of the current stored model,
 *         which means how many calls to CheckPoint we made so far
 * \sa LoadCheckPoint, CheckPoint
 */
inline int VersionNumber();
}  // namespace rabit
// implementation of template functions
#include "./internal/rabit-inl.h"
#endif  // RABIT_RABIT_H_ // NOLINT(*)
