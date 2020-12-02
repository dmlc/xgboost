/*!
 * Copyright by Contributors
 * \file c_api.h
 * \author Tianqi Chen
 * \brief a C style API of rabit.
 */
#ifndef RABIT_C_API_H_
#define RABIT_C_API_H_

#ifdef __cplusplus
#define RABIT_EXTERN_C extern "C"
#include <cstdio>
#else
#define RABIT_EXTERN_C
#include <stdio.h>
#endif  // __cplusplus

#if defined(_MSC_VER) || defined(_WIN32)
#define RABIT_DLL RABIT_EXTERN_C __declspec(dllexport)
#else
#define RABIT_DLL RABIT_EXTERN_C __attribute__ ((visibility ("default")))
#endif  // defined(_MSC_VER) || defined(_WIN32)

/*! \brief rabit unsigned long type */
typedef unsigned long rbt_ulong;  // NOLINT(*)

/*!
 * \brief intialize the rabit module,
 *  call this once before using anything
 *  The additional arguments is not necessary.
 *  Usually rabit will detect settings
 *  from environment variables.
 * \param argc number of arguments in argv
 * \param argv the array of input arguments
 * \return true if rabit is initialized successfully otherwise false
 */
RABIT_DLL bool RabitInit(int argc, char *argv[]);

/*!
 * \brief finalize the rabit engine,
 * call this function after you finished all jobs.
 * \return true if rabit is initialized successfully otherwise false
 */
RABIT_DLL int RabitFinalize(void);

/*!
 * \brief get rank of previous process in ring topology
 * \return rank number of worker
 * */
RABIT_DLL int RabitGetRingPrevRank(void);

/*!
 * \brief get rank of current process
 * \return rank number of worker
 * */
RABIT_DLL int RabitGetRank(void);

/*!
 * \brief get total number of process
 * \return total world size
 * */
RABIT_DLL int RabitGetWorldSize(void);

/*!
 * \brief get rank of current process
 * \return if rabit is distributed
 * */
RABIT_DLL int RabitIsDistributed(void);

/*!
 * \brief print the msg to the tracker,
 *    this function can be used to communicate the information of the progress to
 *    the user who monitors the tracker
 * \param msg the message to be printed
 */
RABIT_DLL int RabitTrackerPrint(const char *msg);
/*!
 * \brief get name of processor
 * \param out_name hold output string
 * \param out_len hold length of output string
 * \param max_len maximum buffer length of input
   */
RABIT_DLL void RabitGetProcessorName(char *out_name,
                                     rbt_ulong *out_len,
                                     rbt_ulong max_len);
/*!
 * \brief broadcast an memory region to all others from root
 *
 *     Example: int a = 1; Broadcast(&a, sizeof(a), root);
 * \param sendrecv_data the pointer to send or recive buffer,
 * \param size the size of the data
 * \param root the root of process
 */
RABIT_DLL int RabitBroadcast(void *sendrecv_data, rbt_ulong size, int root);

/*!
 * \brief Allgather function, each node have a segment of data in the ring of sendrecvbuf,
 *  the data provided by current node k is [slice_begin, slice_end),
 *  the next node's segment must start with slice_end
 *  after the call of Allgather, sendrecvbuf_ contains all the contents including all segments
 *  use a ring based algorithm
 *
 * \param sendrecvbuf buffer for both sending and receiving data, it is a ring conceptually
 * \param total_size total size of data to be gathered
 * \param beginIndex beginning of the current slice in sendrecvbuf of type enum_dtype
 * \param size_node_slice size of the current node slice
 * \param size_prev_slice size of the previous slice i.e. slice of node (rank - 1) % world_size
 * \param enum_dtype the enumeration of data type, see rabit::engine::mpi::DataType in engine.h of rabit include
 * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
 * \sa ReturnType
 */
RABIT_DLL int RabitAllgather(void *sendrecvbuf, size_t total_size,
                             size_t beginIndex, size_t size_node_slice,
                             size_t size_prev_slice, int enum_dtype);

/*!
 * \brief perform in-place allreduce, on sendrecvbuf
 *        this function is NOT thread-safe
 *
 * Example Usage: the following code gives sum of the result
 *     vector<int> data(10);
 *     ...
 *     Allreduce<op::Sum>(&data[0], data.size());
 *     ...
 * \param sendrecvbuf buffer for both sending and recving data
 * \param count number of elements to be reduced
 * \param enum_dtype the enumeration of data type, see rabit::engine::mpi::DataType in engine.h of rabit include
 * \param enum_op the enumeration of operation type, see rabit::engine::mpi::OpType in engine.h of rabit
 * \param prepare_fun Lazy preprocessing function, if it is not NULL, prepare_fun(prepare_arg)
 *                    will be called by the function before performing Allreduce, to intialize the data in sendrecvbuf_.
 *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
 * \param prepare_arg argument used to passed into the lazy preprocessing function
 */
RABIT_DLL int RabitAllreduce(void *sendrecvbuf, size_t count, int enum_dtype,
                             int enum_op, void (*prepare_fun)(void *arg),
                             void *prepare_arg);

/*!
 * \brief load latest check point
 * \param out_global_model hold output of serialized global_model
 * \param out_global_len the output length of serialized global model
 * \param out_local_model hold output of serialized local_model, can be NULL
 * \param out_local_len the output length of serialized local model, can be NULL
 *
 * \return the version number of check point loaded
 *     if returned version == 0, this means no model has been CheckPointed
 *     nothing will be touched
 */
RABIT_DLL int RabitLoadCheckPoint(char **out_global_model,
                                  rbt_ulong *out_global_len,
                                  char **out_local_model,
                                  rbt_ulong *out_local_len);
/*!
 * \brief checkpoint the model, meaning we finished a stage of execution
 *  every time we call check point, there is a version number which will increase by one
 *
 * \param global_model hold content of serialized global_model
 * \param global_len the content length of serialized global model
 * \param local_model hold content of serialized local_model, can be NULL
 * \param local_len the content length of serialized local model, can be NULL
 *
 * NOTE: local_model requires explicit replication of the model for fault-tolerance, which will
 *       bring replication cost in CheckPoint function. global_model do not need explicit replication.
 *       So only CheckPoint with global_model if possible
 */
RABIT_DLL void RabitCheckPoint(const char *global_model,
                               rbt_ulong global_len,
                               const char *local_model,
                               rbt_ulong local_len);
/*!
 * \return version number of current stored model,
 * which means how many calls to CheckPoint we made so far
 * \return rabit version number
 */
RABIT_DLL int RabitVersionNumber(void);


/*!
 * \brief a Dummy function,
 *  used to cause force link of C API  into the  DLL.
 * \code
 * \/\/force link rabit C API library.
 * static int must_link_rabit_ = RabitLinkTag();
 * \endcode
 * \return a dummy integer.
 */
RABIT_DLL int RabitLinkTag(void);

#endif  // RABIT_C_API_H_
