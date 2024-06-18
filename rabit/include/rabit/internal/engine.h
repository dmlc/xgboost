/*!
 *  Copyright (c) 2014 by Contributors
 * \file engine.h
 * \brief This file defines the core interface of rabit library
 * \author Tianqi Chen, Nacho, Tianyi
 */
#ifndef RABIT_INTERNAL_ENGINE_H_
#define RABIT_INTERNAL_ENGINE_H_
#include <string>
#include "rabit/serializable.h"

namespace MPI {  // NOLINT
/*! \brief MPI data type just to be compatible with MPI reduce function*/
class Datatype;
}

/*! \brief namespace of rabit */
namespace rabit {
/*! \brief core interface of the engine */
namespace engine {
/*! \brief interface of core Allreduce engine */
class IEngine {
 public:
  /*!
   * \brief Preprocessing function, that is called before AllReduce,
   *        used to prepare the data used by AllReduce
   * \param arg additional possible argument used to invoke the preprocessor
   */
  typedef void (PreprocFunction) (void *arg);  // NOLINT
  /*!
   * \brief reduce function, the same form of MPI reduce function is used,
   *        to be compatible with MPI interface
   *        In all the functions, the memory is ensured to aligned to 64-bit
   *        which means it is OK to cast src,dst to double* int* etc
   * \param src pointer to source space
   * \param dst pointer to destination reduction
   * \param count total number of elements to be reduced (note this is total number of elements instead of bytes)
   *              the definition of the reduce function should be type aware
   * \param dtype the data type object, to be compatible with MPI reduce
   */
  typedef void (ReduceFunction) (const void *src,  // NOLINT
                                 void *dst, int count,
                                 const MPI::Datatype &dtype);
  /*! \brief virtual destructor */
  ~IEngine() = default;
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
  virtual void Allgather(void *sendrecvbuf,
                         size_t total_size,
                         size_t slice_begin,
                         size_t slice_end,
                         size_t size_prev_slice) = 0;
  /*!
   * \brief performs in-place Allreduce, on sendrecvbuf
   *        this function is NOT thread-safe
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param type_nbytes the number of bytes the type has
   * \param count number of elements to be reduced
   * \param reducer reduce function
   * \param prepare_func Lazy preprocessing function, if it is not NULL, prepare_fun(prepare_arg)
   *                     will be called by the function before performing Allreduce in order to initialize the data in sendrecvbuf.
   *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
   * \param prepare_arg argument used to pass into the lazy preprocessing function
   */
  virtual void Allreduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,
                         ReduceFunction reducer,
                         PreprocFunction prepare_fun = nullptr,
                         void *prepare_arg = nullptr) = 0;
  /*!
   * \brief broadcasts data from root to every other node
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   */
  virtual void Broadcast(void *sendrecvbuf_, size_t size, int root) = 0;
  /*!
   * deprecated
   */
  virtual int LoadCheckPoint() = 0;
  /*!
   * \brief Increase internal version number. Deprecated.
   */
  virtual void CheckPoint() = 0;
  /*!
   * \return version number of the current stored model,
   *         which means how many calls to CheckPoint we made so far
   * \sa LoadCheckPoint, CheckPoint
   */
  virtual int VersionNumber() const = 0;
  /*! \brief gets rank of previous node in ring topology */
  virtual int GetRingPrevRank() const = 0;
  /*! \brief gets rank of current node */
  virtual int GetRank() const = 0;
  /*! \brief gets total number of nodes */
  virtual int GetWorldSize() const = 0;
  /*! \brief whether we run in distribted mode */
  virtual bool IsDistributed() const = 0;
  /*! \brief gets the host name of the current node */
  virtual std::string GetHost() const = 0;
  /*!
   * \brief prints the msg in the tracker,
   *    this function can be used to communicate progress information to
   *    the user who monitors the tracker
   * \param msg message to be printed in the tracker
   */
  virtual void TrackerPrint(const std::string &msg) = 0;
};

/*! \brief initializes the engine module */
bool Init(int argc, char *argv[]);
/*! \brief finalizes the engine module */
bool Finalize();
/*! \brief singleton method to get engine */
IEngine *GetEngine();

/*! \brief namespace that contains stubs to be compatible with MPI */
namespace mpi {
/*!\brief enum of all operators */
enum OpType {
  kMax = 0,
  kMin = 1,
  kSum = 2,
  kBitwiseAND = 3,
  kBitwiseOR = 4,
  kBitwiseXOR = 5,
};
/*!\brief enum of supported data types */
enum DataType {
  kChar = 0,
  kUChar = 1,
  kInt = 2,
  kUInt = 3,
  kLong = 4,
  kULong = 5,
  kFloat = 6,
  kDouble = 7,
  kLongLong = 8,
  kULongLong = 9
};
}  // namespace mpi
/*!
 * \brief Allgather function, each node have a segment of data in the ring of sendrecvbuf,
 *  the data provided by current node k is [slice_begin, slice_end),
 *  the next node's segment must start with slice_end
 *  after the call of Allgather, sendrecvbuf_ contains all the contents including all segments
 *  use a ring based algorithm
 *
 * \param sendrecvbuf buffer for both sending and receiving data, it is a ring conceptually
 * \param total_size total size of data to be gathered
 * \param slice_begin beginning of the current slice
 * \param slice_end end of the current slice
 * \param size_prev_slice size of the previous slice i.e. slice of node (rank - 1) % world_size
 */
void Allgather(void* sendrecvbuf,
                   size_t total_size,
                   size_t slice_begin,
                   size_t slice_end,
                   size_t size_prev_slice);
/*!
 * \brief perform in-place Allreduce, on sendrecvbuf
 *   this is an internal function used by rabit to be able to compile with MPI
 *   do not use this function directly
 * \param sendrecvbuf buffer for both sending and receiving data
 * \param type_nbytes the number of bytes the type has
 * \param count number of elements to be reduced
 * \param reducer reduce function
 * \param dtype the data type
 * \param op the reduce operator type
 * \param prepare_func Lazy preprocessing function, lazy prepare_fun(prepare_arg)
 *                     will be called by the function before performing Allreduce, to initialize the data in sendrecvbuf_.
 *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
 * \param prepare_arg argument used to pass into the lazy preprocessing function.
 */
void Allreduce_(void *sendrecvbuf,  // NOLINT
                size_t type_nbytes,
                size_t count,
                IEngine::ReduceFunction red,
                mpi::DataType dtype,
                mpi::OpType op,
                IEngine::PreprocFunction prepare_fun = nullptr,
                void *prepare_arg = nullptr);
}  // namespace engine
}  // namespace rabit
#endif  // RABIT_INTERNAL_ENGINE_H_
