/*!
 *  Copyright (c) 2014 by Contributors
 * \file engine.h
 * \brief This file defines the core interface of rabit library
 * \author Tianqi Chen, Nacho, Tianyi
 */
#ifndef RABIT_INTERNAL_ENGINE_H_
#define RABIT_INTERNAL_ENGINE_H_
#include <string>
#include "../serializable.h"

namespace MPI {
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
  typedef void (PreprocFunction) (void *arg);
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
  typedef void (ReduceFunction) (const void *src,
                                 void *dst, int count,
                                 const MPI::Datatype &dtype);
  /*! \brief virtual destructor */
  virtual ~IEngine() {}
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
                         PreprocFunction prepare_fun = NULL,
                         void *prepare_arg = NULL) = 0;
  /*!
   * \brief broadcasts data from root to every other node
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   */
  virtual void Broadcast(void *sendrecvbuf_, size_t size, int root) = 0;
  /*!
   * \brief explicitly re-initialize everything before calling LoadCheckPoint
   *    call this function when IEngine throws an exception,
   *    this function should only be used for test purposes
   */
  virtual void InitAfterException(void) = 0;
  /*!
   * \brief loads the latest check point
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller needs to guarantee that the global_model
   *   is the same in all nodes
   * \param local_model pointer to the local model that is specific to current node/rank
   *   this can be NULL when no local model is needed
   *
   * \return the version number of the model loaded
   *     if returned version == 0, this means no model has been CheckPointed
   *     the p_model is not touched, users should do necessary initialization by themselves
   *
   *   Common usage example:
   *      int iter = rabit::LoadCheckPoint(&model);
   *      if (iter == 0) model.InitParameters();
   *      for (i = iter; i < max_iter; ++i) {
   *        do many things, include allreduce
   *        rabit::CheckPoint(model);
   *      }
   *
   * \sa CheckPoint, VersionNumber
   */
  virtual int LoadCheckPoint(Serializable *global_model,
                             Serializable *local_model = NULL) = 0;
  /*!
   * \brief checkpoints the model, meaning a stage of execution was finished
   *  every time we call check point, a version number increases by ones
   *
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller needs to guarantee that the global_model
   *   is the same in every node
   * \param local_model pointer to the local model that is specific to current node/rank
   *   this can be NULL when no local state is needed
   *
   * NOTE: local_model requires explicit replication of the model for fault-tolerance, which will
   *       bring replication cost in CheckPoint function. global_model does not need explicit replication.
   *       So, only CheckPoint with global_model if possible
   *
   * \sa LoadCheckPoint, VersionNumber
   */
  virtual void CheckPoint(const Serializable *global_model,
                          const Serializable *local_model = NULL) = 0;
  /*!
   * \brief This function can be used to replace CheckPoint for global_model only,
   *   when certain condition is met (see detailed explanation).
   *
   *   This is a "lazy" checkpoint such that only the pointer to global_model is
   *   remembered and no memory copy is taken. To use this function, the user MUST ensure that:
   *   The global_model must remain unchanged until the last call of Allreduce/Broadcast in the current version finishes.
   *   In other words, global_model can be changed only between the last call of
   *   Allreduce/Broadcast and LazyCheckPoint in the current version
   *
   *   For example, suppose the calling sequence is:
   *   LazyCheckPoint, code1, Allreduce, code2, Broadcast, code3, LazyCheckPoint
   *
   *   If the user can only change global_model in code3, then LazyCheckPoint can be used to
   *   improve the efficiency of the program.
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller needs to guarantee that global_model
   *   is the same in every node
   * \sa LoadCheckPoint, CheckPoint, VersionNumber
   */
  virtual void LazyCheckPoint(const Serializable *global_model) = 0;
  /*!
   * \return version number of the current stored model,
   *         which means how many calls to CheckPoint we made so far
   * \sa LoadCheckPoint, CheckPoint
   */
  virtual int VersionNumber(void) const = 0;
  /*! \brief gets rank of current node */
  virtual int GetRank(void) const = 0;
  /*! \brief gets total number of nodes */
  virtual int GetWorldSize(void) const = 0;
  /*! \brief whether we run in distribted mode */
  virtual bool IsDistributed(void) const = 0;
  /*! \brief gets the host name of the current node */
  virtual std::string GetHost(void) const = 0;
  /*!
   * \brief prints the msg in the tracker,
   *    this function can be used to communicate progress information to
   *    the user who monitors the tracker
   * \param msg message to be printed in the tracker
   */
  virtual void TrackerPrint(const std::string &msg) = 0;
};

/*! \brief initializes the engine module */
void Init(int argc, char *argv[]);
/*! \brief finalizes the engine module */
void Finalize(void);
/*! \brief singleton method to get engine */
IEngine *GetEngine(void);

/*! \brief namespace that contains stubs to be compatible with MPI */
namespace mpi {
/*!\brief enum of all operators */
enum OpType {
  kMax = 0,
  kMin = 1,
  kSum = 2,
  kBitwiseOR = 3
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
void Allreduce_(void *sendrecvbuf,
                size_t type_nbytes,
                size_t count,
                IEngine::ReduceFunction red,
                mpi::DataType dtype,
                mpi::OpType op,
                IEngine::PreprocFunction prepare_fun = NULL,
                void *prepare_arg = NULL);

/*!
 * \brief handle for customized reducer, used to handle customized reduce
 *  this class is mainly created for compatiblity issues with MPI's customized reduce
 */
class ReduceHandle {
 public:
  // constructor
  ReduceHandle(void);
  // destructor
  ~ReduceHandle(void);
  /*!
   * \brief initialize the reduce function,
   *   with the type the reduce function needs to deal with
   *   the reduce function MUST be communicative
   */
  void Init(IEngine::ReduceFunction redfunc, size_t type_nbytes);
  /*!
   * \brief customized in-place all reduce operation
   * \param sendrecvbuf the in place send-recv buffer
   * \param type_n4bytes size of the type, in terms of 4bytes
   * \param count number of elements to send
   * \param prepare_func Lazy preprocessing function, lazy prepare_fun(prepare_arg)
   *                     will be called by the function before performing Allreduce in order to initialize the data in sendrecvbuf_.
   *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
   * \param prepare_arg argument used to pass into the lazy preprocessing function
   */
  void Allreduce(void *sendrecvbuf,
                 size_t type_nbytes,
                 size_t count,
                 IEngine::PreprocFunction prepare_fun = NULL,
                 void *prepare_arg = NULL);
  /*! \return the number of bytes occupied by the type */
  static int TypeSize(const MPI::Datatype &dtype);

 protected:
  // handle function field
  void *handle_;
  // reduce function of the reducer
  IEngine::ReduceFunction *redfunc_;
  // handle to the type field
  void *htype_;
  // the created type in 4 bytes
  size_t created_type_nbytes_;
};
}  // namespace engine
}  // namespace rabit
#endif  // RABIT_INTERNAL_ENGINE_H_
