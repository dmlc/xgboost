/*!
 * \file engine.h
 * \brief This file defines the core interface of allreduce library
 * \author Tianqi Chen, Nacho, Tianyi
 */
#ifndef RABIT_ENGINE_H
#define RABIT_ENGINE_H
#include "./io.h"

namespace MPI {
/*! \brief MPI data type just to be compatible with MPI reduce function*/
class Datatype;
}

/*! \brief namespace of rabit */
namespace rabit {
/*! \brief core interface of engine */
namespace engine {
/*! \brief interface of core AllReduce engine */
class IEngine {
 public:
  /*! 
   * \brief reduce function, the same form of MPI reduce function is used,
   *        to be compatible with MPI interface
   *        In all the functions, the memory is ensured to aligned to 64-bit
   *        which means it is OK to cast src,dst to double* int* etc
   * \param src pointer to source space
   * \param dst pointer to destination reduction
   * \param count total number of elements to be reduced(note this is total number of elements instead of bytes)
   *              the definition of reduce function should be type aware
   * \param dtype the data type object, to be compatible with MPI reduce
   */
  typedef void (ReduceFunction) (const void *src,
                                 void *dst, int count,
                                 const MPI::Datatype &dtype);
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf 
   *        this function is NOT thread-safe
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   */
  virtual void AllReduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,
                         ReduceFunction reducer) = 0;
  /*!
   * \brief broadcast data from root to all nodes
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   */
  virtual void Broadcast(void *sendrecvbuf_, size_t size, int root) = 0;
  /*!
   * \brief explicitly re-init everything before calling LoadCheckPoint
   *    call this function when IEngine throw an exception out,
   *    this function is only used for test purpose
   */
  virtual void InitAfterException(void) = 0;
  /*!
   * \brief load latest check point
   * \param p_model pointer to the model
   * \return the version number of check point loaded
   *     if returned version == 0, this means no model has been CheckPointed
   *     the p_model is not touched, user should do necessary initialization by themselves
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
  virtual int LoadCheckPoint(utils::ISerializable *p_model) = 0;
  /*!
   * \brief checkpoint the model, meaning we finished a stage of execution
   *  every time we call check point, there is a version number which will increase by one
   * 
   * \param p_model pointer to the model
   * \sa LoadCheckPoint, VersionNumber
   */
  virtual void CheckPoint(const utils::ISerializable &model) = 0;
  /*!
   * \return version number of current stored model,
   *         which means how many calls to CheckPoint we made so far
   * \sa LoadCheckPoint, CheckPoint
   */
  virtual int VersionNumber(void) const = 0;
  /*! \brief get rank of current node */
  virtual int GetRank(void) const = 0;
  /*! \brief get total number of */
  virtual int GetWorldSize(void) const = 0;
  /*! \brief get the host name of current node */  
  virtual std::string GetHost(void) const = 0;
};

/*! \brief intiialize the engine module */
void Init(int argc, char *argv[]);
/*! \brief finalize engine module */
void Finalize(void);
/*! \brief singleton method to get engine */
IEngine *GetEngine(void);

/*! \brief namespace that contains staffs to be compatible with MPI */
namespace mpi {
/*!\brief enum of all operators */
enum OpType {
  kMax, kMin, kSum, kBitwiseOR
};
/*!\brief enum of supported data types */
enum DataType {
  kInt,
  kUInt,
  kDouble,
  kFloat
};
}  // namespace mpi
/*!
 * \brief perform in-place allreduce, on sendrecvbuf 
 *   this is an internal function used by rabit to be able to compile with MPI
 *   do not use this function directly
 * \param sendrecvbuf buffer for both sending and recving data
 * \param type_nbytes the unit number of bytes the type have
 * \param count number of elements to be reduced
 * \param reducer reduce function
 * \param dtype the data type 
 * \param op the reduce operator type
 */
void AllReduce_(void *sendrecvbuf,
                size_t type_nbytes,
                size_t count,
                IEngine::ReduceFunction red,               
                mpi::DataType dtype,
                mpi::OpType op);
}  // namespace engine
}  // namespace rabit
#endif  // RABIT_ENGINE_H
