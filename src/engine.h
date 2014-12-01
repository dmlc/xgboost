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
   * \brief load latest check point
   * \param p_model pointer to the model
   * \return true if there was stored checkpoint and load was successful
   *   false if there was no stored checkpoint, means we are start over gain
   */
  virtual bool LoadCheckPoint(utils::ISerializable *p_model) = 0;
  /*!
   * \brief checkpoint the model, meaning we finished a stage of execution
   * \param p_model pointer to the model
   */
  virtual void CheckPoint(const utils::ISerializable &model) = 0;    
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

}  // namespace engine
}  // namespace rabit
#endif  // RABIT_ENGINE_H
