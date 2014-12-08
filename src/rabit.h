#ifndef RABIT_RABIT_H
#define RABIT_RABIT_H
/*!
 * \file rabit.h
 * \brief This file defines unified Allreduce/Broadcast interface of rabit
 *   The actual implementation is redirected to rabit engine
 *   Code only using this header can also compiled with MPI Allreduce(with no fault recovery),
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#include <string>
#include <vector>
#include "./engine.h"

/*! \brief namespace of rabit */
namespace rabit {
/*! \brief namespace of operator */
namespace op {
/*! \brief maximum value */
struct Max;
/*! \brief minimum value */
struct Min;
/*! \brief perform sum */
struct Sum;
/*! \brief perform bitwise OR */
struct BitOR;
}  // namespace op

/*!
 * \brief intialize the rabit module, call this once function before using anything
 * \param argc number of arguments in argv
 * \param argv the array of input arguments
 */
inline void Init(int argc, char *argv[]);
/*! 
 * \brief finalize the rabit engine, call this function after you finished all jobs 
 */
inline void Finalize(void);
/*! \brief get rank of current process */
inline int GetRank(void);
/*! \brief get total number of process */
inline int GetWorldSize(void);
/*! \brief get name of processor */
inline std::string GetProcessorName(void);
/*!
 * \brief broadcast an memory region to all others from root
 *     Example: int a = 1; Broadcast(&a, sizeof(a), root); 
 * \param sendrecv_data the pointer to send or recive buffer,
 * \param size the size of the data
 * \param root the root of process
 */
inline void Broadcast(void *sendrecv_data, size_t size, int root);
/*!
 * \brief broadcast an std::vector<DType> to all others from root
 * \param sendrecv_data the pointer to send or recive vector,
 *        for receiver, the vector does not need to be pre-allocated
 * \param root the root of process
 * \tparam DType the data type stored in vector, have to be simple data type
 *               that can be directly send by sending the sizeof(DType) data
 */
template<typename DType>
inline void Broadcast(std::vector<DType> *sendrecv_data, int root);
/*!
 * \brief broadcast an std::string to all others from root
 * \param sendrecv_data the pointer to send or recive vector,
 *        for receiver, the vector does not need to be pre-allocated
 * \param root the root of process
 */
inline void Broadcast(std::string *sendrecv_data, int root);
/*!
 * \brief perform in-place allreduce, on sendrecvbuf 
 *        this function is NOT thread-safe
 * Example Usage: the following code gives sum of the result
 *     vector<int> data(10);
 *     ...
 *     Allreduce<op::Sum>(&data[0], data.size());
 *     ...
 * \param sendrecvbuf buffer for both sending and recving data
 * \param count number of elements to be reduced
 * \tparam OP see namespace op, reduce operator 
 * \tparam DType type of data
 */
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count);
/*!
 * \brief load latest check point
 * \param global_model pointer to the globally shared model/state
 *   when calling this function, the caller need to gauranttees that global_model
 *   is the same in all nodes
 * \param local_model pointer to local model, that is specific to current node/rank
 *   this can be NULL when no local model is needed
 * 
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
inline int LoadCheckPoint(utils::ISerializable *global_model,
                          utils::ISerializable *local_model = NULL);
/*!
 * \brief checkpoint the model, meaning we finished a stage of execution
 *  every time we call check point, there is a version number which will increase by one
 * 
 * \param global_model pointer to the globally shared model/state
 *   when calling this function, the caller need to gauranttees that global_model
 *   is the same in all nodes
 * \param local_model pointer to local model, that is specific to current node/rank
 *   this can be NULL when no local state is needed
   * NOTE: local_model requires explicit replication of the model for fault-tolerance, which will
   *       bring replication cost in CheckPoint function. global_model do not need explicit replication.
   *       So only CheckPoint with global_model if possible
   * \sa LoadCheckPoint, VersionNumber
   */
inline void CheckPoint(const utils::ISerializable *global_model,
                       const utils::ISerializable *local_model = NULL);
/*!
 * \return version number of current stored model,
 *         which means how many calls to CheckPoint we made so far
 * \sa LoadCheckPoint, CheckPoint
 */
inline int VersionNumber(void);
}  // namespace rabit
// implementation of template functions
#include "./rabit-inl.h"
#endif  // RABIT_ALLREDUCE_H
