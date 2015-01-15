/*!
 *  Copyright (c) 2014 by Contributors
 * \file rabit.h
 * \brief This file defines unified Allreduce/Broadcast interface of rabit
 *   The actual implementation is redirected to rabit engine
 *   Code only using this header can also compiled with MPI Allreduce(with no fault recovery),
 *
 *   rabit.h and serializable.h is all the user need to use rabit interface
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#ifndef RABIT_RABIT_H_
#define RABIT_RABIT_H_
#include <string>
#include <vector>
// optionally support of lambda function in C++11, if available
#if __cplusplus >= 201103L
#include <functional>
#endif  // C++11
// contains definition of ISerializable
#include "./rabit_serializable.h"
// engine definition of rabit, defines internal implementation
// to use rabit interface, there is no need to read engine.h
// rabit.h and serializable.h are suffice to use the interface
#include "./rabit/engine.h"

/*! \brief namespace of rabit */
namespace rabit {
/*!
 * \brief namespace of reduction operators
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
 * \class rabit::op::BitOR
 * \brief bitwise or reduction operator
 */
struct BitOR;
}  // namespace op
/*!
 * \brief intialize the rabit module, call this once before using anything
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
/*! \brief whether rabit env is in distributed mode */
inline bool IsDistributed(void) {
  return GetWorldSize() != 1;
}
/*! \brief get name of processor */
inline std::string GetProcessorName(void);
/*!
 * \brief print the msg to the tracker,
 *    this function can be used to communicate the information of the progress to
 *    the user who monitors the tracker
 * \param msg the message to be printed
 */
inline void TrackerPrint(const std::string &msg);
#ifndef RABIT_STRICT_CXX98_
/*!
 * \brief print the msg to the tracker, this function may not be available
 *    in very strict c++98 compilers, but is available most of the time
 *    this function can be used to communicate the information of the progress to
 *    the user who monitors the tracker
 * \param fmt the format string
 */
inline void TrackerPrintf(const char *fmt, ...);
#endif
/*!
 * \brief broadcast an memory region to all others from root
 *
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
 *
 * Example Usage: the following code gives sum of the result
 *     vector<int> data(10);
 *     ...
 *     Allreduce<op::Sum>(&data[0], data.size());
 *     ...
 * \param sendrecvbuf buffer for both sending and recving data
 * \param count number of elements to be reduced
 * \param prepare_fun Lazy preprocessing function, if it is not NULL, prepare_fun(prepare_arg)
 *                    will be called by the function before performing Allreduce, to intialize the data in sendrecvbuf_.
 *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
 * \param prepare_arg argument used to passed into the lazy preprocessing function 
 * \tparam OP see namespace op, reduce operator 
 * \tparam DType type of data
 */
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count,
                      void (*prepare_fun)(void *arg) = NULL,
                      void *prepare_arg = NULL);

// C++11 support for lambda prepare function
#if __cplusplus >= 201103L
/*!
 * \brief perform in-place allreduce, on sendrecvbuf
 *        with a prepare function specified by lambda function
 *
 * Example Usage: the following code gives sum of the result
 *     vector<int> data(10);
 *     ...
 *     Allreduce<op::Sum>(&data[0], data.size(), [&]() {
 *                          for (int i = 0; i < 10; ++i) {
 *                            data[i] = i;
 *                          }
 *                        });
 *     ...
 * \param sendrecvbuf buffer for both sending and recving data
 * \param count number of elements to be reduced
 * \param prepare_fun  Lazy lambda preprocessing function, prepare_fun() will be invoked
 *                     will be called by the function before performing Allreduce, to intialize the data in sendrecvbuf_.
 *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
 * \tparam OP see namespace op, reduce operator 
 * \tparam DType type of data
 */
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count,
                      std::function<void()> prepare_fun);
#endif  // C++11
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
inline int LoadCheckPoint(ISerializable *global_model,
                          ISerializable *local_model = NULL);
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
inline void CheckPoint(const ISerializable *global_model,
                       const ISerializable *local_model = NULL);
/*!
 * \brief This function can be used to replace CheckPoint for global_model only,
 *   when certain condition is met(see detailed expplaination).
 * 
 *   This is a "lazy" checkpoint such that only the pointer to global_model is
 *   remembered and no memory copy is taken. To use this function, the user MUST ensure that:
 *   The global_model must remain unchanged util last call of Allreduce/Broadcast in current version finishs.
 *   In another words, global_model model can be changed only between last call of 
 *   Allreduce/Broadcast and LazyCheckPoint in current version
 *   
 *   For example, suppose the calling sequence is:
 *   LazyCheckPoint, code1, Allreduce, code2, Broadcast, code3, LazyCheckPoint
 *   
 *   If user want to use LazyCheckPoint, then the user MUST only change global_model in code3.
 *
 *   The Use of LazyCheckPoint instead of CheckPoint will improve efficiency of the program.
 * \param global_model pointer to the globally shared model/state
 *   when calling this function, the caller need to gauranttees that global_model
 *   is the same in all nodes
 * \sa LoadCheckPoint, CheckPoint, VersionNumber
 */
inline void LazyCheckPoint(const ISerializable *global_model);
/*!
 * \return version number of current stored model,
 *         which means how many calls to CheckPoint we made so far
 * \sa LoadCheckPoint, CheckPoint
 */
inline int VersionNumber(void);
// ----- extensions that allow customized reducer ------
// helper class to do customized reduce, user do not need to know the type
namespace engine {
class ReduceHandle;
}  // namespace engine
/*!
 * \brief template class to make customized reduce and all reduce easy  
 *  Do not use reducer directly in the function you call Finalize, because the destructor can happen after Finalize
 * \tparam DType data type that to be reduced
 *  DType must be a struct, with no pointer, and contains a function Reduce(const DType &d);
 */
template<typename DType>
class Reducer {
 public:
  Reducer(void);
  /*!
   * \brief customized in-place all reduce operation 
   * \param sendrecvbuf the in place send-recv buffer
   * \param count number of elements to be reduced
   * \param prepare_fun Lazy preprocessing function, if it is not NULL, prepare_fun(prepare_arg)
   *                     will be called by the function before performing Allreduce, to intialize the data in sendrecvbuf_.
   *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
   * \param prepare_arg argument used to passed into the lazy preprocessing function 
   */
  inline void Allreduce(DType *sendrecvbuf, size_t count,
                        void (*prepare_fun)(void *arg) = NULL,
                        void *prepare_arg = NULL);
#if __cplusplus >= 201103L
  /*!
   * \brief customized in-place all reduce operation, with lambda function as preprocessor
   * \param sendrecvbuf pointer to the array of objects to be reduced
   * \param count number of elements to be reduced
   * \param prepare_fun lambda function executed to prepare the data, if necessary
   */  
  inline void Allreduce(DType *sendrecvbuf, size_t count,
                        std::function<void()> prepare_fun);
#endif

 private:
  /*! \brief function handle to do reduce */
  engine::ReduceHandle handle_;
};
/*!
 * \brief template class to make customized reduce,
 *  this class defines complex reducer handles all the data structure that can be
 *  serialized/deserialzed into fixed size buffer
 *  Do not use reducer directly in the function you call Finalize, because the destructor can happen after Finalize
 * 
 * \tparam DType data type that to be reduced, DType must contain following functions:
 *   (1) Save(IStream &fs)  (2) Load(IStream &fs) (3) Reduce(const DType &d);
 */
template<typename DType>
class SerializeReducer {
 public:
  SerializeReducer(void);
  /*!
   * \brief customized in-place all reduce operation 
   * \param sendrecvobj pointer to the array of objects to be reduced
   * \param max_nbyte maximum amount of memory needed to serialize each object
   *        this includes budget limit for intermediate and final result
   * \param count number of elements to be reduced
   * \param prepare_fun Lazy preprocessing function, if it is not NULL, prepare_fun(prepare_arg)
   *                     will be called by the function before performing Allreduce, to intialize the data in sendrecvbuf_.
   *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
   * \param prepare_arg argument used to passed into the lazy preprocessing function 
   */
  inline void Allreduce(DType *sendrecvobj,
                        size_t max_nbyte, size_t count,
                        void (*prepare_fun)(void *arg) = NULL,
                        void *prepare_arg = NULL);
// C++11 support for lambda prepare function
#if __cplusplus >= 201103L
  /*!
   * \brief customized in-place all reduce operation, with lambda function as preprocessor
   * \param sendrecvobj pointer to the array of objects to be reduced
   * \param max_nbyte maximum amount of memory needed to serialize each object
   *        this includes budget limit for intermediate and final result
   * \param count number of elements to be reduced
   * \param prepare_fun lambda function executed to prepare the data, if necessary
   */  
  inline void Allreduce(DType *sendrecvobj,
                        size_t max_nbyte, size_t count,
                        std::function<void()> prepare_fun);
#endif

 private:
  /*! \brief function handle to do reduce */
  engine::ReduceHandle handle_;
  /*! \brief temporal buffer used to do reduce*/
  std::string buffer_;
};
}  // namespace rabit
// implementation of template functions
#include "./rabit/rabit-inl.h"
#endif  // RABIT_RABIT_H_
