#ifndef RABIT_RABIT_H
#define RABIT_RABIT_H
/*!
 * \file rabit.h
 * \brief This file defines a template wrapper of engine to give more flexible
 *      AllReduce operations
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#include "./engine.h"

/*! \brief namespace of rabit */
namespace rabit {
/*! \brief namespace of operator */
namespace op {
struct Max {
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) {
    if (dst < src) dst = src;
  }
};
struct Sum {
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) {
    dst += src;
  }
};
struct BitOR {
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) {
    dst |= src;
  }
};
template<typename OP, typename DType>
inline void Reducer(const void *src_, void *dst_, int len, const MPI::Datatype &dtype) {
  const DType *src = (const DType*)src_;
  DType *dst = (DType*)dst_;  
  for (int i = 0; i < len; ++i) {
    OP::Reduce(dst[i], src[i]);
  }
}
}  // namespace op

void Init(int argc, char *argv[]) {
  engine::Init(argc, argv);
}
void Finalize(void) {
  engine::Finalize();
}

/*! \brief get rank of current process */
inline int GetRank(void) {
  return engine::GetEngine()->GetRank();
}
/*! \brief get total number of process */
int GetWorldSize(void) {
  return engine::GetEngine()->GetWorldSize();
}
/*! \brief get name of processor */
std::string GetProcessorName(void) {
  return engine::GetEngine()->GetHost();
}
/*!
 * \brief broadcast an std::string to all others from root
 * \param sendrecv_data the pointer to send or recive buffer,
 *                      receive buffer does not need to be pre-allocated
 *                      and string will be resized to correct length
 * \param root the root of process
 */
inline void Bcast(std::string *sendrecv_data, int root) {
  engine::IEngine *e = engine::GetEngine();
  unsigned len = static_cast<unsigned>(sendrecv_data->length());
  e->Broadcast(&len, sizeof(len), root);
  sendrecv_data->resize(len);
  if (len != 0) {
    e->Broadcast(&(*sendrecv_data)[0], len, root);  
  }
}
/*!
 * \brief perform in-place allreduce, on sendrecvbuf 
 *        this function is NOT thread-safe
 * Example Usage: the following code gives sum of the result
 *     vector<int> data(10);
 *     ...
 *     AllReduce<op::Sum>(&data[0], data.size());
 *     ...
 * \param sendrecvbuf buffer for both sending and recving data
 * \param count number of elements to be reduced
 * \tparam OP see namespace op, reduce operator 
 * \tparam DType type of data
 */
template<typename OP, typename DType>
inline void AllReduce(DType *sendrecvbuf, size_t count) {
  engine::GetEngine()->AllReduce(sendrecvbuf, sizeof(DType), count, op::Reducer<OP,DType>);
}
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
inline int LoadCheckPoint(utils::ISerializable *p_model) {
  return engine::GetEngine()->LoadCheckPoint(p_model);
}
/*!
 * \brief checkpoint the model, meaning we finished a stage of execution
 *  every time we call check point, there is a version number which will increase by one
 * 
 * \param p_model pointer to the model
 * \sa LoadCheckPoint, VersionNumber
 */
inline void CheckPoint(const utils::ISerializable &model) {
  engine::GetEngine()->CheckPoint(model);
}
/*!
 * \return version number of current stored model,
 *         which means how many calls to CheckPoint we made so far
 * \sa LoadCheckPoint, CheckPoint
 */
inline int VersionNumber(void) {
  return engine::GetEngine()->VersionNumber();
}
}  // namespace rabit
#endif  // RABIT_ALLREDUCE_H
