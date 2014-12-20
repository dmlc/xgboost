/*!
 * \file rabit-inl.h
 * \brief implementation of inline template function for rabit interface
 *
 * \author Tianqi Chen
 */
#ifndef RABIT_RABIT_INL_H
#define RABIT_RABIT_INL_H
// use engine for implementation
#include "./engine.h"

namespace rabit {
namespace engine {
namespace mpi {
// template function to translate type to enum indicator
template<typename DType>
inline DataType GetType(void);
template<>
inline DataType GetType<int>(void) {
  return kInt;
}
template<>
inline DataType GetType<unsigned>(void) {
  return kUInt;
}
template<>
inline DataType GetType<float>(void) {
  return kFloat;
}
template<>
inline DataType GetType<double>(void) {
  return kDouble;
}
}  // namespace mpi
}  // namespace engine

namespace op {
struct Max {
  const static engine::mpi::OpType kType = engine::mpi::kMax;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) {
    if (dst < src) dst = src;
  }
};
struct Min {
  const static engine::mpi::OpType kType = engine::mpi::kMin;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) {
    if (dst > src) dst = src;
  }
};
struct Sum {
  const static engine::mpi::OpType kType = engine::mpi::kSum;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) {
    dst += src;
  }
};
struct BitOR {
  const static engine::mpi::OpType kType = engine::mpi::kBitwiseOR;
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
} // namespace op

// intialize the rabit engine
inline void Init(int argc, char *argv[]) {
  engine::Init(argc, argv);
}
// finalize the rabit engine
inline void Finalize(void) {
  engine::Finalize();
}
// get the rank of current process
inline int GetRank(void) {
  return engine::GetEngine()->GetRank();
}
// the the size of the world
inline int GetWorldSize(void) {
  return engine::GetEngine()->GetWorldSize();
}
// get the name of current processor
inline std::string GetProcessorName(void) {
  return engine::GetEngine()->GetHost();
}
// broadcast data to all other nodes from root
inline void Broadcast(void *sendrecv_data, size_t size, int root) {
  engine::GetEngine()->Broadcast(sendrecv_data, size, root);
}
template<typename DType>
inline void Broadcast(std::vector<DType> *sendrecv_data, int root) {
  size_t size = sendrecv_data->size();
  Broadcast(&size, sizeof(size), root);
  if (sendrecv_data->size() != size) {
    sendrecv_data->resize(size);
  }
  if (size != 0) {
    Broadcast(&(*sendrecv_data)[0], size * sizeof(DType), root);
  }
}
inline void Broadcast(std::string *sendrecv_data, int root) {
  size_t size = sendrecv_data->length();
  Broadcast(&size, sizeof(size), root);
  if (sendrecv_data->length() != size) {
    sendrecv_data->resize(size);
  }
  if (size != 0) {
    Broadcast(&(*sendrecv_data)[0], size * sizeof(char), root);
  }
}

// perform inplace Allreduce
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count,
                      void (*prepare_fun)(void *arg), 
                      void *prepare_arg) {
  engine::Allreduce_(sendrecvbuf, sizeof(DType), count, op::Reducer<OP,DType>,
                     engine::mpi::GetType<DType>(), OP::kType, prepare_fun, prepare_arg);
}

// C++11 support for lambda prepare function
#if __cplusplus >= 201103L
inline void InvokeLambda_(void *fun) {
  (*static_cast<std::function<void()>*>(fun))();
}
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count, std::function<void()> prepare_fun) {
  engine::Allreduce_(sendrecvbuf, sizeof(DType), count, op::Reducer<OP,DType>,
                     engine::mpi::GetType<DType>(), OP::kType, InvokeLambda_, &prepare_fun);
}
#endif // C++11

// load latest check point
inline int LoadCheckPoint(ISerializable *global_model,
                          ISerializable *local_model) {
  return engine::GetEngine()->LoadCheckPoint(global_model, local_model);
}
// checkpoint the model, meaning we finished a stage of execution
inline void CheckPoint(const ISerializable *global_model,
                       const ISerializable *local_model) {
  engine::GetEngine()->CheckPoint(global_model, local_model);
}
// return the version number of currently stored model
inline int VersionNumber(void) {
  return engine::GetEngine()->VersionNumber();
}
}  // namespace rabit
#endif
