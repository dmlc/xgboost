/*!
 * Copyright by Contributors
 * \file rabit-inl.h
 * \brief implementation of inline template function for rabit interface
 *
 * \author Tianqi Chen
 */
#ifndef RABIT_INTERNAL_RABIT_INL_H_
#define RABIT_INTERNAL_RABIT_INL_H_
// use engine for implementation
#include <vector>
#include <string>
#include "./io.h"
#include "./utils.h"
#include "../rabit.h"

namespace rabit {
namespace engine {
namespace mpi {
// template function to translate type to enum indicator
template<typename DType>
inline DataType GetType(void);
template<>
inline DataType GetType<char>(void) {
  return kChar;
}
template<>
inline DataType GetType<unsigned char>(void) {
  return kUChar;
}
template<>
inline DataType GetType<int>(void) {
  return kInt;
}
template<>
inline DataType GetType<unsigned int>(void) { // NOLINT(*)
  return kUInt;
}
template<>
inline DataType GetType<long>(void) {  // NOLINT(*)
  return kLong;
}
template<>
inline DataType GetType<unsigned long>(void) { // NOLINT(*)
  return kULong;
}
template<>
inline DataType GetType<float>(void) {
  return kFloat;
}
template<>
inline DataType GetType<double>(void) {
  return kDouble;
}
template<>
inline DataType GetType<long long>(void) { // NOLINT(*)
  return kLongLong;
}
template<>
inline DataType GetType<unsigned long long>(void) { // NOLINT(*)
  return kULongLong;
}
}  // namespace mpi
}  // namespace engine

namespace op {
struct Max {
  static const engine::mpi::OpType kType = engine::mpi::kMax;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) { // NOLINT(*)
    if (dst < src) dst = src;
  }
};
struct Min {
  static const engine::mpi::OpType kType = engine::mpi::kMin;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) { // NOLINT(*)
    if (dst > src) dst = src;
  }
};
struct Sum {
  static const engine::mpi::OpType kType = engine::mpi::kSum;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) { // NOLINT(*)
    dst += src;
  }
};
struct BitOR {
  static const engine::mpi::OpType kType = engine::mpi::kBitwiseOR;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) { // NOLINT(*)
    dst |= src;
  }
};
template<typename OP, typename DType>
inline void Reducer(const void *src_, void *dst_, int len, const MPI::Datatype &dtype) {
  const DType *src = (const DType*)src_;
  DType *dst = (DType*)dst_;  // NOLINT(*)
  for (int i = 0; i < len; ++i) {
    OP::Reduce(dst[i], src[i]);
  }
}
}  // namespace op

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
// whether rabit is distributed
inline bool IsDistributed(void) {
  return engine::GetEngine()->IsDistributed();
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
  engine::Allreduce_(sendrecvbuf, sizeof(DType), count, op::Reducer<OP, DType>,
                     engine::mpi::GetType<DType>(), OP::kType, prepare_fun, prepare_arg);
}

// C++11 support for lambda prepare function
#if DMLC_USE_CXX11
inline void InvokeLambda_(void *fun) {
  (*static_cast<std::function<void()>*>(fun))();
}
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count, std::function<void()> prepare_fun) {
  engine::Allreduce_(sendrecvbuf, sizeof(DType), count, op::Reducer<OP, DType>,
                     engine::mpi::GetType<DType>(), OP::kType, InvokeLambda_, &prepare_fun);
}
#endif  // C++11

// print message to the tracker
inline void TrackerPrint(const std::string &msg) {
  engine::GetEngine()->TrackerPrint(msg);
}
#ifndef RABIT_STRICT_CXX98_
inline void TrackerPrintf(const char *fmt, ...) {
  const int kPrintBuffer = 1 << 10;
  std::string msg(kPrintBuffer, '\0');
  va_list args;
  va_start(args, fmt);
  vsnprintf(&msg[0], kPrintBuffer, fmt, args);
  va_end(args);
  msg.resize(strlen(msg.c_str()));
  TrackerPrint(msg);
}
#endif
// load latest check point
inline int LoadCheckPoint(Serializable *global_model,
                          Serializable *local_model) {
  return engine::GetEngine()->LoadCheckPoint(global_model, local_model);
}
// checkpoint the model, meaning we finished a stage of execution
inline void CheckPoint(const Serializable *global_model,
                       const Serializable *local_model) {
  engine::GetEngine()->CheckPoint(global_model, local_model);
}
// lazy checkpoint the model, only remember the pointer to global_model
inline void LazyCheckPoint(const Serializable *global_model) {
  engine::GetEngine()->LazyCheckPoint(global_model);
}
// return the version number of currently stored model
inline int VersionNumber(void) {
  return engine::GetEngine()->VersionNumber();
}
// ---------------------------------
// Code to handle customized Reduce
// ---------------------------------
// function to perform reduction for Reducer
template<typename DType, void (*freduce)(DType &dst, const DType &src)>
inline void ReducerSafe_(const void *src_, void *dst_, int len_, const MPI::Datatype &dtype) {
  const size_t kUnit = sizeof(DType);
  const char *psrc = reinterpret_cast<const char*>(src_);
  char *pdst = reinterpret_cast<char*>(dst_);
  DType tdst, tsrc;
  for (int i = 0; i < len_; ++i) {
    // use memcpy to avoid alignment issue
    std::memcpy(&tdst, pdst + i * kUnit, sizeof(tdst));
    std::memcpy(&tsrc, psrc + i * kUnit, sizeof(tsrc));
    freduce(tdst, tsrc);
    std::memcpy(pdst + i * kUnit, &tdst, sizeof(tdst));
  }
}
// function to perform reduction for Reducer
template<typename DType, void (*freduce)(DType &dst, const DType &src)> // NOLINT(*)
inline void ReducerAlign_(const void *src_, void *dst_,
                          int len_, const MPI::Datatype &dtype) {
  const DType *psrc = reinterpret_cast<const DType*>(src_);
  DType *pdst = reinterpret_cast<DType*>(dst_);
  for (int i = 0; i < len_; ++i) {
    freduce(pdst[i], psrc[i]);
  }
}
template<typename DType, void (*freduce)(DType &dst, const DType &src)>  // NOLINT(*)
inline Reducer<DType, freduce>::Reducer(void) {
  // it is safe to directly use handle for aligned data types
  if (sizeof(DType) == 8 || sizeof(DType) == 4 || sizeof(DType) == 1) {
    this->handle_.Init(ReducerAlign_<DType, freduce>, sizeof(DType));
  } else {
    this->handle_.Init(ReducerSafe_<DType, freduce>, sizeof(DType));
  }
}
template<typename DType, void (*freduce)(DType &dst, const DType &src)> // NOLINT(*)
inline void Reducer<DType, freduce>::Allreduce(DType *sendrecvbuf, size_t count,
                                               void (*prepare_fun)(void *arg),
                                               void *prepare_arg) {
  handle_.Allreduce(sendrecvbuf, sizeof(DType), count, prepare_fun, prepare_arg);
}
// function to perform reduction for SerializeReducer
template<typename DType>
inline void SerializeReducerFunc_(const void *src_, void *dst_,
                                  int len_, const MPI::Datatype &dtype) {
  int nbytes = engine::ReduceHandle::TypeSize(dtype);
  // temp space
  DType tsrc, tdst;
  for (int i = 0; i < len_; ++i) {
    utils::MemoryFixSizeBuffer fsrc((char*)(src_) + i * nbytes, nbytes); // NOLINT(*)
    utils::MemoryFixSizeBuffer fdst((char*)(dst_) + i * nbytes, nbytes); // NOLINT(*)
    tsrc.Load(fsrc);
    tdst.Load(fdst);
    // govern const check
    tdst.Reduce(static_cast<const DType &>(tsrc), nbytes);
    fdst.Seek(0);
    tdst.Save(fdst);
  }
}
template<typename DType>
inline SerializeReducer<DType>::SerializeReducer(void) {
  handle_.Init(SerializeReducerFunc_<DType>, sizeof(DType));
}
// closure to call Allreduce
template<typename DType>
struct SerializeReduceClosure {
  DType *sendrecvobj;
  size_t max_nbyte, count;
  void (*prepare_fun)(void *arg);
  void *prepare_arg;
  std::string *p_buffer;
  // invoke the closure
  inline void Run(void) {
    if (prepare_fun != NULL) prepare_fun(prepare_arg);
    for (size_t i = 0; i < count; ++i) {
      utils::MemoryFixSizeBuffer fs(BeginPtr(*p_buffer) + i * max_nbyte, max_nbyte);
      sendrecvobj[i].Save(fs);
    }
  }
  inline static void Invoke(void *c) {
    static_cast<SerializeReduceClosure<DType>*>(c)->Run();
  }
};
template<typename DType>
inline void SerializeReducer<DType>::Allreduce(DType *sendrecvobj,
                                               size_t max_nbyte, size_t count,
                                               void (*prepare_fun)(void *arg),
                                               void *prepare_arg) {
  buffer_.resize(max_nbyte * count);
  // setup closure
  SerializeReduceClosure<DType> c;
  c.sendrecvobj = sendrecvobj; c.max_nbyte = max_nbyte; c.count = count;
  c.prepare_fun = prepare_fun; c.prepare_arg = prepare_arg; c.p_buffer = &buffer_;
  // invoke here
  handle_.Allreduce(BeginPtr(buffer_), max_nbyte, count,
                    SerializeReduceClosure<DType>::Invoke, &c);
  for (size_t i = 0; i < count; ++i) {
    utils::MemoryFixSizeBuffer fs(BeginPtr(buffer_) + i * max_nbyte, max_nbyte);
    sendrecvobj[i].Load(fs);
  }
}

#if DMLC_USE_CXX11
template<typename DType, void (*freduce)(DType &dst, const DType &src)>  // NOLINT(*)g
inline void Reducer<DType, freduce>::Allreduce(DType *sendrecvbuf, size_t count,
                                               std::function<void()> prepare_fun) {
  this->Allreduce(sendrecvbuf, count, InvokeLambda_, &prepare_fun);
}
template<typename DType>
inline void SerializeReducer<DType>::Allreduce(DType *sendrecvobj,
                                               size_t max_nbytes, size_t count,
                                               std::function<void()> prepare_fun) {
  this->Allreduce(sendrecvobj, max_nbytes, count, InvokeLambda_, &prepare_fun);
}
#endif
}  // namespace rabit
#endif  // RABIT_INTERNAL_RABIT_INL_H_
