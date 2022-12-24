/*!
 * Copyright (c) 2014-2019 by Contributors
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
#include "rabit/internal/io.h"
#include "rabit/internal/utils.h"
#include "rabit/rabit.h"

namespace rabit {
namespace engine {
namespace mpi {
// template function to translate type to enum indicator
template<typename DType>
inline DataType GetType();
template<>
inline DataType GetType<char>() {
  return kChar;
}
template<>
inline DataType GetType<unsigned char>() {
  return kUChar;
}
template<>
inline DataType GetType<int>() {
  return kInt;
}
template<>
inline DataType GetType<unsigned int>() { // NOLINT(*)
  return kUInt;
}
template<>
inline DataType GetType<long>() {  // NOLINT(*)
  return kLong;
}
template<>
inline DataType GetType<unsigned long>() { // NOLINT(*)
  return kULong;
}
template<>
inline DataType GetType<float>() {
  return kFloat;
}
template<>
inline DataType GetType<double>() {
  return kDouble;
}
template<>
inline DataType GetType<long long>() { // NOLINT(*)
  return kLongLong;
}
template<>
inline DataType GetType<unsigned long long>() { // NOLINT(*)
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
struct BitAND {
  static const engine::mpi::OpType kType = engine::mpi::kBitwiseAND;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) { // NOLINT(*)
    dst &= src;
  }
};
struct BitOR {
  static const engine::mpi::OpType kType = engine::mpi::kBitwiseOR;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) { // NOLINT(*)
    dst |= src;
  }
};
struct BitXOR {
  static const engine::mpi::OpType kType = engine::mpi::kBitwiseXOR;
  template<typename DType>
  inline static void Reduce(DType &dst, const DType &src) { // NOLINT(*)
    dst ^= src;
  }
};
template <typename OP, typename DType>
inline void Reducer(const void *src_, void *dst_, int len, const MPI::Datatype &) {
  const DType *src = static_cast<const DType *>(src_);
  DType *dst = (DType *)dst_;  // NOLINT(*)
  for (int i = 0; i < len; i++) {
    OP::Reduce(dst[i], src[i]);
  }
}
}  // namespace op

// initialize the rabit engine
inline bool Init(int argc, char *argv[]) {
  return engine::Init(argc, argv);
}
// finalize the rabit engine
inline bool Finalize() {
  return engine::Finalize();
}
// get the rank of the previous worker in ring topology
inline int GetRingPrevRank() {
  return engine::GetEngine()->GetRingPrevRank();
}
// get the rank of current process
inline int GetRank() {
  return engine::GetEngine()->GetRank();
}
// the the size of the world
inline int GetWorldSize() {
  return engine::GetEngine()->GetWorldSize();
}
// whether rabit is distributed
inline bool IsDistributed() {
  return engine::GetEngine()->IsDistributed();
}
// get the name of current processor
inline std::string GetProcessorName() {
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
inline void InvokeLambda(void *fun) {
  (*static_cast<std::function<void()>*>(fun))();
}
template<typename OP, typename DType>
inline void Allreduce(DType *sendrecvbuf, size_t count,
                      std::function<void()> prepare_fun) {
  engine::Allreduce_(sendrecvbuf, sizeof(DType), count, op::Reducer<OP, DType>,
                     engine::mpi::GetType<DType>(), OP::kType, InvokeLambda, &prepare_fun);
}

// Performs inplace Allgather
template<typename DType>
inline void Allgather(DType *sendrecvbuf,
                      size_t totalSize,
                      size_t beginIndex,
                      size_t sizeNodeSlice,
                      size_t sizePrevSlice) {
  engine::GetEngine()->Allgather(sendrecvbuf, totalSize * sizeof(DType), beginIndex * sizeof(DType),
                        (beginIndex + sizeNodeSlice) * sizeof(DType),
                        sizePrevSlice * sizeof(DType));
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

#endif  // RABIT_STRICT_CXX98_

// deprecated, planned for removal after checkpoing from JVM package is removed.
inline int LoadCheckPoint() { return engine::GetEngine()->LoadCheckPoint(); }
// deprecated, increase internal version number
inline void CheckPoint() { engine::GetEngine()->CheckPoint(); }
// return the version number of currently stored model
inline int VersionNumber() {
  return engine::GetEngine()->VersionNumber();
}
}  // namespace rabit
#endif  // RABIT_INTERNAL_RABIT_INL_H_
