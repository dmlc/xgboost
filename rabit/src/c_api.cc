// Copyright by Contributors
// implementations in ctypes
#include <rabit/base.h>
#include <cstring>
#include <string>
#include "rabit/rabit.h"
#include "rabit/c_api.h"

#include "../../src/c_api/c_api_error.h"

namespace rabit {
namespace c_api {
// helper use to avoid BitOR operator
template<typename OP, typename DType>
struct FHelper {
  static void
  Allreduce(DType *senrecvbuf_,
            size_t count,
            void (*prepare_fun)(void *arg),
            void *prepare_arg) {
    rabit::Allreduce<OP>(senrecvbuf_, count,
                         prepare_fun, prepare_arg);
  }
};

template<typename DType>
struct FHelper<op::BitOR, DType> {
  static void
  Allreduce(DType *,
            size_t ,
            void (*)(void *arg),
            void *) {
    utils::Error("DataType does not support bitwise or operation");
  }
};

template<typename OP>
void Allreduce(void *sendrecvbuf_,
                size_t count,
                engine::mpi::DataType enum_dtype,
                void (*prepare_fun)(void *arg),
                void *prepare_arg) {
  using namespace engine::mpi;  // NOLINT
  switch (enum_dtype) {
    case kChar:
      rabit::Allreduce<OP>
          (static_cast<char*>(sendrecvbuf_),
           count, prepare_fun, prepare_arg);
      return;
    case kUChar:
      rabit::Allreduce<OP>
          (static_cast<unsigned char*>(sendrecvbuf_),
           count, prepare_fun, prepare_arg);
      return;
    case kInt:
      rabit::Allreduce<OP>
          (static_cast<int*>(sendrecvbuf_),
           count, prepare_fun, prepare_arg);
      return;
    case kUInt:
      rabit::Allreduce<OP>
          (static_cast<unsigned*>(sendrecvbuf_),
           count, prepare_fun, prepare_arg);
      return;
    case kLong:
      rabit::Allreduce<OP>
          (static_cast<long*>(sendrecvbuf_),  // NOLINT(*)
           count, prepare_fun, prepare_arg);
      return;
    case kULong:
      rabit::Allreduce<OP>
          (static_cast<unsigned long*>(sendrecvbuf_),  // NOLINT(*)
           count, prepare_fun, prepare_arg);
      return;
    case kFloat:
      FHelper<OP, float>::Allreduce
          (static_cast<float*>(sendrecvbuf_),
           count, prepare_fun, prepare_arg);
      return;
    case kDouble:
      FHelper<OP, double>::Allreduce
          (static_cast<double*>(sendrecvbuf_),
           count, prepare_fun, prepare_arg);
      return;
    default: utils::Error("unknown data_type");
  }
}
void Allreduce(void *sendrecvbuf,
               size_t count,
               engine::mpi::DataType enum_dtype,
               engine::mpi::OpType enum_op,
               void (*prepare_fun)(void *arg),
               void *prepare_arg) {
  using namespace engine::mpi;  // NOLINT
  switch (enum_op) {
    case kMax:
      Allreduce<op::Max>
          (sendrecvbuf,
           count, enum_dtype,
           prepare_fun, prepare_arg);
      return;
    case kMin:
      Allreduce<op::Min>
          (sendrecvbuf,
           count, enum_dtype,
           prepare_fun, prepare_arg);
      return;
    case kSum:
      Allreduce<op::Sum>
          (sendrecvbuf,
           count, enum_dtype,
           prepare_fun, prepare_arg);
      return;
    case kBitwiseOR:
      Allreduce<op::BitOR>
          (sendrecvbuf,
           count, enum_dtype,
           prepare_fun, prepare_arg);
      return;
    default: utils::Error("unknown enum_op");
  }
}
void Allgather(void *sendrecvbuf_,
               size_t total_size,
               size_t beginIndex,
               size_t size_node_slice,
               size_t size_prev_slice,
               int enum_dtype) {
  using namespace engine::mpi;  // NOLINT
  size_t type_size = 0;
  switch (enum_dtype) {
  case kChar:
    type_size = sizeof(char);
    rabit::Allgather(static_cast<char*>(sendrecvbuf_), total_size * type_size,
      beginIndex * type_size, (beginIndex + size_node_slice) * type_size,
      size_prev_slice * type_size);
    break;
  case kUChar:
    type_size = sizeof(unsigned char);
    rabit::Allgather(static_cast<unsigned char*>(sendrecvbuf_), total_size * type_size,
      beginIndex * type_size, (beginIndex + size_node_slice) * type_size,
      size_prev_slice * type_size);
    break;
  case kInt:
    type_size = sizeof(int);
    rabit::Allgather(static_cast<int*>(sendrecvbuf_), total_size * type_size,
      beginIndex * type_size, (beginIndex + size_node_slice) * type_size,
      size_prev_slice * type_size);
    break;
  case kUInt:
    type_size = sizeof(unsigned);
    rabit::Allgather(static_cast<unsigned*>(sendrecvbuf_), total_size * type_size,
      beginIndex * type_size, (beginIndex + size_node_slice) * type_size,
      size_prev_slice * type_size);
    break;
  case kLong:
    type_size = sizeof(int64_t);
    rabit::Allgather(static_cast<int64_t*>(sendrecvbuf_), total_size * type_size,
      beginIndex * type_size, (beginIndex + size_node_slice) * type_size,
      size_prev_slice * type_size);
    break;
  case kULong:
    type_size = sizeof(uint64_t);
    rabit::Allgather(static_cast<uint64_t*>(sendrecvbuf_), total_size * type_size,
      beginIndex * type_size, (beginIndex + size_node_slice) * type_size,
      size_prev_slice * type_size);
    break;
  case kFloat:
    type_size = sizeof(float);
    rabit::Allgather(static_cast<float*>(sendrecvbuf_), total_size * type_size,
      beginIndex * type_size, (beginIndex + size_node_slice) * type_size,
      size_prev_slice * type_size);
    break;
  case kDouble:
    type_size = sizeof(double);
    rabit::Allgather(static_cast<double*>(sendrecvbuf_), total_size * type_size,
      beginIndex * type_size, (beginIndex + size_node_slice) * type_size,
      size_prev_slice * type_size);
    break;
  default: utils::Error("unknown data_type");
  }
}

// wrapper for serialization
struct ReadWrapper : public Serializable {
  std::string *p_str;
  explicit ReadWrapper(std::string *p_str)
      : p_str(p_str) {}
  void Load(Stream *fi) override {
    uint64_t sz;
    utils::Assert(fi->Read(&sz, sizeof(sz)) != 0,
                 "Read pickle string");
    p_str->resize(sz);
    if (sz != 0) {
      utils::Assert(fi->Read(&(*p_str)[0], sizeof(char) * sz) != 0,
                    "Read pickle string");
    }
  }
  void Save(Stream *) const override {
    utils::Error("not implemented");
  }
};

struct WriteWrapper : public Serializable {
  const char *data;
  size_t length;
  explicit WriteWrapper(const char *data,
                        size_t length)
      : data(data), length(length) {
  }
  void Load(Stream *) override {
    utils::Error("not implemented");
  }
  void Save(Stream *fo) const override {
    uint64_t sz = static_cast<uint16_t>(length);
    fo->Write(&sz, sizeof(sz));
    fo->Write(data, length * sizeof(char));
  }
};
}  // namespace c_api
}  // namespace rabit

RABIT_DLL bool RabitInit(int argc, char *argv[]) {
  auto ret = rabit::Init(argc, argv);
  if (!ret) {
    XGBAPISetLastError("Failed to initialize RABIT.");
  }
  return ret;
}

RABIT_DLL int RabitFinalize() {
  auto ret = rabit::Finalize();
  if (!ret) {
    XGBAPISetLastError("Failed to shutdown RABIT worker.");
  }
  return static_cast<int>(ret);
}

RABIT_DLL int RabitGetRingPrevRank() {
  return rabit::GetRingPrevRank();
}

RABIT_DLL int RabitGetRank() {
  return rabit::GetRank();
}

RABIT_DLL int RabitGetWorldSize() {
  return rabit::GetWorldSize();
}

RABIT_DLL int RabitIsDistributed() {
  return rabit::IsDistributed();
}

RABIT_DLL int RabitTrackerPrint(const char *msg) {
  API_BEGIN()
  std::string m(msg);
  rabit::TrackerPrint(m);
  API_END()
}

RABIT_DLL void RabitGetProcessorName(char *out_name,
                                     rbt_ulong *out_len,
                                     rbt_ulong max_len) {
  std::string s = rabit::GetProcessorName();
  if (s.length() > max_len) {
    s.resize(max_len - 1);
  }
  strcpy(out_name, s.c_str()); // NOLINT(*)
  *out_len = static_cast<rbt_ulong>(s.length());
}

RABIT_DLL int RabitBroadcast(void *sendrecv_data,
                              rbt_ulong size, int root) {
  API_BEGIN()
  rabit::Broadcast(sendrecv_data, size, root);
  API_END()
}

RABIT_DLL int RabitAllgather(void *sendrecvbuf_, size_t total_size,
                              size_t beginIndex, size_t size_node_slice,
                              size_t size_prev_slice, int enum_dtype) {
  API_BEGIN()
  rabit::c_api::Allgather(
      sendrecvbuf_, total_size, beginIndex, size_node_slice, size_prev_slice,
      static_cast<rabit::engine::mpi::DataType>(enum_dtype));
  API_END()
}

RABIT_DLL int RabitAllreduce(void *sendrecvbuf, size_t count, int enum_dtype,
                              int enum_op, void (*prepare_fun)(void *arg),
                              void *prepare_arg) {
  API_BEGIN()
  rabit::c_api::Allreduce(sendrecvbuf, count,
                          static_cast<rabit::engine::mpi::DataType>(enum_dtype),
                          static_cast<rabit::engine::mpi::OpType>(enum_op),
                          prepare_fun, prepare_arg);
  API_END()
}

RABIT_DLL int RabitLoadCheckPoint(char **out_global_model,
                                  rbt_ulong *out_global_len,
                                  char **out_local_model,
                                  rbt_ulong *out_local_len) {
  // no-op as XGBoost 1.3
  using rabit::BeginPtr;
  using namespace rabit::c_api; // NOLINT(*)
  static std::string global_buffer;
  static std::string local_buffer;

  ReadWrapper sg(&global_buffer);
  ReadWrapper sl(&local_buffer);
  int version;

  if (out_local_model == nullptr) {
    version = rabit::LoadCheckPoint(&sg, nullptr);
    *out_global_model = BeginPtr(global_buffer);
    *out_global_len = static_cast<rbt_ulong>(global_buffer.length());
  } else {
    version = rabit::LoadCheckPoint(&sg, &sl);
    *out_global_model = BeginPtr(global_buffer);
    *out_global_len = static_cast<rbt_ulong>(global_buffer.length());
    *out_local_model = BeginPtr(local_buffer);
    *out_local_len = static_cast<rbt_ulong>(local_buffer.length());
  }
  return version;
}

RABIT_DLL void RabitCheckPoint(const char *global_model, rbt_ulong global_len,
                               const char *local_model, rbt_ulong local_len) {
  using namespace rabit::c_api; // NOLINT(*)
  WriteWrapper sg(global_model, global_len);
  WriteWrapper sl(local_model, local_len);
  if (local_model == nullptr) {
    rabit::CheckPoint(&sg, nullptr);
  } else {
    rabit::CheckPoint(&sg, &sl);
  }
}

RABIT_DLL int RabitVersionNumber() {
  return rabit::VersionNumber();
}

RABIT_DLL int RabitLinkTag() {
  return 0;
}
