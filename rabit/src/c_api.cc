// Copyright by Contributors
// implementations in ctypes
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <cstring>
#include <string>
#include "../include/rabit/rabit.h"
#include "../include/rabit/c_api.h"

namespace rabit {
namespace c_api {
// helper use to avoid BitOR operator
template<typename OP, typename DType>
struct FHelper {
  inline static void
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
  inline static void
  Allreduce(DType *senrecvbuf_,
            size_t count,
            void (*prepare_fun)(void *arg),
            void *prepare_arg) {
    utils::Error("DataType does not support bitwise or operation");
  }
};

template<typename OP>
inline void Allreduce_(void *sendrecvbuf_,
                       size_t count,
                       engine::mpi::DataType enum_dtype,
                       void (*prepare_fun)(void *arg),
                       void *prepare_arg) {
  using namespace engine::mpi;
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
inline void Allreduce(void *sendrecvbuf,
                      size_t count,
                      engine::mpi::DataType enum_dtype,
                      engine::mpi::OpType enum_op,
                      void (*prepare_fun)(void *arg),
                      void *prepare_arg) {
  using namespace engine::mpi;
  switch (enum_op) {
    case kMax:
      Allreduce_<op::Max>
          (sendrecvbuf,
           count, enum_dtype,
           prepare_fun, prepare_arg);
      return;
    case kMin:
      Allreduce_<op::Min>
          (sendrecvbuf,
           count, enum_dtype,
           prepare_fun, prepare_arg);
      return;
    case kSum:
      Allreduce_<op::Sum>
          (sendrecvbuf,
           count, enum_dtype,
           prepare_fun, prepare_arg);
      return;
    case kBitwiseOR:
      Allreduce_<op::BitOR>
          (sendrecvbuf,
           count, enum_dtype,
           prepare_fun, prepare_arg);
      return;
    default: utils::Error("unknown enum_op");
  }
}



// wrapper for serialization
struct ReadWrapper : public Serializable {
  std::string *p_str;
  explicit ReadWrapper(std::string *p_str)
      : p_str(p_str) {}
  virtual void Load(Stream *fi) {
    uint64_t sz;
    utils::Assert(fi->Read(&sz, sizeof(sz)) != 0,
                 "Read pickle string");
    p_str->resize(sz);
    if (sz != 0) {
      utils::Assert(fi->Read(&(*p_str)[0], sizeof(char) * sz) != 0,
                    "Read pickle string");
    }
  }
  virtual void Save(Stream *fo) const {
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
  virtual void Load(Stream *fi) {
    utils::Error("not implemented");
  }
  virtual void Save(Stream *fo) const {
    uint64_t sz = static_cast<uint16_t>(length);
    fo->Write(&sz, sizeof(sz));
    fo->Write(data, length * sizeof(char));
  }
};
}  // namespace c_api
}  // namespace rabit

void RabitInit(int argc, char *argv[]) {
  rabit::Init(argc, argv);
}

void RabitFinalize() {
  rabit::Finalize();
}

int RabitGetRank() {
  return rabit::GetRank();
}

int RabitGetWorldSize() {
  return rabit::GetWorldSize();
}

int RabitIsDistributed() {
  return rabit::IsDistributed();
}

void RabitTrackerPrint(const char *msg) {
  std::string m(msg);
  rabit::TrackerPrint(m);
}

void RabitGetProcessorName(char *out_name,
                           rbt_ulong *out_len,
                           rbt_ulong max_len) {
  std::string s = rabit::GetProcessorName();
  if (s.length() > max_len) {
    s.resize(max_len - 1);
  }
  strcpy(out_name, s.c_str()); // NOLINT(*)
  *out_len = static_cast<rbt_ulong>(s.length());
}

void RabitBroadcast(void *sendrecv_data,
                    rbt_ulong size, int root) {
  rabit::Broadcast(sendrecv_data, size, root);
}

void RabitAllreduce(void *sendrecvbuf,
                    size_t count,
                    int enum_dtype,
                    int enum_op,
                    void (*prepare_fun)(void *arg),
                    void *prepare_arg) {
  rabit::c_api::Allreduce
      (sendrecvbuf, count,
       static_cast<rabit::engine::mpi::DataType>(enum_dtype),
       static_cast<rabit::engine::mpi::OpType>(enum_op),
       prepare_fun, prepare_arg);
}

int RabitLoadCheckPoint(char **out_global_model,
                        rbt_ulong *out_global_len,
                        char **out_local_model,
                        rbt_ulong *out_local_len) {
  // NOTE: this function is not thread-safe
  using rabit::BeginPtr;
  using namespace rabit::c_api; // NOLINT(*)
  static std::string global_buffer;
  static std::string local_buffer;

  ReadWrapper sg(&global_buffer);
  ReadWrapper sl(&local_buffer);
  int version;

  if (out_local_model == NULL) {
    version = rabit::LoadCheckPoint(&sg, NULL);
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

void RabitCheckPoint(const char *global_model,
                     rbt_ulong global_len,
                     const char *local_model,
                     rbt_ulong local_len) {
  using namespace rabit::c_api; // NOLINT(*)
  WriteWrapper sg(global_model, global_len);
  WriteWrapper sl(local_model, local_len);
  if (local_model == NULL) {
    rabit::CheckPoint(&sg, NULL);
  } else {
    rabit::CheckPoint(&sg, &sl);
  }
}

int RabitVersionNumber() {
  return rabit::VersionNumber();
}

int RabitLinkTag() {
  return 0;
}
