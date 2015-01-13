// implementations in ctypes
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <rabit.h>
#include <cstring>
#include <string>
#include "./rabit_wrapper.h"
namespace rabit {
namespace wrapper {
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
          (static_cast<long*>(sendrecvbuf_),
           count, prepare_fun, prepare_arg);
      return;
    case kULong:
      rabit::Allreduce<OP>
          (static_cast<unsigned long*>(sendrecvbuf_),
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
}  // namespace wrapper
}  // namespace rabit
extern "C" {
  void RabitInit(int argc, char *argv[]) {
    rabit::Init(argc, argv);
  }
  void RabitFinalize(void) {
    rabit::Finalize();
  }
  int RabitGetRank(void) {
    return rabit::GetRank();
  }
  int RabitGetWorldSize(void) {
    return rabit::GetWorldSize();
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
    strcpy(out_name, s.c_str());
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
    rabit::wrapper::Allreduce
        (sendrecvbuf, count,
         static_cast<rabit::engine::mpi::DataType>(enum_dtype),
         static_cast<rabit::engine::mpi::OpType>(enum_op),
         prepare_fun, prepare_arg);
  }
}
