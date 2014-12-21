/*!
 * \file engine.cc
 * \brief this file governs which implementation of engine we are actually using
 *  provides an singleton of engine interface
 *   
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX

#include <rabit/engine.h>
#include "./allreduce_base.h"
#include "./allreduce_robust.h"

namespace rabit {
namespace engine {
// singleton sync manager
AllreduceRobust manager;

/*! \brief intiialize the synchronization module */
void Init(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      manager.SetParam(name, val);
    }
  }
  manager.Init();
}

/*! \brief finalize syncrhonization module */
void Finalize(void) {
  manager.Shutdown();
}
/*! \brief singleton method to get engine */
IEngine *GetEngine(void) {
  return &manager;
}
// perform in-place allreduce, on sendrecvbuf 
void Allreduce_(void *sendrecvbuf,
                size_t type_nbytes,
                size_t count,
                IEngine::ReduceFunction red,               
                mpi::DataType dtype,
                mpi::OpType op,
                IEngine::PreprocFunction prepare_fun,
                void *prepare_arg) {
  GetEngine()->Allreduce(sendrecvbuf, type_nbytes, count, red, prepare_fun, prepare_arg);
}

// code for reduce handle
ReduceHandle::ReduceHandle(void) : handle_(NULL), htype_(NULL) {
}
ReduceHandle::~ReduceHandle(void) {}

int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return static_cast<int>(dtype.type_size);
}
void ReduceHandle::Init(IEngine::ReduceFunction redfunc, size_t type_nbytes) {
  utils::Assert(handle_ == NULL, "cannot initialize reduce handle twice");
  handle_ = reinterpret_cast<void*>(redfunc);
}
void ReduceHandle::Allreduce(void *sendrecvbuf,
                             size_t type_nbytes, size_t count,
                             IEngine::PreprocFunction prepare_fun,
                             void *prepare_arg) {
  utils::Assert(handle_ != NULL, "must intialize handle to call AllReduce");
  GetEngine()->Allreduce(sendrecvbuf, type_nbytes, count,
                         reinterpret_cast<IEngine::ReduceFunction*>(handle_),
                         prepare_fun, prepare_arg);
}
}  // namespace engine
}  // namespace rabit
