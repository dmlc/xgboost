/*!
 *  Copyright (c) 2014 by Contributors
 * \file engine.cc
 * \brief this file governs which implementation of engine we are actually using
 *  provides an singleton of engine interface
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#include <rabit/base.h>
#include <dmlc/thread_local.h>

#include <memory>
#include "rabit/internal/engine.h"
#include "allreduce_base.h"
#include "allreduce_robust.h"

namespace rabit {
namespace engine {
// singleton sync manager
#ifndef RABIT_USE_BASE
#ifndef RABIT_USE_MOCK
using Manager = AllreduceRobust;
#else
using Manager = AllreduceMock;
#endif  // RABIT_USE_MOCK
#else
using Manager = AllreduceBase;
#endif  // RABIT_USE_BASE

/*! \brief entry to to easily hold returning information */
struct ThreadLocalEntry {
  /*! \brief stores the current engine */
  std::unique_ptr<Manager> engine;
  /*! \brief whether init has been called */
  bool initialized{false};
  /*! \brief constructor */
  ThreadLocalEntry() = default;
};

// define the threadlocal store.
using EngineThreadLocal = dmlc::ThreadLocalStore<ThreadLocalEntry>;

/*! \brief intiialize the synchronization module */
bool Init(int argc, char *argv[]) {
  ThreadLocalEntry* e = EngineThreadLocal::Get();
  if (e->engine.get() == nullptr) {
    e->initialized = true;
    e->engine.reset(new Manager());
    return e->engine->Init(argc, argv);
  } else {
    return true;
  }
}

/*! \brief finalize syncrhonization module */
bool Finalize() {
  ThreadLocalEntry* e = EngineThreadLocal::Get();
  if (e->engine.get() != nullptr) {
    if (e->engine->Shutdown()) {
      e->engine.reset(nullptr);
      e->initialized = false;
      return true;
    } else {
      return false;
    }
  } else {
    return true;
  }
}

/*! \brief singleton method to get engine */
IEngine *GetEngine() {
  // un-initialized default manager.
  static AllreduceBase default_manager;
  ThreadLocalEntry* e = EngineThreadLocal::Get();
  IEngine* ptr = e->engine.get();
  if (ptr == nullptr) {
    utils::Check(!e->initialized, "the rabit has not been initialized");
    return &default_manager;
  } else {
    return ptr;
  }
}

// perform in-place allgather, on sendrecvbuf
void Allgather(void *sendrecvbuf_, size_t total_size,
                   size_t slice_begin,
                   size_t slice_end,
                   size_t size_prev_slice,
                   const char* _file,
                   const int _line,
                   const char* _caller) {
  GetEngine()->Allgather(sendrecvbuf_, total_size, slice_begin,
    slice_end, size_prev_slice, _file, _line, _caller);
}


// perform in-place allreduce, on sendrecvbuf
void Allreduce_(void *sendrecvbuf,  // NOLINT
                size_t type_nbytes,
                size_t count,
                IEngine::ReduceFunction red,
                mpi::DataType dtype,
                mpi::OpType op,
                IEngine::PreprocFunction prepare_fun,
                void *prepare_arg,
                const char* _file,
                const int _line,
                const char* _caller) {
  GetEngine()->Allreduce(sendrecvbuf, type_nbytes, count, red, prepare_fun,
    prepare_arg, _file, _line, _caller);
}

// code for reduce handle
ReduceHandle::ReduceHandle() = default;
ReduceHandle::~ReduceHandle() = default;

int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return static_cast<int>(dtype.type_size);
}

void ReduceHandle::Init(IEngine::ReduceFunction redfunc, size_t type_nbytes) {
  utils::Assert(redfunc_ == nullptr, "cannot initialize reduce handle twice");
  redfunc_ = redfunc;
}

void ReduceHandle::Allreduce(void *sendrecvbuf,
                             size_t type_nbytes, size_t count,
                             IEngine::PreprocFunction prepare_fun,
                             void *prepare_arg,
                             const char* _file,
                             const int _line,
                             const char* _caller) {
  utils::Assert(redfunc_ != nullptr, "must intialize handle to call AllReduce");
  GetEngine()->Allreduce(sendrecvbuf, type_nbytes, count,
                         redfunc_, prepare_fun, prepare_arg,
                         _file, _line, _caller);
}
}  // namespace engine
}  // namespace rabit
