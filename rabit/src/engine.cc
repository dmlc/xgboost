/*!
 *  Copyright (c) 2014 by Contributors
 * \file engine.cc
 * \brief this file governs which implementation of engine we are actually using
 *  provides an singleton of engine interface
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX

#include <memory>
#include "../include/rabit/internal/engine.h"
#include "./allreduce_base.h"
#include "./allreduce_robust.h"
#include "./thread_local.h"

namespace rabit {
namespace engine {
// singleton sync manager
#ifndef RABIT_USE_BASE
#ifndef RABIT_USE_MOCK
typedef AllreduceRobust Manager;
#else
typedef AllreduceMock Manager;
#endif
#else
typedef AllreduceBase Manager;
#endif

/*! \brief entry to to easily hold returning information */
struct ThreadLocalEntry {
  /*! \brief stores the current engine */
  std::unique_ptr<Manager> engine;
  /*! \brief whether init has been called */
  bool initialized;
  /*! \brief constructor */
  ThreadLocalEntry() : initialized(false) {}
};

// define the threadlocal store.
typedef ThreadLocalStore<ThreadLocalEntry> EngineThreadLocal;

/*! \brief intiialize the synchronization module */
void Init(int argc, char *argv[]) {
  ThreadLocalEntry* e = EngineThreadLocal::Get();
  utils::Check(e->engine.get() == nullptr,
               "rabit::Init is already called in this thread");
  e->initialized = true;
  e->engine.reset(new Manager());
  e->engine->Init(argc, argv);
}

/*! \brief finalize syncrhonization module */
void Finalize() {
  ThreadLocalEntry* e = EngineThreadLocal::Get();
  utils::Check(e->engine.get() != nullptr,
               "rabit::Finalize engine is not initialized or already been finalized.");
  e->engine->Shutdown();
  e->engine.reset(nullptr);
}

/*! \brief singleton method to get engine */
IEngine *GetEngine() {
  // un-initialized default manager.
  static AllreduceBase default_manager;
  ThreadLocalEntry* e = EngineThreadLocal::Get();
  IEngine* ptr = e->engine.get();
  if (ptr == nullptr) {
    utils::Check(!e->initialized,
                 "Doing rabit call after Finalize");
    return &default_manager;
  } else {
    return ptr;
  }
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
  GetEngine()->Allreduce(sendrecvbuf, type_nbytes, count,
                         red, prepare_fun, prepare_arg);
}

// code for reduce handle
ReduceHandle::ReduceHandle(void)
  : handle_(NULL), redfunc_(NULL), htype_(NULL) {
}

ReduceHandle::~ReduceHandle(void) {}

int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return static_cast<int>(dtype.type_size);
}

void ReduceHandle::Init(IEngine::ReduceFunction redfunc, size_t type_nbytes) {
  utils::Assert(redfunc_ == NULL, "cannot initialize reduce handle twice");
  redfunc_ = redfunc;
}

void ReduceHandle::Allreduce(void *sendrecvbuf,
                             size_t type_nbytes, size_t count,
                             IEngine::PreprocFunction prepare_fun,
                             void *prepare_arg) {
  utils::Assert(redfunc_ != NULL, "must intialize handle to call AllReduce");
  GetEngine()->Allreduce(sendrecvbuf, type_nbytes, count,
                         redfunc_, prepare_fun, prepare_arg);
}
}  // namespace engine
}  // namespace rabit
