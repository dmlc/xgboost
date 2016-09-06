/*!
 *  Copyright (c) 2014 by Contributors
 * \file engine_empty.cc
 * \brief this file provides a dummy implementation of engine that does nothing
 *  this file provides a way to fall back to single node program without causing too many dependencies
 *  This is usually NOT needed, use engine_mpi or engine for real distributed version
 * \author Tianqi Chen
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX

#include "../include/rabit/internal/engine.h"

namespace rabit {
namespace engine {
/*! \brief EmptyEngine */
class EmptyEngine : public IEngine {
 public:
  EmptyEngine(void) {
    version_number = 0;
  }
  virtual void Allreduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,
                         ReduceFunction reducer,
                         PreprocFunction prepare_fun,
                         void *prepare_arg) {
    utils::Error("EmptyEngine:: Allreduce is not supported,"\
                 "use Allreduce_ instead");
  }
  virtual void Broadcast(void *sendrecvbuf_, size_t size, int root) {
  }
  virtual void InitAfterException(void) {
    utils::Error("EmptyEngine is not fault tolerant");
  }
  virtual int LoadCheckPoint(Serializable *global_model,
                             Serializable *local_model = NULL) {
    return 0;
  }
  virtual void CheckPoint(const Serializable *global_model,
                          const Serializable *local_model = NULL) {
    version_number += 1;
  }
  virtual void LazyCheckPoint(const Serializable *global_model) {
    version_number += 1;
  }
  virtual int VersionNumber(void) const {
    return version_number;
  }
  /*! \brief get rank of current node */
  virtual int GetRank(void) const {
    return 0;
  }
  /*! \brief get total number of */
  virtual int GetWorldSize(void) const {
    return 1;
  }
  /*! \brief whether it is distributed */
  virtual bool IsDistributed(void) const {
    return false;
  }
  /*! \brief get the host name of current node */
  virtual std::string GetHost(void) const {
    return std::string("");
  }
  virtual void TrackerPrint(const std::string &msg) {
    // simply print information into the tracker
    utils::Printf("%s", msg.c_str());
  }

 private:
  int version_number;
};

// singleton sync manager
EmptyEngine manager;

/*! \brief intiialize the synchronization module */
void Init(int argc, char *argv[]) {
}
/*! \brief finalize syncrhonization module */
void Finalize(void) {
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
  if (prepare_fun != NULL) prepare_fun(prepare_arg);
}

// code for reduce handle
ReduceHandle::ReduceHandle(void) : handle_(NULL), htype_(NULL) {
}
ReduceHandle::~ReduceHandle(void) {}

int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return 0;
}
void ReduceHandle::Init(IEngine::ReduceFunction redfunc, size_t type_nbytes) {}
void ReduceHandle::Allreduce(void *sendrecvbuf,
                             size_t type_nbytes, size_t count,
                             IEngine::PreprocFunction prepare_fun,
                             void *prepare_arg) {
  if (prepare_fun != NULL) prepare_fun(prepare_arg);
}
}  // namespace engine
}  // namespace rabit
