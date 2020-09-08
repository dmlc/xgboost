/*!
 *  Copyright (c) 2014 by Contributors
 * \file engine_empty.cc
 * \brief this file provides a dummy implementation of engine that does nothing
 *  this file provides a way to fall back to single node program without causing too many dependencies
 *  This is usually NOT needed, use engine_mpi or engine for real distributed version
 * \author Tianqi Chen
 */
#define NOMINMAX

#include <rabit/base.h>
#include "rabit/internal/engine.h"

namespace rabit {
namespace engine {
/*! \brief EmptyEngine */
class EmptyEngine : public IEngine {
 public:
  EmptyEngine() {
    version_number_ = 0;
  }
  void Allgather(void *sendrecvbuf_, size_t total_size, size_t slice_begin,
                 size_t slice_end, size_t size_prev_slice, const char *_file,
                 const int _line, const char *_caller) override {
    utils::Error("EmptyEngine:: Allgather is not supported");
  }
  int GetRingPrevRank() const override {
    utils::Error("EmptyEngine:: GetRingPrevRank is not supported");
    return -1;
  }
  void Allreduce(void *sendrecvbuf_, size_t type_nbytes, size_t count,
                 ReduceFunction reducer, PreprocFunction prepare_fun,
                 void *prepare_arg, const char *_file, const int _line,
                 const char *_caller) override {
    utils::Error("EmptyEngine:: Allreduce is not supported,"\
                 "use Allreduce_ instead");
  }
  void Broadcast(void *sendrecvbuf_, size_t size, int root,
                 const char* _file, const int _line, const char* _caller) override {
  }
  void InitAfterException() override {
    utils::Error("EmptyEngine is not fault tolerant");
  }
  int LoadCheckPoint(Serializable *global_model,
                     Serializable *local_model = nullptr) override {
    return 0;
  }
  void CheckPoint(const Serializable *global_model,
                          const Serializable *local_model = nullptr) override {
    version_number_ += 1;
  }
  void LazyCheckPoint(const Serializable *global_model) override {
    version_number_ += 1;
  }
  int VersionNumber() const override {
    return version_number_;
  }
  /*! \brief get rank of current node */
  int GetRank() const override {
    return 0;
  }
  /*! \brief get total number of */
  int GetWorldSize() const override {
    return 1;
  }
  /*! \brief whether it is distributed */
  bool IsDistributed() const override {
    return false;
  }
  /*! \brief get the host name of current node */
  std::string GetHost() const override {
    return std::string("");
  }
  void TrackerPrint(const std::string &msg) override {
    // simply print information into the tracker
    utils::Printf("%s", msg.c_str());
  }

 private:
  int version_number_;
};

// singleton sync manager
EmptyEngine manager;

/*! \brief intiialize the synchronization module */
bool Init(int argc, char *argv[]) {
  return true;
}
/*! \brief finalize syncrhonization module */
bool Finalize() {
  return true;
}

/*! \brief singleton method to get engine */
IEngine *GetEngine() {
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
                void *prepare_arg,
                const char* _file,
                const int _line,
                const char* _caller) {
  if (prepare_fun != nullptr) prepare_fun(prepare_arg);
}

// code for reduce handle
ReduceHandle::ReduceHandle()  = default;
ReduceHandle::~ReduceHandle() = default;

int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return 0;
}
void ReduceHandle::Init(IEngine::ReduceFunction redfunc, size_t type_nbytes) {}
void ReduceHandle::Allreduce(void *sendrecvbuf,
                             size_t type_nbytes, size_t count,
                             IEngine::PreprocFunction prepare_fun,
                             void *prepare_arg,
                             const char* _file,
                             const int _line,
                             const char* _caller) {
  if (prepare_fun != nullptr) prepare_fun(prepare_arg);
}
}  // namespace engine
}  // namespace rabit
