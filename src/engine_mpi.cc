/*!
 *  Copyright (c) 2014 by Contributors
 * \file engine_mpi.cc
 * \brief this file gives an implementation of engine interface using MPI,
 *   this will allow rabit program to run with MPI, but do not comes with fault tolerant
 *
 * \author Tianqi Chen
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <mpi.h>
#include <cstdio>
#include "../include/rabit/internal/engine.h"
#include "../include/rabit/internal/utils.h"

namespace rabit {
namespace engine {
/*! \brief implementation of engine using MPI */
class MPIEngine : public IEngine {
 public:
  MPIEngine(void) {
    version_number = 0;
  }
  virtual void Allreduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,
                         ReduceFunction reducer,
                         PreprocFunction prepare_fun,
                         void *prepare_arg) {
    utils::Error("MPIEngine:: Allreduce is not supported,"\
                 "use Allreduce_ instead");
  }
  virtual void Broadcast(void *sendrecvbuf_, size_t size, int root) {
    MPI::COMM_WORLD.Bcast(sendrecvbuf_, size, MPI::CHAR, root);
  }
  virtual void InitAfterException(void) {
    utils::Error("MPI is not fault tolerant");
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
    return MPI::COMM_WORLD.Get_rank();
  }
  /*! \brief get total number of */
  virtual int GetWorldSize(void) const {
    return MPI::COMM_WORLD.Get_size();
  }
  /*! \brief whether it is distributed */
  virtual bool IsDistributed(void) const {
    return true;
  }
  /*! \brief get the host name of current node */
  virtual std::string GetHost(void) const {
    int len;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI::Get_processor_name(name, len);
    name[len] = '\0';
    return std::string(name);
  }
  virtual void TrackerPrint(const std::string &msg) {
    // simply print information into the tracker
    if (GetRank() == 0) {
      utils::Printf("%s", msg.c_str());
    }
  }

 private:
  int version_number;
};

// singleton sync manager
MPIEngine manager;

/*! \brief intiialize the synchronization module */
void Init(int argc, char *argv[]) {
  MPI::Init(argc, argv);
}
/*! \brief finalize syncrhonization module */
void Finalize(void) {
  MPI::Finalize();
}

/*! \brief singleton method to get engine */
IEngine *GetEngine(void) {
  return &manager;
}
// transform enum to MPI data type
inline MPI::Datatype GetType(mpi::DataType dtype) {
  using namespace mpi;
  switch (dtype) {
    case kChar: return MPI::CHAR;
    case kUChar: return MPI::BYTE;
    case kInt: return MPI::INT;
    case kUInt: return MPI::UNSIGNED;
    case kLong: return MPI::LONG;
    case kULong: return MPI::UNSIGNED_LONG;
    case kFloat: return MPI::FLOAT;
    case kDouble: return MPI::DOUBLE;
    case kLongLong: return MPI::LONG_LONG;
    case kULongLong: return MPI::UNSIGNED_LONG_LONG;
  }
  utils::Error("unknown mpi::DataType");
  return MPI::CHAR;
}
// transform enum to MPI OP
inline MPI::Op GetOp(mpi::OpType otype) {
  using namespace mpi;
  switch (otype) {
    case kMax: return MPI::MAX;
    case kMin: return MPI::MIN;
    case kSum: return MPI::SUM;
    case kBitwiseOR: return MPI::BOR;
  }
  utils::Error("unknown mpi::OpType");
  return MPI::MAX;
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
  MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf,
                            count, GetType(dtype), GetOp(op));
}

// code for reduce handle
ReduceHandle::ReduceHandle(void)
    : handle_(NULL), redfunc_(NULL), htype_(NULL) {
}
ReduceHandle::~ReduceHandle(void) {
  if (handle_ != NULL) {
    MPI::Op *op = reinterpret_cast<MPI::Op*>(handle_);
    op->Free();
    delete op;
  }
  if (htype_ != NULL) {
    MPI::Datatype *dtype = reinterpret_cast<MPI::Datatype*>(htype_);
    dtype->Free();
    delete dtype;
  }
}
int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return dtype.Get_size();
}
void ReduceHandle::Init(IEngine::ReduceFunction redfunc, size_t type_nbytes) {
  utils::Assert(handle_ == NULL, "cannot initialize reduce handle twice");
  if (type_nbytes != 0) {
    MPI::Datatype *dtype = new MPI::Datatype();
    if (type_nbytes % 8 == 0) {
      *dtype = MPI::LONG.Create_contiguous(type_nbytes / sizeof(long));  // NOLINT(*)
    } else if (type_nbytes % 4 == 0) {
      *dtype = MPI::INT.Create_contiguous(type_nbytes / sizeof(int));
    } else {
      *dtype = MPI::CHAR.Create_contiguous(type_nbytes);
    }
    dtype->Commit();
    created_type_nbytes_ = type_nbytes;
    htype_ = dtype;
  }
  MPI::Op *op = new MPI::Op();
  MPI::User_function *pf = redfunc;
  op->Init(pf, true);
  handle_ = op;
}
void ReduceHandle::Allreduce(void *sendrecvbuf,
                             size_t type_nbytes, size_t count,
                             IEngine::PreprocFunction prepare_fun,
                             void *prepare_arg) {
  utils::Assert(handle_ != NULL, "must intialize handle to call AllReduce");
  MPI::Op *op = reinterpret_cast<MPI::Op*>(handle_);
  MPI::Datatype *dtype = reinterpret_cast<MPI::Datatype*>(htype_);
  if (created_type_nbytes_ != type_nbytes || dtype == NULL) {
    if (dtype == NULL) {
      dtype = new MPI::Datatype();
    } else {
      dtype->Free();
    }
    if (type_nbytes % 8 == 0) {
      *dtype = MPI::LONG.Create_contiguous(type_nbytes / sizeof(long));  // NOLINT(*)
    } else if (type_nbytes % 4 == 0) {
      *dtype = MPI::INT.Create_contiguous(type_nbytes / sizeof(int));
    } else {
      *dtype = MPI::CHAR.Create_contiguous(type_nbytes);
    }
    dtype->Commit();
    created_type_nbytes_ = type_nbytes;
  }
  if (prepare_fun != NULL) prepare_fun(prepare_arg);
  MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, *dtype, *op);
}
}  // namespace engine
}  // namespace rabit
