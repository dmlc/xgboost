/*!
 * \file engine_mpi.cc
 * \brief this file gives an implementation of engine interface using MPI,
 *   this will allow rabit program to run with MPI, but do not comes with fault tolerant
 *   
 * \author Tianqi Chen
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include "./engine.h"
#include "./utils.h"
#include <mpi.h>

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
                         ReduceFunction reducer) {
    utils::Error("MPIEngine:: Allreduce is not supported, use Allreduce_ instead");
  }
  virtual void Broadcast(void *sendrecvbuf_, size_t size, int root) {   
    MPI::COMM_WORLD.Bcast(sendrecvbuf_, size, MPI::CHAR, root);
  }
  virtual void InitAfterException(void) {
    utils::Error("MPI is not fault tolerant");
  }
  virtual int LoadCheckPoint(utils::ISerializable *p_model) {
    return 0;
  }
  virtual void CheckPoint(const utils::ISerializable &model) {
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
  /*! \brief get the host name of current node */  
  virtual std::string GetHost(void) const {
    int len;
    char name[MPI_MAX_PROCESSOR_NAME];
    MPI::Get_processor_name(name, len);
    name[len] = '\0';
    return std::string(name);
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
  switch(dtype) {
    case kInt: return MPI::INT;
    case kUInt: return MPI::UNSIGNED;
    case kFloat: return MPI::FLOAT;
    case kDouble: return MPI::DOUBLE;
  }
  utils::Error("unknown mpi::DataType");
  return MPI::CHAR;
}
// transform enum to MPI OP
inline MPI::Op GetOp(mpi::OpType otype) {
  using namespace mpi;
  switch(otype) {
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
                mpi::OpType op) {  
  MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, GetType(dtype), GetOp(op));
}
}  // namespace engine
}  // namespace rabit
