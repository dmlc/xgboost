#include "./sync.h"
#include "../utils/utils.h"
#include "mpi.h"

namespace xgboost {
namespace sync {

// code for reduce handle
ReduceHandle::ReduceHandle(void) : handle(NULL) {
}
ReduceHandle::~ReduceHandle(void) {
  if (handle != NULL) {
    MPI::Op *op = reinterpret_cast<MPI::Op*>(handle);
    op->Free();
    delete op;
  }
}
void ReduceHandle::Init(ReduceFunction redfunc, bool commute) {
  utils::Assert(handle == NULL, "cannot initialize reduce handle twice");
  MPI::Op *op = new MPI::Op();
  MPI::User_function *pf = reinterpret_cast<MPI::User_function*>(redfunc);
  op->Init(pf, commute);
  handle = op;
}
void ReduceHandle::AllReduce(void *sendrecvbuf, size_t n4byte) {
  utils::Assert(handle != NULL, "must intialize handle to call AllReduce");  
  MPI::Op *op = reinterpret_cast<MPI::Op*>(handle);
  MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, n4byte, MPI_INT, *op);
}

int GetRank(void) {
  return MPI::COMM_WORLD.Get_rank();
}

void Init(int argc, char *argv[]) {
  MPI::Init(argc, argv);
}

void Finalize(void) {
  MPI::Finalize();
}

void AllReduce_(void *sendrecvbuf, int count, const MPI::Datatype &dtype, ReduceOp op) {
  switch(op) {
    case kBitwiseOR: MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, dtype, MPI::BOR); return;
    case kSum: MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, dtype, MPI::SUM); return;
  }
}

template<>
void AllReduce<uint32_t>(uint32_t *sendrecvbuf, int count, ReduceOp op) {
  AllReduce_(sendrecvbuf, count, MPI::UNSIGNED, op);
}

template<>
void AllReduce<float>(float *sendrecvbuf, int count, ReduceOp op) {
  AllReduce_(sendrecvbuf, count, MPI::FLOAT, op);
}

}  // namespace sync
}  // namespace xgboost
