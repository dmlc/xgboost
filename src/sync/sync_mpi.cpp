#include "./sync.h"
#include "../utils/utils.h"
#include <mpi.h>

// use MPI to implement sync
namespace xgboost {
namespace sync {
int GetRank(void) {
  return MPI::COMM_WORLD.Get_rank();
}

int GetWorldSize(void) {
  return MPI::COMM_WORLD.Get_size();
}

void Init(int argc, char *argv[]) {
  MPI::Init(argc, argv);
}

bool IsDistributed(void) {
  return true;
}

std::string GetProcessorName(void) {
  int len;
  char name[MPI_MAX_PROCESSOR_NAME];
  MPI::Get_processor_name(name, len);
  name[len] = '\0';
  return std::string(name);
}

void Finalize(void) {
  MPI::Finalize();
}

void AllReduce_(void *sendrecvbuf, size_t count, const MPI::Datatype &dtype, ReduceOp op) {
  switch(op) {
    case kBitwiseOR: MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, dtype, MPI::BOR); return;
    case kSum: MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, dtype, MPI::SUM); return;
    case kMax: MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, dtype, MPI::MAX); return;
  }
}

template<>
void AllReduce<uint32_t>(uint32_t *sendrecvbuf, size_t count, ReduceOp op) {
  AllReduce_(sendrecvbuf, count, MPI::UNSIGNED, op);
}

template<>
void AllReduce<float>(float *sendrecvbuf, size_t count, ReduceOp op) {
  AllReduce_(sendrecvbuf, count, MPI::FLOAT, op);
}

void Bcast(std::string *sendrecv_data, int root) {
  unsigned len = static_cast<unsigned>(sendrecv_data->length());
  MPI::COMM_WORLD.Bcast(&len, 1, MPI::UNSIGNED, root);
  sendrecv_data->resize(len);
  if (len != 0) {
    MPI::COMM_WORLD.Bcast(&(*sendrecv_data)[0], len, MPI::CHAR, root);  
  }
}

// code for reduce handle
ReduceHandle::ReduceHandle(void) : handle(NULL), htype(NULL) {
}
ReduceHandle::~ReduceHandle(void) {
  if (handle != NULL) {
    MPI::Op *op = reinterpret_cast<MPI::Op*>(handle);
    op->Free();
    delete op;
  }
  if (htype != NULL) {
    MPI::Datatype *dtype = reinterpret_cast<MPI::Datatype*>(htype);
    dtype->Free();
    delete dtype;
  }
}
int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return dtype.Get_size();
}
void ReduceHandle::Init(ReduceFunction redfunc, size_t type_n4bytes, bool commute) {
  utils::Assert(handle == NULL, "cannot initialize reduce handle twice");
  if (type_n4bytes != 0) {
    MPI::Datatype *dtype = new MPI::Datatype();
    *dtype = MPI::INT.Create_contiguous(type_n4bytes);
    dtype->Commit();
    created_type_n4bytes = type_n4bytes;
    htype = dtype;
  }
  
  MPI::Op *op = new MPI::Op();
  MPI::User_function *pf = redfunc;
  op->Init(pf, commute);
  handle = op;
}
void ReduceHandle::AllReduce(void *sendrecvbuf, size_t type_n4bytes, size_t count) {
  utils::Assert(handle != NULL, "must intialize handle to call AllReduce");
  MPI::Op *op = reinterpret_cast<MPI::Op*>(handle);
  MPI::Datatype *dtype = reinterpret_cast<MPI::Datatype*>(htype);
  if (created_type_n4bytes != type_n4bytes || dtype == NULL) {
    if (dtype == NULL) {
      dtype = new MPI::Datatype();
    } else {
      dtype->Free();
    }
    *dtype = MPI::INT.Create_contiguous(type_n4bytes);
    dtype->Commit();
    created_type_n4bytes = type_n4bytes;
  }
  MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, *dtype, *op);
}
}  // namespace sync
}  // namespace xgboost
