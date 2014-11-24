#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include "./sync.h"
#include "../utils/utils.h"
// no synchronization module, single thread mode does not need it anyway
namespace xgboost {
namespace sync {
int GetRank(void) {
  return 0;
}

void Init(int argc, char *argv[]) {
}

void Finalize(void) {
}

bool IsDistributed(void) {
  return false;
}

int GetWorldSize(void) {
  return 1;
}

std::string GetProcessorName(void) {
  return std::string("");
}

template<>
void AllReduce<uint32_t>(uint32_t *sendrecvbuf, size_t count, ReduceOp op) {
}

template<>
void AllReduce<float>(float *sendrecvbuf, size_t count, ReduceOp op) {
}

void Bcast(std::string *sendrecv_data, int root) {
}

ReduceHandle::ReduceHandle(void) : handle(NULL) {}
ReduceHandle::~ReduceHandle(void) {}
int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return 0;
}
void ReduceHandle::Init(ReduceFunction redfunc, size_t type_n4bytes, bool commute) {}
void ReduceHandle::AllReduce(void *sendrecvbuf, size_t type_n4bytes, size_t n4byte) {}
}  // namespace sync
}  // namespace xgboost

