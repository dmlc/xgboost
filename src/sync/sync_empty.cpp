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
template<>
void AllReduce<uint32_t>(uint32_t *sendrecvbuf, int count, ReduceOp op) {
}
template<>
void AllReduce<float>(float *sendrecvbuf, int count, ReduceOp op) {
}
void Bcast(std::string *sendrecv_data, int root) {
}
ReduceHandle::ReduceHandle(void) : handle(NULL) {}
ReduceHandle::~ReduceHandle(void) {}
void ReduceHandle::Init(ReduceFunction redfunc, bool commute) {}
void ReduceHandle::AllReduce(void *sendrecvbuf, size_t n4byte) {}
}  // namespace sync
}  // namespace xgboost

