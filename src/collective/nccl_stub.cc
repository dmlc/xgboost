#include "nccl_stub.h"

#include <dlfcn.h>
#include <nccl.h>

#include "xgboost/logging.h"
#include "xgboost/string_view.h"  // for StringView

namespace xgboost::collective {
NcclStub::NcclStub(std::string path) : path_{std::move(path)} {
  handle_ = dlopen(path_.c_str(), RTLD_LAZY);
  StringView msg{"Failed to load nccl:"};
  CHECK(handle_) << msg << dlerror();
  allreduce_ = reinterpret_cast<decltype(allreduce_)>(dlsym(handle_, "ncclAllReduce"));
  CHECK(allreduce_) << msg << dlerror();
  broadcast_ = reinterpret_cast<decltype(broadcast_)>(dlsym(handle_, "ncclBroadcast"));
  CHECK(broadcast_) << msg << dlerror();
  allgather_ = reinterpret_cast<decltype(allgather_)>(dlsym(handle_, "ncclAllGather"));
  CHECK(allgather_) << msg << dlerror();
  comm_init_rank_ = reinterpret_cast<decltype(comm_init_rank_)>(dlsym(handle_, "ncclCommInitRank"));
  CHECK(comm_init_rank_) << msg << dlerror();
  comm_destroy_ = reinterpret_cast<decltype(comm_destroy_)>(dlsym(handle_, "ncclCommDestroy"));
  CHECK(comm_destroy_) << msg << dlerror();
  get_uniqueid_ = reinterpret_cast<decltype(get_uniqueid_)>(dlsym(handle_, "ncclGetUniqueId"));
  CHECK(get_uniqueid_) << msg << dlerror();
  send_ = reinterpret_cast<decltype(send_)>(dlsym(handle_, "ncclSend"));
  CHECK(send_) << msg << dlerror();
  recv_ = reinterpret_cast<decltype(recv_)>(dlsym(handle_, "ncclRecv"));
  CHECK(recv_) << msg << dlerror();
};

NcclStub::~NcclStub() { CHECK_EQ(dlclose(handle_), 0) << dlerror(); }
}  // namespace xgboost::collective
