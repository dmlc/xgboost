/**
 * Copyright 2023, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include "nccl_stub.h"

#include <dlfcn.h>
#include <nccl.h>

#include <string>  // for string

#include "xgboost/logging.h"

namespace xgboost::collective {
NcclStub::NcclStub(StringView path) : path_{std::move(path)} {
#if defined(XGBOOST_USE_DLOPEN_NCCL)
  handle_ = dlopen(path_.c_str(), RTLD_LAZY);
  std::string msg{"Failed to load nccl from path: `" + path_ + "`. Error:"};
  msg += R"m(
This usually happens when XGBoost is intalled from PyPI (using pip) and can be fixed by:
- Run `pip install pip install nvidia-nccl-cu12`

If you are using a customized XGBoost, please make sure one of the following is true:
- XGBoost is NOT compiled with the `USE_DLOPEN_NCCL` flag.
- The `dmlc_nccl_path` parameter is set to full NCCL path when initializing the collective.

Since 2.1.0, XGBoost can optionally load `libnccl.so` from the environment to reduce the
binary size for some repositories with limited capacity.

)m";
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
  group_start_ = reinterpret_cast<decltype(group_start_)>(dlsym(handle_, "ncclGroupStart"));
  CHECK(group_start_) << msg << dlerror();
  group_end_ = reinterpret_cast<decltype(group_end_)>(dlsym(handle_, "ncclGroupEnd"));
  CHECK(group_end_) << msg << dlerror();
  get_error_string_ =
      reinterpret_cast<decltype(get_error_string_)>(dlsym(handle_, "ncclGetErrorString"));
  get_version_ = reinterpret_cast<decltype(get_version_)>(dlsym(handle_, "ncclGetVersion"));
  CHECK(get_version_);

  std::int32_t v;
  CHECK_EQ(get_version_(&v), ncclSuccess);
  auto patch = v % 100;
  auto minor = (v / 100) % 100;
  auto major = v / 10000;

  LOG(INFO) << "Loaded shared NCCL [" << major << "." << minor << "." << patch << "]:`" << path_
            << "`" << std::endl;
#else
  handle_ = nullptr;
  allreduce_ = ncclAllReduce;
  broadcast_ = ncclBroadcast;
  allgather_ = ncclAllGather;
  comm_init_rank_ = ncclCommInitRank;
  comm_destroy_ = ncclCommDestroy;
  get_uniqueid_ = ncclGetUniqueId;
  send_ = ncclSend;
  recv_ = ncclRecv;
  group_start_ = ncclGroupStart;
  group_end_ = ncclGroupEnd;
  get_error_string_ = ncclGetErrorString;
  get_version_ = ncclGetVersion;
#endif
};

NcclStub::~NcclStub() {
  if (handle_) {
    auto rc = dlclose(handle_);
    if (rc != 0) {
      LOG(WARNING) << "Failed to close NCCL handle:" << dlerror();
    }
  }
  handle_ = nullptr;
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL)
