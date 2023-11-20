/**
 * Copyright 2023, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include "nccl_stub.h"

#include <cuda.h>   // for CUDA_VERSION
#include <dlfcn.h>  // for dlclose, dlsym, dlopen
#include <nccl.h>

#include <cstdint>  // for int32_t
#include <sstream>  // for stringstream
#include <string>   // for string
#include <utility>  // for move

#include "xgboost/logging.h"

namespace xgboost::collective {
NcclStub::NcclStub(StringView path) : path_{std::move(path)} {
#if defined(XGBOOST_USE_DLOPEN_NCCL)
  CHECK(!path_.empty()) << "Empty path for NCCL.";

  auto cu_major = (CUDA_VERSION) / 1000;
  std::stringstream ss;
  ss << R"m(

If XGBoost is installed from PyPI with pip, the error can fixed by:
- Run `pip install nvidia-nccl-cu)m"
     << cu_major << "` (Or with any CUDA version that's higher than " << cu_major << ").";
  ss << R"m(

If you are using the XGBoost package from conda-forge, please open an issue. This error
should not have happened.

If you are using a customized XGBoost, please make sure one of the followings is true:
- XGBoost is NOT compiled with the `USE_DLOPEN_NCCL` flag.
- The `dmlc_nccl_path` parameter is set to full NCCL path when initializing the collective.

If you are not using distributed training with XGBoost yet this error comes up, please
open an issue.

Since 2.1.0, in order to reduce the binary size for some restricted repositories (like
PyPI), XGBoost can optionally load the `libnccl.so.2` shared object from the environment
using `dlopen`. If you are seeing this error, it means XGBoost failed to find the correct
NCCL installation in the current environment.

)m";
  auto help = ss.str();
  std::string msg{"Failed to load NCCL from path: `" + path_ + "`. Error:\n  "};

  auto safe_load = [&](auto t, StringView name) {
    std::stringstream errs;
    auto ptr = reinterpret_cast<decltype(t)>(dlsym(handle_, name.c_str()));
    if (!ptr) {
      errs << "Failed to load NCCL symbol `" << name << "` from " << path_ << ". Error:\n  "
           << dlerror() << help;
      LOG(FATAL) << errs.str();
    }
    return ptr;
  };

  handle_ = dlopen(path_.c_str(), RTLD_LAZY);
  CHECK(handle_) << msg << dlerror() << help;

  allreduce_ = safe_load(allreduce_, "ncclAllReduce");
  broadcast_ = safe_load(broadcast_, "ncclBroadcast");
  allgather_ = safe_load(allgather_, "ncclAllGather");
  comm_init_rank_ = safe_load(comm_init_rank_, "ncclCommInitRank");
  comm_destroy_ = safe_load(comm_destroy_, "ncclCommDestroy");
  get_uniqueid_ = safe_load(get_uniqueid_, "ncclGetUniqueId");
  send_ = safe_load(send_, "ncclSend");
  recv_ = safe_load(recv_, "ncclRecv");
  group_start_ = safe_load(group_start_, "ncclGroupStart");
  group_end_ = safe_load(group_end_, "ncclGroupEnd");
  get_error_string_ = safe_load(get_error_string_, "ncclGetErrorString");
  get_version_ = safe_load(get_version_, "ncclGetVersion");

  std::int32_t v;
  CHECK_EQ(get_version_(&v), ncclSuccess);
  auto patch = v % 100;
  auto minor = (v / 100) % 100;
  auto major = v / 10000;

  LOG(INFO) << "Loaded shared NCCL " << major << "." << minor << "." << patch << ":`" << path_
            << "`" << std::endl;
#else
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

NcclStub::~NcclStub() {  // NOLINT
#if defined(XGBOOST_USE_DLOPEN_NCCL)
  if (handle_) {
    auto rc = dlclose(handle_);
    if (rc != 0) {
      LOG(WARNING) << "Failed to close NCCL handle:" << dlerror();
    }
  }
  handle_ = nullptr;
#endif  // defined(XGBOOST_USE_DLOPEN_NCCL)
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL)
