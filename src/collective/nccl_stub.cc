/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include "nccl_stub.h"

#if defined(XGBOOST_USE_DLOPEN_NCCL)

#include <dlfcn.h>  // for dlclose, dlsym, dlopen

#include <cstdint>  // for int32_t

#include "xgboost/logging.h"

#endif  // defined(XGBOOST_USE_DLOPEN_NCCL)

#include <cuda.h>              // for CUDA_VERSION
#include <cuda_runtime_api.h>  // for cudaPeekAtLastError
#include <nccl.h>
#include <thrust/system/cuda/error.h>  // for cuda_category
#include <thrust/system_error.h>       // for system_error

#include <memory>   // for shared_ptr
#include <sstream>  // for stringstream
#include <string>   // for string
#include <thread>   // for this_thread
#include <utility>  // for move

#include "../common/error_msg.h"  // for OldNccl
#include "../common/timer.h"      // for Timer

namespace xgboost::collective {
[[nodiscard]] Result NcclStub::GetNcclResult(ncclResult_t code) const {
  if (code == ncclSuccess || code == ncclInProgress) {
    return Success();
  }

  std::stringstream ss;
  ss << "NCCL failure: " << this->GetErrorString(code) << ".";
  if (code == ncclUnhandledCudaError) {
    // nccl usually preserves the last error so we can get more details.
    auto err = cudaPeekAtLastError();
    ss << "  CUDA error: " << thrust::system_error(err, thrust::cuda_category()).what() << "\n";
  } else if (code == ncclSystemError) {
    ss << "  This might be caused by a network configuration issue. Please consider specifying "
          "the network interface for NCCL via environment variables listed in its reference: "
          "`https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html`.\n";
  }
  return Fail(ss.str());
}

NcclStub::NcclStub(StringView path) : path_{std::move(path)} {
#if defined(XGBOOST_USE_DLOPEN_NCCL)
  CHECK(!path_.empty()) << "Empty path for NCCL.";

  auto cu_major = (CUDA_VERSION) / 1000;
  std::stringstream ss;
  ss << R"m(

If XGBoost is installed from PyPI with pip, the error can fixed by:

- Run `pip install nvidia-nccl-cu)m"
     << cu_major << "` (Or with any CUDA version that's compatible with " << cu_major << ").";
  ss << R"m(

Otherwise, please refer to:

  https://xgboost.readthedocs.io/en/stable/tutorials/dask.html#troubleshooting

for more info, or open an issue on GitHub. Starting from XGBoost 2.1.0, the PyPI package
no long bundles NCCL in the binary wheel.

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
  if (!handle_) {
    LOG(FATAL) << msg << dlerror() << help;
  }

  allreduce_ = safe_load(allreduce_, "ncclAllReduce");
  broadcast_ = safe_load(broadcast_, "ncclBroadcast");
  allgather_ = safe_load(allgather_, "ncclAllGather");
  comm_init_rank_ = safe_load(comm_init_rank_, "ncclCommInitRank");
  comm_init_rank_config_ = safe_load(comm_init_rank_config_, "ncclCommInitRankConfig");
  comm_destroy_ = safe_load(comm_destroy_, "ncclCommDestroy");
  comm_finalize_ = safe_load(comm_finalize_, "ncclCommFinalize");
  comm_get_async_error_ = safe_load(comm_get_async_error_, "ncclCommGetAsyncError");
  comm_abort_ = safe_load(comm_abort_, "ncclCommAbort");
  get_uniqueid_ = safe_load(get_uniqueid_, "ncclGetUniqueId");
  send_ = safe_load(send_, "ncclSend");
  recv_ = safe_load(recv_, "ncclRecv");
  group_start_ = safe_load(group_start_, "ncclGroupStart");
  group_end_ = safe_load(group_end_, "ncclGroupEnd");
  get_error_string_ = safe_load(get_error_string_, "ncclGetErrorString");
  get_version_ = safe_load(get_version_, "ncclGetVersion");
#else
  allreduce_ = ncclAllReduce;
  broadcast_ = ncclBroadcast;
  allgather_ = ncclAllGather;
  comm_init_rank_ = ncclCommInitRank;
  comm_init_rank_config_ = ncclCommInitRankConfig;
  comm_destroy_ = ncclCommDestroy;
  comm_finalize_ = ncclCommFinalize;
  comm_get_async_error_ = ncclCommGetAsyncError;
  comm_abort_ = ncclCommAbort;
  get_uniqueid_ = ncclGetUniqueId;
  send_ = ncclSend;
  recv_ = ncclRecv;
  group_start_ = ncclGroupStart;
  group_end_ = ncclGroupEnd;
  get_error_string_ = ncclGetErrorString;
  get_version_ = ncclGetVersion;
#endif

  std::int32_t major = 0, minor = 0, patch = 0;
  SafeColl(this->GetVersion(&major, &minor, &patch));
  LOG(INFO) << "Loaded shared NCCL " << major << "." << minor << "." << patch << ":`" << path_
            << "`" << std::endl;

  error::CheckOldNccl(major, minor, patch);
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

[[nodiscard]] Result BusyWait(std::shared_ptr<NcclStub> nccl, ncclComm_t comm,
                              std::chrono::seconds timeout) {
  using namespace std::chrono_literals;  // NOLINT
  common::Timer timer;
  ncclResult_t async_error = ncclSuccess;
  timer.Start();
  do {
    auto rc = nccl->CommGetAsyncError(comm, &async_error);
    if (!rc.OK()) {
      return rc;
    }
    if (async_error == ncclInProgress) {
      if (timer.Duration().count() < timeout.count()) {
        std::this_thread::sleep_for(20ms);
      } else {
        return Fail("Timeout, elapsed:" + std::to_string(timer.Duration().count()));
      }
    }
  } while (async_error == ncclInProgress);

  return nccl->GetNcclResult(async_error);
}
}  // namespace xgboost::collective
#endif  // defined(XGBOOST_USE_NCCL)
