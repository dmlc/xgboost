/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#if defined(XGBOOST_USE_NCCL)
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <atomic>  // for atomic
#include <memory>  // for shared_ptr
#include <string>  // for string

#include "xgboost/collective/result.h"  // for Result
#include "xgboost/string_view.h"        // for StringView

namespace xgboost::collective {
/**
 * @brief A stub for NCCL to facilitate dynamic loading.
 */
class NcclStub {
#if defined(XGBOOST_USE_DLOPEN_NCCL)
  void* handle_{nullptr};
#endif  // defined(XGBOOST_USE_DLOPEN_NCCL)
  std::string path_;
  std::atomic<bool> aborted_{false};

  decltype(ncclAllReduce)* allreduce_{nullptr};
  decltype(ncclBroadcast)* broadcast_{nullptr};
  decltype(ncclAllGather)* allgather_{nullptr};
  decltype(ncclCommInitRank)* comm_init_rank_{nullptr};
  decltype(ncclCommInitRankConfig)* comm_init_rank_config_{nullptr};
  decltype(ncclCommDestroy)* comm_destroy_{nullptr};
  decltype(ncclCommFinalize)* comm_finalize_{nullptr};
  decltype(ncclCommGetAsyncError)* comm_get_async_error_{nullptr};
  decltype(ncclCommAbort)* comm_abort_{nullptr};
  decltype(ncclGetUniqueId)* get_uniqueid_{nullptr};
  decltype(ncclSend)* send_{nullptr};
  decltype(ncclRecv)* recv_{nullptr};
  decltype(ncclGroupStart)* group_start_{nullptr};
  decltype(ncclGroupEnd)* group_end_{nullptr};
  decltype(ncclGetErrorString)* get_error_string_{nullptr};
  decltype(ncclGetVersion)* get_version_{nullptr};

 public:
  [[nodiscard]] Result GetNcclResult(ncclResult_t code) const;

 public:
  explicit NcclStub(StringView path);
  ~NcclStub();

  [[nodiscard]] Result Allreduce(const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                 cudaStream_t stream) const {
    return this->GetNcclResult(allreduce_(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  [[nodiscard]] Result Broadcast(const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, int root, ncclComm_t comm,
                                 cudaStream_t stream) const {
    return this->GetNcclResult(broadcast_(sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  [[nodiscard]] Result Allgather(const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype, ncclComm_t comm,
                                 cudaStream_t stream) const {
    return this->GetNcclResult(allgather_(sendbuff, recvbuff, sendcount, datatype, comm, stream));
  }
  [[nodiscard]] Result CommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                                    int rank) const {
    return this->GetNcclResult(this->comm_init_rank_(comm, nranks, commId, rank));
  }
  [[nodiscard]] Result CommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                                          int rank, ncclConfig_t* config) const {
    return this->GetNcclResult(this->comm_init_rank_config_(comm, nranks, commId, rank, config));
  }
  [[nodiscard]] Result CommDestroy(ncclComm_t comm) const {
    if (this->Aborted()) {
      return Success();
    }
    return this->GetNcclResult(comm_destroy_(comm));
  }
  [[nodiscard]] Result CommFinalize(ncclComm_t comm) const {
    if (this->Aborted()) {
      return Success();
    }
    return this->GetNcclResult(comm_finalize_(comm));
  }
  [[nodiscard]] bool Aborted() const { return this->aborted_; }

  [[nodiscard]] Result CommGetAsyncError(ncclComm_t comm, ncclResult_t* async_error) const {
    if (this->Aborted()) {
      *async_error = ncclSuccess;
      return Success();
    }
    return this->GetNcclResult(comm_get_async_error_(comm, async_error));
  }
  [[nodiscard]] Result CommAbort(ncclComm_t comm) {
    if (this->Aborted()) {
      return Success();
    }
    this->aborted_ = true;
    return this->GetNcclResult(comm_abort_(comm));
  }
  [[nodiscard]] Result GetUniqueId(ncclUniqueId* uniqueId) const {
    return this->GetNcclResult(get_uniqueid_(uniqueId));
  }
  [[nodiscard]] Result Send(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
                            ncclComm_t comm, cudaStream_t stream) {
    return this->GetNcclResult(send_(sendbuff, count, datatype, peer, comm, stream));
  }
  [[nodiscard]] Result Recv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                            ncclComm_t comm, cudaStream_t stream) const {
    return this->GetNcclResult(recv_(recvbuff, count, datatype, peer, comm, stream));
  }
  [[nodiscard]] Result GroupStart() const { return this->GetNcclResult(group_start_()); }
  [[nodiscard]] Result GroupEnd() const { return this->GetNcclResult(group_end_()); }
  [[nodiscard]] const char* GetErrorString(ncclResult_t result) const {
    return get_error_string_(result);
  }
  [[nodiscard]] Result GetVersion(std::int32_t* major, std::int32_t* minor,
                                  std::int32_t* patch) const {
    std::int32_t v = 0;
    auto rc = this->GetNcclResult(get_version_(&v));
    if (!rc.OK()) {
      return rc;
    }

    if (major) {
      *major = v / 10000;
    }
    if (minor) {
      *minor = v % 10000 / 100;
    }
    if (patch) {
      *patch = v % 100;
    }
    return rc;
  }
};

[[nodiscard]] Result BusyWait(std::shared_ptr<NcclStub> nccl, ncclComm_t comm,
                              std::chrono::seconds timeout);
}  // namespace xgboost::collective

#endif  // defined(XGBOOST_USE_NCCL)
