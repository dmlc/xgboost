/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#if defined(XGBOOST_USE_NCCL)
#include <cuda_runtime_api.h>
#include <nccl.h>

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

  decltype(ncclAllReduce)* allreduce_{nullptr};
  decltype(ncclBroadcast)* broadcast_{nullptr};
  decltype(ncclAllGather)* allgather_{nullptr};
  decltype(ncclCommInitRank)* comm_init_rank_{nullptr};
  decltype(ncclCommDestroy)* comm_destroy_{nullptr};
  decltype(ncclGetUniqueId)* get_uniqueid_{nullptr};
  decltype(ncclSend)* send_{nullptr};
  decltype(ncclRecv)* recv_{nullptr};
  decltype(ncclGroupStart)* group_start_{nullptr};
  decltype(ncclGroupEnd)* group_end_{nullptr};
  decltype(ncclGetErrorString)* get_error_string_{nullptr};
  decltype(ncclGetVersion)* get_version_{nullptr};

 public:
  Result GetNcclResult(ncclResult_t code) const;

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
  [[nodiscard]] Result CommDestroy(ncclComm_t comm) const {
    return this->GetNcclResult(comm_destroy_(comm));
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
};
}  // namespace xgboost::collective

#endif  // defined(XGBOOST_USE_NCCL)
