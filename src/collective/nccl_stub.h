/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once
#if defined(XGBOOST_USE_NCCL)
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <string>  // for string

#include "xgboost/string_view.h"  // for StringView

namespace xgboost::collective {
class NcclStub {
  void* handle_{nullptr};
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

 public:
  explicit NcclStub(StringView path);
  ~NcclStub();

  [[nodiscard]] ncclResult_t Allreduce(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                       cudaStream_t stream) {
    return this->allreduce_(sendbuff, recvbuff, count, datatype, op, comm, stream);
  }
  [[nodiscard]] ncclResult_t Broadcast(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, int root, ncclComm_t comm,
                                       cudaStream_t stream) {
    return this->broadcast_(sendbuff, recvbuff, count, datatype, root, comm, stream);
  }
  [[nodiscard]] ncclResult_t Allgather(const void* sendbuff, void* recvbuff, size_t sendcount,
                                       ncclDataType_t datatype, ncclComm_t comm,
                                       cudaStream_t stream) {
    return this->allgather_(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
  [[nodiscard]] ncclResult_t CommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                                          int rank) {
    return this->comm_init_rank_(comm, nranks, commId, rank);
  }
  [[nodiscard]] ncclResult_t CommDestroy(ncclComm_t comm) { return this->comm_destroy_(comm); }

  [[nodiscard]] ncclResult_t GetUniqueId(ncclUniqueId* uniqueId) {
    return this->get_uniqueid_(uniqueId);
  }
  [[nodiscard]] ncclResult_t Send(const void* sendbuff, size_t count, ncclDataType_t datatype,
                                  int peer, ncclComm_t comm, cudaStream_t stream) {
    return send_(sendbuff, count, datatype, peer, comm, stream);
  }
  [[nodiscard]] ncclResult_t Recv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                                  ncclComm_t comm, cudaStream_t stream) {
    return recv_(recvbuff, count, datatype, peer, comm, stream);
  }
  [[nodiscard]] ncclResult_t GroupStart() { return group_start_(); }
  [[nodiscard]] ncclResult_t GroupEnd() { return group_end_(); }

  [[nodiscard]] const char* GetErrorString(ncclResult_t result) {
    return get_error_string_(result);
  }
};
}  // namespace xgboost::collective

#endif  // defined(XGBOOST_USE_NCCL)
