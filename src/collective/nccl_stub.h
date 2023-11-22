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
  explicit NcclStub(StringView path);
  ~NcclStub();

  [[nodiscard]] ncclResult_t Allreduce(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                                       cudaStream_t stream) const {
    CHECK(allreduce_);
    return this->allreduce_(sendbuff, recvbuff, count, datatype, op, comm, stream);
  }
  [[nodiscard]] ncclResult_t Broadcast(const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype, int root, ncclComm_t comm,
                                       cudaStream_t stream) const {
    CHECK(broadcast_);
    return this->broadcast_(sendbuff, recvbuff, count, datatype, root, comm, stream);
  }
  [[nodiscard]] ncclResult_t Allgather(const void* sendbuff, void* recvbuff, size_t sendcount,
                                       ncclDataType_t datatype, ncclComm_t comm,
                                       cudaStream_t stream) const {
    CHECK(allgather_);
    return this->allgather_(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  }
  [[nodiscard]] ncclResult_t CommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId,
                                          int rank) const {
    CHECK(comm_init_rank_);
    return this->comm_init_rank_(comm, nranks, commId, rank);
  }
  [[nodiscard]] ncclResult_t CommDestroy(ncclComm_t comm) const {
    CHECK(comm_destroy_);
    return this->comm_destroy_(comm);
  }

  [[nodiscard]] ncclResult_t GetUniqueId(ncclUniqueId* uniqueId) const {
    CHECK(get_uniqueid_);
    return this->get_uniqueid_(uniqueId);
  }
  [[nodiscard]] ncclResult_t Send(const void* sendbuff, size_t count, ncclDataType_t datatype,
                                  int peer, ncclComm_t comm, cudaStream_t stream) {
    CHECK(send_);
    return send_(sendbuff, count, datatype, peer, comm, stream);
  }
  [[nodiscard]] ncclResult_t Recv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                                  ncclComm_t comm, cudaStream_t stream) const {
    CHECK(recv_);
    return recv_(recvbuff, count, datatype, peer, comm, stream);
  }
  [[nodiscard]] ncclResult_t GroupStart() const {
    CHECK(group_start_);
    return group_start_();
  }
  [[nodiscard]] ncclResult_t GroupEnd() const {
    CHECK(group_end_);
    return group_end_();
  }

  [[nodiscard]] const char* GetErrorString(ncclResult_t result) const {
    return get_error_string_(result);
  }
};
}  // namespace xgboost::collective

#endif  // defined(XGBOOST_USE_NCCL)
