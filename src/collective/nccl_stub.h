/**
 * Copyright 2023, XGBoost Contributors
 */
#pragma once

#include <cuda_runtime_api.h>
#include <nccl.h>

#include <string>  // for string

namespace xgboost::collective {
class NcclStub {
  void* handle_{nullptr};
  std::string path_;

  ncclResult_t (*allreduce_)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
                             cudaStream_t);
  ncclResult_t (*broadcast_)(const void*, void*, size_t, ncclDataType_t, int, ncclComm_t,
                             cudaStream_t);
  ncclResult_t (*allgather_)(const void*, void*, size_t, ncclDataType_t, ncclComm_t, cudaStream_t);

  ncclResult_t (*comm_init_rank_)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
  ncclResult_t (*comm_destroy_)(ncclComm_t comm);
  ncclResult_t (*get_uniqueid_)(ncclUniqueId* uniqueId);

  ncclResult_t (*send_)(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
                        ncclComm_t comm, cudaStream_t stream);
  ncclResult_t (*recv_)(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                        ncclComm_t comm, cudaStream_t stream);
  const char*  (*get_error_string_)(ncclResult_t result);

 public:
  explicit NcclStub(std::string path);
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
  [[nodiscard]] const char* GetErrorString(ncclResult_t result) {
    return get_error_string_(result);
  }
};
}  // namespace xgboost::collective
