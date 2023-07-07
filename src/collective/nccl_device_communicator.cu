/*!
 * Copyright 2023 XGBoost contributors
 */
#if defined(XGBOOST_USE_NCCL)
#include "nccl_device_communicator.cuh"

namespace xgboost {
namespace collective {

NcclDeviceCommunicator::NcclDeviceCommunicator(int device_ordinal, bool needs_sync)
    : device_ordinal_{device_ordinal},
      needs_sync_{needs_sync},
      world_size_{GetWorldSize()},
      rank_{GetRank()} {
  if (device_ordinal_ < 0) {
    LOG(FATAL) << "Invalid device ordinal: " << device_ordinal_;
  }
  if (world_size_ == 1) {
    return;
  }

  std::vector<uint64_t> uuids(world_size_ * kUuidLength, 0);
  auto s_uuid = xgboost::common::Span<uint64_t>{uuids.data(), uuids.size()};
  auto s_this_uuid = s_uuid.subspan(rank_ * kUuidLength, kUuidLength);
  GetCudaUUID(s_this_uuid);

  // TODO(rongou): replace this with allgather.
  Allreduce(uuids.data(), uuids.size(), DataType::kUInt64, Operation::kSum);

  std::vector<xgboost::common::Span<uint64_t, kUuidLength>> converted(world_size_);
  size_t j = 0;
  for (size_t i = 0; i < uuids.size(); i += kUuidLength) {
    converted[j] = xgboost::common::Span<uint64_t, kUuidLength>{uuids.data() + i, kUuidLength};
    j++;
  }

  auto iter = std::unique(converted.begin(), converted.end());
  auto n_uniques = std::distance(converted.begin(), iter);

  CHECK_EQ(n_uniques, world_size_)
      << "Multiple processes within communication group running on same CUDA "
      << "device is not supported. " << PrintUUID(s_this_uuid) << "\n";

  nccl_unique_id_ = GetUniqueId();
  dh::safe_cuda(cudaSetDevice(device_ordinal_));
  dh::safe_nccl(ncclCommInitRank(&nccl_comm_, world_size_, nccl_unique_id_, rank_));
  dh::safe_cuda(cudaStreamCreate(&cuda_stream_));
}

NcclDeviceCommunicator::~NcclDeviceCommunicator() {
  if (world_size_ == 1) {
    return;
  }
  if (cuda_stream_) {
    dh::safe_cuda(cudaStreamDestroy(cuda_stream_));
  }
  if (nccl_comm_) {
    dh::safe_nccl(ncclCommDestroy(nccl_comm_));
  }
  if (xgboost::ConsoleLogger::ShouldLog(xgboost::ConsoleLogger::LV::kDebug)) {
    LOG(CONSOLE) << "======== NCCL Statistics========";
    LOG(CONSOLE) << "AllReduce calls: " << allreduce_calls_;
    LOG(CONSOLE) << "AllReduce total MiB communicated: " << allreduce_bytes_ / 1048576;
  }
}

namespace {
ncclDataType_t GetNcclDataType(DataType const &data_type) {
  ncclDataType_t result{ncclInt8};
  switch (data_type) {
    case DataType::kInt8:
      result = ncclInt8;
      break;
    case DataType::kUInt8:
      result = ncclUint8;
      break;
    case DataType::kInt32:
      result = ncclInt32;
      break;
    case DataType::kUInt32:
      result = ncclUint32;
      break;
    case DataType::kInt64:
      result = ncclInt64;
      break;
    case DataType::kUInt64:
      result = ncclUint64;
      break;
    case DataType::kFloat:
      result = ncclFloat;
      break;
    case DataType::kDouble:
      result = ncclDouble;
      break;
    default:
      LOG(FATAL) << "Unknown data type.";
  }
  return result;
}

bool IsBitwiseOp(Operation const &op) {
  return op == Operation::kBitwiseAND || op == Operation::kBitwiseOR ||
         op == Operation::kBitwiseXOR;
}

ncclRedOp_t GetNcclRedOp(Operation const &op) {
  ncclRedOp_t result{ncclMax};
  switch (op) {
    case Operation::kMax:
      result = ncclMax;
      break;
    case Operation::kMin:
      result = ncclMin;
      break;
    case Operation::kSum:
      result = ncclSum;
      break;
    default:
      LOG(FATAL) << "Unsupported reduce operation.";
  }
  return result;
}

template <typename Func>
void RunBitwiseAllreduce(char *out_buffer, char const *device_buffer, Func func, int world_size,
                         std::size_t size, cudaStream_t stream) {
  dh::LaunchN(size, stream, [=] __device__(std::size_t idx) {
    auto result = device_buffer[idx];
    for (auto rank = 1; rank < world_size; rank++) {
      result = func(result, device_buffer[rank * size + idx]);
    }
    out_buffer[idx] = result;
  });
}
}  // anonymous namespace

void NcclDeviceCommunicator::BitwiseAllReduce(void *send_receive_buffer, std::size_t count,
                                              DataType data_type, Operation op) {
  auto const size = count * GetTypeSize(data_type);
  dh::caching_device_vector<char> buffer(size * world_size_);
  auto *device_buffer = buffer.data().get();

  // First gather data from all the workers.
  dh::safe_nccl(ncclAllGather(send_receive_buffer, device_buffer, count, GetNcclDataType(data_type),
                              nccl_comm_, cuda_stream_));
  if (needs_sync_) {
    dh::safe_cuda(cudaStreamSynchronize(cuda_stream_));
  }

  // Then reduce locally.
  auto *out_buffer = static_cast<char *>(send_receive_buffer);
  switch (op) {
    case Operation::kBitwiseAND:
      RunBitwiseAllreduce(out_buffer, device_buffer, thrust::bit_and<char>(), world_size_, size,
                          cuda_stream_);
      break;
    case Operation::kBitwiseOR:
      RunBitwiseAllreduce(out_buffer, device_buffer, thrust::bit_or<char>(), world_size_, size,
                          cuda_stream_);
      break;
    case Operation::kBitwiseXOR:
      RunBitwiseAllreduce(out_buffer, device_buffer, thrust::bit_xor<char>(), world_size_, size,
                          cuda_stream_);
      break;
    default:
      LOG(FATAL) << "Not a bitwise reduce operation.";
  }
}

void NcclDeviceCommunicator::AllReduce(void *send_receive_buffer, std::size_t count,
                                       DataType data_type, Operation op) {
  if (world_size_ == 1) {
    return;
  }

  dh::safe_cuda(cudaSetDevice(device_ordinal_));
  if (IsBitwiseOp(op)) {
    BitwiseAllReduce(send_receive_buffer, count, data_type, op);
  } else {
    dh::safe_nccl(ncclAllReduce(send_receive_buffer, send_receive_buffer, count,
                                GetNcclDataType(data_type), GetNcclRedOp(op), nccl_comm_,
                                cuda_stream_));
  }
  allreduce_bytes_ += count * GetTypeSize(data_type);
  allreduce_calls_ += 1;
}

void NcclDeviceCommunicator::AllGatherV(void const *send_buffer, size_t length_bytes,
                                        std::vector<std::size_t> *segments,
                                        dh::caching_device_vector<char> *receive_buffer) {
  if (world_size_ == 1) {
    return;
  }

  dh::safe_cuda(cudaSetDevice(device_ordinal_));

  segments->clear();
  segments->resize(world_size_, 0);
  segments->at(rank_) = length_bytes;
  Allreduce(segments->data(), segments->size(), DataType::kUInt64, Operation::kMax);
  auto total_bytes = std::accumulate(segments->cbegin(), segments->cend(), 0UL);
  receive_buffer->resize(total_bytes);

  size_t offset = 0;
  dh::safe_nccl(ncclGroupStart());
  for (int32_t i = 0; i < world_size_; ++i) {
    size_t as_bytes = segments->at(i);
    dh::safe_nccl(ncclBroadcast(send_buffer, receive_buffer->data().get() + offset, as_bytes,
                                ncclChar, i, nccl_comm_, cuda_stream_));
    offset += as_bytes;
  }
  dh::safe_nccl(ncclGroupEnd());
}

void NcclDeviceCommunicator::Synchronize() {
  if (world_size_ == 1) {
    return;
  }
  dh::safe_cuda(cudaSetDevice(device_ordinal_));
  dh::safe_cuda(cudaStreamSynchronize(cuda_stream_));
}

}  // namespace collective
}  // namespace xgboost
#endif
