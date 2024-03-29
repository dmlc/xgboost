/**
 * Copyright 2022-2024, XGBoost contributors
 */
#pragma once
#include <string>
#include <vector>

#include "communicator.h"

namespace xgboost {
namespace collective {

/**
 * \brief Initialize the collective communicator.
 *
 *  Currently the communicator API is experimental, function signatures may change in the future
 *  without notice.
 *
 *  Call this once before using anything.
 *
 *  The additional configuration is not required. Usually the communicator will detect settings
 *  from environment variables.
 *
 * \param json_config JSON encoded configuration. Accepted JSON keys are:
 *   - xgboost_communicator: The type of the communicator. Can be set as an environment variable.
 *     * rabit: Use Rabit. This is the default if the type is unspecified.
 *     * mpi: Use MPI.
 *     * federated: Use the gRPC interface for Federated Learning.
 * Only applicable to the Rabit communicator (these are case-sensitive):
 *   - rabit_tracker_uri: Hostname of the tracker.
 *   - rabit_tracker_port: Port number of the tracker.
 *   - rabit_task_id: ID of the current task, can be used to obtain deterministic rank assignment.
 *   - rabit_world_size: Total number of workers.
 *   - rabit_hadoop_mode: Enable Hadoop support.
 *   - rabit_tree_reduce_minsize: Minimal size for tree reduce.
 *   - rabit_reduce_ring_mincount: Minimal count to perform ring reduce.
 *   - rabit_reduce_buffer: Size of the reduce buffer.
 *   - rabit_bootstrap_cache: Size of the bootstrap cache.
 *   - rabit_debug: Enable debugging.
 *   - rabit_timeout: Enable timeout.
 *   - rabit_timeout_sec: Timeout in seconds.
 *   - rabit_enable_tcp_no_delay: Enable TCP no delay on Unix platforms.
 * Only applicable to the Rabit communicator (these are case-sensitive, and can be set as
 * environment variables):
 *   - DMLC_TRACKER_URI: Hostname of the tracker.
 *   - DMLC_TRACKER_PORT: Port number of the tracker.
 *   - DMLC_TASK_ID: ID of the current task, can be used to obtain deterministic rank assignment.
 *   - DMLC_ROLE: Role of the current task, "worker" or "server".
 *   - DMLC_NUM_ATTEMPT: Number of attempts after task failure.
 *   - DMLC_WORKER_CONNECT_RETRY: Number of retries to connect to the tracker.
 * Only applicable to the Federated communicator (use upper case for environment variables, use
 * lower case for runtime configuration):
 *   - federated_server_address: Address of the federated server.
 *   - federated_world_size: Number of federated workers.
 *   - federated_rank: Rank of the current worker.
 *   - federated_server_cert: Server certificate file path. Only needed for the SSL mode.
 *   - federated_client_key: Client key file path. Only needed for the SSL mode.
 *   - federated_client_cert: Client certificate file path. Only needed for the SSL mode.
 */
inline void Init(Json const &config) { Communicator::Init(config); }

/*!
 * \brief Finalize the collective communicator.
 *
 * Call this function after you finished all jobs.
 */
inline void Finalize() { Communicator::Finalize(); }

/*!
 * \brief Get rank of current process.
 *
 * \return Rank of the worker.
 */
inline int GetRank() { return Communicator::Get()->GetRank(); }

/*!
 * \brief Get total number of processes.
 *
 * \return Total world size.
 */
inline int GetWorldSize() { return Communicator::Get()->GetWorldSize(); }

/*!
 * \brief Get if the communicator is distributed.
 *
 * \return True if the communicator is distributed.
 */
inline bool IsDistributed() { return Communicator::Get()->IsDistributed(); }

/*!
 * \brief Get if the communicator is federated.
 *
 * \return True if the communicator is federated.
 */
inline bool IsFederated() { return Communicator::Get()->IsFederated(); }

/*!
 * \brief Print the message to the communicator.
 *
 * This function can be used to communicate the information of the progress to the user who monitors
 * the communicator.
 *
 * \param message The message to be printed.
 */
inline void Print(char const *message) { Communicator::Get()->Print(message); }

inline void Print(std::string const &message) { Communicator::Get()->Print(message); }

/*!
 * \brief Get the name of the processor.
 *
 * \return Name of the processor.
 */
inline std::string GetProcessorName() { return Communicator::Get()->GetProcessorName(); }

/*!
 * \brief Broadcast a memory region to all others from root.  This function is NOT thread-safe.
 *
 * Example:
 *   int a = 1;
 *   Broadcast(&a, sizeof(a), root);
 *
 * \param send_receive_buffer Pointer to the send or receive buffer.
 * \param size Size of the data.
 * \param root The process rank to broadcast from.
 */
inline void Broadcast(void *send_receive_buffer, size_t size, int root) {
  Communicator::Get()->Broadcast(send_receive_buffer, size, root);
}

inline void Broadcast(std::string *sendrecv_data, int root) {
  size_t size = sendrecv_data->length();
  Broadcast(&size, sizeof(size), root);
  if (sendrecv_data->length() != size) {
    sendrecv_data->resize(size);
  }
  if (size != 0) {
    Broadcast(&(*sendrecv_data)[0], size * sizeof(char), root);
  }
}

/**
 * @brief Gathers a single value all processes and distributes the result to all processes.
 *
 * @param input The single value.
 */
template <typename T>
inline std::vector<T> Allgather(T const &input) {
  std::string_view str_input{reinterpret_cast<char const *>(&input), sizeof(T)};
  auto const output = Communicator::Get()->AllGather(str_input);
  CHECK_EQ(output.size() % sizeof(T), 0);
  std::vector<T> result(output.size() / sizeof(T));
  std::memcpy(reinterpret_cast<void *>(result.data()), output.data(), output.size());
  return result;
}

/**
 * @brief Gathers data from all processes and distributes it to all processes.
 *
 * This assumes all ranks have the same size.
 *
 * @param input Buffer storing the data.
 */
template <typename T>
inline std::vector<T> Allgather(std::vector<T> const &input) {
  if (input.empty()) {
    return input;
  }
  std::string_view str_input{reinterpret_cast<char const *>(input.data()),
                             input.size() * sizeof(T)};
  auto const output = Communicator::Get()->AllGather(str_input);
  CHECK_EQ(output.size() % sizeof(T), 0);
  std::vector<T> result(output.size() / sizeof(T));
  std::memcpy(reinterpret_cast<void *>(result.data()), output.data(), output.size());
  return result;
}

/**
 * @brief Gathers variable-length data from all processes and distributes it to all processes.
 * @param input Buffer storing the data.
 */
template <typename T>
inline std::vector<T> AllgatherV(std::vector<T> const &input) {
  std::string_view str_input{reinterpret_cast<char const *>(input.data()),
                             input.size() * sizeof(T)};
  auto const output = Communicator::Get()->AllGatherV(str_input);
  CHECK_EQ(output.size() % sizeof(T), 0);
  std::vector<T> result(output.size() / sizeof(T));
  if (!output.empty()) {
    std::memcpy(reinterpret_cast<void *>(result.data()), output.data(), output.size());
  }
  return result;
}

/**
 * @brief Gathers variable-length data from all processes and distributes it to all processes.
 *
 * @param inputs All the inputs from the local worker. The number of inputs can vary
 *               across different workers. Along with which, the size of each vector in
 *               the input can also vary.
 *
 * @return The AllgatherV result, containing vectors from all workers.
 */
[[nodiscard]] std::vector<std::vector<char>> VectorAllgatherV(
    std::vector<std::vector<char>> const &input);

/**
 * @brief Gathers variable-length strings from all processes and distributes them to all processes.
 * @param input Variable-length list of variable-length strings.
 */
inline std::vector<std::string> AllgatherStrings(std::vector<std::string> const &input) {
  std::size_t total_size{0};
  for (auto const &s : input) {
    total_size += s.length() + 1;  // +1 for null-terminators
  }
  std::string flat_string;
  flat_string.reserve(total_size);
  for (auto const &s : input) {
    flat_string.append(s);
    flat_string.push_back('\0');  // Append a null-terminator after each string
  }

  auto const output = Communicator::Get()->AllGatherV(flat_string);

  std::vector<std::string> result;
  std::size_t start_index = 0;
  // Iterate through the output, find each null-terminated substring.
  for (std::size_t i = 0; i < output.size(); i++) {
    if (output[i] == '\0') {
      // Construct a std::string from the char* substring
      result.emplace_back(&output[start_index]);
      // Move to the next substring
      start_index = i + 1;
    }
  }
  return result;
}

/*!
 * \brief Perform in-place allreduce. This function is NOT thread-safe.
 *
 * Example Usage: the following code gives sum of the result
 *     vector<int> data(10);
 *     ...
 *     Allreduce(&data[0], data.size(), DataType:kInt32, Op::kSum);
 *     ...
 * \param send_receive_buffer Buffer for both sending and receiving data.
 * \param count Number of elements to be reduced.
 * \param data_type Enumeration of data type, see xgboost::collective::DataType in communicator.h.
 * \param op Enumeration of operation type, see xgboost::collective::Operation in communicator.h.
 */
inline void Allreduce(void *send_receive_buffer, size_t count, int data_type, int op) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, static_cast<DataType>(data_type),
                                 static_cast<Operation>(op));
}

inline void Allreduce(void *send_receive_buffer, size_t count, DataType data_type, Operation op) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, data_type, op);
}

template <Operation op>
inline void Allreduce(int8_t *send_receive_buffer, size_t count) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kInt8, op);
}

template <Operation op>
inline void Allreduce(uint8_t *send_receive_buffer, size_t count) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kUInt8, op);
}

template <Operation op>
inline void Allreduce(int32_t *send_receive_buffer, size_t count) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kInt32, op);
}

template <Operation op>
inline void Allreduce(uint32_t *send_receive_buffer, size_t count) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kUInt32, op);
}

template <Operation op>
inline void Allreduce(int64_t *send_receive_buffer, size_t count) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kInt64, op);
}

template <Operation op>
inline void Allreduce(uint64_t *send_receive_buffer, size_t count) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kUInt64, op);
}

// Specialization for size_t, which is implementation defined, so it might or might not
// be one of uint64_t/uint32_t/unsigned long long/unsigned long.
template <Operation op, typename T,
          typename = std::enable_if_t<std::is_same<size_t, T>{} && !std::is_same<uint64_t, T>{}> >
inline void Allreduce(T *send_receive_buffer, size_t count) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kUInt64, op);
}

template <Operation op>
inline void Allreduce(float *send_receive_buffer, size_t count) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kFloat, op);
}

template <Operation op>
inline void Allreduce(double *send_receive_buffer, size_t count) {
  Communicator::Get()->AllReduce(send_receive_buffer, count, DataType::kDouble, op);
}
}  // namespace collective
}  // namespace xgboost
