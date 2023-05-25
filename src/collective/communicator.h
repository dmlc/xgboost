/*!
 * Copyright 2022 XGBoost contributors
 */
#pragma once
#include <xgboost/json.h>
#include <xgboost/logging.h>

#include <memory>
#include <string>

namespace xgboost {
namespace collective {

/** @brief Defines the integral and floating data types. */
enum class DataType {
  kInt8 = 0,
  kUInt8 = 1,
  kInt32 = 2,
  kUInt32 = 3,
  kInt64 = 4,
  kUInt64 = 5,
  kFloat = 6,
  kDouble = 7
};

/** @brief Get the size of the data type. */
inline std::size_t GetTypeSize(DataType data_type) {
  std::size_t size{0};
  switch (data_type) {
    case DataType::kInt8:
      size = sizeof(std::int8_t);
      break;
    case DataType::kUInt8:
      size = sizeof(std::uint8_t);
      break;
    case DataType::kInt32:
      size = sizeof(std::int32_t);
      break;
    case DataType::kUInt32:
      size = sizeof(std::uint32_t);
      break;
    case DataType::kInt64:
      size = sizeof(std::int64_t);
      break;
    case DataType::kUInt64:
      size = sizeof(std::uint64_t);
      break;
    case DataType::kFloat:
      size = sizeof(float);
      break;
    case DataType::kDouble:
      size = sizeof(double);
      break;
    default:
      LOG(FATAL) << "Unknown data type.";
  }
  return size;
}

/** @brief Defines the reduction operation. */
enum class Operation {
  kMax = 0,
  kMin = 1,
  kSum = 2,
  kBitwiseAND = 3,
  kBitwiseOR = 4,
  kBitwiseXOR = 5
};

class DeviceCommunicator;

enum class CommunicatorType { kUnknown, kRabit, kFederated, kInMemory };

/** \brief Case-insensitive string comparison. */
inline int CompareStringsCaseInsensitive(const char *s1, const char *s2) {
#ifdef _MSC_VER
  return _stricmp(s1, s2);
#else   // _MSC_VER
  return strcasecmp(s1, s2);
#endif  // _MSC_VER
}

/**
 * @brief A communicator class that handles collective communication.
 */
class Communicator {
 public:
  /**
   * @brief Initialize the communicator. This can only be done once.
   *
   * @param config JSON configuration for the communicator.
   */
  static void Init(Json const &config);

  /** @brief Finalize the communicator. */
  static void Finalize();

  /** @brief Get the communicator instance. */
  static Communicator *Get() { return communicator_.get(); }

#if defined(XGBOOST_USE_CUDA)
  /**
   * @brief Get the device communicator.
   *
   * @param device_ordinal ID of the device.
   * @return An instance of device communicator.
   */
  static DeviceCommunicator *GetDevice(int device_ordinal);
#endif

  virtual ~Communicator() = default;

  /** @brief Get the total number of processes. */
  int GetWorldSize() const { return world_size_; }

  /** @brief Get the rank of the current processes. */
  int GetRank() const { return rank_; }

  /** @brief Whether the communicator is running in distributed mode. */
  virtual bool IsDistributed() const = 0;

  /** @brief Whether the communicator is running in federated mode. */
  virtual bool IsFederated() const = 0;

  /**
   * @brief Gathers data from all processes and distributes it to all processes.
   *
   * This assumes all ranks have the same size, and input data has been sliced into the
   * corresponding position.
   *
   * @param send_receive_buffer Buffer storing the data.
   * @param size                Size of the data in bytes.
   */
  virtual void AllGather(void *send_receive_buffer, std::size_t size) = 0;

  /**
   * @brief Combines values from all processes and distributes the result back to all processes.
   *
   * @param send_receive_buffer Buffer storing the data.
   * @param count               Number of elements in the buffer.
   * @param data_type           Data type stored in the buffer.
   * @param op                  The operation to perform.
   */
  virtual void AllReduce(void *send_receive_buffer, std::size_t count, DataType data_type,
                         Operation op) = 0;

  /**
   * @brief Broadcasts a message from the process with rank `root` to all other processes of the
   * group.
   *
   * @param send_receive_buffer Buffer storing the data.
   * @param size                Size of the data in bytes.
   * @param root                Rank of broadcast root.
   */
  virtual void Broadcast(void *send_receive_buffer, std::size_t size, int root) = 0;

  /**
   * @brief Gets the name of the processor.
   */
  virtual std::string GetProcessorName() = 0;

  /**
   * @brief Prints the message.
   */
  virtual void Print(std::string const &message) = 0;

  /** @brief Get the communicator type from environment variables. Visible for testing. */
  static CommunicatorType GetTypeFromEnv() {
    auto *env = std::getenv("XGBOOST_COMMUNICATOR");
    if (env != nullptr) {
      return StringToType(env);
    } else {
      return CommunicatorType::kUnknown;
    }
  }

  /** @brief Get the communicator type from runtime configuration. Visible for testing. */
  static CommunicatorType GetTypeFromConfig(Json const &config) {
    auto const &j_upper = config["XGBOOST_COMMUNICATOR"];
    if (IsA<String const>(j_upper)) {
      return StringToType(get<String const>(j_upper).c_str());
    }
    auto const &j_lower = config["xgboost_communicator"];
    if (IsA<String const>(j_lower)) {
      return StringToType(get<String const>(j_lower).c_str());
    }
    return CommunicatorType::kUnknown;
  }

 protected:
  /**
   * @brief Construct a new communicator.
   *
   * @param world_size Total number of processes.
   * @param rank       Rank of the current process.
   */
  Communicator(int world_size, int rank) : world_size_(world_size), rank_(rank) {
    if (world_size < 1) {
      LOG(FATAL) << "World size " << world_size << " is less than 1.";
    }
    if (rank < 0) {
      LOG(FATAL) << "Rank " << rank << " is less than 0.";
    }
    if (rank >= world_size) {
      LOG(FATAL) << "Rank " << rank << " is greater than world_size - 1: " << world_size - 1 << ".";
    }
  }

  /**
   * @brief Shuts down the communicator.
   */
  virtual void Shutdown() = 0;

 private:
  static CommunicatorType StringToType(char const *str) {
    CommunicatorType result = CommunicatorType::kUnknown;
    if (!CompareStringsCaseInsensitive("rabit", str)) {
      result = CommunicatorType::kRabit;
    } else if (!CompareStringsCaseInsensitive("federated", str)) {
      result = CommunicatorType::kFederated;
    } else if (!CompareStringsCaseInsensitive("in-memory", str)) {
      result = CommunicatorType::kInMemory;
    } else {
      LOG(FATAL) << "Unknown communicator type " << str;
    }
    return result;
  }

  static thread_local std::unique_ptr<Communicator> communicator_;
  static thread_local CommunicatorType type_;
#if defined(XGBOOST_USE_CUDA)
  static thread_local std::unique_ptr<DeviceCommunicator> device_communicator_;
#endif

  int const world_size_;
  int const rank_;
};

}  // namespace collective
}  // namespace xgboost
