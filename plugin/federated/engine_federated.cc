/*!
 * Copyright 2022 XGBoost contributors
 */
#include <cstdio>
#include <fstream>
#include <sstream>

#include "federated_client.h"
#include "rabit/internal/engine.h"
#include "rabit/internal/utils.h"

namespace MPI {  // NOLINT
// MPI data type to be compatible with existing MPI interface
class Datatype {
 public:
  size_t type_size;
  explicit Datatype(size_t type_size) : type_size(type_size) {}
};
}  // namespace MPI

namespace rabit {
namespace engine {

/*! \brief implementation of engine using federated learning */
class FederatedEngine : public IEngine {
 public:
  void Init(int argc, char *argv[]) {
    // Parse environment variables first.
    for (auto const &env_var : env_vars_) {
      char const *value = getenv(env_var.c_str());
      if (value != nullptr) {
        SetParam(env_var, value);
      }
    }
    // Command line argument overrides.
    for (int i = 0; i < argc; ++i) {
      std::string const key_value = argv[i];
      auto const delimiter = key_value.find('=');
      if (delimiter != std::string::npos) {
        SetParam(key_value.substr(0, delimiter), key_value.substr(delimiter + 1));
      }
    }
    utils::Printf("Connecting to federated server %s, world size %d, rank %d",
                  server_address_.c_str(), world_size_, rank_);
    client_.reset(new xgboost::federated::FederatedClient(server_address_, rank_, server_cert_,
                                                          client_key_, client_cert_));
  }

  void Finalize() { client_.reset(); }

  void Allgather(void *sendrecvbuf, size_t total_size, size_t slice_begin, size_t slice_end,
                 size_t size_prev_slice) override {
    throw std::logic_error("FederatedEngine:: Allgather is not supported");
  }

  std::string Allgather(void *sendbuf, size_t total_size) {
    std::string const send_buffer(reinterpret_cast<char *>(sendbuf), total_size);
    return client_->Allgather(send_buffer);
  }

  void Allreduce(void *sendrecvbuf, size_t type_nbytes, size_t count, ReduceFunction reducer,
                 PreprocFunction prepare_fun, void *prepare_arg) override {
    throw std::logic_error("FederatedEngine:: Allreduce is not supported, use Allreduce_ instead");
  }

  void Allreduce(void *sendrecvbuf, size_t size, mpi::DataType dtype, mpi::OpType op) {
    auto *buffer = reinterpret_cast<char *>(sendrecvbuf);
    std::string const send_buffer(buffer, size);
    auto const receive_buffer = client_->Allreduce(send_buffer, GetDataType(dtype), GetOp(op));
    receive_buffer.copy(buffer, size);
  }

  int GetRingPrevRank() const override {
    throw std::logic_error("FederatedEngine:: GetRingPrevRank is not supported");
  }

  void Broadcast(void *sendrecvbuf, size_t size, int root) override {
    if (world_size_ == 1) return;
    auto *buffer = reinterpret_cast<char *>(sendrecvbuf);
    std::string const send_buffer(buffer, size);
    auto const receive_buffer = client_->Broadcast(send_buffer, root);
    if (rank_ != root) {
      receive_buffer.copy(buffer, size);
    }
  }

  int LoadCheckPoint() override {
    return 0;
  }

  void CheckPoint() override {
    version_number_ += 1;
  }

  int VersionNumber() const override { return version_number_; }

  /*! \brief get rank of current node */
  int GetRank() const override { return rank_; }

  /*! \brief get total number of */
  int GetWorldSize() const override { return world_size_; }

  /*! \brief whether it is distributed */
  bool IsDistributed() const override { return true; }

  /*! \brief get the host name of current node */
  std::string GetHost() const override { return "rank" + std::to_string(rank_); }

  void TrackerPrint(const std::string &msg) override {
    // simply print information into the tracker
    utils::Printf("%s", msg.c_str());
  }

 private:
  /** @brief Transform mpi::DataType to xgboost::federated::DataType. */
  static xgboost::federated::DataType GetDataType(mpi::DataType data_type) {
    switch (data_type) {
      case mpi::kChar:
        return xgboost::federated::CHAR;
      case mpi::kUChar:
        return xgboost::federated::UCHAR;
      case mpi::kInt:
        return xgboost::federated::INT;
      case mpi::kUInt:
        return xgboost::federated::UINT;
      case mpi::kLong:
        return xgboost::federated::LONG;
      case mpi::kULong:
        return xgboost::federated::ULONG;
      case mpi::kFloat:
        return xgboost::federated::FLOAT;
      case mpi::kDouble:
        return xgboost::federated::DOUBLE;
      case mpi::kLongLong:
        return xgboost::federated::LONGLONG;
      case mpi::kULongLong:
        return xgboost::federated::ULONGLONG;
    }
    utils::Error("unknown mpi::DataType");
    return xgboost::federated::CHAR;
  }

  /** @brief Transform mpi::OpType to enum to MPI OP */
  static xgboost::federated::ReduceOperation GetOp(mpi::OpType op_type) {
    switch (op_type) {
      case mpi::kMax:
        return xgboost::federated::MAX;
      case mpi::kMin:
        return xgboost::federated::MIN;
      case mpi::kSum:
        return xgboost::federated::SUM;
      case mpi::kBitwiseOR:
        utils::Error("Bitwise OR is not supported");
        return xgboost::federated::MAX;
    }
    utils::Error("unknown mpi::OpType");
    return xgboost::federated::MAX;
  }

  void SetParam(std::string const &name, std::string const &val) {
    if (!strcasecmp(name.c_str(), "FEDERATED_SERVER_ADDRESS")) {
      server_address_ = val;
    } else if (!strcasecmp(name.c_str(), "FEDERATED_WORLD_SIZE")) {
      world_size_ = std::stoi(val);
    } else if (!strcasecmp(name.c_str(), "FEDERATED_RANK")) {
      rank_ = std::stoi(val);
    } else if (!strcasecmp(name.c_str(), "FEDERATED_SERVER_CERT")) {
      server_cert_ = ReadFile(val);
    } else if (!strcasecmp(name.c_str(), "FEDERATED_CLIENT_KEY")) {
      client_key_ = ReadFile(val);
    } else if (!strcasecmp(name.c_str(), "FEDERATED_CLIENT_CERT")) {
      client_cert_ = ReadFile(val);
    }
  }

  static std::string ReadFile(std::string const &path) {
    auto stream = std::ifstream(path.data());
    std::ostringstream out;
    out << stream.rdbuf();
    return out.str();
  }

  // clang-format off
  std::vector<std::string> const env_vars_{
      "FEDERATED_SERVER_ADDRESS",
      "FEDERATED_WORLD_SIZE",
      "FEDERATED_RANK",
      "FEDERATED_SERVER_CERT",
      "FEDERATED_CLIENT_KEY",
      "FEDERATED_CLIENT_CERT" };
  // clang-format on
  std::string server_address_{"localhost:9091"};
  int world_size_{1};
  int rank_{0};
  std::string server_cert_{};
  std::string client_key_{};
  std::string client_cert_{};
  std::unique_ptr<xgboost::federated::FederatedClient> client_{};
  int version_number_{0};
};

// Singleton federated engine.
FederatedEngine engine;  // NOLINT(cert-err58-cpp)

/*! \brief initialize the synchronization module */
bool Init(int argc, char *argv[]) {
  try {
    engine.Init(argc, argv);
    return true;
  } catch (std::exception const &e) {
    fprintf(stderr, " failed in federated Init %s\n", e.what());
    return false;
  }
}

/*! \brief finalize synchronization module */
bool Finalize() {
  try {
    engine.Finalize();
    return true;
  } catch (const std::exception &e) {
    fprintf(stderr, "failed in federated shutdown %s\n", e.what());
    return false;
  }
}

/*! \brief singleton method to get engine */
IEngine *GetEngine() { return &engine; }

// perform in-place allreduce, on sendrecvbuf
void Allreduce_(void *sendrecvbuf, size_t type_nbytes, size_t count, IEngine::ReduceFunction red,
                mpi::DataType dtype, mpi::OpType op, IEngine::PreprocFunction prepare_fun,
                void *prepare_arg) {
  if (prepare_fun != nullptr) prepare_fun(prepare_arg);
  if (engine.GetWorldSize() == 1) return;
  engine.Allreduce(sendrecvbuf, type_nbytes * count, dtype, op);
}
}  // namespace engine
}  // namespace rabit
