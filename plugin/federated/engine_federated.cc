#define NOMINMAX
#include <mpi.h>
#include <rabit/base.h>

#include <cstdio>
#include <string>
#include <vector>

#include "federated_client.h"
#include "rabit/internal/engine.h"
#include "rabit/internal/utils.h"

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
    // Command line argument override.
    for (int i = 0; i < argc; ++i) {
      std::string key_value = argv[i];
      auto delimiter = key_value.find('=');
      if (delimiter != std::string::npos) {
        SetParam(key_value.substr(0, delimiter), key_value.substr(delimiter + 1));
      }
    }
    utils::Printf("Connecting to federated server %s, world size %d, rank %d",
                  server_address_.c_str(), world_size_, rank_);
    client_.reset(new xgboost::federated::FederatedClient(server_address_, rank_));
  }

  void Allgather(void *sendrecvbuf_, size_t total_size, size_t slice_begin, size_t slice_end,
                 size_t size_prev_slice) override {
    throw std::logic_error("FederatedEngine:: Allgather is not supported");
  }

  void Allreduce(void *sendrecvbuf_, size_t type_nbytes, size_t count, ReduceFunction reducer,
                 PreprocFunction prepare_fun, void *prepare_arg) override {
    throw std::logic_error("FederatedEngine:: Allreduce is not supported, use Allreduce_ instead");
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Allreduce_(void *sendrecvbuf, size_t size, mpi::DataType dtype, mpi::OpType op) {
    auto *buffer = reinterpret_cast<char *>(sendrecvbuf);
    std::string send_buffer(buffer, size);
    auto const receive_buffer = client_->Allreduce(send_buffer, GetDataType(dtype), GetOp(op));
    receive_buffer.copy(buffer, size);
  }

  int GetRingPrevRank() const override {
    throw std::logic_error("FederatedEngine:: GetRingPrevRank is not supported");
  }

  void Broadcast(void *sendrecvbuf, size_t size, int root) override {
    auto *buffer = reinterpret_cast<char *>(sendrecvbuf);
    std::string send_buffer(buffer, size);
    auto const receive_buffer = client_->Broadcast(send_buffer, root);
    if (rank_ != root) {
      receive_buffer.copy(buffer, size);
    }
  }

  int LoadCheckPoint(Serializable *global_model, Serializable *local_model = nullptr) override {
    return 0;
  }

  void CheckPoint(const Serializable *global_model,
                  const Serializable *local_model = nullptr) override {
    version_number_ += 1;
  }

  void LazyCheckPoint(const Serializable *global_model) override { version_number_ += 1; }

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
    if (GetRank() == 0) {
      utils::Printf("%s", msg.c_str());
    }
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
    }
  }

  std::vector<std::string> const env_vars_{"FEDERATED_SERVER_ADDRESS", "FEDERATED_WORLD_SIZE",
                                           "FEDERATED_RANK"};
  std::string server_address_{"localhost:9091"};
  int world_size_{1};
  int rank_{0};
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
    fprintf(stderr, " failed in Federated Init %s\n", e.what());
    return false;
  }
}

/*! \brief finalize synchronization module */
bool Finalize() { return true; }

/*! \brief singleton method to get engine */
IEngine *GetEngine() { return &engine; }

// perform in-place allreduce, on sendrecvbuf
void Allreduce_(void *sendrecvbuf, size_t type_nbytes, size_t count, IEngine::ReduceFunction red,
                mpi::DataType dtype, mpi::OpType op, IEngine::PreprocFunction prepare_fun,
                void *prepare_arg) {
  if (prepare_fun != nullptr) prepare_fun(prepare_arg);
  engine.Allreduce_(sendrecvbuf, type_nbytes * count, dtype, op);
}

// code for reduce handle
ReduceHandle::ReduceHandle(void) : handle_(NULL), redfunc_(NULL), htype_(NULL) {}

ReduceHandle::~ReduceHandle(void) {
  /* !WARNING!

     A handle can be held by a tree method/Learner from xgboost.  The booster might not be
     freed until program exit, while (good) users call rabit.finalize() before reaching
     the end of program.  So op->Free() might be called after finalization and results
     into following error:

      ```
        Attempting to use an MPI routine after finalizing MPICH
      ```

     Here we skip calling Free if MPI has already been finalized to workaround the issue.
     It can be a potential leak of memory.  The best way to resolve it is to eliminate all
     use of long living handle.
  */
  int finalized = 0;
  //  CHECK_EQ(MPI_Finalized(&finalized), MPI_SUCCESS);
  if (handle_ != NULL) {
    //    MPI::Op *op = reinterpret_cast<MPI::Op *>(handle_);
    //    if (!finalized) {
    //      op->Free();
    //    }
    //    delete op;
  }
  if (htype_ != NULL) {
    MPI::Datatype *dtype = reinterpret_cast<MPI::Datatype *>(htype_);
    if (!finalized) {
      //      dtype->Free();
    }
    //    delete dtype;
  }
}

int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return 0;
  //  return dtype.Get_size();
}

void ReduceHandle::Init(IEngine::ReduceFunction redfunc, size_t type_nbytes) {
  utils::Assert(handle_ == NULL, "cannot initialize reduce handle twice");
  if (type_nbytes != 0) {
    //    MPI::Datatype *dtype = new MPI::Datatype();
    //    if (type_nbytes % 8 == 0) {
    //      *dtype = MPI::LONG.Create_contiguous(type_nbytes / sizeof(long));  // NOLINT(*)
    //    } else if (type_nbytes % 4 == 0) {
    //      *dtype = MPI::INT.Create_contiguous(type_nbytes / sizeof(int));
    //    } else {
    //      *dtype = MPI::CHAR.Create_contiguous(type_nbytes);
    //    }
    //    dtype->Commit();
    created_type_nbytes_ = type_nbytes;
    //    htype_ = dtype;
  }
  //  MPI::Op *op = new MPI::Op();
  //  MPI::User_function *pf = redfunc;
  //  op->Init(pf, true);
  //  handle_ = op;
}

void ReduceHandle::Allreduce(void *sendrecvbuf, size_t type_nbytes, size_t count,
                             IEngine::PreprocFunction prepare_fun, void *prepare_arg) {
  utils::Assert(handle_ != NULL, "must initialize handle to call AllReduce");
  //  MPI::Op *op = reinterpret_cast<MPI::Op *>(handle_);
  //  MPI::Datatype *dtype = reinterpret_cast<MPI::Datatype *>(htype_);
  //  if (created_type_nbytes_ != type_nbytes || dtype == NULL) {
  //    if (dtype == NULL) {
  //      dtype = new MPI::Datatype();
  //    } else {
  //      dtype->Free();
  //    }
  //    if (type_nbytes % 8 == 0) {
  //      *dtype = MPI::LONG.Create_contiguous(type_nbytes / sizeof(long));  // NOLINT(*)
  //    } else if (type_nbytes % 4 == 0) {
  //      *dtype = MPI::INT.Create_contiguous(type_nbytes / sizeof(int));
  //    } else {
  //      *dtype = MPI::CHAR.Create_contiguous(type_nbytes);
  //    }
  //    dtype->Commit();
  //    created_type_nbytes_ = type_nbytes;
  //  }
  //  if (prepare_fun != NULL) prepare_fun(prepare_arg);
  //  MPI::COMM_WORLD.Allreduce(MPI_IN_PLACE, sendrecvbuf, count, *dtype, *op);
}

}  // namespace engine
}  // namespace rabit
