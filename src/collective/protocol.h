/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#pragma once
#include <cstdint>  // for int32_t
#include <string>   // for string
#include <utility>  // for move

#include "xgboost/collective/result.h"  // for Result
#include "xgboost/collective/socket.h"  // for TCPSocket
#include "xgboost/json.h"               // for Json

namespace xgboost::collective::proto {
struct PeerInfo {
  std::string host;
  std::int32_t port{-1};
  std::int32_t rank{-1};

  PeerInfo() = default;
  PeerInfo(std::string host, std::int32_t port, std::int32_t rank)
      : host{std::move(host)}, port{port}, rank{rank} {}

  explicit PeerInfo(Json const& peer)
      : host{get<String>(peer["host"])},
        port{static_cast<std::int32_t>(get<Integer const>(peer["port"]))},
        rank{static_cast<std::int32_t>(get<Integer const>(peer["rank"]))} {}

  [[nodiscard]] Json ToJson() const {
    Json info{Object{}};
    info["rank"] = rank;
    info["host"] = String{host};
    info["port"] = Integer{port};
    return info;
  }

  [[nodiscard]] auto HostPort() const { return host + ":" + std::to_string(this->port); }
};

struct Magic {
  static constexpr std::int32_t kMagic = 0xff99;

  [[nodiscard]] Result Verify(xgboost::collective::TCPSocket* p_sock) {
    std::int32_t magic{kMagic};
    std::size_t n_sent{0};
    return Success() << [&] {
      return p_sock->SendAll(&magic, sizeof(magic), &n_sent);
    } << [&] {
      if (n_sent != sizeof(magic)) {
        return Fail("Failed to verify.");
      }
      return Success();
    } << [&] {
      magic = 0;
      return p_sock->RecvAll(&magic, sizeof(magic), &n_sent);
    } << [&] {
      if (n_sent != sizeof(magic)) {
        return Fail("Failed to verify.");
      }
      if (magic != kMagic) {
        return xgboost::collective::Fail("Invalid verification number.");
      }
      return Success();
    };
  }
};

// Basic commands for communication between workers and the tracker.
enum class CMD : std::int32_t {
  kInvalid = 0,
  kStart = 1,
  kShutdown = 2,
  kError = 3,
  kPrint = 4,
};

struct Connect {
  [[nodiscard]] Result WorkerSend(TCPSocket* tracker, std::int32_t world, std::int32_t rank,
                                  std::string task_id) const {
    Json jinit{Object{}};
    jinit["world_size"] = Integer{world};
    jinit["rank"] = Integer{rank};
    jinit["task_id"] = String{task_id};
    std::string msg;
    Json::Dump(jinit, &msg);
    auto n_bytes = tracker->Send(msg);
    if (n_bytes != msg.size()) {
      return Fail("Failed to send init command from worker.");
    }
    return Success();
  }
  [[nodiscard]] Result TrackerRecv(TCPSocket* sock, std::int32_t* world, std::int32_t* rank,
                                   std::string* task_id) const {
    std::string init;
    auto rc = sock->Recv(&init);
    if (!rc.OK()) {
      return Fail("Connect protocol failed.", std::move(rc));
    }
    auto jinit = Json::Load(StringView{init});
    *world = get<Integer const>(jinit["world_size"]);
    *rank = get<Integer const>(jinit["rank"]);
    *task_id = get<String const>(jinit["task_id"]);
    return Success();
  }
};

class Start {
 private:
  [[nodiscard]] Result TrackerSend(std::int32_t world, TCPSocket* worker) const {
    Json jcmd{Object{}};
    jcmd["world_size"] = Integer{world};
    auto scmd = Json::Dump(jcmd);
    auto n_bytes = worker->Send(scmd);
    if (n_bytes != scmd.size()) {
      return Fail("Failed to send init command from tracker.");
    }
    return Success();
  }

 public:
  [[nodiscard]] Result WorkerSend(std::int32_t lport, TCPSocket* tracker,
                                  std::int32_t eport) const {
    Json jcmd{Object{}};
    jcmd["cmd"] = Integer{static_cast<std::int32_t>(CMD::kStart)};
    jcmd["port"] = Integer{lport};
    jcmd["error_port"] = Integer{eport};
    auto scmd = Json::Dump(jcmd);
    auto n_bytes = tracker->Send(scmd);
    if (n_bytes != scmd.size()) {
      return Fail("Failed to send init command from worker.");
    }
    return Success();
  }
  [[nodiscard]] Result WorkerRecv(TCPSocket* tracker, std::int32_t* p_world) const {
    std::string scmd;
    auto rc = tracker->Recv(&scmd);
    if (!rc.OK()) {
      return Fail("Failed to recv init command from tracker.", std::move(rc));
    }
    auto jcmd = Json::Load(scmd);
    auto world = get<Integer const>(jcmd["world_size"]);
    if (world <= 0) {
      return Fail("Invalid world size.");
    }
    *p_world = world;
    return rc;
  }
  [[nodiscard]] Result TrackerHandle(Json jcmd, std::int32_t* recv_world, std::int32_t world,
                                     std::int32_t* p_port, TCPSocket* p_sock,
                                     std::int32_t* eport) const {
    *p_port = get<Integer const>(jcmd["port"]);
    if (*p_port <= 0) {
      return Fail("Invalid port.");
    }
    if (*recv_world != -1) {
      return Fail("Invalid initialization sequence.");
    }
    *recv_world = world;
    *eport = get<Integer const>(jcmd["error_port"]);
    return TrackerSend(world, p_sock);
  }
};

// Protocol for communicating with the tracker for printing message.
struct Print {
  [[nodiscard]] Result WorkerSend(TCPSocket* tracker, std::string msg) const {
    Json jcmd{Object{}};
    jcmd["cmd"] = Integer{static_cast<std::int32_t>(CMD::kPrint)};
    jcmd["msg"] = String{std::move(msg)};
    auto scmd = Json::Dump(jcmd);
    auto n_bytes = tracker->Send(scmd);
    if (n_bytes != scmd.size()) {
      return Fail("Failed to send print command from worker.");
    }
    return Success();
  }
  [[nodiscard]] Result TrackerHandle(Json jcmd, std::string* p_msg) const {
    if (!IsA<String>(jcmd["msg"])) {
      return Fail("Invalid print command.");
    }
    auto msg = get<String const>(jcmd["msg"]);
    *p_msg = msg;
    return Success();
  }
};

// Protocol for communicating with the tracker during error.
struct ErrorCMD {
  [[nodiscard]] Result WorkerSend(TCPSocket* tracker, Result const& res) const {
    auto msg = res.Report();
    auto code = res.Code().value();
    Json jcmd{Object{}};
    jcmd["msg"] = String{std::move(msg)};
    jcmd["code"] = Integer{code};
    jcmd["cmd"] = Integer{static_cast<std::int32_t>(CMD::kError)};
    auto scmd = Json::Dump(jcmd);
    auto n_bytes = tracker->Send(scmd);
    if (n_bytes != scmd.size()) {
      return Fail("Failed to send error command from worker.");
    }
    return Success();
  }
  [[nodiscard]] Result TrackerHandle(Json jcmd, std::string* p_msg, int* p_code) const {
    if (!IsA<String>(jcmd["msg"]) || !IsA<Integer>(jcmd["code"])) {
      return Fail("Invalid error command.");
    }
    auto msg = get<String const>(jcmd["msg"]);
    auto code = get<Integer const>(jcmd["code"]);
    *p_msg = msg;
    *p_code = code;
    return Success();
  }
};

// Protocol for communicating with the tracker during shutdown.
struct ShutdownCMD {
  [[nodiscard]] Result Send(TCPSocket* peer) const {
    Json jcmd{Object{}};
    jcmd["cmd"] = Integer{static_cast<std::int32_t>(proto::CMD::kShutdown)};
    auto scmd = Json::Dump(jcmd);
    auto n_bytes = peer->Send(scmd);
    if (n_bytes != scmd.size()) {
      return Fail("Failed to send shutdown command from worker.");
    }
    return Success();
  }
};

// Protocol for communicating with the local error handler during error or shutdown. Only
// one protocol that doesn't have the tracker involved.
struct Error {
  constexpr static std::int32_t ShutdownSignal() { return 0; }
  constexpr static std::int32_t ErrorSignal() { return -1; }

  [[nodiscard]] Result SignalError(TCPSocket* worker) const {
    std::int32_t err{ErrorSignal()};
    std::size_t n_sent{0};
    return Success() << [&] {
      return worker->SendAll(&err, sizeof(err), &n_sent);
    } << [&] {
      if (n_sent == sizeof(err)) {
        return Success();
      }
      return Fail("Failed to send error signal");
    };
  }
  // self is localhost, we are sending the signal to the error handling thread for it to
  // close.
  [[nodiscard]] Result SignalShutdown(TCPSocket* self) const {
    std::int32_t err{ShutdownSignal()};
    std::size_t n_sent{0};
    return Success() << [&] {
      return self->SendAll(&err, sizeof(err), &n_sent);
    } << [&] {
      if (n_sent == sizeof(err)) {
        return Success();
      }
      return Fail("Failed to send shutdown signal");
    };
  }
  // get signal, either for error or for shutdown.
  [[nodiscard]] Result RecvSignal(TCPSocket* peer, bool* p_is_error) const {
    std::int32_t err{ShutdownSignal()};
    std::size_t n_recv{0};
    return Success() << [&] {
      return peer->RecvAll(&err, sizeof(err), &n_recv);
    } << [&] {
      if (n_recv == sizeof(err)) {
        *p_is_error = err == 1;
        return Success();
      }
      return Fail("Failed to receive error signal.");
    };
  }
};
}  // namespace xgboost::collective::proto
