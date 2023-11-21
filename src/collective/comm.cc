/**
 * Copyright 2023, XGBoost Contributors
 */
#include "comm.h"

#include <algorithm>  // for copy
#include <chrono>     // for seconds
#include <cstdlib>    // for exit
#include <memory>     // for shared_ptr
#include <string>     // for string
#include <utility>    // for move, forward

#include "../common/common.h"           // for AssertGPUSupport
#include "allgather.h"                  // for RingAllgather
#include "protocol.h"                   // for kMagic
#include "xgboost/base.h"               // for XGBOOST_STRICT_R_MODE
#include "xgboost/collective/socket.h"  // for TCPSocket
#include "xgboost/json.h"               // for Json, Object
#include "xgboost/string_view.h"        // for StringView

namespace xgboost::collective {
Comm::Comm(std::string const& host, std::int32_t port, std::chrono::seconds timeout,
           std::int32_t retry, std::string task_id)
    : timeout_{timeout},
      retry_{retry},
      tracker_{host, port, -1},
      task_id_{std::move(task_id)},
      loop_{std::shared_ptr<Loop>{new Loop{timeout}}} {}

Result ConnectTrackerImpl(proto::PeerInfo info, std::chrono::seconds timeout, std::int32_t retry,
                          std::string const& task_id, TCPSocket* out, std::int32_t rank,
                          std::int32_t world) {
  // Get information from the tracker
  CHECK(!info.host.empty());
  TCPSocket& tracker = *out;
  return Success() << [&] {
    auto rc = Connect(info.host, info.port, retry, timeout, out);
    if (rc.OK()) {
      return rc;
    } else {
      return Fail("Failed to connect to the tracker.", std::move(rc));
    }
  } << [&] {
    return tracker.NonBlocking(false);
  } << [&] {
    return tracker.RecvTimeout(timeout);
  } << [&] {
    return proto::Magic{}.Verify(&tracker);
  } << [&] {
    return proto::Connect{}.WorkerSend(&tracker, world, rank, task_id);
  } << [&] {
    LOG(INFO) << "Task " << task_id << " connected to the tracker";
    return Success();
  };
}

[[nodiscard]] Result Comm::ConnectTracker(TCPSocket* out) const {
  return ConnectTrackerImpl(this->TrackerInfo(), this->Timeout(), this->retry_, this->task_id_, out,
                            this->Rank(), this->World());
}

[[nodiscard]] Result ConnectWorkers(Comm const& comm, TCPSocket* listener, std::int32_t lport,
                                    proto::PeerInfo ninfo, std::chrono::seconds timeout,
                                    std::int32_t retry,
                                    std::vector<std::shared_ptr<TCPSocket>>* out_workers) {
  auto next = std::make_shared<TCPSocket>();
  auto prev = std::make_shared<TCPSocket>();

  auto rc = Success() << [&] {
    auto rc = Connect(ninfo.host, ninfo.port, retry, timeout, next.get());
    if (!rc.OK()) {
      return Fail("Bootstrap failed to connect to ring next.", std::move(rc));
    }
    return rc;
  } << [&] {
    return next->NonBlocking(true);
  } << [&] {
    SockAddrV4 addr;
    return listener->Accept(prev.get(), &addr);
  } << [&] { return prev->NonBlocking(true); };
  if (!rc.OK()) {
    return rc;
  }

  // exchange host name and port
  std::vector<std::int8_t> buffer(HOST_NAME_MAX * comm.World(), 0);
  auto s_buffer = common::Span{buffer.data(), buffer.size()};
  auto next_host = s_buffer.subspan(HOST_NAME_MAX * comm.Rank(), HOST_NAME_MAX);
  if (next_host.size() < ninfo.host.size()) {
    return Fail("Got an invalid host name.");
  }
  std::copy(ninfo.host.cbegin(), ninfo.host.cend(), next_host.begin());

  auto prev_ch = std::make_shared<Channel>(comm, prev);
  auto next_ch = std::make_shared<Channel>(comm, next);

  auto block = [&] {
    for (auto ch : {prev_ch, next_ch}) {
      auto rc = ch->Block();
      if (!rc.OK()) {
        return rc;
      }
    }
    return Success();
  };

  rc = std::move(rc) << [&] {
    return cpu_impl::RingAllgather(comm, s_buffer, HOST_NAME_MAX, 0, prev_ch, next_ch);
  } << [&] { return block(); };
  if (!rc.OK()) {
    return Fail("Failed to get host names from peers.", std::move(rc));
  }

  std::vector<std::int32_t> peers_port(comm.World(), -1);
  peers_port[comm.Rank()] = ninfo.port;
  rc = std::move(rc) << [&] {
    auto s_ports = common::Span{reinterpret_cast<std::int8_t*>(peers_port.data()),
                                peers_port.size() * sizeof(ninfo.port)};
    return cpu_impl::RingAllgather(comm, s_ports, sizeof(ninfo.port), 0, prev_ch, next_ch);
  } << [&] { return block(); };
  if (!rc.OK()) {
    return Fail("Failed to get the port from peers.", std::move(rc));
  }

  std::vector<proto::PeerInfo> peers(comm.World());
  for (auto r = 0; r < comm.World(); ++r) {
    auto nhost = s_buffer.subspan(HOST_NAME_MAX * r, HOST_NAME_MAX);
    auto nport = peers_port[r];
    auto nrank = BootstrapNext(r, comm.World());

    peers[nrank] = {std::string{reinterpret_cast<char const*>(nhost.data())}, nport, nrank};
  }
  CHECK_EQ(peers[comm.Rank()].port, lport);
  for (auto const& p : peers) {
    CHECK_NE(p.port, -1);
  }

  std::vector<std::shared_ptr<TCPSocket>>& workers = *out_workers;
  workers.resize(comm.World());

  for (std::int32_t r = (comm.Rank() + 1); r < comm.World(); ++r) {
    auto const& peer = peers[r];
    std::shared_ptr<TCPSocket> worker{TCPSocket::CreatePtr(comm.Domain())};
    rc = std::move(rc)
         << [&] { return Connect(peer.host, peer.port, retry, timeout, worker.get()); }
         << [&] { return worker->RecvTimeout(timeout); };
    if (!rc.OK()) {
      return rc;
    }

    auto rank = comm.Rank();
    auto n_bytes = worker->SendAll(&rank, sizeof(comm.Rank()));
    if (n_bytes != sizeof(comm.Rank())) {
      return Fail("Failed to send rank.");
    }
    workers[r] = std::move(worker);
  }

  for (std::int32_t r = 0; r < comm.Rank(); ++r) {
    SockAddrV4 addr;
    auto peer = std::shared_ptr<TCPSocket>(TCPSocket::CreatePtr(comm.Domain()));
    rc = std::move(rc) << [&] { return listener->Accept(peer.get(), &addr); }
                       << [&] { return peer->RecvTimeout(timeout); };
    if (!rc.OK()) {
      return rc;
    }
    std::int32_t rank{-1};
    auto n_bytes = peer->RecvAll(&rank, sizeof(rank));
    if (n_bytes != sizeof(comm.Rank())) {
      return Fail("Failed to recv rank.");
    }
    workers[rank] = std::move(peer);
  }

  for (std::int32_t r = 0; r < comm.World(); ++r) {
    if (r == comm.Rank()) {
      continue;
    }
    CHECK(workers[r]);
  }

  return Success();
}

RabitComm::RabitComm(std::string const& host, std::int32_t port, std::chrono::seconds timeout,
                     std::int32_t retry, std::string task_id, StringView nccl_path)
    : HostComm{std::move(host), port, timeout, retry, std::move(task_id)},
      nccl_path_{std::move(nccl_path)} {
  auto rc = this->Bootstrap(timeout_, retry_, task_id_);
  CHECK(rc.OK()) << rc.Report();
}

#if !defined(XGBOOST_USE_NCCL)
Comm* RabitComm::MakeCUDAVar(Context const*, std::shared_ptr<Coll>) const {
  common::AssertGPUSupport();
  common::AssertNCCLSupport();
  return nullptr;
}
#endif  //  !defined(XGBOOST_USE_NCCL)

[[nodiscard]] Result RabitComm::Bootstrap(std::chrono::seconds timeout, std::int32_t retry,
                                          std::string task_id) {
  TCPSocket tracker;
  std::int32_t world{-1};
  auto rc = ConnectTrackerImpl(this->TrackerInfo(), timeout, retry, task_id, &tracker, this->Rank(),
                               world);
  if (!rc.OK()) {
    return Fail("Bootstrap failed.", std::move(rc));
  }

  this->domain_ = tracker.Domain();

  // Start command
  TCPSocket listener = TCPSocket::Create(tracker.Domain());
  std::int32_t lport = listener.BindHost();
  listener.Listen();

  // create worker for listening to error notice.
  auto domain = tracker.Domain();
  std::shared_ptr<TCPSocket> error_sock{TCPSocket::CreatePtr(domain)};
  auto eport = error_sock->BindHost();
  error_sock->Listen();
  error_worker_ = std::thread{[error_sock = std::move(error_sock)] {
    auto conn = error_sock->Accept();
    // On Windows, accept returns a closed socket after finalize.
    if (conn.IsClosed()) {
      return;
    }
    LOG(WARNING) << "Another worker is running into error.";
#if !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0
    // exit is nicer than abort as the former performs cleanups.
    std::exit(-1);
#else
    LOG(FATAL) << "abort";
#endif
  }};
  error_worker_.detach();

  proto::Start start;
  rc = std::move(rc) << [&] { return start.WorkerSend(lport, &tracker, eport); }
                     << [&] { return start.WorkerRecv(&tracker, &world); };
  if (!rc.OK()) {
    return rc;
  }
  this->world_ = world;

  // get ring neighbors
  std::string snext;
  tracker.Recv(&snext);
  auto jnext = Json::Load(StringView{snext});

  proto::PeerInfo ninfo{jnext};

  // get the rank of this worker
  this->rank_ = BootstrapPrev(ninfo.rank, world);
  this->tracker_.rank = rank_;

  std::vector<std::shared_ptr<TCPSocket>> workers;
  rc = ConnectWorkers(*this, &listener, lport, ninfo, timeout, retry, &workers);
  if (!rc.OK()) {
    return rc;
  }

  CHECK(this->channels_.empty());
  for (auto& w : workers) {
    if (w) {
      rc = std::move(rc) << [&] { return w->SetNoDelay(); } << [&] { return w->NonBlocking(true); }
                         << [&] { return w->SetKeepAlive(); };
    }
    if (!rc.OK()) {
      return rc;
    }
    this->channels_.emplace_back(std::make_shared<Channel>(*this, w));
  }
  return rc;
}

RabitComm::~RabitComm() noexcept(false) {
  if (!this->IsDistributed()) {
    return;
  }
  auto rc = this->Shutdown();
  if (!rc.OK()) {
    LOG(WARNING) << rc.Report();
  }
}

[[nodiscard]] Result RabitComm::Shutdown() {
  TCPSocket tracker;
  return Success() << [&] {
    return ConnectTrackerImpl(tracker_, timeout_, retry_, task_id_, &tracker, Rank(), World());
  } << [&] {
    return this->Block();
  } << [&] {
    Json jcmd{Object{}};
    jcmd["cmd"] = Integer{static_cast<std::int32_t>(proto::CMD::kShutdown)};
    auto scmd = Json::Dump(jcmd);
    auto n_bytes = tracker.Send(scmd);
    if (n_bytes != scmd.size()) {
      return Fail("Faled to send cmd.");
    }
    return Success();
  };
}

[[nodiscard]] Result RabitComm::LogTracker(std::string msg) const {
  TCPSocket out;
  proto::Print print;
  return Success() << [&] { return this->ConnectTracker(&out); }
                   << [&] { return print.WorkerSend(&out, msg); };
}

[[nodiscard]] Result RabitComm::SignalError(Result const& res) {
  TCPSocket out;
  return Success() << [&] { return this->ConnectTracker(&out); }
                   << [&] { return proto::ErrorCMD{}.WorkerSend(&out, res); };
}
}  // namespace xgboost::collective
