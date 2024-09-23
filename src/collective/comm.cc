/**
 * Copyright 2023-2024, XGBoost Contributors
 */
#include "comm.h"

#include <algorithm>  // for copy
#include <chrono>     // for seconds
#include <cstdint>    // for int32_t
#include <cstdlib>    // for exit
#include <memory>     // for shared_ptr
#include <string>     // for string
#include <thread>     // for thread
#include <utility>    // for move, forward
#if !defined(XGBOOST_USE_NCCL)
#include "../common/common.h"           // for AssertNCCLSupport
#endif                                  // !defined(XGBOOST_USE_NCCL)
#include "allgather.h"                  // for RingAllgather
#include "protocol.h"                   // for kMagic
#include "topo.h"                       // for BootstrapNext
#include "xgboost/base.h"               // for XGBOOST_STRICT_R_MODE
#include "xgboost/collective/socket.h"  // for TCPSocket
#include "xgboost/json.h"               // for Json, Object
#include "xgboost/string_view.h"        // for StringView

namespace xgboost::collective {
Comm::Comm(std::string const& host, std::int32_t port, std::chrono::seconds timeout,
           std::int32_t retry, std::string task_id)
    : timeout_{timeout}, retry_{retry}, tracker_{host, port, -1}, task_id_{std::move(task_id)} {}

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

// Connect ring and tree neighbors
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
    SockAddress addr;
    return listener->Accept(prev.get(), &addr);
  } << [&] {
    return prev->NonBlocking(true);
  };
  if (!rc.OK()) {
    return Fail("Bootstrap failed to recv from ring prev.", std::move(rc));
  }

  // Exchange host name and port
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
  } << [&] {
    return block();
  };
  if (!rc.OK()) {
    return Fail("Failed to get host names from peers.", std::move(rc));
  }

  std::vector<std::int32_t> peers_port(comm.World(), -1);
  peers_port[comm.Rank()] = ninfo.port;
  rc = std::move(rc) << [&] {
    auto s_ports = common::Span{reinterpret_cast<std::int8_t*>(peers_port.data()),
                                peers_port.size() * sizeof(ninfo.port)};
    return cpu_impl::RingAllgather(comm, s_ports, sizeof(ninfo.port), 0, prev_ch, next_ch);
  } << [&] {
    return block();
  };
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
  workers[BootstrapNext(comm.Rank(), comm.World())] = next;
  if (BootstrapNext(comm.Rank(), comm.World()) == BootstrapPrev(comm.Rank(), comm.World())) {
    if (comm.Rank() == 0) {
      if (comm.World() == 2) {
        workers[BootstrapNext(comm.Rank(), comm.World())] = prev;
      } else {
        CHECK_EQ(comm.World(), 1);
      }
    }
  } else {
    workers[BootstrapPrev(comm.Rank(), comm.World())] = prev;
  }

  /**
   * Construct tree.
   */
  // All workers connect to rank 0 so that we can always use rank 0 as broadcast root.
  if (comm.Rank() == 0) {
    for (std::int32_t i = 0; i < comm.World() - 3; ++i) {
      auto worker = std::make_shared<TCPSocket>();
      SockAddress addr;
      rc = listener->Accept(worker.get(), &addr);
      if (!rc.OK()) {
        return Fail("Failed to accept for rank 0.", std::move(rc));
      }
      std::int32_t r{-1};
      std::size_t n_bytes{0};
      rc = worker->RecvAll(&r, sizeof(r), &n_bytes);
      if (!rc.OK()) {
        return Fail("Failed to recv rank.", std::move(rc));
      }
      if (n_bytes != sizeof(r)) {
        return Fail("Failed to recv rank due to size.", std::move(rc));
      }
      workers[r] = worker;
    }
  } else {
    if (!workers[0]) {
      auto worker = std::make_shared<TCPSocket>();
      rc = std::move(rc) << [&] {
        return Connect(peers[0].host, peers[0].port, retry, timeout, worker.get());
      } << [&] {
        auto rank = comm.Rank();
        std::size_t n_bytes = 0;
        auto rc = worker->SendAll(&rank, sizeof(rank), &n_bytes);
        if (n_bytes != sizeof(rank)) {
          return Fail("Failed to send rank due to size.", std::move(rc));
        }
        return rc;
      };
      if (!rc.OK()) {
        return Fail("Failed to connect to root.", std::move(rc));
      }
      workers[0] = worker;
    }
  }
  // Binomial tree connect
  std::int32_t const kDepth = std::ceil(std::log2(static_cast<double>(comm.World()))) - 1;
  if (comm.Rank() != 0) {
    auto prank = ParentRank(comm.Rank(), kDepth);
    if (!workers[prank]) {  // Skip if it's part of the ring.
      auto parent = std::make_shared<TCPSocket>();
      SockAddress addr;
      rc = listener->Accept(parent.get(), &addr);
      if (!rc.OK()) {
        return Fail("Failed to recv connection from tree parent.", std::move(rc));
      }
      workers[prank] = parent;
    }
  }

  for (std::int32_t i = kDepth; i >= 0; --i) {
    if (comm.Rank() % (1 << (i + 1)) == 0 && comm.Rank() + (1 << i) < comm.World()) {
      auto peer = comm.Rank() + (1 << i);
      if (workers[peer]) {  // skip if it's part of the ring.
        continue;
      }
      auto worker = std::make_shared<TCPSocket>();
      rc = std::move(rc) << [&] {
        return Connect(peers[peer].host, peers[peer].port, retry, timeout, worker.get());
      } << [&] {
        return worker->RecvTimeout(timeout);
      };
      if (!rc.OK()) {
        return Fail("Failed to connect to tree neighbor", std::move(rc));
      }
      workers[peer] = worker;
    }
  }

  return Success();
}

namespace {
std::string InitLog(std::string task_id, std::int32_t rank) {
  if (task_id.empty()) {
    return "Rank " + std::to_string(rank);
  }
  return "Task " + task_id + " got rank " + std::to_string(rank);
}
}  // namespace

RabitComm::RabitComm(std::string const& tracker_host, std::int32_t tracker_port,
                     std::chrono::seconds timeout, std::int32_t retry, std::string task_id,
                     StringView nccl_path)
    : HostComm{tracker_host, tracker_port, timeout, retry, std::move(task_id)},
      nccl_path_{std::move(nccl_path)} {
  if (this->TrackerInfo().host.empty()) {
    // Not in a distributed environment.
    LOG(CONSOLE) << InitLog(task_id_, rank_);
    return;
  }

  loop_.reset(new Loop{std::chrono::seconds{timeout_}});  // NOLINT
  auto rc = this->Bootstrap(timeout_, retry_, task_id_);
  if (!rc.OK()) {
    this->ResetState();
    SafeColl(Fail("Failed to bootstrap the communication group.", std::move(rc)));
  }
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
  std::int32_t lport{0};
  rc = std::move(rc) << [&] {
    return listener.BindHost(&lport);
  } << [&] {
    return listener.Listen();
  };
  if (!rc.OK()) {
    return rc;
  }

  // create worker for listening to error notice.
  auto domain = tracker.Domain();
  std::shared_ptr<TCPSocket> error_sock{TCPSocket::CreatePtr(domain)};
  std::int32_t eport{0};
  rc = std::move(rc) << [&] {
    return error_sock->BindHost(&eport);
  } << [&] {
    return error_sock->Listen();
  };
  if (!rc.OK()) {
    return rc;
  }
  error_port_ = eport;

  error_worker_ = std::thread{[error_sock = std::move(error_sock)] {
    TCPSocket conn;
    SockAddress addr;
    auto rc = error_sock->Accept(&conn, &addr);
    // On Linux, a shutdown causes an invalid argument error;
    if (rc.Code() == std::errc::invalid_argument) {
      return;
    }
    // On Windows, accept returns a closed socket after finalize.
    if (conn.IsClosed()) {
      return;
    }
    // The error signal is from the tracker, while shutdown signal is from the shutdown method
    // of the RabitComm class (this).
    bool is_error{false};
    rc = proto::Error{}.RecvSignal(&conn, &is_error);
    if (!rc.OK()) {
      LOG(WARNING) << rc.Report();
      return;
    }
    if (!is_error) {
      return;  // shutdown
    }

    LOG(WARNING) << "Another worker is running into error.";
#if !defined(XGBOOST_STRICT_R_MODE) || XGBOOST_STRICT_R_MODE == 0
    // exit is nicer than abort as the former performs cleanups.
    std::exit(-1);
#else
    LOG(FATAL) << "abort";
#endif
  }};
  // The worker thread is detached here to avoid the need to handle it later during
  // destruction. For C++, if a thread is not joined or detached, it will segfault during
  // destruction.
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
  rc = tracker.Recv(&snext);
  if (!rc.OK()) {
    return Fail("Failed to receive the rank for the next worker.", std::move(rc));
  }
  auto jnext = Json::Load(StringView{snext});

  proto::PeerInfo ninfo{jnext};
  // get the rank of this worker
  this->rank_ = BootstrapPrev(ninfo.rank, world);
  this->tracker_.rank = rank_;

  std::vector<std::shared_ptr<TCPSocket>> workers;
  rc = ConnectWorkers(*this, &listener, lport, ninfo, timeout, retry, &workers);
  if (!rc.OK()) {
    return Fail("Failed to connect to other workers.", std::move(rc));
  }

  CHECK(this->channels_.empty());
  for (auto& w : workers) {
    if (w) {
      rc = std::move(rc) << [&] {
        return w->SetNoDelay();
      } << [&] {
        return w->NonBlocking(true);
      } << [&] {
        return w->SetKeepAlive();
      };
    }
    if (!rc.OK()) {
      return rc;
    }
    this->channels_.emplace_back(std::make_shared<Channel>(*this, w));
  }

  LOG(CONSOLE) << InitLog(task_id_, rank_);
  return rc;
}

RabitComm::~RabitComm() noexcept(false) {
  if (!this->IsDistributed()) {
    return;
  }
  LOG(WARNING) << "The communicator is being destroyed without a call to shutdown first. This can "
                  "lead to undefined behaviour.";
  auto rc = this->Shutdown();
  if (!rc.OK()) {
    LOG(WARNING) << rc.Report();
  }
}

[[nodiscard]] Result RabitComm::Shutdown() {
  if (!this->IsDistributed()) {
    return Success();
  }
  // Tell the tracker that this worker is shutting down.
  TCPSocket tracker;
  // Tell the error hanlding thread that we are shutting down.
  TCPSocket err_client;

  auto rc = Success() << [&] {
    return ConnectTrackerImpl(tracker_, timeout_, retry_, task_id_, &tracker, Rank(), World());
  } << [&] {
    return this->Block();
  } << [&] {
    return proto::ShutdownCMD{}.Send(&tracker);
  } << [&] {
    this->channels_.clear();
    return Success();
  } << [&] {
    // Use tracker address to determine whether we want to use IPv6.
    auto taddr = MakeSockAddress(xgboost::StringView{this->tracker_.host}, this->tracker_.port);
    // Shutdown the error handling thread. We signal the thread through socket,
    // alternatively, we can get the native handle and use pthread_cancel. But using a
    // socket seems to be clearer as we know what's happening.
    auto const& addr = taddr.IsV4() ? SockAddrV4::Loopback().Addr() : SockAddrV6::Loopback().Addr();
    // We use hardcoded 10 seconds and 1 retry here since we are just connecting to a
    // local socket. For a normal OS, this should be enough time to schedule the
    // connection.
    auto rc = Connect(StringView{addr}, this->error_port_, 1,
                      std::min(std::chrono::seconds{10}, timeout_), &err_client);
    this->ResetState();
    if (!rc.OK()) {
      return Fail("Failed to connect to the error socket.", std::move(rc));
    }
    return rc;
  } << [&] {
    // We put error thread shutdown at the end so that we have a better chance to finish
    // the previous more important steps.
    return proto::Error{}.SignalShutdown(&err_client);
  };
  if (!rc.OK()) {
    return Fail("Failed to shutdown.", std::move(rc));
  }
  return rc;
}

[[nodiscard]] Result RabitComm::LogTracker(std::string msg) const {
  if (!this->IsDistributed()) {
    LOG(CONSOLE) << msg;
    return Success();
  }
  TCPSocket out;
  proto::Print print;
  return Success() << [&] { return this->ConnectTracker(&out); }
                   << [&] { return print.WorkerSend(&out, msg); };
}

[[nodiscard]] Result RabitComm::SignalError(Result const& res) {
  TCPSocket tracker;
  return Success() << [&] {
    return this->ConnectTracker(&tracker);
  } << [&] {
    return proto::ErrorCMD{}.WorkerSend(&tracker, res);
  };
}
}  // namespace xgboost::collective
