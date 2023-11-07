/**
 * Copyright 2023, XGBoost Contributors
 */
#if defined(__unix__) || defined(__APPLE__)
#include <netdb.h>       // gethostbyname
#include <sys/socket.h>  // socket, AF_INET6, AF_INET, connect, getsockname
#endif                   // defined(__unix__) || defined(__APPLE__)

#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif  // !defined(NOMINMAX)

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#endif  // defined(_WIN32)

#include <algorithm>  // for sort
#include <chrono>     // for seconds, ms
#include <cstdint>    // for int32_t
#include <string>     // for string
#include <utility>    // for move, forward

#include "../common/json_utils.h"
#include "comm.h"
#include "protocol.h"  // for kMagic, PeerInfo
#include "tracker.h"
#include "xgboost/collective/result.h"  // for Result, Fail, Success
#include "xgboost/collective/socket.h"  // for GetHostName, FailWithCode, MakeSockAddress, ...
#include "xgboost/json.h"

namespace xgboost::collective {
Tracker::Tracker(Json const& config)
    : n_workers_{static_cast<std::int32_t>(
          RequiredArg<Integer const>(config, "n_workers", __func__))},
      port_{static_cast<std::int32_t>(OptionalArg<Integer const>(config, "port", Integer::Int{0}))},
      timeout_{std::chrono::seconds{OptionalArg<Integer const>(
          config, "timeout", static_cast<std::int64_t>(collective::DefaultTimeoutSec()))}} {}

Result Tracker::WaitUntilReady() const {
  using namespace std::chrono_literals;  // NOLINT

  // Busy waiting. The function is mostly for waiting for the OS to launch an async
  // thread, which should be reasonably fast.
  common::Timer timer;
  timer.Start();
  while (!this->Ready()) {
    auto ela = timer.Duration().count();
    if (ela > this->Timeout().count()) {
      return Fail("Failed to start tracker, timeout:" + std::to_string(this->Timeout().count()) +
                  " seconds.");
    }
    std::this_thread::sleep_for(100ms);
  }

  return Success();
}

RabitTracker::WorkerProxy::WorkerProxy(std::int32_t world, TCPSocket sock, SockAddrV4 addr)
    : sock_{std::move(sock)} {
  std::int32_t rank{0};
  Json jcmd;
  std::int32_t port{0};

  rc_ = Success() << [&] { return proto::Magic{}.Verify(&sock_); } << [&] {
    return proto::Connect{}.TrackerRecv(&sock_, &world_, &rank, &task_id_);
  } << [&] {
    std::string cmd;
    sock_.Recv(&cmd);
    jcmd = Json::Load(StringView{cmd});
    cmd_ = static_cast<proto::CMD>(get<Integer const>(jcmd["cmd"]));
    return Success();
  } << [&] {
    if (cmd_ == proto::CMD::kStart) {
      proto::Start start;
      return start.TrackerHandle(jcmd, &world_, world, &port, &sock_, &eport_);
    } else if (cmd_ == proto::CMD::kPrint) {
      proto::Print print;
      return print.TrackerHandle(jcmd, &msg_);
    } else if (cmd_ == proto::CMD::kError) {
      proto::ErrorCMD error;
      return error.TrackerHandle(jcmd, &msg_, &code_);
    }
    return Success();
  } << [&] {
    auto host = addr.Addr();
    info_ = proto::PeerInfo{host, port, rank};
    return Success();
  };
}

RabitTracker::RabitTracker(Json const& config) : Tracker{config} {
  std::string self;
  auto rc = collective::GetHostAddress(&self);
  auto host = OptionalArg<String>(config, "host", self);

  host_ = host;
  listener_ = TCPSocket::Create(SockDomain::kV4);
  rc = listener_.Bind(host, &this->port_);
  CHECK(rc.OK()) << rc.Report();
  listener_.Listen();
}

Result RabitTracker::Bootstrap(std::vector<WorkerProxy>* p_workers) {
  auto& workers = *p_workers;

  std::sort(workers.begin(), workers.end(), WorkerCmp{});

  std::vector<std::thread> bootstrap_threads;
  for (std::int32_t r = 0; r < n_workers_; ++r) {
    auto& worker = workers[r];
    auto next = BootstrapNext(r, n_workers_);
    auto const& next_w = workers[next];
    bootstrap_threads.emplace_back([next, &worker, &next_w] {
      auto jnext = proto::PeerInfo{next_w.Host(), next_w.Port(), next}.ToJson();
      std::string str;
      Json::Dump(jnext, &str);
      worker.Send(StringView{str});
    });
  }

  for (auto& t : bootstrap_threads) {
    t.join();
  }

  for (auto const& w : workers) {
    worker_error_handles_.emplace_back(w.Host(), w.ErrorPort());
  }
  return Success();
}

[[nodiscard]] std::future<Result> RabitTracker::Run() {
  // a state machine to keep track of consistency.
  struct State {
    std::int32_t const n_workers;

    std::int32_t n_shutdown{0};
    bool during_restart{false};
    bool running{false};
    std::vector<WorkerProxy> pending;

    explicit State(std::int32_t world) : n_workers{world} {}
    State(State const& that) = delete;
    State& operator=(State&& that) = delete;

    // modifiers
    void Start(WorkerProxy&& worker) {
      CHECK_LT(pending.size(), n_workers);
      CHECK_LE(n_shutdown, n_workers);
      CHECK(!running);

      pending.emplace_back(std::forward<WorkerProxy>(worker));

      CHECK_LE(pending.size(), n_workers);
    }
    void Shutdown() {
      CHECK_GE(n_shutdown, 0);
      CHECK_LT(n_shutdown, n_workers);

      running = false;
      ++n_shutdown;

      CHECK_LE(n_shutdown, n_workers);
    }
    void Error() {
      CHECK_LE(pending.size(), n_workers);
      CHECK_LE(n_shutdown, n_workers);

      running = false;
      during_restart = true;
    }
    void Bootstrap() {
      CHECK_EQ(pending.size(), n_workers);
      CHECK_LE(n_shutdown, n_workers);

      running = true;

      // A reset.
      n_shutdown = 0;
      during_restart = false;
      pending.clear();
    }

    // observers
    [[nodiscard]] bool Ready() const {
      CHECK_LE(pending.size(), n_workers);
      return static_cast<std::int32_t>(pending.size()) == n_workers;
    }
    [[nodiscard]] bool ShouldContinue() const {
      CHECK_LE(pending.size(), n_workers);
      CHECK_LE(n_shutdown, n_workers);
      // - Without error, we should shutdown after all workers are offline.
      // - With error, all workers are offline, and we have during_restart as true.
      return n_shutdown != n_workers || during_restart;
    }
  };

  auto handle_error = [&](WorkerProxy const& worker) {
    auto msg = worker.Msg();
    auto code = worker.Code();
    LOG(WARNING) << "Recieved error from [" << worker.Host() << ":" << worker.Rank() << "]: " << msg
                 << " code:" << code;
    auto host = worker.Host();
    // We signal all workers for the error, if they haven't aborted already.
    for (auto& w : worker_error_handles_) {
      if (w.first == host) {
        continue;
      }
      TCPSocket out;
      // Connecting to the error port as a signal for exit.
      //
      // retry is set to 1, just let the worker timeout or error. Otherwise the
      // tracker and the worker might be waiting for each other.
      auto rc = Connect(w.first, w.second, 1, timeout_, &out);
      if (!rc.OK()) {
        return Fail("Failed to inform workers to stop.");
      }
    }
    return Success();
  };

  return std::async(std::launch::async, [this, handle_error] {
    State state{this->n_workers_};

    while (state.ShouldContinue()) {
      TCPSocket sock;
      SockAddrV4 addr;
      this->ready_ = true;
      auto rc = listener_.Accept(&sock, &addr);
      if (!rc.OK()) {
        return Fail("Failed to accept connection.", std::move(rc));
      }

      auto worker = WorkerProxy{n_workers_, std::move(sock), std::move(addr)};
      if (!worker.Status().OK()) {
        return Fail("Failed to initialize worker proxy.", std::move(worker.Status()));
      }
      switch (worker.Command()) {
        case proto::CMD::kStart: {
          if (state.running) {
            // Something went wrong with one of the workers. It got disconnected without
            // notice.
            state.Error();
            rc = handle_error(worker);
            if (!rc.OK()) {
              return Fail("Failed to handle abort.", std::move(rc));
            }
          }

          state.Start(std::move(worker));
          if (state.Ready()) {
            rc = this->Bootstrap(&state.pending);
            state.Bootstrap();
          }
          if (!rc.OK()) {
            return rc;
          }
          continue;
        }
        case proto::CMD::kShutdown: {
          if (state.during_restart) {
            // The worker can still send shutdown after call to `std::exit`.
            continue;
          }
          state.Shutdown();
          continue;
        }
        case proto::CMD::kError: {
          if (state.during_restart) {
            // Ignore further errors.
            continue;
          }
          state.Error();
          rc = handle_error(worker);
          continue;
        }
        case proto::CMD::kPrint: {
          LOG(CONSOLE) << worker.Msg();
          continue;
        }
        case proto::CMD::kInvalid:
        default: {
          return Fail("Invalid command received.");
        }
      }
    }
    ready_ = false;
    return Success();
  });
}

[[nodiscard]] Json RabitTracker::WorkerArgs() const {
  auto rc = this->WaitUntilReady();
  CHECK(rc.OK()) << rc.Report();

  Json args{Object{}};
  args["DMLC_TRACKER_URI"] = String{host_};
  args["DMLC_TRACKER_PORT"] = this->Port();
  return args;
}

[[nodiscard]] Result GetHostAddress(std::string* out) {
  auto rc = GetHostName(out);
  if (!rc.OK()) {
    return rc;
  }
  auto host = gethostbyname(out->c_str());

  // get ip address from host
  std::string ip;
  rc = INetNToP(host, &ip);
  if (!rc.OK()) {
    return rc;
  }

  if (!(ip.size() >= 4 && ip.substr(0, 4) == "127.")) {
    // return if this is a public IP address.
    // not entirely accurate, we have other reserved IPs
    *out = ip;
    return Success();
  }

  // Create an UDP socket to prob the public IP address, it's fine even if it's
  // unreachable.
  auto sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock == -1) {
    return Fail("Failed to create socket.");
  }

  auto paddr = MakeSockAddress(StringView{"10.255.255.255"}, 1);
  sockaddr const* addr_handle = reinterpret_cast<const sockaddr*>(&paddr.V4().Handle());
  socklen_t addr_len{sizeof(paddr.V4().Handle())};
  auto err = connect(sock, addr_handle, addr_len);
  if (err != 0) {
    return system::FailWithCode("Failed to find IP address.");
  }

  // get the IP address from socket desrciptor
  struct sockaddr_in addr;
  socklen_t len = sizeof(addr);
  if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&addr), &len) == -1) {
    return Fail("Failed to get sock name.");
  }
  ip = inet_ntoa(addr.sin_addr);

  err = system::CloseSocket(sock);
  if (err != 0) {
    return system::FailWithCode("Failed to close socket.");
  }

  *out = ip;
  return Success();
}
}  // namespace xgboost::collective
