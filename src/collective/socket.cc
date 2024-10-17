/**
 * Copyright 2022-2024, XGBoost Contributors
 */
#include "xgboost/collective/socket.h"

#include <algorithm>     // for max
#include <array>         // for array
#include <cstddef>       // std::size_t
#include <cstdint>       // std::int32_t
#include <cstring>       // std::memcpy, std::memset
#include <filesystem>    // for path
#include <system_error>  // for error_code, system_category
#include <thread>        // for sleep_for

#include "rabit/internal/socket.h"      // for PollHelper
#include "xgboost/collective/result.h"  // for Result

#if defined(__unix__) || defined(__APPLE__)
#include <netdb.h>  // getaddrinfo, freeaddrinfo
#endif              // defined(__unix__) || defined(__APPLE__)

namespace xgboost::collective {
SockAddress MakeSockAddress(StringView host, in_port_t port) {
  struct addrinfo hints;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  int sig = getaddrinfo(host.c_str(), nullptr, &hints, &res);
  if (sig != 0) {
    LOG(FATAL) << "Failed to get addr info for: " << host
      << ", error: " << gai_strerror(sig);
    return {};
  }
  if (res->ai_family == static_cast<std::int32_t>(SockDomain::kV4)) {
    sockaddr_in addr;
    std::memcpy(&addr, res->ai_addr, res->ai_addrlen);
    addr.sin_port = htons(port);
    auto v = SockAddrV4{addr};
    freeaddrinfo(res);
    return SockAddress{v};
  } else if (res->ai_family == static_cast<std::int32_t>(SockDomain::kV6)) {
    sockaddr_in6 addr;
    std::memcpy(&addr, res->ai_addr, res->ai_addrlen);

    addr.sin6_port = htons(port);
    auto v = SockAddrV6{addr};
    freeaddrinfo(res);
    return SockAddress{v};
  } else {
    LOG(FATAL) << "Failed to get addr info for: " << host;
  }

  return SockAddress{};
}

SockAddrV4 SockAddrV4::Loopback() { return MakeSockAddress("127.0.0.1", 0).V4(); }
SockAddrV4 SockAddrV4::InaddrAny() { return MakeSockAddress("0.0.0.0", 0).V4(); }

SockAddrV6 SockAddrV6::Loopback() { return MakeSockAddress("::1", 0).V6(); }
SockAddrV6 SockAddrV6::InaddrAny() { return MakeSockAddress("::", 0).V6(); }

[[nodiscard]] Result TCPSocket::Listen(std::int32_t backlog) {
  backlog = std::max(backlog, 256);
  if (listen(this->handle_, backlog) != 0) {
    return system::FailWithCode("Failed to listen.");
  }
  return Success();
}

std::size_t TCPSocket::Send(StringView str) {
  CHECK(!this->IsClosed());
  CHECK_LT(str.size(), std::numeric_limits<std::int32_t>::max());
  std::int32_t len = static_cast<std::int32_t>(str.size());
  std::size_t n_bytes{0};
  auto rc = Success() << [&] {
    return this->SendAll(&len, sizeof(len), &n_bytes);
  } << [&] {
    if (n_bytes != sizeof(len)) {
      return Fail("Failed to send string length.");
    }
    return Success();
  } << [&] {
    return this->SendAll(str.c_str(), str.size(), &n_bytes);
  } << [&] {
    if (n_bytes != str.size()) {
      return Fail("Failed to send string.");
    }
    return Success();
  };
  SafeColl(rc);
  return n_bytes;
}

[[nodiscard]] Result TCPSocket::Recv(std::string *p_str) {
  CHECK(!this->IsClosed());
  std::int32_t len;
  std::size_t n_bytes{0};
  return Success() << [&] {
    return this->RecvAll(&len, sizeof(len), &n_bytes);
  } << [&] {
    if (n_bytes != sizeof(len)) {
      return Fail("Failed to recv string length.");
    }
    return Success();
  } << [&] {
    p_str->resize(len);
    return this->RecvAll(&(*p_str)[0], len, &n_bytes);
  } << [&] {
    if (static_cast<std::remove_reference_t<decltype(len)>>(n_bytes) != len) {
      return Fail("Failed to recv string.");
    }
    return Success();
  };
}

[[nodiscard]] Result Connect(xgboost::StringView host, std::int32_t port, std::int32_t retry,
                             std::chrono::seconds timeout,
                             xgboost::collective::TCPSocket *out_conn) {
  auto addr = MakeSockAddress(xgboost::StringView{host}, port);
  auto &conn = *out_conn;

  sockaddr const *addr_handle{nullptr};
  socklen_t addr_len{0};
  if (addr.IsV4()) {
    addr_handle = reinterpret_cast<const sockaddr *>(&addr.V4().Handle());
    addr_len = sizeof(addr.V4().Handle());
  } else {
    addr_handle = reinterpret_cast<const sockaddr *>(&addr.V6().Handle());
    addr_len = sizeof(addr.V6().Handle());
  }

  conn = TCPSocket::Create(addr.Domain());
  CHECK_EQ(static_cast<std::int32_t>(conn.Domain()), static_cast<std::int32_t>(addr.Domain()));
  auto non_blocking = conn.NonBlocking();
  auto rc = conn.NonBlocking(true);
  if (!rc.OK()) {
    return Fail("Failed to set socket option.", std::move(rc));
  }

  Result last_error;
  auto log_failure = [&host, &last_error, port](Result err, char const *file, std::int32_t line) {
    last_error = std::move(err);
    LOG(WARNING) << std::filesystem::path{file}.filename().string() << "(" << line
                 << "): Failed to connect to:" << host << ":" << port
                 << " Error:" << last_error.Report();
  };

  for (std::int32_t attempt = 0; attempt < std::max(retry, 1); ++attempt) {
    if (attempt > 0) {
      LOG(WARNING) << "Retrying connection to " << host << " for the " << attempt << " time.";
      std::this_thread::sleep_for(std::chrono::seconds{attempt << 1});
    }

    auto rc = connect(conn.Handle(), addr_handle, addr_len);
    if (rc == 0) {
      return conn.NonBlocking(non_blocking);
    }

    auto errcode = system::LastError();
    if (!system::ErrorWouldBlock(errcode)) {
      log_failure(Fail("connect failed.", std::error_code{errcode, std::system_category()}),
                  __FILE__, __LINE__);
      continue;
    }

    rabit::utils::PollHelper poll;
    poll.WatchWrite(conn);
    auto result = poll.Poll(timeout);
    if (!result.OK()) {
      // poll would fail if there's a socket error, we log the root cause instead of the
      // poll failure.
      auto sockerr = conn.GetSockError();
      if (!sockerr.OK()) {
        result = std::move(sockerr);
      }
      log_failure(std::move(result), __FILE__, __LINE__);
      continue;
    }
    if (!poll.CheckWrite(conn)) {
      log_failure(Fail("poll failed.", std::error_code{errcode, std::system_category()}), __FILE__,
                  __LINE__);
      continue;
    }
    result = conn.GetSockError();
    if (!result.OK()) {
      log_failure(std::move(result), __FILE__, __LINE__);
      continue;
    }

    return conn.NonBlocking(non_blocking);
  }

  std::stringstream ss;
  ss << "Failed to connect to " << host << ":" << port;
  auto close_rc = conn.Close();
  return Fail(ss.str(), std::move(close_rc) + std::move(last_error));
}

[[nodiscard]] Result GetHostName(std::string *p_out) {
  std::array<char, HOST_NAME_MAX> buf;
  if (gethostname(&buf[0], HOST_NAME_MAX) != 0) {
    return system::FailWithCode("Failed to get host name.");
  }
  *p_out = buf.data();
  return Success();
}
}  // namespace xgboost::collective
