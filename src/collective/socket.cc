/*!
 * Copyright (c) 2022 by XGBoost Contributors
 */
#include "xgboost/collective/socket.h"

#if defined(__unix__) || defined(__APPLE__)
#include <netdb.h>  // getaddrinfo
#endif              // defined(__unix__) || defined(__APPLE__)

namespace xgboost {
namespace collective {
SockAddress MakeSockAddress(StringView host, in_port_t port) {
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_protocol = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  int sig = getaddrinfo(host.c_str(), nullptr, &hints, &res);
  if (sig != 0) {
    return {};
  }
  if (res->ai_family == AF_INET) {
    sockaddr_in addr;
    memcpy(&addr, res->ai_addr, res->ai_addrlen);
    addr.sin_port = htons(port);
    auto v = SockAddrV4{addr};
    freeaddrinfo(res);
    return SockAddress{v};
  } else if (res->ai_family == AF_INET6) {
    sockaddr_in6 addr;
    memcpy(&addr, res->ai_addr, res->ai_addrlen);

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

std::error_code Connect(SockAddress const &addr, TCPSocket *out) {
  sockaddr const *addr_handle{nullptr};
  socklen_t addr_len{0};
  if (addr.IsV4()) {
    addr_handle = reinterpret_cast<const sockaddr *>(&addr.V4().Handle());
    addr_len = sizeof(addr.V4().Handle());
  } else {
    addr_handle = reinterpret_cast<const sockaddr *>(&addr.V6().Handle());
    addr_len = sizeof(addr.V6().Handle());
  }
  auto socket = TCPSocket::Create(addr.Domain());
  CHECK_EQ(static_cast<std::int32_t>(socket.Domain()), static_cast<std::int32_t>(addr.Domain()));
  auto rc = connect(socket.Handle(), addr_handle, addr_len);
  if (rc != 0) {
    return std::error_code{errno, std::system_category()};
  }
  *out = std::move(socket);
  return std::make_error_code(std::errc{});
}
}  // namespace collective
}  // namespace xgboost
