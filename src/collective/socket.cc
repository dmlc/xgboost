/*!
 * Copyright (c) 2022 by XGBoost Contributors
 */
#include "xgboost/collective/socket.h"

#include <cstddef>       // std::size_t
#include <cstdint>       // std::int32_t
#include <cstring>       // std::memcpy, std::memset
#include <system_error>  // std::error_code, std::system_category

#if defined(__unix__) || defined(__APPLE__)
#include <netdb.h>  // getaddrinfo, freeaddrinfo
#endif              // defined(__unix__) || defined(__APPLE__)

namespace xgboost {
namespace collective {
SockAddress MakeSockAddress(StringView host, in_port_t port) {
  struct addrinfo hints;
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_protocol = SOCK_STREAM;
  struct addrinfo *res = nullptr;
  int sig = getaddrinfo(host.c_str(), nullptr, &hints, &res);
  if (sig != 0) {
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

std::size_t TCPSocket::Send(StringView str) {
  CHECK(!this->IsClosed());
  CHECK_LT(str.size(), std::numeric_limits<std::int32_t>::max());
  std::int32_t len = static_cast<std::int32_t>(str.size());
  CHECK_EQ(this->SendAll(&len, sizeof(len)), sizeof(len)) << "Failed to send string length.";
  auto bytes = this->SendAll(str.c_str(), str.size());
  CHECK_EQ(bytes, str.size()) << "Failed to send string.";
  return bytes;
}

std::size_t TCPSocket::Recv(std::string *p_str) {
  CHECK(!this->IsClosed());
  std::int32_t len;
  CHECK_EQ(this->RecvAll(&len, sizeof(len)), sizeof(len)) << "Failed to recv string length.";
  p_str->resize(len);
  auto bytes = this->RecvAll(&(*p_str)[0], len);
  CHECK_EQ(bytes, len) << "Failed to recv string.";
  return bytes;
}

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
