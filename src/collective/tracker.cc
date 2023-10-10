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

#include <string>  // for string

#include "xgboost/collective/result.h"  // for Result, Fail, Success
#include "xgboost/collective/socket.h"  // for GetHostName, FailWithCode, MakeSockAddress, ...

namespace xgboost::collective {
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
