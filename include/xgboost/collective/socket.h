/**
 * Copyright (c) 2022-2023, XGBoost Contributors
 */
#pragma once

#if !defined(NOMINMAX) && defined(_WIN32)
#define NOMINMAX
#endif                   // !defined(NOMINMAX)

#include <cerrno>        // errno, EINTR, EBADF
#include <climits>       // HOST_NAME_MAX
#include <cstddef>       // std::size_t
#include <cstdint>       // std::int32_t, std::uint16_t
#include <cstring>       // memset
#include <limits>        // std::numeric_limits
#include <string>        // std::string
#include <system_error>  // std::error_code, std::system_category
#include <utility>       // std::swap

#if !defined(xgboost_IS_MINGW)

#if defined(__MINGW32__)
#define xgboost_IS_MINGW 1
#endif  // defined(__MINGW32__)

#endif  // xgboost_IS_MINGW

#if defined(_WIN32)

#include <winsock2.h>
#include <ws2tcpip.h>

using in_port_t = std::uint16_t;

#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif  // _MSC_VER

#if !defined(xgboost_IS_MINGW)
using ssize_t = int;
#endif                    // !xgboost_IS_MINGW()

#else                     // UNIX

#include <arpa/inet.h>    // inet_ntop
#include <fcntl.h>        // fcntl, F_GETFL, O_NONBLOCK
#include <netinet/in.h>   // sockaddr_in6, sockaddr_in, in_port_t, INET6_ADDRSTRLEN, INET_ADDRSTRLEN
#include <netinet/in.h>   // IPPROTO_TCP
#include <netinet/tcp.h>  // TCP_NODELAY
#include <sys/socket.h>  // socket, SOL_SOCKET, SO_ERROR, MSG_WAITALL, recv, send, AF_INET6, AF_INET
#include <unistd.h>      // close

#if defined(__sun) || defined(sun)
#include <sys/sockio.h>
#endif                            // defined(__sun) || defined(sun)

#endif                            // defined(_WIN32)

#include "xgboost/base.h"         // XGBOOST_EXPECT
#include "xgboost/logging.h"      // LOG
#include "xgboost/string_view.h"  // StringView

#if !defined(HOST_NAME_MAX)
#define HOST_NAME_MAX 256  // macos
#endif

namespace xgboost {

#if defined(xgboost_IS_MINGW)
// see the dummy implementation of `poll` in rabit for more info.
inline void MingWError() { LOG(FATAL) << "Distributed training on mingw is not supported."; }
#endif  // defined(xgboost_IS_MINGW)

namespace system {
inline std::int32_t LastError() {
#if defined(_WIN32)
  return WSAGetLastError();
#else
  int errsv = errno;
  return errsv;
#endif
}

#if defined(__GLIBC__)
inline auto ThrowAtError(StringView fn_name, std::int32_t errsv = LastError(),
                         std::int32_t line = __builtin_LINE(),
                         char const *file = __builtin_FILE()) {
  auto err = std::error_code{errsv, std::system_category()};
  LOG(FATAL) << "\n"
             << file << "(" << line << "): Failed to call `" << fn_name << "`: " << err.message()
             << std::endl;
}
#else
inline auto ThrowAtError(StringView fn_name, std::int32_t errsv = LastError()) {
  auto err = std::error_code{errsv, std::system_category()};
  LOG(FATAL) << "Failed to call `" << fn_name << "`: " << err.message() << std::endl;
}
#endif  // defined(__GLIBC__)

#if defined(_WIN32)
using SocketT = SOCKET;
#else
using SocketT = int;
#endif  // defined(_WIN32)

#if !defined(xgboost_CHECK_SYS_CALL)
#define xgboost_CHECK_SYS_CALL(exp, expected)         \
  do {                                                \
    if (XGBOOST_EXPECT((exp) != (expected), false)) { \
      ::xgboost::system::ThrowAtError(#exp);          \
    }                                                 \
  } while (false)
#endif  // !defined(xgboost_CHECK_SYS_CALL)

inline std::int32_t CloseSocket(SocketT fd) {
#if defined(_WIN32)
  return closesocket(fd);
#else
  return close(fd);
#endif
}

inline bool LastErrorWouldBlock() {
  int errsv = LastError();
#ifdef _WIN32
  return errsv == WSAEWOULDBLOCK;
#else
  return errsv == EAGAIN || errsv == EWOULDBLOCK;
#endif  // _WIN32
}

inline void SocketStartup() {
#if defined(_WIN32)
  WSADATA wsa_data;
  if (WSAStartup(MAKEWORD(2, 2), &wsa_data) == -1) {
    ThrowAtError("WSAStartup");
  }
  if (LOBYTE(wsa_data.wVersion) != 2 || HIBYTE(wsa_data.wVersion) != 2) {
    WSACleanup();
    LOG(FATAL) << "Could not find a usable version of Winsock.dll";
  }
#endif  // defined(_WIN32)
}

inline void SocketFinalize() {
#if defined(_WIN32)
  WSACleanup();
#endif  // defined(_WIN32)
}

#if defined(_WIN32) && defined(xgboost_IS_MINGW)
// dummy definition for old mysys32.
inline const char *inet_ntop(int, const void *, char *, socklen_t) {  // NOLINT
  MingWError();
  return nullptr;
}
#else
using ::inet_ntop;
#endif  // defined(_WIN32) && defined(xgboost_IS_MINGW)

}  // namespace system

namespace collective {
class SockAddress;

enum class SockDomain : std::int32_t { kV4 = AF_INET, kV6 = AF_INET6 };

/**
 * \brief Parse host address and return a SockAddress instance. Supports IPv4 and IPv6
 *        host.
 */
SockAddress MakeSockAddress(StringView host, in_port_t port);

class SockAddrV6 {
  sockaddr_in6 addr_;

 public:
  explicit SockAddrV6(sockaddr_in6 addr) : addr_{addr} {}
  SockAddrV6() { std::memset(&addr_, '\0', sizeof(addr_)); }

  static SockAddrV6 Loopback();
  static SockAddrV6 InaddrAny();

  in_port_t Port() const { return ntohs(addr_.sin6_port); }

  std::string Addr() const {
    char buf[INET6_ADDRSTRLEN];
    auto const *s = system::inet_ntop(static_cast<std::int32_t>(SockDomain::kV6), &addr_.sin6_addr,
                                      buf, INET6_ADDRSTRLEN);
    if (s == nullptr) {
      system::ThrowAtError("inet_ntop");
    }
    return {buf};
  }
  sockaddr_in6 const &Handle() const { return addr_; }
};

class SockAddrV4 {
 private:
  sockaddr_in addr_;

 public:
  explicit SockAddrV4(sockaddr_in addr) : addr_{addr} {}
  SockAddrV4() { std::memset(&addr_, '\0', sizeof(addr_)); }

  static SockAddrV4 Loopback();
  static SockAddrV4 InaddrAny();

  in_port_t Port() const { return ntohs(addr_.sin_port); }

  std::string Addr() const {
    char buf[INET_ADDRSTRLEN];
    auto const *s = system::inet_ntop(static_cast<std::int32_t>(SockDomain::kV4), &addr_.sin_addr,
                                      buf, INET_ADDRSTRLEN);
    if (s == nullptr) {
      system::ThrowAtError("inet_ntop");
    }
    return {buf};
  }
  sockaddr_in const &Handle() const { return addr_; }
};

/**
 * \brief Address for TCP socket, can be either IPv4 or IPv6.
 */
class SockAddress {
 private:
  SockAddrV6 v6_;
  SockAddrV4 v4_;
  SockDomain domain_{SockDomain::kV4};

 public:
  SockAddress() = default;
  explicit SockAddress(SockAddrV6 const &addr) : v6_{addr}, domain_{SockDomain::kV6} {}
  explicit SockAddress(SockAddrV4 const &addr) : v4_{addr} {}

  auto Domain() const { return domain_; }

  bool IsV4() const { return Domain() == SockDomain::kV4; }
  bool IsV6() const { return !IsV4(); }

  auto const &V4() const { return v4_; }
  auto const &V6() const { return v6_; }
};

/**
 * \brief TCP socket for simple communication.
 */
class TCPSocket {
 public:
  using HandleT = system::SocketT;

 private:
  HandleT handle_{InvalidSocket()};
  // There's reliable no way to extract domain from a socket without first binding that
  // socket on macos.
#if defined(__APPLE__)
  SockDomain domain_{SockDomain::kV4};
#endif

  constexpr static HandleT InvalidSocket() { return -1; }

  explicit TCPSocket(HandleT newfd) : handle_{newfd} {}

 public:
  TCPSocket() = default;
  /**
   * \brief Return the socket domain.
   */
  auto Domain() const -> SockDomain {
    auto ret_iafamily = [](std::int32_t domain) {
      switch (domain) {
        case AF_INET:
          return SockDomain::kV4;
        case AF_INET6:
          return SockDomain::kV6;
        default: {
          LOG(FATAL) << "Unknown IA family.";
        }
      }
      return SockDomain::kV4;
    };

#if defined(_WIN32)
    WSAPROTOCOL_INFOA info;
    socklen_t len = sizeof(info);
    xgboost_CHECK_SYS_CALL(
        getsockopt(handle_, SOL_SOCKET, SO_PROTOCOL_INFO, reinterpret_cast<char *>(&info), &len),
        0);
    return ret_iafamily(info.iAddressFamily);
#elif defined(__APPLE__)
    return domain_;
#elif defined(__unix__)
#ifndef __PASE__
    std::int32_t domain;
    socklen_t len = sizeof(domain);
    xgboost_CHECK_SYS_CALL(
        getsockopt(handle_, SOL_SOCKET, SO_DOMAIN, reinterpret_cast<char *>(&domain), &len), 0);
    return ret_iafamily(domain);
#else
    struct sockaddr sa;
    socklen_t sizeofsa = sizeof(sa);
    xgboost_CHECK_SYS_CALL(getsockname(handle_, &sa, &sizeofsa), 0);
    if (sizeofsa < sizeof(uchar_t) * 2) {
      return ret_iafamily(AF_INET);
    }
    return ret_iafamily(sa.sa_family);
#endif  // __PASE__
#else
    LOG(FATAL) << "Unknown platform.";
    return ret_iafamily(AF_INET);
#endif  // platforms
  }

  bool IsClosed() const { return handle_ == InvalidSocket(); }

  /** \brief get last error code if any */
  std::int32_t GetSockError() const {
    std::int32_t error = 0;
    socklen_t len = sizeof(error);
    xgboost_CHECK_SYS_CALL(
        getsockopt(handle_, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&error), &len), 0);
    return error;
  }
  /** \brief check if anything bad happens */
  bool BadSocket() const {
    if (IsClosed()) return true;
    std::int32_t err = GetSockError();
    if (err == EBADF || err == EINTR) return true;
    return false;
  }

  void SetNonBlock() {
    bool non_block{true};
#if defined(_WIN32)
    u_long mode = non_block ? 1 : 0;
    xgboost_CHECK_SYS_CALL(ioctlsocket(handle_, FIONBIO, &mode), NO_ERROR);
#else
    std::int32_t flag = fcntl(handle_, F_GETFL, 0);
    if (flag == -1) {
      system::ThrowAtError("fcntl");
    }
    if (non_block) {
      flag |= O_NONBLOCK;
    } else {
      flag &= ~O_NONBLOCK;
    }
    if (fcntl(handle_, F_SETFL, flag) == -1) {
      system::ThrowAtError("fcntl");
    }
#endif  // _WIN32
  }

  void SetKeepAlive() {
    std::int32_t keepalive = 1;
    xgboost_CHECK_SYS_CALL(setsockopt(handle_, SOL_SOCKET, SO_KEEPALIVE,
                                      reinterpret_cast<char *>(&keepalive), sizeof(keepalive)),
                           0);
  }

  void SetNoDelay() {
    std::int32_t tcp_no_delay = 1;
    xgboost_CHECK_SYS_CALL(
        setsockopt(handle_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char *>(&tcp_no_delay),
                   sizeof(tcp_no_delay)),
        0);
  }

  /**
   * \brief Accept new connection, returns a new TCP socket for the new connection.
   */
  TCPSocket Accept() {
    HandleT newfd = accept(handle_, nullptr, nullptr);
    if (newfd == InvalidSocket()) {
      system::ThrowAtError("accept");
    }
    TCPSocket newsock{newfd};
    return newsock;
  }

  ~TCPSocket() {
    if (!IsClosed()) {
      Close();
    }
  }

  TCPSocket(TCPSocket const &that) = delete;
  TCPSocket(TCPSocket &&that) noexcept(true) { std::swap(this->handle_, that.handle_); }
  TCPSocket &operator=(TCPSocket const &that) = delete;
  TCPSocket &operator=(TCPSocket &&that) {
    std::swap(this->handle_, that.handle_);
    return *this;
  }
  /**
   * \brief Return the native socket file descriptor.
   */
  HandleT const &Handle() const { return handle_; }
  /**
   * \brief Listen to incoming requests. Should be called after bind.
   */
  void Listen(std::int32_t backlog = 16) { xgboost_CHECK_SYS_CALL(listen(handle_, backlog), 0); }
  /**
   * \brief Bind socket to INADDR_ANY, return the port selected by the OS.
   */
  in_port_t BindHost() {
    if (Domain() == SockDomain::kV6) {
      auto addr = SockAddrV6::InaddrAny();
      auto handle = reinterpret_cast<sockaddr const *>(&addr.Handle());
      xgboost_CHECK_SYS_CALL(
          bind(handle_, handle, sizeof(std::remove_reference_t<decltype(addr.Handle())>)), 0);

      sockaddr_in6 res_addr;
      socklen_t addrlen = sizeof(res_addr);
      xgboost_CHECK_SYS_CALL(
          getsockname(handle_, reinterpret_cast<sockaddr *>(&res_addr), &addrlen), 0);
      return ntohs(res_addr.sin6_port);
    } else {
      auto addr = SockAddrV4::InaddrAny();
      auto handle = reinterpret_cast<sockaddr const *>(&addr.Handle());
      xgboost_CHECK_SYS_CALL(
          bind(handle_, handle, sizeof(std::remove_reference_t<decltype(addr.Handle())>)), 0);

      sockaddr_in res_addr;
      socklen_t addrlen = sizeof(res_addr);
      xgboost_CHECK_SYS_CALL(
          getsockname(handle_, reinterpret_cast<sockaddr *>(&res_addr), &addrlen), 0);
      return ntohs(res_addr.sin_port);
    }
  }
  /**
   * \brief Send data, without error then all data should be sent.
   */
  auto SendAll(void const *buf, std::size_t len) {
    char const *_buf = reinterpret_cast<const char *>(buf);
    std::size_t ndone = 0;
    while (ndone < len) {
      ssize_t ret = send(handle_, _buf, len - ndone, 0);
      if (ret == -1) {
        if (system::LastErrorWouldBlock()) {
          return ndone;
        }
        system::ThrowAtError("send");
      }
      _buf += ret;
      ndone += ret;
    }
    return ndone;
  }
  /**
   * \brief Receive data, without error then all data should be received.
   */
  auto RecvAll(void *buf, std::size_t len) {
    char *_buf = reinterpret_cast<char *>(buf);
    std::size_t ndone = 0;
    while (ndone < len) {
      ssize_t ret = recv(handle_, _buf, len - ndone, MSG_WAITALL);
      if (ret == -1) {
        if (system::LastErrorWouldBlock()) {
          return ndone;
        }
        system::ThrowAtError("recv");
      }
      if (ret == 0) {
        return ndone;
      }
      _buf += ret;
      ndone += ret;
    }
    return ndone;
  }
  /**
   * \brief Send data using the socket
   * \param buf the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually sent return -1 if error occurs
   */
  auto Send(const void *buf_, std::size_t len, std::int32_t flags = 0) {
    const char *buf = reinterpret_cast<const char *>(buf_);
    return send(handle_, buf, len, flags);
  }
  /**
   * \brief receive data using the socket
   * \param buf the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually received return -1 if error occurs
   */
  auto Recv(void *buf, std::size_t len, std::int32_t flags = 0) {
    char *_buf = reinterpret_cast<char *>(buf);
    return recv(handle_, _buf, len, flags);
  }
  /**
   * \brief Send string, format is matched with the Python socket wrapper in RABIT.
   */
  std::size_t Send(StringView str);
  /**
   * \brief Receive string, format is matched with the Python socket wrapper in RABIT.
   */
  std::size_t Recv(std::string *p_str);
  /**
   * \brief Close the socket, called automatically in destructor if the socket is not closed.
   */
  void Close() {
    if (InvalidSocket() != handle_) {
      xgboost_CHECK_SYS_CALL(system::CloseSocket(handle_), 0);
      handle_ = InvalidSocket();
    }
  }
  /**
   * \brief Create a TCP socket on specified domain.
   */
  static TCPSocket Create(SockDomain domain) {
#if defined(xgboost_IS_MINGW)
    MingWError();
    return {};
#else
    auto fd = socket(static_cast<std::int32_t>(domain), SOCK_STREAM, 0);
    if (fd == InvalidSocket()) {
      system::ThrowAtError("socket");
    }

    TCPSocket socket{fd};
#if defined(__APPLE__)
    socket.domain_ = domain;
#endif  // defined(__APPLE__)
    return socket;
#endif  // defined(xgboost_IS_MINGW)
  }
};

/**
 * \brief Connect to remote address, returns the error code if failed (no exception is
 *        raised so that we can retry).
 */
std::error_code Connect(SockAddress const &addr, TCPSocket *out);

/**
 * \brief Get the local host name.
 */
inline std::string GetHostName() {
  char buf[HOST_NAME_MAX];
  xgboost_CHECK_SYS_CALL(gethostname(&buf[0], HOST_NAME_MAX), 0);
  return buf;
}
}  // namespace collective
}  // namespace xgboost

#undef xgboost_CHECK_SYS_CALL

#if defined(xgboost_IS_MINGW)
#undef xgboost_IS_MINGW
#endif
