/**
 * Copyright 2022-2025, XGBoost Contributors
 */
#pragma once

#include <cerrno>        // errno, EINTR, EBADF
#include <climits>       // HOST_NAME_MAX
#include <cstddef>       // std::size_t
#include <cstdint>       // std::int32_t, std::uint16_t
#include <cstring>       // memset
#include <string>        // std::string
#include <system_error>  // std::error_code, std::system_category
#include <utility>       // std::swap

#if defined(__linux__)
#include <sys/ioctl.h>  // for TIOCOUTQ, FIONREAD
#endif                  // defined(__linux__)

#if defined(_WIN32)
// Guard the include.
#include <xgboost/windefs.h>
// Socket API
#include <winsock2.h>
#include <ws2tcpip.h>

using in_port_t = std::uint16_t;

#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif  // _MSC_VER

#if !defined(xgboost_IS_MINGW)
using ssize_t = int;
#endif  // !xgboost_IS_MINGW()

#else  // UNIX

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

#include "xgboost/base.h"               // XGBOOST_EXPECT
#include "xgboost/collective/result.h"  // for Result
#include "xgboost/logging.h"            // LOG
#include "xgboost/string_view.h"        // StringView

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

[[nodiscard]] inline collective::Result FailWithCode(std::string msg) {
  return collective::Fail(std::move(msg), std::error_code{LastError(), std::system_category()});
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
#define INVALID_SOCKET -1
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

inline std::int32_t ShutdownSocket(SocketT fd) {
#if defined(_WIN32)
  auto rc = shutdown(fd, SD_BOTH);
  if (rc != 0 && LastError() == WSANOTINITIALISED) {
    return 0;
  }
#else
  auto rc = shutdown(fd, SHUT_RDWR);
  if (rc != 0 && LastError() == ENOTCONN) {
    return 0;
  }
#endif
  return rc;
}

inline bool ErrorWouldBlock(std::int32_t errsv) noexcept(true) {
#ifdef _WIN32
  return errsv == WSAEWOULDBLOCK;
#else
  return errsv == EAGAIN || errsv == EWOULDBLOCK || errsv == EINPROGRESS;
#endif  // _WIN32
}

inline bool LastErrorWouldBlock() {
  int errsv = LastError();
  return ErrorWouldBlock(errsv);
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

  [[nodiscard]] in_port_t Port() const { return ntohs(addr_.sin_port); }

  [[nodiscard]] std::string Addr() const {
    char buf[INET_ADDRSTRLEN];
    auto const *s = system::inet_ntop(static_cast<std::int32_t>(SockDomain::kV4), &addr_.sin_addr,
                                      buf, INET_ADDRSTRLEN);
    if (s == nullptr) {
      system::ThrowAtError("inet_ntop");
    }
    return {buf};
  }
  [[nodiscard]] sockaddr_in const &Handle() const { return addr_; }
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

  [[nodiscard]] auto Domain() const { return domain_; }

  [[nodiscard]] bool IsV4() const { return Domain() == SockDomain::kV4; }
  [[nodiscard]] bool IsV6() const { return !IsV4(); }

  [[nodiscard]] auto const &V4() const { return v4_; }
  [[nodiscard]] auto const &V6() const { return v6_; }
};

/**
 * \brief TCP socket for simple communication.
 */
class TCPSocket {
 public:
  using HandleT = system::SocketT;

 private:
  HandleT handle_{InvalidSocket()};
  bool non_blocking_{false};
  // There's reliable no way to extract domain from a socket without first binding that
  // socket on macos.
#if defined(__APPLE__)
  SockDomain domain_{SockDomain::kV4};
#endif

  constexpr static HandleT InvalidSocket() { return INVALID_SOCKET; }

  explicit TCPSocket(HandleT newfd) : handle_{newfd} {}

 public:
  TCPSocket() = default;
  /**
   * \brief Return the socket domain.
   */
  [[nodiscard]] auto Domain() const -> SockDomain {
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
    WSAPROTOCOL_INFOW info;
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
        getsockopt(this->Handle(), SOL_SOCKET, SO_DOMAIN, reinterpret_cast<char *>(&domain), &len),
        0);
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

  [[nodiscard]] bool IsClosed() const { return handle_ == InvalidSocket(); }

  /** @brief get last error code if any */
  [[nodiscard]] Result GetSockError() const {
    std::int32_t optval = 0;
    socklen_t len = sizeof(optval);
    auto ret = getsockopt(handle_, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&optval), &len);
    if (ret != 0) {
      auto errc = std::error_code{system::LastError(), std::system_category()};
      return Fail("Failed to retrieve socket error.", std::move(errc));
    }
    if (optval != 0) {
      auto errc = std::error_code{optval, std::system_category()};
      return Fail("Socket error.", std::move(errc));
    }
    return Success();
  }

  /** \brief check if anything bad happens */
  [[nodiscard]] bool BadSocket() const {
    if (IsClosed()) {
      return true;
    }
    auto err = GetSockError();
    if (err.Code() == std::error_code{EBADF, std::system_category()} ||  // NOLINT
        err.Code() == std::error_code{EINTR, std::system_category()}) {  // NOLINT
      return true;
    }
    return false;
  }

  [[nodiscard]] Result NonBlocking(bool non_block) {
#if defined(_WIN32)
    u_long mode = non_block ? 1 : 0;
    if (ioctlsocket(handle_, FIONBIO, &mode) != NO_ERROR) {
      return system::FailWithCode("Failed to set socket to non-blocking.");
    }
#else
    std::int32_t flag = fcntl(handle_, F_GETFL, 0);
    auto rc = flag;
    if (rc == -1) {
      return system::FailWithCode("Failed to get socket flag.");
    }
    if (non_block) {
      flag |= O_NONBLOCK;
    } else {
      flag &= ~O_NONBLOCK;
    }
    rc = fcntl(handle_, F_SETFL, flag);
    if (rc == -1) {
      return system::FailWithCode("Failed to set socket to non-blocking.");
    }
#endif  // _WIN32
    non_blocking_ = non_block;
    return Success();
  }
  [[nodiscard]] bool NonBlocking() const { return non_blocking_; }
  [[nodiscard]] Result RecvTimeout(std::chrono::seconds timeout) {
    // https://stackoverflow.com/questions/2876024/linux-is-there-a-read-or-recv-from-socket-with-timeout
#if defined(_WIN32)
    DWORD tv = timeout.count() * 1000;
    auto rc =
        setsockopt(Handle(), SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char *>(&tv), sizeof(tv));
#else
    struct timeval tv;
    tv.tv_sec = timeout.count();
    tv.tv_usec = 0;
    auto rc = setsockopt(Handle(), SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char const *>(&tv),
                         sizeof(tv));
#endif
    if (rc != 0) {
      return system::FailWithCode("Failed to set timeout on recv.");
    }
    return Success();
  }

  [[nodiscard]] Result SetBufSize(std::int32_t n_bytes) {
    auto rc = setsockopt(this->Handle(), SOL_SOCKET, SO_SNDBUF, reinterpret_cast<char *>(&n_bytes),
                         sizeof(n_bytes));
    if (rc != 0) {
      return system::FailWithCode("Failed to set send buffer size.");
    }
    rc = setsockopt(this->Handle(), SOL_SOCKET, SO_RCVBUF, reinterpret_cast<char *>(&n_bytes),
                    sizeof(n_bytes));
    if (rc != 0) {
      return system::FailWithCode("Failed to set recv buffer size.");
    }
    return Success();
  }

  [[nodiscard]] Result SendBufSize(std::int32_t *n_bytes) {
    socklen_t optlen;
    auto rc = getsockopt(this->Handle(), SOL_SOCKET, SO_SNDBUF, reinterpret_cast<char *>(n_bytes),
                         &optlen);
    if (rc != 0 || optlen != sizeof(std::int32_t)) {
      return system::FailWithCode("getsockopt");
    }
    return Success();
  }
  [[nodiscard]] Result RecvBufSize(std::int32_t *n_bytes) {
    socklen_t optlen;
    auto rc = getsockopt(this->Handle(), SOL_SOCKET, SO_RCVBUF, reinterpret_cast<char *>(n_bytes),
                         &optlen);
    if (rc != 0 || optlen != sizeof(std::int32_t)) {
      return system::FailWithCode("getsockopt");
    }
    return Success();
  }
#if defined(__linux__)
  [[nodiscard]] Result PendingSendSize(std::int32_t *n_bytes) const {
    return ioctl(this->Handle(), TIOCOUTQ, n_bytes) == 0 ? Success()
                                                         : system::FailWithCode("ioctl");
  }
  [[nodiscard]] Result PendingRecvSize(std::int32_t *n_bytes) const {
    return ioctl(this->Handle(), FIONREAD, n_bytes) == 0 ? Success()
                                                         : system::FailWithCode("ioctl");
  }
#endif  // defined(__linux__)

  [[nodiscard]] Result SetKeepAlive() {
    std::int32_t keepalive = 1;
    auto rc = setsockopt(handle_, SOL_SOCKET, SO_KEEPALIVE, reinterpret_cast<char *>(&keepalive),
                         sizeof(keepalive));
    if (rc != 0) {
      return system::FailWithCode("Failed to set TCP keeaplive.");
    }
    return Success();
  }

  [[nodiscard]] Result SetNoDelay(std::int32_t no_delay = 1) {
    auto rc = setsockopt(handle_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char *>(&no_delay),
                         sizeof(no_delay));
    if (rc != 0) {
      return system::FailWithCode("Failed to set TCP no delay.");
    }
    return Success();
  }

  /**
   * \brief Accept new connection, returns a new TCP socket for the new connection.
   */
  TCPSocket Accept() {
    SockAddress addr;
    TCPSocket newsock;
    auto rc = this->Accept(&newsock, &addr);
    SafeColl(rc);
    return newsock;
  }

  [[nodiscard]] Result Accept(TCPSocket *out, SockAddress *addr) {
#if defined(_WIN32)
    auto interrupt = WSAEINTR;
#else
    auto interrupt = EINTR;
#endif
    if (this->Domain() == SockDomain::kV4) {
      struct sockaddr_in caddr;
      socklen_t caddr_len = sizeof(caddr);
      HandleT newfd = accept(Handle(), reinterpret_cast<sockaddr *>(&caddr), &caddr_len);
      if (newfd == InvalidSocket() && system::LastError() != interrupt) {
        return system::FailWithCode("Failed to accept.");
      }
      *addr = SockAddress{SockAddrV4{caddr}};
      *out = TCPSocket{newfd};
    } else {
      struct sockaddr_in6 caddr;
      socklen_t caddr_len = sizeof(caddr);
      HandleT newfd = accept(Handle(), reinterpret_cast<sockaddr *>(&caddr), &caddr_len);
      if (newfd == InvalidSocket() && system::LastError() != interrupt) {
        return system::FailWithCode("Failed to accept.");
      }
      *addr = SockAddress{SockAddrV6{caddr}};
      *out = TCPSocket{newfd};
    }
    // On MacOS, this is automatically set to async socket if the parent socket is async
    // We make sure all socket are blocking by default.
    //
    // On Windows, a closed socket is returned during shutdown. We guard against it when
    // setting non-blocking.
    if (!out->IsClosed()) {
      return out->NonBlocking(false);
    }
    return Success();
  }

  ~TCPSocket() {
    if (!IsClosed()) {
      auto rc = this->Close();
      if (!rc.OK()) {
        LOG(WARNING) << rc.Report();
      }
    }
  }

  TCPSocket(TCPSocket const &that) = delete;
  TCPSocket(TCPSocket &&that) noexcept(true) { std::swap(this->handle_, that.handle_); }
  TCPSocket &operator=(TCPSocket const &that) = delete;
  TCPSocket &operator=(TCPSocket &&that) noexcept(true) {
    std::swap(this->handle_, that.handle_);
    return *this;
  }
  /**
   * @brief Return the native socket file descriptor.
   */
  [[nodiscard]] HandleT const &Handle() const { return handle_; }
  /**
   * @brief Listen to incoming requests. Should be called after bind.
   *
   *   Both the default and minimum backlog is set to 256.
   */
  [[nodiscard]] Result Listen(std::int32_t backlog = 256);
  /**
   * @brief Bind socket to INADDR_ANY, return the port selected by the OS.
   */
  [[nodiscard]] Result BindHost(std::int32_t* p_out) {
    // Use int32 instead of in_port_t for consistency. We take port as parameter from
    // users using other languages, the port is usually stored and passed around as int.
    if (Domain() == SockDomain::kV6) {
      auto addr = SockAddrV6::InaddrAny();
      auto handle = reinterpret_cast<sockaddr const *>(&addr.Handle());
      if (bind(handle_, handle, sizeof(std::remove_reference_t<decltype(addr.Handle())>)) != 0) {
        return system::FailWithCode("bind failed.");
      }

      sockaddr_in6 res_addr;
      socklen_t addrlen = sizeof(res_addr);
      if (getsockname(handle_, reinterpret_cast<sockaddr *>(&res_addr), &addrlen) != 0) {
        return system::FailWithCode("getsockname failed.");
      }
      *p_out = ntohs(res_addr.sin6_port);
    } else {
      auto addr = SockAddrV4::InaddrAny();
      auto handle = reinterpret_cast<sockaddr const *>(&addr.Handle());
      if (bind(handle_, handle, sizeof(std::remove_reference_t<decltype(addr.Handle())>)) != 0) {
        return system::FailWithCode("bind failed.");
      }

      sockaddr_in res_addr;
      socklen_t addrlen = sizeof(res_addr);
      if (getsockname(handle_, reinterpret_cast<sockaddr *>(&res_addr), &addrlen) != 0) {
        return system::FailWithCode("getsockname failed.");
      }
      *p_out = ntohs(res_addr.sin_port);
    }

    return Success();
  }

  [[nodiscard]] auto Port() const {
    if (this->Domain() == SockDomain::kV4) {
      sockaddr_in res_addr;
      socklen_t addrlen = sizeof(res_addr);
      auto code = getsockname(handle_, reinterpret_cast<sockaddr *>(&res_addr), &addrlen);
      if (code != 0) {
        return std::make_pair(system::FailWithCode("getsockname"), std::int32_t{0});
      }
      return std::make_pair(Success(), std::int32_t{ntohs(res_addr.sin_port)});
    } else {
      sockaddr_in6 res_addr;
      socklen_t addrlen = sizeof(res_addr);
      auto code = getsockname(handle_, reinterpret_cast<sockaddr *>(&res_addr), &addrlen);
      if (code != 0) {
        return std::make_pair(system::FailWithCode("getsockname"), std::int32_t{0});
      }
      return std::make_pair(Success(), std::int32_t{ntohs(res_addr.sin6_port)});
    }
  }
  /**
   * @brief Bind the socket to the address.
   *
   * @param ip[in]        The IP address.
   * @param port [in,out] Let the system choose a port if this parameter is set to 0.
   */
  [[nodiscard]] Result Bind(StringView ip, std::int32_t *port) {
    // bind socket handle_ to ip
    auto addr = MakeSockAddress(ip, *port);
    std::int32_t errc{0};
    if (addr.IsV4()) {
      auto handle = reinterpret_cast<sockaddr const *>(&addr.V4().Handle());
      errc = bind(handle_, handle, sizeof(std::remove_reference_t<decltype(addr.V4().Handle())>));
    } else {
      auto handle = reinterpret_cast<sockaddr const *>(&addr.V6().Handle());
      errc = bind(handle_, handle, sizeof(std::remove_reference_t<decltype(addr.V6().Handle())>));
    }
    if (errc != 0) {
      return system::FailWithCode("Failed to bind socket.");
    }
    auto [rc, new_port] = this->Port();
    if (!rc.OK()) {
      return std::move(rc);
    }
    if (*port == 0) {
      *port = new_port;
      return Success();
    }
    if (*port != new_port) {
      return Fail("Got an invalid port from bind.");
    }
    return Success();
  }

  /**
   * @brief Send data, without error then all data should be sent.
   */
  [[nodiscard]] Result SendAll(void const *buf, std::size_t len, std::size_t *n_sent) {
    char const *_buf = reinterpret_cast<const char *>(buf);
    std::size_t &ndone = *n_sent;
    ndone = 0;
    while (ndone < len) {
      ssize_t ret = send(handle_, _buf, len - ndone, 0);
      if (ret == -1) {
        if (system::LastErrorWouldBlock()) {
          return Success();
        }
        return system::FailWithCode("send");
      }
      _buf += ret;
      ndone += ret;
    }
    return Success();
  }
  /**
   * @brief Receive data, without error then all data should be received.
   */
  [[nodiscard]] Result RecvAll(void *buf, std::size_t len, std::size_t *n_recv) {
    char *_buf = reinterpret_cast<char *>(buf);
    std::size_t &ndone = *n_recv;
    ndone = 0;
    while (ndone < len) {
      ssize_t ret = recv(handle_, _buf, len - ndone, MSG_WAITALL);
      if (ret == -1) {
        if (system::LastErrorWouldBlock()) {
          return Success();
        }
        return system::FailWithCode("recv");
      }
      if (ret == 0) {
        return Success();
      }
      _buf += ret;
      ndone += ret;
    }
    return Success();
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
    char *_buf = static_cast<char *>(buf);
    // See https://github.com/llvm/llvm-project/issues/104241 for skipped tidy analysis
    // NOLINTBEGIN(clang-analyzer-unix.BlockInCriticalSection)
    return recv(handle_, _buf, len, flags);
    // NOLINTEND(clang-analyzer-unix.BlockInCriticalSection)
  }
  /**
   * \brief Send string, format is matched with the Python socket wrapper in RABIT.
   */
  std::size_t Send(StringView str);
  /**
   * @brief Receive string, format is matched with the Python socket wrapper in RABIT.
   */
  [[nodiscard]] Result Recv(std::string *p_str);
  /**
   * @brief Close the socket, called automatically in destructor if the socket is not closed.
   */
  [[nodiscard]] Result Close() {
    if (InvalidSocket() != handle_) {
      auto rc = system::CloseSocket(handle_);
#if defined(_WIN32)
      // it's possible that we close TCP sockets after finalizing WSA due to detached thread.
      if (rc != 0 && system::LastError() != WSANOTINITIALISED) {
        return system::FailWithCode("Failed to close the socket.");
      }
#else
      if (rc != 0) {
        return system::FailWithCode("Failed to close the socket.");
      }
#endif
      handle_ = InvalidSocket();
    }
    return Success();
  }
  /**
   * @brief Call shutdown on the socket.
   */
  [[nodiscard]] Result Shutdown() {
    if (this->IsClosed()) {
      return Success();
    }
    auto rc = system::ShutdownSocket(this->Handle());
#if defined(_WIN32)
    // Windows cannot shutdown a socket if it's not connected.
    if (rc == -1 && system::LastError() == WSAENOTCONN) {
      return Success();
    }
#endif
    if (rc != 0) {
      return system::FailWithCode("Failed to shutdown socket.");
    }
    return Success();
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

  static TCPSocket *CreatePtr(SockDomain domain) {
#if defined(xgboost_IS_MINGW)
    MingWError();
    return nullptr;
#else
    auto fd = socket(static_cast<std::int32_t>(domain), SOCK_STREAM, 0);
    if (fd == InvalidSocket()) {
      system::ThrowAtError("socket");
    }
    auto socket = new TCPSocket{fd};

#if defined(__APPLE__)
    socket->domain_ = domain;
#endif  // defined(__APPLE__)
    return socket;
#endif  // defined(xgboost_IS_MINGW)
  }
};

/**
 * @brief Connect to remote address, returns the error code if failed.
 *
 * @param host   Host IP address.
 * @param port   Connection port.
 * @param retry  Number of retries to attempt.
 * @param timeout  Timeout of each connection attempt.
 * @param out_conn Output socket if the connection is successful. Value is invalid and undefined if
 *                 the connection failed.
 *
 * @return Connection status.
 */
[[nodiscard]] Result Connect(xgboost::StringView host, std::int32_t port, std::int32_t retry,
                             std::chrono::seconds timeout,
                             xgboost::collective::TCPSocket *out_conn);

/**
 * @brief Get the local host name.
 */
[[nodiscard]] Result GetHostName(std::string *p_out);

/**
 * @brief inet_ntop
 */
template <typename H>
Result INetNToP(H const &host, std::string *p_out) {
  std::string &ip = *p_out;
  switch (host->h_addrtype) {
    case AF_INET: {
      auto addr = reinterpret_cast<struct in_addr *>(host->h_addr_list[0]);
      char str[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, addr, str, INET_ADDRSTRLEN);
      ip = str;
      break;
    }
    case AF_INET6: {
      auto addr = reinterpret_cast<struct in6_addr *>(host->h_addr_list[0]);
      char str[INET6_ADDRSTRLEN];
      inet_ntop(AF_INET6, addr, str, INET6_ADDRSTRLEN);
      ip = str;
      break;
    }
    default: {
      return Fail("Invalid address type.");
    }
  }
  return Success();
}
}  // namespace collective
}  // namespace xgboost

#undef xgboost_CHECK_SYS_CALL
