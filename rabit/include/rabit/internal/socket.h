/*!
 *  Copyright (c) 2014-2019 by Contributors
 * \file socket.h
 * \brief this file aims to provide a wrapper of sockets
 * \author Tianqi Chen
 */
#ifndef RABIT_INTERNAL_SOCKET_H_
#define RABIT_INTERNAL_SOCKET_H_
#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>

#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif  // _MSC_VER

#else

#include <fcntl.h>
#include <netdb.h>
#include <cerrno>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#endif  // defined(_WIN32)

#include <string>
#include <cstring>
#include <vector>
#include <chrono>
#include <unordered_map>
#include "utils.h"

#if defined(_WIN32) && !defined(__MINGW32__)
typedef int ssize_t;
#endif  // defined(_WIN32) || defined(__MINGW32__)

#if defined(_WIN32)
using sock_size_t = int;

#else

#include <sys/poll.h>
using SOCKET = int;
using sock_size_t = size_t;  // NOLINT
#endif  // defined(_WIN32)

#define IS_MINGW() defined(__MINGW32__)

#if IS_MINGW()
inline void MingWError() {
  throw dmlc::Error("Distributed training on mingw is not supported.");
}
#endif  // IS_MINGW()

#if IS_MINGW() && !defined(POLLRDNORM) && !defined(POLLRDBAND)
/*
 * On later mingw versions poll should be supported (with bugs).  See:
 * https://stackoverflow.com/a/60623080
 *
 * But right now the mingw distributed with R 3.6 doesn't support it.
 * So we just give a warning and provide dummy implementation to get
 * compilation passed.  Otherwise we will have to provide a stub for
 * RABIT.
 *
 * Even on mingw version that has these structures and flags defined,
 * functions like `send` and `listen` might have unresolved linkage to
 * their implementation.  So supporting mingw is quite difficult at
 * the time of writing.
 */
#pragma message("Distributed training on mingw is not supported.")
typedef struct pollfd {
  SOCKET fd;
  short  events;
  short  revents;
} WSAPOLLFD, *PWSAPOLLFD, *LPWSAPOLLFD;

// POLLRDNORM | POLLRDBAND
#define POLLIN    (0x0100 | 0x0200)
#define POLLPRI    0x0400
// POLLWRNORM
#define POLLOUT    0x0010

inline const char *inet_ntop(int, const void *, char *, size_t) {
  MingWError();
  return nullptr;
}
#endif  // IS_MINGW() && !defined(POLLRDNORM) && !defined(POLLRDBAND)

namespace rabit {
namespace utils {

static constexpr int kInvalidSocket = -1;

template <typename PollFD>
int PollImpl(PollFD *pfd, int nfds, std::chrono::seconds timeout) {
#if defined(_WIN32)

#if IS_MINGW()
  MingWError();
  return -1;
#else
  return WSAPoll(pfd, nfds, std::chrono::milliseconds(timeout).count());
#endif  // IS_MINGW()

#else
  return poll(pfd, nfds, std::chrono::milliseconds(timeout).count());
#endif  // IS_MINGW()
}

/*! \brief data structure for network address */
struct SockAddr {
  sockaddr_in addr;
  // constructor
  SockAddr() = default;
  SockAddr(const char *url, int port) {
    this->Set(url, port);
  }
  inline static std::string GetHostName() {
    std::string buf; buf.resize(256);
#if !IS_MINGW()
    utils::Check(gethostname(&buf[0], 256) != -1, "fail to get host name");
#endif  // IS_MINGW()
    return std::string(buf.c_str());
  }
  /*!
   * \brief set the address
   * \param url the url of the address
   * \param port the port of address
   */
  inline void Set(const char *host, int port) {
#if !IS_MINGW()
    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_protocol = SOCK_STREAM;
    addrinfo *res = nullptr;
    int sig = getaddrinfo(host, nullptr, &hints, &res);
    Check(sig == 0 && res != nullptr, "cannot obtain address of %s", host);
    Check(res->ai_family == AF_INET, "Does not support IPv6");
    memcpy(&addr, res->ai_addr, res->ai_addrlen);
    addr.sin_port = htons(port);
    freeaddrinfo(res);
#endif  // !IS_MINGW()
  }
  /*! \brief return port of the address*/
  inline int Port() const {
    return ntohs(addr.sin_port);
  }
  /*! \return a string representation of the address */
  inline std::string AddrStr() const {
    std::string buf; buf.resize(256);
#ifdef _WIN32
    const char *s = inet_ntop(AF_INET, (PVOID)&addr.sin_addr,
                    &buf[0], buf.length());
#else
    const char *s = inet_ntop(AF_INET, &addr.sin_addr,
                              &buf[0], buf.length());
#endif  // _WIN32
    Assert(s != nullptr, "cannot decode address");
    return std::string(s);
  }
};

/*!
 * \brief base class containing common operations of TCP and UDP sockets
 */
class Socket {
 public:
  /*! \brief the file descriptor of socket */
  SOCKET sockfd;
  // default conversion to int
  operator SOCKET() const {  // NOLINT
    return sockfd;
  }
  /*!
   * \return last error of socket operation
   */
  inline static int GetLastError() {
#ifdef _WIN32

#if IS_MINGW()
    MingWError();
    return -1;
#else
    return WSAGetLastError();
#endif  // IS_MINGW()

#else
    return errno;
#endif  // _WIN32
  }
  /*! \return whether last error was would block */
  inline static bool LastErrorWouldBlock() {
    int errsv = GetLastError();
#ifdef _WIN32
    return errsv == WSAEWOULDBLOCK;
#else
    return errsv == EAGAIN || errsv == EWOULDBLOCK;
#endif  // _WIN32
  }
  /*!
   * \brief start up the socket module
   *   call this before using the sockets
   */
  inline static void Startup() {
#ifdef _WIN32
#if !IS_MINGW()
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) == -1) {
      Socket::Error("Startup");
    }
    if (LOBYTE(wsa_data.wVersion) != 2 || HIBYTE(wsa_data.wVersion) != 2) {
    WSACleanup();
    utils::Error("Could not find a usable version of Winsock.dll\n");
    }
#endif  // !IS_MINGW()
#endif  // _WIN32
  }
  /*!
   * \brief shutdown the socket module after use, all sockets need to be closed
   */
  inline static void Finalize() {
#ifdef _WIN32
#if !IS_MINGW()
    WSACleanup();
#endif  // !IS_MINGW()
#endif  // _WIN32
  }
  /*!
   * \brief set this socket to use non-blocking mode
   * \param non_block whether set it to be non-block, if it is false
   *        it will set it back to block mode
   */
  inline void SetNonBlock(bool non_block) {
#ifdef _WIN32
#if !IS_MINGW()
    u_long mode = non_block ? 1 : 0;
    if (ioctlsocket(sockfd, FIONBIO, &mode) != NO_ERROR) {
      Socket::Error("SetNonBlock");
    }
#endif  // !IS_MINGW()
#else
    int flag = fcntl(sockfd, F_GETFL, 0);
    if (flag == -1) {
      Socket::Error("SetNonBlock-1");
    }
    if (non_block) {
      flag |= O_NONBLOCK;
    } else {
      flag &= ~O_NONBLOCK;
    }
    if (fcntl(sockfd, F_SETFL, flag) == -1) {
      Socket::Error("SetNonBlock-2");
    }
#endif  // _WIN32
  }
  /*!
   * \brief bind the socket to an address
   * \param addr
   */
  inline void Bind(const SockAddr &addr) {
#if !IS_MINGW()
    if (bind(sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
             sizeof(addr.addr)) == -1) {
      Socket::Error("Bind");
    }
#endif  // !IS_MINGW()
  }
  /*!
   * \brief try bind the socket to host, from start_port to end_port
   * \param start_port starting port number to try
   * \param end_port ending port number to try
   * \return the port successfully bind to, return -1 if failed to bind any port
   */
  inline int TryBindHost(int start_port, int end_port) {
    // TODO(tqchen) add prefix check
#if !IS_MINGW()
    for (int port = start_port; port < end_port; ++port) {
      SockAddr addr("0.0.0.0", port);
      if (bind(sockfd, reinterpret_cast<sockaddr*>(&addr.addr),
               sizeof(addr.addr)) == 0) {
        return port;
      }
#if defined(_WIN32)
      if (WSAGetLastError() != WSAEADDRINUSE) {
        Socket::Error("TryBindHost");
      }
#else
      if (errno != EADDRINUSE) {
        Socket::Error("TryBindHost");
      }
#endif  // defined(_WIN32)
    }
#endif  // !IS_MINGW()
    return -1;
  }
  /*! \brief get last error code if any */
  inline int GetSockError() const {
    int error = 0;
    socklen_t len = sizeof(error);
#if !IS_MINGW()
    if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR,
                   reinterpret_cast<char *>(&error), &len) != 0) {
      Error("GetSockError");
    }
#else
    // undefined reference to `_imp__getsockopt@20'
    MingWError();
#endif  // !IS_MINGW()
    return error;
  }
  /*! \brief check if anything bad happens */
  inline bool BadSocket() const {
    if (IsClosed()) return true;
    int err = GetSockError();
    if (err == EBADF || err == EINTR) return true;
    return false;
  }
  /*! \brief check if socket is already closed */
  inline bool IsClosed() const {
    return sockfd == kInvalidSocket;
  }
  /*! \brief close the socket */
  inline void Close() {
    if (sockfd != kInvalidSocket) {
#ifdef _WIN32
#if !IS_MINGW()
      closesocket(sockfd);
#endif  // !IS_MINGW()
#else
      close(sockfd);
#endif
      sockfd = kInvalidSocket;
    } else {
      Error("Socket::Close double close the socket or close without create");
    }
  }
  // report an socket error
  inline static void Error(const char *msg) {
    int errsv = GetLastError();
#ifdef _WIN32
    utils::Error("Socket %s Error:WSAError-code=%d", msg, errsv);
#else
    utils::Error("Socket %s Error:%s", msg, strerror(errsv));
#endif
  }

 protected:
  explicit Socket(SOCKET sockfd) : sockfd(sockfd) {
  }
};

/*!
 * \brief a wrapper of TCP socket that hopefully be cross platform
 */
class TCPSocket : public Socket{
 public:
  // constructor
  TCPSocket() : Socket(kInvalidSocket) {
  }
  explicit TCPSocket(SOCKET sockfd) : Socket(sockfd) {
  }
  /*!
   * \brief enable/disable TCP keepalive
   * \param keepalive whether to set the keep alive option on
   */
  void SetKeepAlive(bool keepalive) {
#if !IS_MINGW()
    int opt = static_cast<int>(keepalive);
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE,
                   reinterpret_cast<char*>(&opt), sizeof(opt)) < 0) {
      Socket::Error("SetKeepAlive");
    }
#endif  // !IS_MINGW()
  }
  inline void SetLinger(int timeout = 0) {
#if !IS_MINGW()
    struct linger sl;
    sl.l_onoff = 1;    /* non-zero value enables linger option in kernel */
    sl.l_linger = timeout;    /* timeout interval in seconds */
    if (setsockopt(sockfd, SOL_SOCKET, SO_LINGER, reinterpret_cast<char*>(&sl), sizeof(sl)) == -1) {
      Socket::Error("SO_LINGER");
    }
#endif  // !IS_MINGW()
  }
  /*!
   * \brief create the socket, call this before using socket
   * \param af domain
   */
  inline void Create(int af = PF_INET) {
#if !IS_MINGW()
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd == kInvalidSocket) {
      Socket::Error("Create");
    }
#endif  // !IS_MINGW()
  }
  /*!
   * \brief perform listen of the socket
   * \param backlog backlog parameter
   */
  inline void Listen(int backlog = 16) {
#if !IS_MINGW()
    listen(sockfd, backlog);
#endif  // !IS_MINGW()
  }
  /*! \brief get a new connection */
  TCPSocket Accept() {
#if !IS_MINGW()
    SOCKET newfd = accept(sockfd, nullptr, nullptr);
    if (newfd == kInvalidSocket) {
      Socket::Error("Accept");
    }
    return TCPSocket(newfd);
#else
    return TCPSocket();
#endif // !IS_MINGW()
  }
  /*!
   * \brief decide whether the socket is at OOB mark
   * \return 1 if at mark, 0 if not, -1 if an error occured
   */
  inline int AtMark() const {
#if !IS_MINGW()

#ifdef _WIN32
    unsigned long atmark;  // NOLINT(*)
    if (ioctlsocket(sockfd, SIOCATMARK, &atmark) != NO_ERROR) return -1;
#else
    int atmark;
    if (ioctl(sockfd, SIOCATMARK, &atmark) == -1) return -1;
#endif  // _WIN32

    return static_cast<int>(atmark);

#else
    return -1;
#endif  // !IS_MINGW()
  }
  /*!
   * \brief connect to an address
   * \param addr the address to connect to
   * \return whether connect is successful
   */
  inline bool Connect(const SockAddr &addr) {
#if !IS_MINGW()
    return connect(sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
                   sizeof(addr.addr)) == 0;
#else
    return false;
#endif  // !IS_MINGW()
  }
  /*!
   * \brief send data using the socket
   * \param buf the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually sent
   *         return -1 if error occurs
   */
  inline ssize_t Send(const void *buf_, size_t len, int flag = 0) {
    const char *buf = reinterpret_cast<const char*>(buf_);
#if !IS_MINGW()
    return send(sockfd, buf, static_cast<sock_size_t>(len), flag);
#else
    return 0;
#endif  // !IS_MINGW()
  }
  /*!
   * \brief receive data using the socket
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually received
   *         return -1 if error occurs
   */
  inline ssize_t Recv(void *buf_, size_t len, int flags = 0) {
    char *buf = reinterpret_cast<char*>(buf_);
#if !IS_MINGW()
    return recv(sockfd, buf, static_cast<sock_size_t>(len), flags);
#else
    return 0;
#endif  // !IS_MINGW()
  }
  /*!
   * \brief peform block write that will attempt to send all data out
   *    can still return smaller than request when error occurs
   * \param buf the pointer to the buffer
   * \param len the size of the buffer
   * \return size of data actually sent
   */
  inline size_t SendAll(const void *buf_, size_t len) {
    const char *buf = reinterpret_cast<const char*>(buf_);
    size_t ndone = 0;
#if !IS_MINGW()
    while (ndone <  len) {
      ssize_t ret = send(sockfd, buf, static_cast<ssize_t>(len - ndone), 0);
      if (ret == -1) {
        if (LastErrorWouldBlock()) return ndone;
        Socket::Error("SendAll");
      }
      buf += ret;
      ndone += ret;
    }
#endif  // !IS_MINGW()
    return ndone;
  }
  /*!
   * \brief peforma block read that will attempt to read all data
   *    can still return smaller than request when error occurs
   * \param buf_ the buffer pointer
   * \param len length of data to recv
   * \return size of data actually sent
   */
  inline size_t RecvAll(void *buf_, size_t len) {
    char *buf = reinterpret_cast<char*>(buf_);
    size_t ndone = 0;
#if !IS_MINGW()
    while (ndone <  len) {
      ssize_t ret = recv(sockfd, buf,
                         static_cast<sock_size_t>(len - ndone), MSG_WAITALL);
      if (ret == -1) {
        if (LastErrorWouldBlock()) return ndone;
        Socket::Error("RecvAll");
      }
      if (ret == 0) return ndone;
      buf += ret;
      ndone += ret;
    }
#endif  // !IS_MINGW()
    return ndone;
  }
  /*!
   * \brief send a string over network
   * \param str the string to be sent
   */
  inline void SendStr(const std::string &str) {
    int len = static_cast<int>(str.length());
    utils::Assert(this->SendAll(&len, sizeof(len)) == sizeof(len),
                  "error during send SendStr");
    if (len != 0) {
      utils::Assert(this->SendAll(str.c_str(), str.length()) == str.length(),
                    "error during send SendStr");
    }
  }
  /*!
   * \brief recv a string from network
   * \param out_str the string to receive
   */
  inline void RecvStr(std::string *out_str) {
    int len;
    utils::Assert(this->RecvAll(&len, sizeof(len)) == sizeof(len),
                  "error during send RecvStr");
    out_str->resize(len);
    if (len != 0) {
      utils::Assert(this->RecvAll(&(*out_str)[0], len) == out_str->length(),
                    "error during send SendStr");
    }
  }
};

/*! \brief helper data structure to perform poll */
struct PollHelper {
 public:
  /*!
   * \brief add file descriptor to watch for read
   * \param fd file descriptor to be watched
   */
  inline void WatchRead(SOCKET fd) {
    auto& pfd = fds[fd];
    pfd.fd = fd;
    pfd.events |= POLLIN;
  }
  /*!
   * \brief add file descriptor to watch for write
   * \param fd file descriptor to be watched
   */
  inline void WatchWrite(SOCKET fd) {
    auto& pfd = fds[fd];
    pfd.fd = fd;
    pfd.events |= POLLOUT;
  }
  /*!
   * \brief add file descriptor to watch for exception
   * \param fd file descriptor to be watched
   */
  inline void WatchException(SOCKET fd) {
    auto& pfd = fds[fd];
    pfd.fd = fd;
    pfd.events |= POLLPRI;
  }
  /*!
   * \brief Check if the descriptor is ready for read
   * \param fd file descriptor to check status
   */
  inline bool CheckRead(SOCKET fd) const {
    const auto& pfd = fds.find(fd);
    return pfd != fds.end() && ((pfd->second.events & POLLIN) != 0);
  }
  /*!
   * \brief Check if the descriptor is ready for write
   * \param fd file descriptor to check status
   */
  inline bool CheckWrite(SOCKET fd) const {
    const auto& pfd = fds.find(fd);
    return pfd != fds.end() && ((pfd->second.events & POLLOUT) != 0);
  }

  /*!
   * \brief peform poll on the set defined, read, write, exception
   * \param timeout specify timeout in milliseconds(ms) if negative, means poll will block
   * \return
   */
  inline void Poll(std::chrono::seconds timeout) {  // NOLINT(*)
    std::vector<pollfd> fdset;
    fdset.reserve(fds.size());
    for (auto kv : fds) {
      fdset.push_back(kv.second);
    }
    int ret = PollImpl(fdset.data(), fdset.size(), timeout);
    if (ret == 0) {
      LOG(FATAL) << "Poll timeout";
    } else if (ret < 0) {
      Socket::Error("Poll");
    } else {
      for (auto& pfd : fdset) {
        auto revents = pfd.revents & pfd.events;
        if (!revents) {
          fds.erase(pfd.fd);
        } else {
          fds[pfd.fd].events = revents;
        }
      }
    }
  }

  std::unordered_map<SOCKET, pollfd> fds;
};
}  // namespace utils
}  // namespace rabit

#if IS_MINGW() && !defined(POLLRDNORM) && !defined(POLLRDBAND)
#undef POLLIN
#undef POLLPRI
#undef POLLOUT
#endif  // IS_MINGW()

#endif  // RABIT_INTERNAL_SOCKET_H_
