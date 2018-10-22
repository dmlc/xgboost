/*!
 *  Copyright (c) 2014 by Contributors
 * \file socket.h
 * \brief this file aims to provide a wrapper of sockets
 * \author Tianqi Chen
 */
#ifndef RABIT_SOCKET_H_
#define RABIT_SOCKET_H_
#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif
#else
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#endif
#include <string>
#include <cstring>
#include <vector>
#include <unordered_map>
#include "../include/rabit/internal/utils.h"

#if defined(_WIN32)
typedef int ssize_t;
typedef int sock_size_t;

static inline int poll(struct pollfd *pfd, int nfds,
                       int timeout) { return WSAPoll ( pfd, nfds, timeout ); }
#else
#include <sys/poll.h>
typedef int SOCKET;
typedef size_t sock_size_t;
const int INVALID_SOCKET = -1;
#endif

namespace rabit {
namespace utils {
/*! \brief data structure for network address */
struct SockAddr {
  sockaddr_in addr;
  // constructor
  SockAddr(void) {}
  SockAddr(const char *url, int port) {
    this->Set(url, port);
  }
  inline static std::string GetHostName(void) {
    std::string buf; buf.resize(256);
    utils::Check(gethostname(&buf[0], 256) != -1, "fail to get host name");
    return std::string(buf.c_str());
  }
  /*!
   * \brief set the address
   * \param url the url of the address
   * \param port the port of address
   */
  inline void Set(const char *host, int port) {
    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_protocol = SOCK_STREAM;
    addrinfo *res = NULL;
    int sig = getaddrinfo(host, NULL, &hints, &res);
    Check(sig == 0 && res != NULL, "cannot obtain address of %s", host);
    Check(res->ai_family == AF_INET, "Does not support IPv6");
    memcpy(&addr, res->ai_addr, res->ai_addrlen);
    addr.sin_port = htons(port);
    freeaddrinfo(res);
  }
  /*! \brief return port of the address*/
  inline int port(void) const {
    return ntohs(addr.sin_port);
  }
  /*! \return a string representation of the address */
  inline std::string AddrStr(void) const {
    std::string buf; buf.resize(256);
#ifdef _WIN32
    const char *s = inet_ntop(AF_INET, (PVOID)&addr.sin_addr,
                    &buf[0], buf.length());
#else
    const char *s = inet_ntop(AF_INET, &addr.sin_addr,
                              &buf[0], buf.length());
#endif
    Assert(s != NULL, "cannot decode address");
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
  inline operator SOCKET() const {
    return sockfd;
  }
  /*!
   * \return last error of socket operation
   */
  inline static int GetLastError(void) {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
  }
  /*! \return whether last error was would block */
  inline static bool LastErrorWouldBlock(void) {
    int errsv = GetLastError();
#ifdef _WIN32
    return errsv == WSAEWOULDBLOCK;
#else
    return errsv == EAGAIN || errsv == EWOULDBLOCK;
#endif
  }
  /*!
   * \brief start up the socket module
   *   call this before using the sockets
   */
  inline static void Startup(void) {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) == -1) {
    Socket::Error("Startup");
    }
    if (LOBYTE(wsa_data.wVersion) != 2 || HIBYTE(wsa_data.wVersion) != 2) {
    WSACleanup();
    utils::Error("Could not find a usable version of Winsock.dll\n");
    }
#endif
  }
  /*!
   * \brief shutdown the socket module after use, all sockets need to be closed
   */
  inline static void Finalize(void) {
#ifdef _WIN32
    WSACleanup();
#endif
  }
  /*!
   * \brief set this socket to use non-blocking mode
   * \param non_block whether set it to be non-block, if it is false
   *        it will set it back to block mode
   */
  inline void SetNonBlock(bool non_block) {
#ifdef _WIN32
    u_long mode = non_block ? 1 : 0;
    if (ioctlsocket(sockfd, FIONBIO, &mode) != NO_ERROR) {
      Socket::Error("SetNonBlock");
    }
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
#endif
  }
  /*!
   * \brief bind the socket to an address
   * \param addr
   */
  inline void Bind(const SockAddr &addr) {
    if (bind(sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
             sizeof(addr.addr)) == -1) {
      Socket::Error("Bind");
    }
  }
  /*!
   * \brief try bind the socket to host, from start_port to end_port
   * \param start_port starting port number to try
   * \param end_port ending port number to try
   * \return the port successfully bind to, return -1 if failed to bind any port
   */
  inline int TryBindHost(int start_port, int end_port) {
    // TODO(tqchen) add prefix check
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
#endif
    }

    return -1;
  }
  /*! \brief get last error code if any */
  inline int GetSockError(void) const {
    int error = 0;
    socklen_t len = sizeof(error);
    if (getsockopt(sockfd,  SOL_SOCKET, SO_ERROR,
            reinterpret_cast<char*>(&error), &len) != 0) {
      Error("GetSockError");
    }
    return error;
  }
  /*! \brief check if anything bad happens */
  inline bool BadSocket(void) const {
    if (IsClosed()) return true;
    int err = GetSockError();
    if (err == EBADF || err == EINTR) return true;
    return false;
  }
  /*! \brief check if socket is already closed */
  inline bool IsClosed(void) const {
    return sockfd == INVALID_SOCKET;
  }
  /*! \brief close the socket */
  inline void Close(void) {
    if (sockfd != INVALID_SOCKET) {
#ifdef _WIN32
      closesocket(sockfd);
#else
      close(sockfd);
#endif
      sockfd = INVALID_SOCKET;
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
  TCPSocket(void) : Socket(INVALID_SOCKET) {
  }
  explicit TCPSocket(SOCKET sockfd) : Socket(sockfd) {
  }
  /*!
   * \brief enable/disable TCP keepalive
   * \param keepalive whether to set the keep alive option on
   */
  inline void SetKeepAlive(bool keepalive) {
    int opt = static_cast<int>(keepalive);
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE,
                   reinterpret_cast<char*>(&opt), sizeof(opt)) < 0) {
      Socket::Error("SetKeepAlive");
    }
  }
  /*!
   * \brief create the socket, call this before using socket
   * \param af domain
   */
  inline void Create(int af = PF_INET) {
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd == INVALID_SOCKET) {
      Socket::Error("Create");
    }
  }
  /*!
   * \brief perform listen of the socket
   * \param backlog backlog parameter
   */
  inline void Listen(int backlog = 16) {
    listen(sockfd, backlog);
  }
  /*! \brief get a new connection */
  TCPSocket Accept(void) {
    SOCKET newfd = accept(sockfd, NULL, NULL);
    if (newfd == INVALID_SOCKET) {
      Socket::Error("Accept");
    }
    return TCPSocket(newfd);
  }
  /*!
   * \brief decide whether the socket is at OOB mark
   * \return 1 if at mark, 0 if not, -1 if an error occured
   */
  inline int AtMark(void) const {
#ifdef _WIN32
    unsigned long atmark;  // NOLINT(*)
    if (ioctlsocket(sockfd, SIOCATMARK, &atmark) != NO_ERROR) return -1;
#else
    int atmark;
    if (ioctl(sockfd, SIOCATMARK, &atmark) == -1) return -1;
#endif
    return static_cast<int>(atmark);
  }
  /*!
   * \brief connect to an address
   * \param addr the address to connect to
   * \return whether connect is successful
   */
  inline bool Connect(const SockAddr &addr) {
    return connect(sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
                   sizeof(addr.addr)) == 0;
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
    return send(sockfd, buf, static_cast<sock_size_t>(len), flag);
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
    return recv(sockfd, buf, static_cast<sock_size_t>(len), flags);
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
    while (ndone <  len) {
      ssize_t ret = send(sockfd, buf, static_cast<ssize_t>(len - ndone), 0);
      if (ret == -1) {
        if (LastErrorWouldBlock()) return ndone;
        Socket::Error("SendAll");
      }
      buf += ret;
      ndone += ret;
    }
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
   * \brief Check if the descriptor has any exception
   * \param fd file descriptor to check status
   */
  inline bool CheckExcept(SOCKET fd) const {
    const auto& pfd = fds.find(fd);
    return pfd != fds.end() && ((pfd->second.events & POLLPRI) != 0);
  }
  /*!
   * \brief wait for exception event on a single descriptor
   * \param fd the file descriptor to wait the event for
   * \param timeout the timeout counter, can be negative, which means wait until the event happen
   * \return 1 if success, 0 if timeout, and -1 if error occurs
   */
  inline static int WaitExcept(SOCKET fd, long timeout = -1) { // NOLINT(*)
    pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLPRI;
    return poll(&pfd, 1, timeout);
  }

  /*!
   * \brief peform poll on the set defined, read, write, exception
   * \param timeout specify timeout in milliseconds(ms) if negative, means poll will block
   * \return
   */
  inline void Poll(long timeout = -1) {  // NOLINT(*)
    std::vector<pollfd> fdset;
    fdset.reserve(fds.size());
    for (auto kv : fds) {
      fdset.push_back(kv.second);
    }
    int ret = poll(fdset.data(), fdset.size(), timeout);
    if (ret == -1) {
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
#endif  // RABIT_SOCKET_H_
