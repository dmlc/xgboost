#ifndef XGBOOST_UTILS_SOCKET_H
#define XGBOOST_UTILS_SOCKET_H
/*!
 * \file socket.h
 * \brief this file aims to provide a wrapper of sockets
 * \author Tianqi Chen
 */
#if defined(_WIN32)
#include <winsock2.h>
#else
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#endif
#include <string>
#include <cstring>
#include "./utils.h"

namespace xgboost {
namespace utils {
#if defined(_WIN32)
typedef int ssize_t;
#else
typedef int SOCKET;
const int INVALID_SOCKET = -1;
#endif

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
    hostent *hp = gethostbyname(host);
    Check(hp != NULL, "cannot obtain address of %s", host);
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    memcpy(&addr.sin_addr, hp->h_addr_list[0], hp->h_length);
  }
  /*! \brief return port of the address*/
  inline int port(void) const {
    return ntohs(addr.sin_port);
  }
  /*! \return a string representation of the address */
  inline std::string AddrStr(void) const {
    std::string buf; buf.resize(256);
    const char *s = inet_ntop(AF_INET, &addr.sin_addr, &buf[0], buf.length());
    Assert(s != NULL, "cannot decode address");
    return std::string(s);
  }
};
/*! 
 * \brief a wrapper of TCP socket that hopefully be cross platform
 */
class TCPSocket {
 public:
  /*! \brief the file descriptor of socket */
  SOCKET sockfd;
  // constructor
  TCPSocket(void) : sockfd(INVALID_SOCKET) {
  }
  explicit TCPSocket(SOCKET sockfd) : sockfd(sockfd) {
  }
  ~TCPSocket(void) {
    // do nothing in destructor
    // user need to take care of close
  }
  // default conversion to int
  inline operator SOCKET() const {
    return sockfd;
  }
  /*!
   * \brief create the socket, call this before using socket
   * \param af domain
   */
  inline void Create(int af = PF_INET) {
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd == INVALID_SOCKET) {
      SockError("Create");
    }
  }
  /*!
   * \brief start up the socket module
   *   call this before using the sockets
   */
  inline static void Startup(void) {
  }
  /*! 
   * \brief shutdown the socket module after use, all sockets need to be closed
   */  
  inline static void Finalize(void) {
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
      SockError("SetNonBlock");
	}
#else
    int flag = fcntl(sockfd, F_GETFL, 0);
    if (flag == -1) {
      SockError("SetNonBlock-1");
    }
    if (non_block) {
      flag |= O_NONBLOCK;
    } else {
      flag &= ~O_NONBLOCK;
    }
    if (fcntl(sockfd, F_SETFL, flag) == -1) {
      SockError("SetNonBlock-2");
    }
#endif
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
      SockError("Accept");
    }
    return TCPSocket(newfd);
  }
  /*! 
   * \brief bind the socket to an address 
   * \param addr
   */
  inline void Bind(const SockAddr &addr) {
    if (bind(sockfd, (sockaddr*)&addr.addr, sizeof(addr.addr)) == -1) {
      SockError("Bind");
    }
  }
  /*! 
   * \brief try bind the socket to host, from start_port to end_port
   * \param start_port starting port number to try
   * \param end_port ending port number to try
   * \param out_addr the binding address, if successful
   * \return whether the binding is successful
   */
  inline int TryBindHost(int start_port, int end_port) {    
    for (int port = start_port; port < end_port; ++port) {
      SockAddr addr("0.0.0.0", port);
      if (bind(sockfd, (sockaddr*)&addr.addr, sizeof(addr.addr)) == 0) {
        return port;
      }
      if (errno != EADDRINUSE) {
        SockError("TryBindHost");
      }
    }
    return -1;
  }                      
  /*! 
   * \brief connect to an address 
   * \param addr the address to connect to
   */
  inline void Connect(const SockAddr &addr) {
    if (connect(sockfd, (sockaddr*)&addr.addr, sizeof(addr.addr)) == -1) {
      SockError("Connect");
    }
  }
  /*! \brief close the connection */
  inline void Close(void) {
    if (sockfd != -1) {
#ifdef _WIN32
      closesocket(sockfd);
#else
	  close(sockfd);
#endif
	  sockfd = INVALID_SOCKET;
    } else {
      Error("TCPSocket::Close double close the socket or close without create");
    }
  }
  /*!
   * \brief send data using the socket 
   * \param buf the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually sent
   */
  inline size_t Send(const void *buf, size_t len, int flag = 0) {
    if (len == 0) return 0;
    ssize_t ret = send(sockfd, buf, len, flag);
    if (ret == -1) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
      SockError("Send");
    }
    return ret;
  }  
  /*! 
   * \brief receive data using the socket 
   * \param buf the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually received 
   */
  inline size_t Recv(void *buf, size_t len, int flags = 0) {
    if (len == 0) return 0;    
    ssize_t ret = recv(sockfd, buf, len, flags);
    if (ret == -1) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
      SockError("Recv");
    }
    return ret;
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
        if (errno == EAGAIN || errno == EWOULDBLOCK) return ndone;
        SockError("Recv");
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
      ssize_t ret = recv(sockfd, buf, len - ndone, MSG_WAITALL);
      if (ret == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return ndone;
        SockError("Recv");
      }
      if (ret == 0) return ndone;
      buf += ret;
      ndone += ret;
    }
    return ndone;
  }

 private:
  // report an socket error
  inline static void SockError(const char *msg) {
    int errsv = errno;
    Error("Socket %s Error:%s", msg, strerror(errsv));
  }
};
/*! \brief helper data structure to perform select */
struct SelectHelper {
 public:
  SelectHelper(void) {
    this->Clear();
  }
  /*!
   * \brief add file descriptor to watch for read 
   * \param fd file descriptor to be watched
   */
  inline void WatchRead(SOCKET fd) {
    read_fds.push_back(fd);
    if (fd > maxfd) maxfd = fd;
  }
  /*!
   * \brief add file descriptor to watch for write
   * \param fd file descriptor to be watched
   */
  inline void WatchWrite(SOCKET fd) {
    write_fds.push_back(fd);
    if (fd > maxfd) maxfd = fd;
  }
  /*!
   * \brief Check if the descriptor is ready for read
   * \param fd file descriptor to check status
   */
  inline bool CheckRead(SOCKET fd) const {
    return FD_ISSET(fd, &read_set);
  }
  /*!
   * \brief Check if the descriptor is ready for write
   * \param fd file descriptor to check status
   */
  inline bool CheckWrite(SOCKET fd) const {
    return FD_ISSET(fd, &write_set);
  }
  /*!
   * \brief clear all the monitored descriptors
   */
  inline void Clear(void) {
    read_fds.clear();
    write_fds.clear();
    maxfd = 0;
  }
  /*!
   * \brief peform select on the set defined
   * \param timeout specify timeout in micro-seconds(ms) if equals 0, means select will always block
   * \return number of active descriptors selected
   */
  inline int Select(long timeout = 0) {
    FD_ZERO(&read_set);
    FD_ZERO(&write_set);
    for (size_t i = 0; i < read_fds.size(); ++i) {
      FD_SET(read_fds[i], &read_set);
    } 
    for (size_t i = 0; i < write_fds.size(); ++i) {
      FD_SET(write_fds[i], &write_set);
    }
    int ret;
    if (timeout == 0) {
      ret = select(maxfd + 1, &read_set, &write_set, NULL, NULL);
    } else {
      timeval tm;
      tm.tv_usec = (timeout % 1000) * 1000;
      tm.tv_sec = timeout / 1000;
      ret = select(maxfd + 1, &read_set, &write_set, NULL, &tm);
    }
    if (ret == -1) {
      int errsv = errno;
      Error("Select Error: %s", strerror(errsv));
    }
    return ret;
  }
  
 private:
  int maxfd; 
  fd_set read_set, write_set;
  std::vector<int> read_fds, write_fds;
};
}
}
#endif
