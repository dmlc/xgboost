#ifndef XGBOOST_UTILS_SOCKET_H
#define XGBOOST_UTILS_SOCKET_H
/*!
 * \file socket.h
 * \brief this file aims to provide a wrapper of sockets
 * \author Tianqi Chen
 */
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <string>
#include <cstring>
#include "./utils.h"

namespace xgboost {
namespace utils {

/*! \brief data structure for network address */
struct SockAddr {
  sockaddr_in addr;
  // constructor
  SockAddr(void) {}
  SockAddr(const char *url, int port) {
    this->Set(url, port);
  }
  /*! 
   * \brief set the address
   * \param url the url of the address
   * \param port the port of address
   */
  inline void Set(const char *url, int port) {
    hostent *hp = gethostbyname(url);
    Check(hp != NULL, "cannot obtain address of %s", url);
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    memcpy(&addr.sin_addr, hp->h_addr_list[0], hp->h_length);
  }
  /*! \return a string representation of the address */
  inline std::string ToString(void) const {
    std::string buf; buf.resize(256);
    const char *s = inet_ntop(AF_INET, &addr, &buf[0], buf.length());
    Assert(s != NULL, "cannot decode address");
    std::string res = s;
    sprintf(&buf[0], "%u", ntohs(addr.sin_port));
    res += ":" + buf;
    return res;
  }
};
/*! 
 * \brief a wrapper of TCP socket that hopefully be cross platform
 */
class TCPSocket {
 public:
  /*! \brief the file descriptor of socket */
  int sockfd;
  // constructor
  TCPSocket(void) {}
  // default conversion to int
  inline operator int() const {
    return sockfd;
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
   * \brief set this socket to use async I/O 
   */
  inline void SetAsync(void) {
    if (fcntl(sockfd, fcntl(sockfd, F_GETFL) | O_NONBLOCK) == -1) {
      SockError("SetAsync", errno);
    }
  }
  /*!
   * \brief perform listen of the socket
   * \param backlog backlog parameter
   */
  inline void Listen(int backlog = 16) {
    listen(sockfd, backlog);
  }
  /*! 
   * \brief bind the socket to an address 
   * \param 3
   */
  inline void Bind(const SockAddr &addr) {
    if (bind(sockfd, (sockaddr*)&addr.addr, sizeof(addr.addr)) == -1) {
      SockError("Bind", errno);
    }
  }
  /*! 
   * \brief connect to an address 
   * \param addr the address to connect to
   */
  inline void Connect(const SockAddr &addr) {
    if (connect(sockfd, (sockaddr*)&addr.addr, sizeof(addr.addr)) == -1) {
      SockError("Connect", errno);
    }
  }
  /*! \brief close the connection */
  inline void Close(void) {
    close(sockfd);
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
    if (ret == -1) SockError("Send", errno);
    return ret;
  }
  /*! 
   * \brief send data using the socket 
   * \param buf the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually received 
   */
  inline size_t Recv(void *buf, size_t len, int flags = 0) {
    if (len == 0) return 0;
    ssize_t ret = recv(sockfd, buf, len, flags);
    if (ret == -1) SockError("Recv", errno);
    return ret;
   }
 private:
  // report an socket error
  inline static void SockError(const char *msg, int errsv) {
    char buf[256];    
    Error("Socket %s Error:%s", msg, strerror_r(errsv, buf, sizeof(buf)));
  }
};
/*! \brief helper data structure to perform select */
struct SelectHelper {
 public:
  SelectHelper(void) {}
  /*!
   * \brief add file descriptor to watch for read 
   * \param fd file descriptor to be watched
   */
  inline void WatchRead(int fd) {
    FD_SET(fd, &read_set);
    if (fd > maxfd) maxfd = fd;
  }
  /*!
   * \brief add file descriptor to watch for write
   * \param fd file descriptor to be watched
   */
  inline void WatchWrite(int fd) {
    FD_SET(fd, &write_set);
    if (fd > maxfd) maxfd = fd;
  }
  /*!
   * \brief Check if the descriptor is ready for read
   * \param 
   */
  inline bool CheckRead(int fd) const {
    return FD_ISSET(fd, &read_set);
  }
  inline bool CheckWrite(int fd) const {
    return FD_ISSET(fd, &write_set);
  }
  inline void Clear(void) {
    FD_ZERO(&read_set);
    FD_ZERO(&write_set);
    maxfd = 0;
  }
  /*!
   * \brief peform select on the set defined
   * \param timeout specify timeout in micro-seconds(ms) if equals 0, means select will always block
   * \return number of active descriptors selected
   */
  inline int Select(long timeout = 0) {
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
      char buf[256];
      Error("Select Error:%s", strerror_r(errsv, buf, sizeof(buf)));      
    }
    return ret;
  }
  
 private:
  int maxfd; 
  fd_set read_set, write_set;
};
}
}
#endif
