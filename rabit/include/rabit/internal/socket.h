/**
 *  Copyright 2014-2023, XGBoost Contributors
 * \file socket.h
 * \author Tianqi Chen
 */
#ifndef RABIT_INTERNAL_SOCKET_H_
#define RABIT_INTERNAL_SOCKET_H_
#include "xgboost/collective/result.h"
#include "xgboost/collective/socket.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>

#else

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>

#endif  // defined(_WIN32)

#include <chrono>
#include <cstring>
#include <string>
#include <system_error>  // make_error_code, errc
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)

#include <sys/poll.h>

using SOCKET = int;
using sock_size_t = size_t;  // NOLINT
#endif  // !defined(_WIN32)

#define IS_MINGW() defined(__MINGW32__)

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

#endif  // IS_MINGW() && !defined(POLLRDNORM) && !defined(POLLRDBAND)

namespace rabit {
namespace utils {

template <typename PollFD>
int PollImpl(PollFD* pfd, int nfds, std::chrono::seconds timeout) noexcept(true) {
#if defined(_WIN32)

#if IS_MINGW()
  xgboost::MingWError();
  return -1;
#else
  return WSAPoll(pfd, nfds, std::chrono::milliseconds(timeout).count());
#endif  // IS_MINGW()

#else
  return poll(pfd, nfds, std::chrono::milliseconds(timeout).count());
#endif  // IS_MINGW()
}

template <typename E>
std::enable_if_t<std::is_integral_v<E>, xgboost::collective::Result> PollError(E const& revents) {
  if ((revents & POLLERR) != 0) {
    return xgboost::system::FailWithCode("Poll error condition.");
  }
  if ((revents & POLLNVAL) != 0) {
    return xgboost::system::FailWithCode("Invalid polling request.");
  }
  return xgboost::collective::Success();
}

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
  void WatchRead(xgboost::collective::TCPSocket const &socket) { this->WatchRead(socket.Handle()); }

  /*!
   * \brief add file descriptor to watch for write
   * \param fd file descriptor to be watched
   */
  inline void WatchWrite(SOCKET fd) {
    auto& pfd = fds[fd];
    pfd.fd = fd;
    pfd.events |= POLLOUT;
  }
  void WatchWrite(xgboost::collective::TCPSocket const &socket) {
    this->WatchWrite(socket.Handle());
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
  void WatchException(xgboost::collective::TCPSocket const &socket) {
    this->WatchException(socket.Handle());
  }
  /*!
   * \brief Check if the descriptor is ready for read
   * \param fd file descriptor to check status
   */
  [[nodiscard]] bool CheckRead(SOCKET fd) const {
    const auto& pfd = fds.find(fd);
    return pfd != fds.end() && ((pfd->second.events & POLLIN) != 0);
  }
  [[nodiscard]] bool CheckRead(xgboost::collective::TCPSocket const& socket) const {
    return this->CheckRead(socket.Handle());
  }

  /*!
   * \brief Check if the descriptor is ready for write
   * \param fd file descriptor to check status
   */
  [[nodiscard]] bool CheckWrite(SOCKET fd) const {
    const auto& pfd = fds.find(fd);
    return pfd != fds.end() && ((pfd->second.events & POLLOUT) != 0);
  }
  [[nodiscard]] bool CheckWrite(xgboost::collective::TCPSocket const& socket) const {
    return this->CheckWrite(socket.Handle());
  }
  /**
   * @brief perform poll on the set defined, read, write, exception
   *
   * @param timeout specify timeout in seconds. Block if negative.
   */
  [[nodiscard]] xgboost::collective::Result Poll(std::chrono::seconds timeout,
                                                 bool check_error = true) {
    std::vector<pollfd> fdset;
    fdset.reserve(fds.size());
    for (auto kv : fds) {
      fdset.push_back(kv.second);
    }
    std::int32_t ret = PollImpl(fdset.data(), fdset.size(), timeout);
    if (ret == 0) {
      return xgboost::collective::Fail("Poll timeout.", std::make_error_code(std::errc::timed_out));
    } else if (ret < 0) {
      return xgboost::system::FailWithCode("Poll failed.");
    }

    for (auto& pfd : fdset) {
      auto result = PollError(pfd.revents);
      if (check_error && !result.OK()) {
        return result;
      }

      auto revents = pfd.revents & pfd.events;
      if (!revents) {
        // FIXME(jiamingy): remove this once rabit is replaced.
        fds.erase(pfd.fd);
      } else {
        fds[pfd.fd].events = revents;
      }
    }
    return xgboost::collective::Success();
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
