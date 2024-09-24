/**
 *  Copyright 2014-2024, XGBoost Contributors
 * \file socket.h
 * \author Tianqi Chen
 */
#pragma once
#include <xgboost/collective/result.h>
#include <xgboost/collective/socket.h>

#if defined(_WIN32)
#include <xgboost/windefs.h>
// Socket API
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

#include <poll.h>

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
  short  events;  // NOLINT
  short  revents;  // NOLINT
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
  // For Windows and Linux, negative timeout means infinite timeout. For freebsd,
  // INFTIM(-1) should be used instead.
#if defined(_WIN32)

#if IS_MINGW()
  xgboost::MingWError();
  return -1;
#else
  return WSAPoll(pfd, nfds, std::chrono::milliseconds(timeout).count());
#endif  // IS_MINGW()

#else
  return poll(pfd, nfds, timeout.count() < 0 ? -1 : std::chrono::milliseconds(timeout).count());
#endif  // IS_MINGW()
}

template <typename E>
std::enable_if_t<std::is_integral_v<E>, xgboost::collective::Result> PollError(E const& revents) {
  if ((revents & POLLERR) != 0) {
    auto err = errno;
    auto str = strerror(err);
    return xgboost::system::FailWithCode(std::string{"Poll error condition:"} +  // NOLINT
                                         std::string{str} +                      // NOLINT
                                         " code:" + std::to_string(err));
  }
  if ((revents & POLLNVAL) != 0) {
    return xgboost::system::FailWithCode("Invalid polling request.");
  }
  if ((revents & POLLHUP) != 0) {
    // Excerpt from the Linux manual:
    //
    // Note that when reading from a channel such as a pipe or a stream socket, this event
    // merely indicates that the peer closed its end of the channel.Subsequent reads from
    // the channel will return 0 (end of file) only after all outstanding data in the
    // channel has been consumed.
    //
    // We don't usually have a barrier for exiting workers, it's normal to have one end
    // exit while the other still reading data.
    return xgboost::collective::Success();
  }
#if defined(POLLRDHUP)
  // Linux only flag
  if ((revents & POLLRDHUP) != 0) {
    return xgboost::system::FailWithCode("Poll hung up on the other end.");
  }
#endif  // defined(POLLRDHUP)
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
      return xgboost::collective::Fail(
          "Poll timeout:" + std::to_string(timeout.count()) + " seconds.",
          std::make_error_code(std::errc::timed_out));
    } else if (ret < 0) {
      return xgboost::system::FailWithCode("Poll failed, nfds:" + std::to_string(fdset.size()));
    }

    for (auto& pfd : fdset) {
      auto result = PollError(pfd.revents);
      if (check_error && !result.OK()) {
        return result;
      }

      auto revents = pfd.revents & pfd.events;
      fds[pfd.fd].events = revents;
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
