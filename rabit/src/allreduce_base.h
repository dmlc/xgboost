/*!
 *  Copyright (c) 2014 by Contributors
 * \file allreduce_base.h
 * \brief Basic implementation of AllReduce
 *   using TCP non-block socket and tree-shape reduction.
 *
 *   This implementation provides basic utility of AllReduce and Broadcast
 *   without considering node failure
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#ifndef RABIT_ALLREDUCE_BASE_H_
#define RABIT_ALLREDUCE_BASE_H_

#include <functional>
#include <future>
#include <vector>
#include <string>
#include <algorithm>
#include "rabit/internal/utils.h"
#include "rabit/internal/engine.h"
#include "rabit/internal/socket.h"

#ifdef RABIT_CXXTESTDEFS_H
#define private   public
#define protected public
#endif  // RABIT_CXXTESTDEFS_H


namespace MPI {  // NOLINT
// MPI data type to be compatible with existing MPI interface
class Datatype {
 public:
  size_t type_size;
  explicit Datatype(size_t type_size) : type_size(type_size) {}
};
}
namespace rabit {
namespace engine {

/*! \brief implementation of basic Allreduce engine */
class AllreduceBase : public IEngine {
 public:
  // magic number to verify server
  static const int kMagic = 0xff99;
  // constant one byte out of band message to indicate error happening
  AllreduceBase();
  virtual ~AllreduceBase() = default;
  // initialize the manager
  virtual bool Init(int argc, char* argv[]);
  // shutdown the engine
  virtual bool Shutdown();
  /*!
   * \brief set parameters to the engine
   * \param name parameter name
   * \param val parameter value
   */
  virtual void SetParam(const char *name, const char *val);
  /*!
   * \brief print the msg in the tracker,
   *    this function can be used to communicate the information of the progress to
   *    the user who monitors the tracker
   * \param msg message to be printed in the tracker
   */
  void TrackerPrint(const std::string &msg) override;

  /*! \brief get rank of previous node in ring topology*/
  int GetRingPrevRank() const override {
    return ring_prev->rank;
  }
  /*! \brief get rank */
  int GetRank() const override {
    return rank;
  }
  /*! \brief get rank */
  int GetWorldSize() const override {
    if (world_size == -1) return 1;
    return world_size;
  }
  /*! \brief whether is distributed or not */
  bool IsDistributed() const override {
    return tracker_uri != "NULL";
  }
  /*! \brief get rank */
  std::string GetHost() const override {
    return host_uri;
  }

  /*!
  * \brief internal Allgather function, each node have a segment of data in the ring of sendrecvbuf,
  *  the data provided by current node k is [slice_begin, slice_end),
  *  the next node's segment must start with slice_end
  *  after the call of Allgather, sendrecvbuf_ contains all the contents including all segments
  *  use a ring based algorithm
  *
  * \param sendrecvbuf_ buffer for both sending and receiving data, it is a ring conceptually
  * \param total_size total size of data to be gathered
  * \param slice_begin beginning of the current slice
  * \param slice_end end of the current slice
  * \param size_prev_slice size of the previous slice i.e. slice of node (rank - 1) % world_size
  */
  void Allgather(void *sendrecvbuf_, size_t total_size, size_t slice_begin,
                 size_t slice_end, size_t size_prev_slice) override {
    if (world_size == 1 || world_size == -1) {
      return;
    }
    utils::Assert(TryAllgatherRing(sendrecvbuf_, total_size, slice_begin,
                                   slice_end, size_prev_slice) == kSuccess,
                  "AllgatherRing failed");
  }
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf
   *        this function is NOT thread-safe
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   * \param prepare_func Lazy preprocessing function, lazy prepare_fun(prepare_arg)
   *                     will be called by the function before performing Allreduce, to initialize the data in sendrecvbuf_.
   *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
   * \param prepare_arg argument used to passed into the lazy preprocessing function
   */
  void Allreduce(void *sendrecvbuf_, size_t type_nbytes, size_t count,
                 ReduceFunction reducer, PreprocFunction prepare_fun = nullptr,
                 void *prepare_arg = nullptr) override {
    if (prepare_fun != nullptr) prepare_fun(prepare_arg);
    if (world_size == 1 || world_size == -1) return;
    utils::Assert(TryAllreduce(sendrecvbuf_, type_nbytes, count, reducer) ==
                      kSuccess,
                  "Allreduce failed");
  }
  /*!
   * \brief broadcast data from root to all nodes
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   * \param _file caller file name used to generate unique cache key
   * \param _line caller line number used to generate unique cache key
   * \param _caller caller function name used to generate unique cache key
   */
  void Broadcast(void *sendrecvbuf_, size_t total_size, int root) override {
    if (world_size == 1 || world_size == -1) return;
    utils::Assert(TryBroadcast(sendrecvbuf_, total_size, root) == kSuccess,
                  "Broadcast failed");
  }
  /*!
   * \brief deprecated
   * \sa CheckPoint, VersionNumber
   */
  int LoadCheckPoint() override { return 0; }

  // deprecated, increase internal version number
  void CheckPoint() override { version_number += 1; }
  /*!
   * \return version number of current stored model,
   *         which means how many calls to CheckPoint we made so far
   * \sa LoadCheckPoint, CheckPoint
   */
  int VersionNumber() const override {
    return version_number;
  }
  /*!
   * \brief report current status to the job tracker
   * depending on the job tracker we are in
   */
  inline void ReportStatus() const {
    if (hadoop_mode != 0) {
      LOG(CONSOLE) << "reporter:status:Rabit Phase[" << version_number << "] Operation " << seq_counter << "\n";
    }
  }

 protected:
  /*! \brief enumeration of possible returning results from Try functions */
  enum ReturnTypeEnum {
    /*! \brief execution is successful */
    kSuccess,
    /*! \brief a link was reset by peer */
    kConnReset,
    /*! \brief received a zero length message */
    kRecvZeroLen,
    /*! \brief a neighbor node go down, the connection is dropped */
    kSockError,
    /*!
     * \brief another node which is not my neighbor go down,
     *   get Out-of-Band exception notification from my neighbor
     */
    kGetExcept
  };
  /*! \brief struct return type to avoid implicit conversion to int/bool */
  struct ReturnType {
    /*! \brief internal return type */
    ReturnTypeEnum value;
    // constructor
    ReturnType() = default;
    ReturnType(ReturnTypeEnum value) : value(value) {}  // NOLINT(*)
    inline bool operator==(const ReturnTypeEnum &v) const {
      return value == v;
    }
    inline bool operator!=(const ReturnTypeEnum &v) const {
      return value != v;
    }
  };
  /*! \brief translate errno to return type */
  static ReturnType Errno2Return() {
    int errsv = xgboost::system::LastError();
    if (errsv == EAGAIN || errsv == EWOULDBLOCK || errsv == 0) return kSuccess;
#ifdef _WIN32
    if (errsv == WSAEWOULDBLOCK) return kSuccess;
    if (errsv == WSAECONNRESET) return kConnReset;
#endif  // _WIN32
    if (errsv == ECONNRESET) return kConnReset;
    return kSockError;
  }
  // link record to a neighbor
  struct LinkRecord {
   public:
    // socket to get data from/to link
    xgboost::collective::TCPSocket sock;
    // rank of the node in this link
    int rank;
    // size of data readed from link
    size_t size_read;
    // size of data sent to the link
    size_t size_write;
    // pointer to buffer head
    char *buffer_head {nullptr};
    // buffer size, in bytes
    size_t buffer_size {0};
    // constructor
    LinkRecord() = default;
    // initialize buffer
    void InitBuffer(size_t type_nbytes, size_t count,
                    size_t reduce_buffer_size) {
      size_t n = (type_nbytes * count + 7)/ 8;
      auto to = Min(reduce_buffer_size, n);
      buffer_.resize(to);
      // make sure align to type_nbytes
      buffer_size =
          buffer_.size() * sizeof(uint64_t) / type_nbytes * type_nbytes;
      utils::Assert(type_nbytes <= buffer_size,
                    "too large type_nbytes=%lu, buffer_size=%lu",
                    type_nbytes, buffer_size);
      // set buffer head
      buffer_head = reinterpret_cast<char*>(BeginPtr(buffer_));
    }
    // reset the recv and sent size
    inline void ResetSize() {
      size_write = size_read = 0;
    }
    /*!
     * \brief read data into ring-buffer, with care not to existing useful override data
     *  position after protect_start
     * \param protect_start all data start from protect_start is still needed in buffer
     *                      read shall not override this
     * \param max_size_read maximum logical amount we can read, size_read cannot exceed this value
     * \return the type of reading
     */
    inline ReturnType ReadToRingBuffer(size_t protect_start, size_t max_size_read) {
      utils::Assert(buffer_head != nullptr, "ReadToRingBuffer: buffer not allocated");
      utils::Assert(size_read <= max_size_read, "ReadToRingBuffer: max_size_read check");
      size_t ngap = size_read - protect_start;
      utils::Assert(ngap <= buffer_size, "Allreduce: boundary check");
      size_t offset = size_read % buffer_size;
      size_t nmax = max_size_read - size_read;
      nmax = Min(nmax, buffer_size - ngap);
      nmax = Min(nmax, buffer_size - offset);
      if (nmax == 0) return kSuccess;
      ssize_t len = sock.Recv(buffer_head + offset, nmax);
      // length equals 0, remote disconnected
      if (len == 0) {
        sock.Close(); return kRecvZeroLen;
      }
      if (len == -1) return Errno2Return();
      size_read += static_cast<size_t>(len);
      return kSuccess;
    }
    /*!
     * \brief read data into array,
     * this function can not be used together with ReadToRingBuffer
     * a link can either read into the ring buffer, or existing array
     * \param max_size maximum size of array
     * \return true if it is an successful read, false if there is some error happens, check errno
     */
    inline ReturnType ReadToArray(void *recvbuf_, size_t max_size) {
      if (max_size == size_read) return kSuccess;
      char *p = static_cast<char*>(recvbuf_);
      ssize_t len = sock.Recv(p + size_read, max_size - size_read);
      // length equals 0, remote disconnected
      if (len == 0) {
        sock.Close(); return kRecvZeroLen;
      }
      if (len == -1) return Errno2Return();
      size_read += static_cast<size_t>(len);
      return kSuccess;
    }
    /*!
     * \brief write data in array to sock
     * \param sendbuf_ head of array
     * \param max_size maximum size of array
     * \return true if it is an successful write, false if there is some error happens, check errno
     */
    inline ReturnType WriteFromArray(const void *sendbuf_, size_t max_size) {
      const char *p = static_cast<const char*>(sendbuf_);
      ssize_t len = sock.Send(p + size_write, max_size - size_write);
      if (len == -1) return Errno2Return();
      size_write += static_cast<size_t>(len);
      return kSuccess;
    }

   private:
    // recv buffer to get data from child
    // aligned with 64 bits, will be able to perform 64 bits operations freely
    std::vector<uint64_t> buffer_;
  };
  /*!
   * \brief simple data structure that works like a vector
   *  but takes reference instead of space
   */
  struct RefLinkVector {
    std::vector<LinkRecord*> plinks;
    inline LinkRecord &operator[](size_t i) {
      return *plinks[i];
    }
    inline size_t Size() const {
      return plinks.size();
    }
  };
  /*!
   * \brief initialize connection to the tracker
   * \return a socket that initializes the connection
   */
  xgboost::collective::TCPSocket ConnectTracker() const;
  /*!
   * \brief connect to the tracker to fix the the missing links
   *   this function is also used when the engine start up
   * \param cmd possible command to sent to tracker
   */
  bool ReConnectLinks(const char *cmd = "start");
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf, this function can fail, and will return the cause of failure
   *
   * NOTE on Allreduce:
   *    The kSuccess TryAllreduce does NOT mean every node have successfully finishes TryAllreduce.
   *    It only means the current node get the correct result of Allreduce.
   *    However, it means every node finishes LAST call(instead of this one) of Allreduce/Bcast
   *
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryAllreduce(void *sendrecvbuf_,
                          size_t type_nbytes,
                          size_t count,
                          ReduceFunction reducer);
  /*!
   * \brief broadcast data from root to all nodes, this function can fail,and will return the cause of failure
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryBroadcast(void *sendrecvbuf_, size_t size, int root);
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf,
   * this function implements tree-shape reduction
   *
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryAllreduceTree(void *sendrecvbuf_,
                              size_t type_nbytes,
                              size_t count,
                              ReduceFunction reducer);
  /*!
   * \brief internal Allgather function, each node have a segment of data in the ring of sendrecvbuf,
   *  the data provided by current node k is [slice_begin, slice_end),
   *  the next node's segment must start with slice_end
   *  after the call of Allgather, sendrecvbuf_ contains all the contents including all segments
   *  use a ring based algorithm
   *
   * \param sendrecvbuf_ buffer for both sending and receiving data, it is a ring conceptually
   * \param total_size total size of data to be gathered
   * \param slice_begin beginning of the current slice
   * \param slice_end end of the current slice
   * \param size_prev_slice size of the previous slice i.e. slice of node (rank - 1) % world_size
   * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryAllgatherRing(void *sendrecvbuf_, size_t total_size,
                              size_t slice_begin, size_t slice_end,
                              size_t size_prev_slice);
  /*!
   * \brief perform in-place allreduce, reduce on the sendrecvbuf,
   *
   *  after the function, node k get k-th segment of the reduction result
   *  the k-th segment is defined by [k * step, min((k + 1) * step,count) )
   *  where step = ceil(count / world_size)
   *
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
   * \sa ReturnType, TryAllreduce
   */
  ReturnType TryReduceScatterRing(void *sendrecvbuf_,
                                  size_t type_nbytes,
                                  size_t count,
                                  ReduceFunction reducer);
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf
   *  use a ring based algorithm, reduce-scatter + allgather
   *
   * \param sendrecvbuf_ buffer for both sending and receiving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryAllreduceRing(void *sendrecvbuf_,
                              size_t type_nbytes,
                              size_t count,
                              ReduceFunction reducer);
  /*!
   * \brief function used to report error when a link goes wrong
   * \param link the pointer to the link who causes the error
   * \param err the error type
   */
  inline ReturnType ReportError(LinkRecord *link, ReturnType err) {
    err_link = link; return err;
  }
  //---- data structure related to model ----
  // call sequence counter, records how many calls we made so far
  // from last call to CheckPoint, LoadCheckPoint
  int seq_counter{0}; // NOLINT
  // version number of model
  int version_number {0};  // NOLINT
  // whether the job is running in Hadoop
  bool hadoop_mode;  // NOLINT
  //---- local data related to link ----
  // index of parent link, can be -1, meaning this is root of the tree
  int parent_index;  // NOLINT
  // rank of parent node, can be -1
  int parent_rank;  // NOLINT
  // sockets of all links this connects to
  std::vector<LinkRecord> all_links;  // NOLINT
  // used to record the link where things goes wrong
  LinkRecord *err_link;  // NOLINT
  // all the links in the reduction tree connection
  RefLinkVector tree_links;  // NOLINT
  // pointer to links in the ring
  LinkRecord *ring_prev, *ring_next;  // NOLINT
  //----- meta information-----
  // list of enviroment variables that are of possible interest
  std::vector<std::string> env_vars;  // NOLINT
  // unique identifier of the possible job this process is doing
  // used to assign ranks, optional, default to NULL
  std::string task_id;  // NOLINT
  // uri of current host, to be set by Init
  std::string host_uri;  // NOLINT
  // uri of tracker
  std::string tracker_uri;  // NOLINT
  // role in dmlc jobs
  std::string dmlc_role;  // NOLINT
  // port of tracker address
  int tracker_port;  // NOLINT
  // reduce buffer size
  size_t reduce_buffer_size;  // NOLINT
  // reduction method
  int reduce_method;  // NOLINT
  // minimum count of cells to use ring based method
  size_t reduce_ring_mincount;  // NOLINT
  // minimum block size per tree reduce
  size_t tree_reduce_minsize;  // NOLINT
  // current rank
  int rank;  // NOLINT
  // world size
  int world_size;  // NOLINT
  // connect retry time
  int connect_retry;  // NOLINT
  // by default, if rabit worker not recover in half an hour exit
  std::chrono::seconds timeout_sec{std::chrono::seconds{1800}}; // NOLINT
  // flag to enable rabit_timeout
  bool rabit_timeout = false;  // NOLINT
  // Enable TCP node delay
  bool rabit_enable_tcp_no_delay = false;  // NOLINT
};
}  // namespace engine
}  // namespace rabit
#endif  // RABIT_ALLREDUCE_BASE_H_
