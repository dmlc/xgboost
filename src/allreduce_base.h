/*!
 * \file allreduce_base.h
 * \brief Basic implementation of AllReduce
 *   using TCP non-block socket and tree-shape reduction.
 *
 *   This implementation provides basic utility of AllReduce and Broadcast
 *   without considering node failure
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#ifndef RABIT_ALLREDUCE_BASE_H
#define RABIT_ALLREDUCE_BASE_H

#include <vector>
#include <string>
#include "./utils.h"
#include "./socket.h"
#include "./engine.h"

namespace MPI {
// MPI data type to be compatible with existing MPI interface
class Datatype {
 public:
  size_t type_size;
  Datatype(size_t type_size) : type_size(type_size) {}
};
}
namespace rabit {
namespace engine {
/*! \brief implementation of basic Allreduce engine */
class AllreduceBase : public IEngine {
 public:
  // magic number to verify server
  const static int kMagic = 0xff99;
  // constant one byte out of band message to indicate error happening
  AllreduceBase(void);
  virtual ~AllreduceBase(void) {}
  // initialize the manager
  void Init(void);
  // shutdown the engine
  virtual void Shutdown(void);
  /*!
   * \brief set parameters to the engine 
   * \param name parameter name
   * \param val parameter value
   */
  virtual void SetParam(const char *name, const char *val);
  /*! \brief get rank */
  virtual int GetRank(void) const {
    return rank;
  }
  /*! \brief get rank */
  virtual int GetWorldSize(void) const {
    return world_size;
  }
  /*! \brief get rank */
  virtual std::string GetHost(void) const {
    return host_uri;
  }
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf 
   *        this function is NOT thread-safe
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   */  
  virtual void Allreduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,           
                         ReduceFunction reducer) {
    utils::Assert(TryAllreduce(sendrecvbuf_, type_nbytes, count, reducer) == kSuccess,
                  "Allreduce failed");
  }
  /*!
   * \brief broadcast data from root to all nodes
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   */
  virtual void Broadcast(void *sendrecvbuf_, size_t total_size, int root) {
    utils::Assert(TryBroadcast(sendrecvbuf_, total_size, root) == kSuccess,
                  "Broadcast failed");
  }
  /*!
   * \brief load latest check point
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller need to gauranttees that global_model
   *   is the same in all nodes
   * \param local_model pointer to local model, that is specific to current node/rank
   *   this can be NULL when no local model is needed
   *
   * \return the version number of check point loaded
   *     if returned version == 0, this means no model has been CheckPointed
   *     the p_model is not touched, user should do necessary initialization by themselves
   *   
   *   Common usage example:
   *      int iter = rabit::LoadCheckPoint(&model);
   *      if (iter == 0) model.InitParameters();
   *      for (i = iter; i < max_iter; ++i) {
   *        do many things, include allreduce
   *        rabit::CheckPoint(model);
   *      } 
   *
   * \sa CheckPoint, VersionNumber
   */
  virtual int LoadCheckPoint(utils::ISerializable *global_model,
                             utils::ISerializable *local_model = NULL) {
    return 0;
  }
  /*!
   * \brief checkpoint the model, meaning we finished a stage of execution
   *  every time we call check point, there is a version number which will increase by one
   * 
   * \param global_model pointer to the globally shared model/state
   *   when calling this function, the caller need to gauranttees that global_model
   *   is the same in all nodes
   * \param local_model pointer to local model, that is specific to current node/rank
   *   this can be NULL when no local state is needed
   *
   * NOTE: local_model requires explicit replication of the model for fault-tolerance, which will
   *       bring replication cost in CheckPoint function. global_model do not need explicit replication.
   *       So only CheckPoint with global_model if possible
   *
   * \sa LoadCheckPoint, VersionNumber
   */
  virtual void CheckPoint(const utils::ISerializable *global_model,
                          const utils::ISerializable *local_model = NULL) {
    version_number += 1;
  }
  /*!
   * \return version number of current stored model,
   *         which means how many calls to CheckPoint we made so far
   * \sa LoadCheckPoint, CheckPoint
   */
  virtual int VersionNumber(void) const {
    return version_number;
  }
  /*!
   * \brief explicitly re-init everything before calling LoadCheckPoint
   *    call this function when IEngine throw an exception out,
   *    this function is only used for test purpose
   */
  virtual void InitAfterException(void) {
    utils::Error("InitAfterException: not implemented");
  }
  /*! 
   * \brief report current status to the job tracker 
   * depending on the job tracker we are in
   */
  inline void ReportStatus(void) const {
    if (hadoop_mode != 0) {
      fprintf(stderr, "reporter:status:Rabit Phase[%03d] Operation %03d\n",
              version_number, seq_counter);
    }
  }

 protected:
  /*! \brief enumeration of possible returning results from Try functions */
  enum ReturnType {
    /*! \brief execution is successful */
    kSuccess,
    /*! \brief a neighbor node go down, the connection is dropped */
    kSockError,
    /*! 
     * \brief another node which is not my neighbor go down,
     *   get Out-of-Band exception notification from my neighbor
     */
    kGetExcept
  };
  // link record to a neighbor
  struct LinkRecord {
   public:
    // socket to get data from/to link
    utils::TCPSocket sock;
    // rank of the node in this link
    int rank;
    // size of data readed from link
    size_t size_read;
    // size of data sent to the link
    size_t size_write;
    // pointer to buffer head
    char *buffer_head;
    // buffer size, in bytes
    size_t buffer_size;
    // constructor
    LinkRecord(void) {}
    // initialize buffer
    inline void InitBuffer(size_t type_nbytes, size_t count, size_t reduce_buffer_size) {
      size_t n = (type_nbytes * count + 7)/ 8;
      buffer_.resize(std::min(reduce_buffer_size, n));
      // make sure align to type_nbytes
      buffer_size = buffer_.size() * sizeof(uint64_t) / type_nbytes * type_nbytes;
      utils::Assert(type_nbytes <= buffer_size, "too large type_nbytes=%lu, buffer_size=%lu", type_nbytes, buffer_size);
      // set buffer head
      buffer_head = reinterpret_cast<char*>(BeginPtr(buffer_));
    }
    // reset the recv and sent size
    inline void ResetSize(void) {
      size_write = size_read = 0;
    }
    /*!
     * \brief read data into ring-buffer, with care not to existing useful override data
     *  position after protect_start
     * \param protect_start all data start from protect_start is still needed in buffer
     *                      read shall not override this 
     * \return true if it is an successful read, false if there is some error happens, check errno
     */
    inline bool ReadToRingBuffer(size_t protect_start) {
      size_t ngap = size_read - protect_start;
      utils::Assert(ngap <= buffer_size, "Allreduce: boundary check");
      size_t offset = size_read % buffer_size;
      size_t nmax = std::min(buffer_size - ngap, buffer_size - offset);      
      if (nmax == 0) return true;
      ssize_t len = sock.Recv(buffer_head + offset, nmax);
      // length equals 0, remote disconnected
      if (len == 0) {
        sock.Close(); return false;
      }
      if (len == -1) return errno == EAGAIN || errno == EWOULDBLOCK;
      size_read += static_cast<size_t>(len);
      return true;
    }    
    /*!
     * \brief read data into array,
     * this function can not be used together with ReadToRingBuffer
     * a link can either read into the ring buffer, or existing array
     * \param max_size maximum size of array
     * \return true if it is an successful read, false if there is some error happens, check errno
     */
    inline bool ReadToArray(void *recvbuf_, size_t max_size) {
      if (max_size == size_read) return true;
      char *p = static_cast<char*>(recvbuf_);
      ssize_t len = sock.Recv(p + size_read, max_size - size_read);
      // length equals 0, remote disconnected
      if (len == 0) {
        sock.Close(); return false;
      }
      if (len == -1) return errno == EAGAIN || errno == EWOULDBLOCK;
      size_read += static_cast<size_t>(len);
      return true;
    }
    /*!
     * \brief write data in array to sock
     * \param sendbuf_ head of array
     * \param max_size maximum size of array
     * \return true if it is an successful write, false if there is some error happens, check errno
     */
    inline bool WriteFromArray(const void *sendbuf_, size_t max_size) {
      const char *p = static_cast<const char*>(sendbuf_);
      ssize_t len = sock.Send(p + size_write, max_size - size_write);
      if (len == -1) return errno == EAGAIN || errno == EWOULDBLOCK;
      size_write += static_cast<size_t>(len);
      return true;
    }

   private:
    // recv buffer to get data from child
    // aligned with 64 bits, will be able to perform 64 bits operations freely
    std::vector<uint64_t> buffer_;
  };
  /*!
   * \brief connect to the tracker to fix the the missing links
   *   this function is also used when the engine start up
   * \param cmd possible command to sent to tracker
   */
  void ReConnectLinks(const char *cmd = "start");
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf, this function can fail, and will return the cause of failure
   *
   * NOTE on Allreduce:
   *    The kSuccess TryAllreduce does NOT mean every node have successfully finishes TryAllreduce.
   *    It only means the current node get the correct result of Allreduce.
   *    However, it means every node finishes LAST call(instead of this one) of Allreduce/Bcast
   * 
   * \param sendrecvbuf_ buffer for both sending and recving data
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
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryBroadcast(void *sendrecvbuf_, size_t size, int root);
  //---- data structure related to model ----
  // call sequence counter, records how many calls we made so far
  // from last call to CheckPoint, LoadCheckPoint
  int seq_counter;
  // version number of model
  int version_number;
  // whether the job is running in hadoop
  int hadoop_mode;
  //---- local data related to link ----
  // index of parent link, can be -1, meaning this is root of the tree
  int parent_index;
  // rank of parent node, can be -1
  int parent_rank;
  // sockets of all links
  std::vector<LinkRecord> links;
  // pointer to someplace in the ring
  LinkRecord *ring_prev, *ring_next;
  //----- meta information-----
  // unique identifier of the possible job this process is doing
  // used to assign ranks, optional, default to NULL
  std::string task_id;
  // uri of current host, to be set by Init
  std::string host_uri;
  // uri of tracker
  std::string tracker_uri;
  // port of tracker address
  int tracker_port;
  // port of slave process
  int slave_port, nport_trial;
  // reduce buffer size
  size_t reduce_buffer_size;
  // current rank
  int rank;
  // world size
  int world_size;
};
}  // namespace engine
}  // namespace rabit
#endif  // RABIT_ALLREDUCE_BASE_H
