/*!
 * \file engine_base.h
 * \brief Basic implementation of AllReduce
 *   using TCP non-block socket and tree-shape reduction.
 *
 *   This implementation provides basic utility of AllReduce and Broadcast
 *   without considering node failure
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#ifndef ALLREDUCE_ENGINE_BASE_H
#define ALLREDUCE_ENGINE_BASE_H

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

namespace engine {
/*! \brief implementation of basic AllReduce engine */
class AllReduceBase : public IEngine {
 public:
  // magic number to verify server
  const static int kMagic = 0xff99;
  // constant one byte out of band message to indicate error happening
  AllReduceBase(void);
  virtual ~AllReduceBase(void) {}
  // shutdown the engine
  void Shutdown(void);
  // initialize the manager
  void Init(void);
  /*! \brief set parameters to the sync manager */
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
  virtual void AllReduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,           
                         ReduceFunction reducer) {
    utils::Assert(TryAllReduce(sendrecvbuf_, type_nbytes, count, reducer) == kSuccess,
                  "AllReduce failed");
  }
  /*!
   * \brief broadcast data from root to all nodes
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   */
  virtual void Broadcast(void *sendrecvbuf_, size_t total_size, int root) {
    utils::Assert(TryBroadcast(sendrecvbuf_, total_size, root) == kSuccess,
                  "AllReduce failed");
  }
  /*! 
   * \brief load latest check point
   * \param p_model pointer to the model
   * \return true if there was stored checkpoint and load was successful
   *   false if there was no stored checkpoint, means we are start over gain
   */  
  virtual bool LoadCheckPoint(utils::ISerializable *p_model) {
    return false;
  }
  /*!
   * \brief checkpoint the model, meaning we finished a stage of execution
   * \param p_model pointer to the model
   */
  virtual void CheckPoint(const utils::ISerializable &model) {
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
      utils::Assert(ngap <= buffer_size, "AllReduce: boundary check");
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
   * \brief perform in-place allreduce, on sendrecvbuf, this function can fail, and will return the cause of failure
   *
   * NOTE on AllReduce:
   *    The kSuccess TryAllReduce does NOT mean every node have successfully finishes TryAllReduce.
   *    It only means the current node get the correct result of AllReduce.
   *    However, it means every node finishes LAST call(instead of this one) of AllReduce/Bcast
   * 
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param type_nbytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
   * \sa ReturnType
   */
  ReturnType TryAllReduce(void *sendrecvbuf_,
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
  //---- local data related to link ----
  // index of parent link, can be -1, meaning this is root of the tree
  int parent_index;
  // sockets of all links
  std::vector<LinkRecord> links;
  //----- meta information-----
  // uri of current host, to be set by Init
  std::string host_uri;
  // uri of master
  std::string master_uri;
  // port of master address
  int master_port;
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
#endif  // ALLREDUCE_ENGINE_BASE_H
