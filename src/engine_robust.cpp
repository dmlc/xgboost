/*!
 * \file engine_robust.cpp
 * \brief Robust implementation of AllReduce 
 *   using TCP non-block socket and tree-shape reduction.
 *
 *   This implementation considers the failure of nodes
 *   
 * \author Tianqi, Nacho, Tianyi
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <vector>
#include <string>
#include <cstring>
#include "./utils.h"
#include "./engine.h"
#include "./socket.h"

namespace MPI {
// MPI data type to be compatible with existing MPI interface
class Datatype {
 public:
  size_t type_size;
  Datatype(size_t type_size) : type_size(type_size) {}
};
}

namespace engine {
/*! \brief implementation of fault tolerant all reduce engine */
class AllReduceManager : public IEngine {
 public:
  // magic number to verify server
  const static int kMagic = 0xff99;
  // constant one byte out of band message to indicate error happening
  // and mark for channel cleanup
  const static char kOOBReset = 95;
  // and mark for channel cleanup
  const static char kOOBResetAck = 97;

  AllReduceManager(void) {
    master_uri = "NULL";
    master_port = 9000;
    host_uri = "";
    slave_port = 9010;
    nport_trial = 1000;
    rank = 0;
    world_size = 1;
    this->SetParam("reduce_buffer", "256MB");
  }
  ~AllReduceManager(void) {
  }
  inline void Shutdown(void) {
    for (size_t i = 0; i < links.size(); ++i) {
      links[i].sock.Close();
    }
    links.clear();
    utils::TCPSocket::Finalize();
  }
  /*! \brief set parameters to the sync manager */
  inline void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "master_uri")) master_uri = val;
    if (!strcmp(name, "master_port")) master_port = atoi(val);
    if (!strcmp(name, "reduce_buffer")) {
      char unit;
      unsigned long amount;
      if (sscanf(val, "%lu%c", &amount, &unit) == 2) {
        switch (unit) {
          case 'B': reduce_buffer_size = (amount + 7)/ 8; break;
          case 'K': reduce_buffer_size = amount << 7UL; break;
          case 'M': reduce_buffer_size = amount << 17UL; break;
          case 'G': reduce_buffer_size = amount << 27UL; break;
          default: utils::Error("invalid format for reduce buffer");
        }
      } else {
        utils::Error("invalid format for reduce_buffer, shhould be {integer}{unit}, unit can be {B, KB, MB, GB}");
      }
    }
  }
  // initialize the manager
  inline void Init(void) {
    utils::Socket::Startup();
    // single node mode
    if (master_uri == "NULL") return;
    utils::Assert(links.size() == 0, "can only call Init once");
    int magic = kMagic;
    int nchild = 0, nparent = 0;
    this->host_uri = utils::SockAddr::GetHostName();
    // get information from master
    utils::TCPSocket master;
    master.Create();
    if (!master.Connect(utils::SockAddr(master_uri.c_str(), master_port))) {
      utils::Socket::Error("Connect");
    }
    utils::Assert(master.SendAll(&magic, sizeof(magic)) == sizeof(magic), "sync::Init failure 1");
    utils::Assert(master.RecvAll(&magic, sizeof(magic)) == sizeof(magic), "sync::Init failure 2");
    utils::Check(magic == kMagic, "sync::Invalid master message, init failure");
    utils::Assert(master.RecvAll(&rank, sizeof(rank)) == sizeof(rank), "sync::Init failure 3");
    utils::Assert(master.RecvAll(&world_size, sizeof(world_size)) == sizeof(world_size), "sync::Init failure 4");
    utils::Assert(master.RecvAll(&nparent, sizeof(nparent)) == sizeof(nparent), "sync::Init failure 5");
    utils::Assert(master.RecvAll(&nchild, sizeof(nchild)) == sizeof(nchild), "sync::Init failure 6");
    utils::Assert(nchild >= 0, "in correct number of childs");
    utils::Assert(nparent == 1 || nparent == 0, "in correct number of parent");

    // create listen
    utils::TCPSocket sock_listen;
    sock_listen.Create();
    int port = sock_listen.TryBindHost(slave_port, slave_port + nport_trial);
    utils::Check(port != -1, "sync::Init fail to bind the ports specified");
    sock_listen.Listen();

    if (nparent != 0) {
      parent_index = 0;
      links.push_back(LinkRecord());
      int len, hport;
      std::string hname;
      utils::Assert(master.RecvAll(&len, sizeof(len)) == sizeof(len), "sync::Init failure 9");
      hname.resize(len);
      utils::Assert(len != 0, "string must not be empty");
      utils::Assert(master.RecvAll(&hname[0], len) == static_cast<size_t>(len), "sync::Init failure 10");
      utils::Assert(master.RecvAll(&hport, sizeof(hport)) == sizeof(hport), "sync::Init failure 11");
      links[0].sock.Create();
      links[0].sock.Connect(utils::SockAddr(hname.c_str(), hport));      
      utils::Assert(links[0].sock.SendAll(&magic, sizeof(magic)) == sizeof(magic), "sync::Init failure 12");
      utils::Assert(links[0].sock.RecvAll(&magic, sizeof(magic)) == sizeof(magic), "sync::Init failure 13");
      utils::Check(magic == kMagic, "sync::Init failure, parent magic number mismatch");
      parent_index = 0;
    } else {
      parent_index = -1;
    }
    // send back socket listening port to master
    utils::Assert(master.SendAll(&port, sizeof(port)) == sizeof(port), "sync::Init failure 14");
    // close connection to master
    master.Close();
    // accept links from childs
    for (int i = 0; i < nchild; ++i) {
      LinkRecord r; 
      while (true) {
        r.sock = sock_listen.Accept();
        if (r.sock.RecvAll(&magic, sizeof(magic)) == sizeof(magic) && magic == kMagic) {
          utils::Assert(r.sock.SendAll(&magic, sizeof(magic)) == sizeof(magic), "sync::Init failure 15");
          break;
        } else {         
          // not a valid child
          r.sock.Close();
        }
      }
      links.push_back(r);
    }
    // close listening sockets
    sock_listen.Close();
    // setup selecter
    for (size_t i = 0; i < links.size(); ++i) {
      // set the socket to non-blocking mode
      links[i].sock.SetNonBlock(true);
    }
    // done
  }
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
  virtual void AllReduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,           
                         ReduceFunction reducer) {
    while (true) {
      if (rank == rand() % 3) TryResetLinks();
      ReturnType ret = TryAllReduce(sendrecvbuf_, type_nbytes, count, reducer);
      if (ret == kSuccess) return;
      if (ret == kSockError) {
        utils::Error("error occur during all reduce\n");
      }
      utils::Check(TryResetLinks() == kSuccess, "error when reset links");      
    }
  }
  /*!
   * \brief broadcast data from root to all nodes
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param size the size of the data to be broadcasted
   * \param root the root worker id to broadcast the data
   */    
  virtual void Broadcast(void *sendrecvbuf_, size_t total_size, int root) {
    if (links.size() == 0) return;
    // number of links
    const int nlink = static_cast<int>(links.size());
    // size of space already read from data
    size_t size_in = 0;
    // input link, -2 means unknown yet, -1 means this is root
    int in_link = -2;

    // initialize the link statistics
    for (int i = 0; i < nlink; ++i) {
      links[i].ResetSize();
    }
    // root have all the data
    if (this->rank == root) {
      size_in = total_size;
      in_link = -1;
    }
    
    // while we have not passed the messages out
    while(true) {
      // select helper
      utils::SelectHelper selecter;
      for (size_t i = 0; i < links.size(); ++i) {
        selecter.WatchRead(links[i].sock);
        selecter.WatchWrite(links[i].sock);
        selecter.WatchException(links[i].sock);        
      }
      if (in_link == -2) {
        // probe in-link
        for (int i = 0; i < nlink; ++i) {
          if (selecter.CheckRead(links[i].sock)) {
            if (!links[i].ReadToArray(sendrecvbuf_, total_size)) {
              utils::Socket::Error("Recv");
            }
            size_in = links[i].size_read;
            if (size_in != 0) {
              in_link = i; break;
            }
          }
        }
      } else {
        // read from in link
        if (in_link >= 0 && selecter.CheckRead(links[in_link].sock)) {
          if(!links[in_link].ReadToArray(sendrecvbuf_, total_size)) {
            utils::Socket::Error("Recv");
          }
          size_in = links[in_link].size_read;
        }
      }
      size_t nfinished = total_size;
      // send data to all out-link
      for (int i = 0; i < nlink; ++i) {
        if (i != in_link) {
          if (selecter.CheckWrite(links[i].sock)) {
            if (!links[i].WriteFromArray(sendrecvbuf_, size_in)) {
              utils::Socket::Error("Send");
            }
          }
          nfinished = std::min(nfinished, links[i].size_write);
        }
      }
      // check boundary condition
      if (nfinished >= total_size) break;
    }
  }
  virtual bool LoadCheckPoint(utils::ISerializable *p_model) {
    return false;
  }
  virtual void CheckPoint(const utils::ISerializable &model) {
  }
  
 protected:
  // possible returning type from the Try Functions
  enum ReturnType {
    kSuccess,
    kSockError,
    kGetExcept
  };
  // possible state of the server
  enum ServerState {
    kNormal,
    kConnDrop,
    kRecover
  };
  // cleanup the links, by sending OOB message
  inline ReturnType TryResetLinks(void) {
    // number of links
    const int nlink = static_cast<int>(links.size());
    for (int i = 0; i < nlink; ++i) {
      links[i].InitBuffer(sizeof(int), 1 << 10, reduce_buffer_size);
      links[i].ResetSize();
    }
    printf("[%d] start to reset link\n", rank);    
    while (true) {
      printf("[%d] loop\n", rank);
      bool finished = true;
      for (int i = 0; i < nlink; ++i) {
        if (links[i].sock.BadSocket()) continue;
        if (links[i].size_write == 0) {
          char sig = kOOBReset;
          ssize_t len = links[i].sock.Send(&sig, sizeof(sig), MSG_OOB);
          // error will be filtered in next loop
          if (len != -1) {
            links[i].size_write += len;
            printf("[%d] send OOB success\n", rank);
          }
        }
        // need to send OOB to every other link
        if (links[i].size_write == 0) finished = false;
      }
      if (finished) break;
    }
    printf("[%d] finish send all OOB\n", rank);
    // wait for incoming except from all links
    for (int i = 0; i < nlink; ++ i) {
      if (links[i].sock.BadSocket()) continue;
      printf("[%d] wait except\n", rank);
      if (utils::SelectHelper::WaitExcept(links[i].sock) == -1) {
        utils::Socket::Error("select");
      }
      printf("[%d] finish wait except\n", rank);
    }
    printf("[%d] start to discard link\n", rank);
    // read and discard data from all channels until pass mark
    while (true) {
      utils::SelectHelper rsel;
      bool finished = true;
      for (int i = 0; i < nlink; ++i) {
        if (links[i].sock.BadSocket()) continue;
        if (links[i].size_read == 0) {
          int atmark = links[i].sock.AtMark();
          if (atmark < 0)  return kSockError;
          if (atmark == 1) {
            char oob_msg;
            ssize_t len = links[i].sock.Recv(&oob_msg, sizeof(oob_msg), MSG_OOB);
            if (len == -1 && errno != EAGAIN && errno != EWOULDBLOCK) {
              finished = false; continue;
            }
            utils::Assert(oob_msg == kOOBReset, "wrong oob msg");
            links[i].size_read = 1;
          } else  {
            finished = false;
            rsel.WatchRead(links[i].sock);
          }
        }
      }
      if (finished) break;
      // wait to read from the channels to discard data
      rsel.Select();
      printf("[%d] select finish read from\n", rank);
      for (int i = 0; i < nlink; ++i) {
        if (links[i].sock.BadSocket()) continue;
        if (rsel.CheckRead(links[i].sock)) {                        
          ssize_t len = links[i].sock.Recv(links[i].buffer_head, links[i].buffer_size);
          // zero length, remote closed the connection, close socket
          if (len == 0) {
            links[i].sock.Close();
          } else if (len == -1) {
            // when error happens here, oob_clear will remember
            if (errno == EAGAIN && errno == EWOULDBLOCK) printf("would block\n");
          } else {
            printf("[%d] discard %ld bytes\n", rank, len);
          }
        }
      }
    }
    printf("[%d] discard all success\n", rank);
    // start synchronization step
    for (int i = 0; i < nlink; ++i) {
      links[i].ResetSize();
    }
    while (true) {
      // selecter for TryResetLinks
      utils::SelectHelper rsel;
      for (int i = 0; i < nlink; ++i) {
        if (links[i].sock.BadSocket()) continue;
        if (links[i].size_read == 0) rsel.WatchRead(links[i].sock);
        if (links[i].size_write == 0) rsel.WatchWrite(links[i].sock);
      }
      printf("[%d] before select\n", rank);
      rsel.Select();
      printf("[%d] after select\n", rank);
      bool finished = true;
      for (int i = 0; i < nlink; ++i) {
        if (links[i].sock.BadSocket()) continue;
        if (links[i].size_read == 0 && rsel.CheckRead(links[i].sock)) {
          char ack;
          links[i].ReadToArray(&ack, sizeof(ack));
          if (links[i].size_read != 0) {
            utils::Assert(ack == kOOBResetAck, "expect ack message");
          }
        }
        if (links[i].size_write == 0 && rsel.CheckWrite(links[i].sock)) {
          char ack = kOOBResetAck;
          links[i].WriteFromArray(&ack, sizeof(ack));
        }
        if (links[i].size_read == 0 || links[i].size_write == 0) finished = false;
      }
      if (finished) break;
    }
    printf("[%d] after the read write data success\n", rank);
    for (int i = 0; i < nlink; ++i) {
      if (links[i].sock.BadSocket()) return kSockError;
    }    
    return kSuccess;
  }
  // Run AllReduce, return if success
  inline ReturnType TryAllReduce(void *sendrecvbuf_,
                                 size_t type_nbytes,
                                 size_t count,
                                 ReduceFunction reducer) {
    if (links.size() == 0) return kSuccess;
    // total size of message
    const size_t total_size = type_nbytes * count;
    // number of links
    const int nlink = static_cast<int>(links.size());
    // send recv buffer
    char *sendrecvbuf = reinterpret_cast<char*>(sendrecvbuf_);
    // size of space that we already performs reduce in up pass
    size_t size_up_reduce = 0;
    // size of space that we have already passed to parent
    size_t size_up_out = 0;
    // size of message we received, and send in the down pass
    size_t size_down_in = 0;
    // initialize the link ring-buffer and pointer
    for (int i = 0; i < nlink; ++i) {
      if (i != parent_index) {
        links[i].InitBuffer(type_nbytes, count, reduce_buffer_size);
      }
      links[i].ResetSize();
    }
    // if no childs, no need to reduce
    if (nlink == static_cast<int>(parent_index != -1)) {
      size_up_reduce = total_size;
    }
    
    // while we have not passed the messages out
    while (true) {
      // select helper
      utils::SelectHelper selecter;
      for (size_t i = 0; i < links.size(); ++i) {
        selecter.WatchRead(links[i].sock);
        selecter.WatchWrite(links[i].sock);
        selecter.WatchException(links[i].sock);        
      }
      // select must return 
      selecter.Select();
      // exception handling
      for (int i = 0; i < nlink; ++i) {
        // recive OOB message from some link 
        if (selecter.CheckExcept(links[i].sock)) return kGetExcept;
      }
      // read data from childs
      for (int i = 0; i < nlink; ++i) {
        if (i != parent_index && selecter.CheckRead(links[i].sock)) {
          if (!links[i].ReadToRingBuffer(size_up_out)) return kSockError;
        }
      }
      // this node have childs, peform reduce
      if (nlink > static_cast<int>(parent_index != -1)) {
        size_t buffer_size = 0;
        // do upstream reduce
        size_t max_reduce = total_size;
        for (int i = 0; i < nlink; ++i) {
          if (i != parent_index) {
            max_reduce= std::min(max_reduce, links[i].size_read);
            utils::Assert(buffer_size == 0 || buffer_size == links[i].buffer_size,
                          "buffer size inconsistent");
            buffer_size = links[i].buffer_size;
          }
        }
        utils::Assert(buffer_size != 0, "must assign buffer_size");
        // round to type_n4bytes
        max_reduce = (max_reduce / type_nbytes * type_nbytes);
        // peform reduce, can be at most two rounds
        while (size_up_reduce < max_reduce) {
          // start position
          size_t start = size_up_reduce % buffer_size;
          // peform read till end of buffer
          size_t nread = std::min(buffer_size - start, max_reduce - size_up_reduce);          
          utils::Assert(nread % type_nbytes == 0, "AllReduce: size check");
          for (int i = 0; i < nlink; ++i) {
            if (i != parent_index) {
              reducer(links[i].buffer_head + start,
                      sendrecvbuf + size_up_reduce,
                      static_cast<int>(nread / type_nbytes),
                      MPI::Datatype(type_nbytes));
            }
          }
          size_up_reduce += nread;
        }
      }
      if (parent_index != -1) {
        // pass message up to parent, can pass data that are already been reduced
        if (selecter.CheckWrite(links[parent_index].sock)) {              
          ssize_t len = links[parent_index].sock.
              Send(sendrecvbuf + size_up_out, size_up_reduce - size_up_out);
          if (len != -1) {
            size_up_out += static_cast<size_t>(len);
          } else {
            if (errno != EAGAIN && errno != EWOULDBLOCK) return kSockError;
          }
        }
        // read data from parent
        if (selecter.CheckRead(links[parent_index].sock) && total_size > size_down_in) {
          ssize_t len = links[parent_index].sock.
              Recv(sendrecvbuf + size_down_in, total_size - size_down_in);
          if (len == 0) {
            links[parent_index].sock.Close(); return kSockError;
          }
          if (len != -1) {
            size_down_in += static_cast<size_t>(len);
            utils::Assert(size_down_in <= size_up_out, "AllReduce: boundary error");
          } else {
            if (errno != EAGAIN && errno != EWOULDBLOCK) return kSockError;
          }
        }
      } else {
        // this is root, can use reduce as most recent point
        size_down_in = size_up_out = size_up_reduce;
      }
      // check if we finished the job of message passing
      size_t nfinished = size_down_in;
      // can pass message down to childs
      for (int i = 0; i < nlink; ++i) {
        if (i != parent_index) {
          if (selecter.CheckWrite(links[i].sock)) {
            if (!links[i].WriteFromArray(sendrecvbuf, size_down_in)) return kSockError;
          }
          nfinished = std::min(links[i].size_write, nfinished);
        }
      }
      // check boundary condition
      if (nfinished >= total_size) break;
    }
    return kSuccess;
  }
  
 private:
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
      if (max_size == size_read ) return true;
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
  // data structure to remember result of Bcast and AllReduce calls
  class ResultBuffer {
   public:
    // constructor
    ResultBuffer(void) {
      this->Clear();
    }
    // clear the existing record
    inline void Clear(void) {
      seqno_.clear(); size_.clear();
      rptr_.clear(); rptr_.push_back(0);
      data_.clear();
    }
    // allocate temporal space for 
    inline void *AllocTemp(size_t type_nbytes, size_t count) {
      size_t size = type_nbytes * count;
      size_t nhop = (size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
      utils::Assert(nhop != 0, "cannot allocate 0 size memory");
      data_.resize(rptr_.back() + nhop);
      return BeginPtr(data_) + rptr_.back();
    }
    // push the result in temp to the 
    inline void PushTemp(int seqid, size_t type_nbytes, size_t count) {
      size_t size = type_nbytes * count;
      size_t nhop = (size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
      if (seqno_.size() != 0) {
        utils::Assert(seqno_.back() < seqid, "PushTemp seqid inconsistent");
      }
      seqno_.push_back(seqid);
      rptr_.push_back(rptr_.back() + nhop);
      size_.push_back(size);
      utils::Assert(data_.size() == rptr_.back(), "PushTemp inconsistent");
    }
    // return the stored result of seqid, if any 
    inline void* Query(int seqid, size_t *p_size) {
      size_t idx = std::lower_bound(seqno_.begin(), seqno_.end(), seqid) - seqno_.begin();
      if (idx == seqno_.size() || seqno_[idx] != seqid) return NULL;
      *p_size = size_[idx];
      return BeginPtr(data_) + rptr_[idx];
    }
   private:
    // sequence number of each 
    std::vector<int> seqno_;
    // pointer to the positions
    std::vector<size_t> rptr_;
    // actual size of each buffer
    std::vector<size_t> size_;
    // content of the buffer
    std::vector<uint64_t> data_;    
  };
  //---- recovery data structure ----
  // call sequence counter, records how many calls we made so far
  // from last call to CheckPoint, LoadCheckPoint
  int seq_counter;
  // result buffer
  ResultBuffer resbuf;
  // model that is saved from last CheckPoint
  std::string check_point;
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

// singleton sync manager
AllReduceManager manager;

/*! \brief intiialize the synchronization module */
void Init(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      manager.SetParam(name, val);
    }
  }
  manager.Init();
}

/*! \brief finalize syncrhonization module */
void Finalize(void) {
  manager.Shutdown();
}
/*! \brief singleton method to get engine */
IEngine *GetEngine(void) {
  return &manager;
}
}  // namespace engine
