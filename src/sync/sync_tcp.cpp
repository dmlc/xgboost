/*!
 * \file sync_tcp.cpp
 * \brief implementation of sync AllReduce using TCP sockets
 *   with use non-block socket and tree-shape reduction
 * \author Tianqi Chen
 */
#include <vector>
#include <string>
#include <cstring>
#include "./sync.h"
#include "../utils/socket.h"

namespace MPI {
class Datatype {
 public:
  size_t type_size;
  Datatype(size_t type_size) : type_size(type_size) {}
};
}
namespace xgboost {
namespace sync {
/*! \brief implementation of sync goes to here */
class SyncManager {  
 public:
  const static int kMagic = 0xff99;
  SyncManager(void) {
    master_uri = "NULL";
    master_port = 9000;
    host_uri = "";
    slave_port = 9010;
    nport_trial = 1000;
    rank = 0;
    world_size = 1;
    this->SetParam("reduce_buffer", "256MB");
  }
  ~SyncManager(void) {
    this->Shutdown();
  }
  inline void Shutdown(void) {
    for (size_t i = 0; i < links.size(); ++i) {
      links[i].sock.Close();
    }
    links.clear();
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
  /*! \brief get rank */
  inline int GetRank(void) const {
    return rank;
  }
  /*! \brief check whether its distributed mode */
  inline bool IsDistributed(void) const {
    return links.size() != 0;
  }
  /*! \brief get rank */
  inline int GetWorldSize(void) const {
    return world_size;
  }
  /*! \brief get rank */
  inline std::string GetHost(void) const {
    return host_uri;
  }
  // initialize the manager
  inline void Init(void) {
    // single node mode
    if (master_uri == "NULL") return;
    utils::Assert(links.size() == 0, "can only call Init once");
    int magic = kMagic;
    int nchild = 0, nparent = 0;
    this->host_uri = utils::SockAddr::GetHostName();
    // get information from master
    utils::TCPSocket master;
    master.Create();
    master.Connect(utils::SockAddr(master_uri.c_str(), master_port));
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
    selecter.Clear();
    for (size_t i = 0; i < links.size(); ++i) {
      // set the socket to non-blocking mode
      links[i].sock.SetNonBlock(true);
      selecter.WatchRead(links[i].sock);
      selecter.WatchWrite(links[i].sock);
    }
    // done
  }
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf 
   *        this function is NOT thread-safe
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param type_n4bytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   */
  inline void AllReduce(void *sendrecvbuf_,
                        size_t type_nbytes,
                        size_t count,
                        ReduceHandle::ReduceFunction reducer) {
    if (links.size() == 0) return;
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
    while(true) {
      selecter.Select();
      // read data from childs
      for (int i = 0; i < nlink; ++i) {
        if (i != parent_index && selecter.CheckRead(links[i].sock)) {
          links[i].ReadToRingBuffer(size_up_out);
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
                      nread / type_nbytes,
                      MPI::Datatype(type_nbytes));
            }
          }
          size_up_reduce += nread;
        }
      }
      if (parent_index != -1) {
        // pass message up to parent, can pass data that are already been reduced
        if (selecter.CheckWrite(links[parent_index].sock)) {
          size_up_out += links[parent_index].sock.
              Send(sendrecvbuf + size_up_out, size_up_reduce - size_up_out);
        }
        // read data from parent
        if (selecter.CheckRead(links[parent_index].sock)) {
          size_down_in +=  links[parent_index].sock.
              Recv(sendrecvbuf + size_down_in, total_size - size_down_in);
          utils::Assert(size_down_in <= size_up_out, "AllReduce: boundary error");
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
            links[i].WriteFromArray(sendrecvbuf, size_down_in);
          }
          nfinished = std::min(links[i].size_write, nfinished);
        }
      }
      // check boundary condition
      if (nfinished >= total_size) break;
    }
  }
  /*!
   * \brief broadcast data from root to all nodes
   * \param sendrecvbuf_ buffer for both sending and recving data
   * \param type_n4bytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   */  
  inline void Bcast(void *sendrecvbuf_,
                    size_t total_size,
                    int root) {
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
      selecter.Select();
      if (in_link == -2) {
        // probe in-link
        for (int i = 0; i < nlink; ++i) {
          if (selecter.CheckRead(links[i].sock)) {
            links[i].ReadToArray(sendrecvbuf_, total_size);
            size_in = links[i].size_read;
            if (size_in != 0) {
              in_link = i; break;
            }
          }
        }
      } else {
        // read from in link
        if (in_link >= 0 && selecter.CheckRead(links[in_link].sock)) {
          links[in_link].ReadToArray(sendrecvbuf_, total_size);
          size_in = links[in_link].size_read;
        }
      }
      size_t nfinished = total_size;
      // send data to all out-link
      for (int i = 0; i < nlink; ++i) {
        if (i != in_link) {
          if (selecter.CheckWrite(links[i].sock)) {
            links[i].WriteFromArray(sendrecvbuf_, size_in);
          }
          nfinished = std::min(nfinished, links[i].size_write);
        }
      }
      // check boundary condition
      if (nfinished >= total_size) break;
    }
  }
 private:  
  // an independent child record
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
    // initialize buffer
    inline void InitBuffer(size_t type_nbytes, size_t count, size_t reduce_buffer_size) {
      size_t n = (type_nbytes * count + 7)/ 8;
      buffer_.resize(std::min(reduce_buffer_size, n));
      // make sure align to type_nbytes
      buffer_size = buffer_.size() * sizeof(uint64_t) / type_nbytes * type_nbytes;
      utils::Assert(type_nbytes < buffer_size, "too large type_nbytes=%lu, buffer_size", type_nbytes, buffer_size);
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
     */
    inline void ReadToRingBuffer(size_t protect_start) {
      size_t ngap = size_read - protect_start;
      utils::Assert(ngap <= buffer_size, "AllReduce: boundary check");
      size_t offset = size_read % buffer_size;
      size_t nmax = std::min(buffer_size - ngap, buffer_size - offset);
      size_read += sock.Recv(buffer_head + offset, nmax);
    }
    /*!
     * \brief read data into array,
     * this function can not be used together with ReadToRingBuffer
     * a link can either read into the ring buffer, or existing array
     * \param max_size maximum size of array
     */
    inline void ReadToArray(void *recvbuf_, size_t max_size) {
      char *p = static_cast<char*>(recvbuf_);
      size_read += sock.Recv(p + size_read, max_size - size_read);
    }
    /*!
     * \brief write data in array to sock
     * \param sendbuf_ head of array
     * \param max_size maximum size of array
     */
    inline void WriteFromArray(const void *sendbuf_, size_t max_size) {
      const char *p = static_cast<const char*>(sendbuf_);
      size_write += sock.Send(p + size_write, max_size - size_write);
    }

   private:
    // recv buffer to get data from child
    // aligned with 64 bits, will be able to perform 64 bits operations freely
    std::vector<uint64_t> buffer_;
  };
  //------------------
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
  // index of parent link, can be -1, meaning this is root of the tree
  int parent_index;
  // sockets of all links
  std::vector<LinkRecord> links;
  // select helper
  utils::SelectHelper selecter;
};

// singleton sync manager
SyncManager manager;

/*! \brief get rank of current process */
int GetRank(void) {
  return manager.GetRank();
}
/*! \brief get total number of process */
int GetWorldSize(void) {
  return manager.GetWorldSize();
}

/*! \brief get name of processor */
std::string GetProcessorName(void) {
  return manager.GetHost();
}
bool IsDistributed(void) {
  return manager.IsDistributed();
}
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

// this can only be used for data that was smaller than 64 bit
template<typename DType>
inline void ReduceSum(const void *src_, void *dst_, int len, const MPI::Datatype &dtype) {
  const DType *src = (const DType*)src_;
  DType *dst = (DType*)dst_;  
  for (int i = 0; i < len; ++i) {
    dst[i] += src[i];
  }
}
template<typename DType>
inline void ReduceMax(const void *src_, void *dst_, int len, const MPI::Datatype &dtype) {
  const DType *src = (const DType*)src_;
  DType *dst = (DType*)dst_;  
  for (int i = 0; i < len; ++i) {
    if (src[i] > dst[i]) dst[i] = src[i];
  }
}
template<typename DType>
inline void ReduceBitOR(const void *src_, void *dst_, int len, const MPI::Datatype &dtype) {
  const DType *src = (const DType*)src_;
  DType *dst = (DType*)dst_;  
  for (int i = 0; i < len; ++i) {
    dst[i] |= src[i];
  }
}

template<>
void AllReduce<uint32_t>(uint32_t *sendrecvbuf, int count, ReduceOp op) {
  typedef uint32_t DType;
  switch(op) {
    case kBitwiseOR: manager.AllReduce(sendrecvbuf, sizeof(DType), count, ReduceBitOR<DType>); return;
    default: utils::Error("reduce op not supported");
  }
}

template<>
void AllReduce<float>(float *sendrecvbuf, int count, ReduceOp op) {
  typedef float DType;
  switch(op) {
    case kSum: manager.AllReduce(sendrecvbuf, sizeof(DType), count, ReduceSum<DType>); return;
    case kMax: manager.AllReduce(sendrecvbuf, sizeof(DType), count, ReduceMax<DType>); return;
    default: utils::Error("unknown ReduceOp");
  }
}

void Bcast(std::string *sendrecv_data, int root) {
  unsigned len = static_cast<unsigned>(sendrecv_data->length());
  manager.Bcast(&len, sizeof(len), root);
  sendrecv_data->resize(len);
  if (len != 0) {
    manager.Bcast(&(*sendrecv_data)[0], len, root);  
  }
}

// code for reduce handle
ReduceHandle::ReduceHandle(void) : handle(NULL), htype(NULL) {
}
ReduceHandle::~ReduceHandle(void) {}

int ReduceHandle::TypeSize(const MPI::Datatype &dtype) {
  return dtype.type_size;
}
void ReduceHandle::Init(ReduceFunction redfunc, size_t type_n4bytes, bool commute) {
  utils::Assert(handle == NULL, "cannot initialize reduce handle twice");
  handle = reinterpret_cast<void*>(redfunc);
}
void ReduceHandle::AllReduce(void *sendrecvbuf, size_t type_n4bytes, size_t count) {
  utils::Assert(handle != NULL, "must intialize handle to call AllReduce");
  manager.AllReduce(sendrecvbuf, type_n4bytes * 4, count, reinterpret_cast<ReduceFunction*>(handle));
}

}  // namespace sync
}  // namespace xgboost
