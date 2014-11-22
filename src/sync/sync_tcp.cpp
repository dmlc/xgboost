/*!
 * \file sync_tcp.cpp
 * \brief implementation of sync AllReduce using TCP sockets
 *   with use async socket and tree-shape reduction
 * \author Tianqi Chen
 */
#include "./sync.h"
#include "../utils/socket.h"

namespace MPI {
struct Datatype {
  size_t type_size;
  Datatype(size_t type_size) : type_size(type_size) {}
};
}
namespace xgboost {
namespace sync {
/*! \brief implementation of sync goes to here */
class SyncManager {  
 public:
  // initialize the manager
  inline void Init(int argc, char *argv[]) {    
  }
  /*!
   * \brief perform in-place allreduce, on sendrecvbuf 
   *        this function is not thread-safe
   * \param sendrecvbuf buffer for both sending and recving data
   * \param type_n4bytes the unit number of bytes the type have
   * \param count number of elements to be reduced
   * \param reducer reduce function
   */
  inline void AllReduce(void *sendrecvbuf_,
                        size_t type_nbytes,
                        size_t count,
                        ReduceHandle::ReduceFunction reducer) {
    if (parent.size() == 0 && childs.size() == 0) return;
    char *sendrecvbuf = reinterpret_cast<char*>(sendrecvbuf_);
    // total size of message
    const size_t total_size = type_nbytes * count;
    // size of space that we already performs reduce in up pass
    size_t size_up_reduce = 0;
    // size of space that we have already passed to parent
    size_t size_up_out = 0;
    // size of message we received, and send in the down pass
    size_t size_down_in = 0;
    // initialize the send buffer
    for (size_t i = 0; i < childs.size(); ++i) {
      childs[i].Init(type_nbytes, count);
    }
    // if no childs, no need to reduce
    if (childs.size() == 0) size_up_reduce = total_size;    
    // while we have not passed the messages out
    while(true) {
      selecter.Select();
      // read data from childs
      for (size_t i = 0; i < childs.size(); ++i) {
        if (selecter.CheckRead(childs[i].sock)) {
          childs[i].Read(size_up_out);
        }
      }
      // peform reduce
      if (childs.size() != 0) {
        const size_t buffer_size = childs[0].buffer_size;
        // do upstream reduce
        size_t min_read = childs[0].size_read;
        for (size_t i = 1; i < childs.size(); ++i) {
          min_read = std::min(min_read, childs[i].size_read);
        }
        // align to type_nbytes
        min_read = (min_read / type_nbytes * type_nbytes);
        // start position
        size_t start = size_up_reduce % buffer_size;
        // peform read till end of buffer
        if (start + min_read - size_up_reduce > buffer_size) {
          const size_t nread = buffer_size - start;
          utils::Assert(nread % type_nbytes == 0, "AllReduce: size check");
          for (size_t i = 0; i < childs.size(); ++i) {
            reducer(childs[i].buffer_head + start,
                    sendrecvbuf + size_up_reduce,
                    nread / type_nbytes,
                    MPI::Datatype(type_nbytes));
          }
          size_up_reduce += nread;
          start = 0;
        }
        // peform second phase of reduce
        const size_t nread = min_read - size_up_reduce;
        if (nread != 0) {
          utils::Assert(nread % type_nbytes == 0, "AllReduce: size check");
          for (size_t i = 0; i < childs.size(); ++i) {
            reducer(childs[i].buffer_head + start,
                    sendrecvbuf + size_up_reduce,
                    nread / type_nbytes,
                    MPI::Datatype(type_nbytes));
          }
        }
        size_up_reduce += nread;
      }
      if (parent.size() != 0) {
        // can pass message up to parent
        if (selecter.CheckWrite(parent[0])) {
          size_up_out += parent[0]
              .Send(sendrecvbuf + size_up_out, size_up_reduce - size_up_out);
        }
        // read data from parent
        if (selecter.CheckRead(parent[0])) {
          size_down_in +=  parent[0]
              .Recv(sendrecvbuf + size_down_in, total_size - size_down_in);
          utils::Assert(size_down_in <= size_up_out, "AllReduce: boundary error");
        }
      } else {
        // this is root, can use reduce as most recent point
        size_down_in = size_up_out = size_up_reduce;
      }
      // check if we finished the job of message passing
      size_t nfinished = size_down_in;
      // can pass message down to childs
      for (size_t i = 0; i < childs.size(); ++i) {
        if (selecter.CheckWrite(childs[i].sock)) {
          childs[i].size_write += childs[i].sock
              .Send(sendrecvbuf + childs[i].size_write, size_down_in - childs[i].size_write);
        }
        nfinished = std::min(childs[i].size_write, nfinished);
      }
      // check boundary condition
      if (nfinished >= total_size) {
        utils::Assert(nfinished == total_size, "AllReduce: nfinished check");
        break;
      }
    }
  }
  inline void Bcast(std::string *sendrecv_data, int root) {
    if (parent.size() == 0 && childs.size() == 0) return;
    // message send to parent
    size_t size_up_out = 0;
    // all messages received
    size_t size_in = 0;
    // all headers received so far
    size_t header_in = 0;
    // total size of data
    size_t total_size;
    // input channel, -1 means parent, -2 means unknown yet
    // otherwise its child index
    int in_channel = -2;
    // root already reads all data in
    if (root == rank) {
      in_channel = -3;
      total_size = size_in = sendrecv_data->length();
      header_in = sizeof(total_size);
    }
    // initialize write position
    for (size_t i = 0; i < childs.size(); ++i) {
      childs[i].size_write = 0;
    }
    const int nchilds = static_cast<int>(childs.size());

    while (true) {
      selecter.Select();
      if (selecter.CheckRead(parent[0])) {
        utils::Assert(in_channel == -2 || in_channel == -1, "invalid in channel");
        this->BcastRecvData(parent[0], sendrecv_data,
                            header_in, size_in, total_size);
        if (header_in != 0) in_channel = -1;
      }
      for (int i = 0; i < nchilds; ++i) {
        if (selecter.CheckRead(childs[i].sock)) {
          utils::Assert(in_channel == -2 || in_channel == i, "invalid in channel");
          this->BcastRecvData(parent[0], sendrecv_data,
                              header_in, size_in, total_size);
          if (header_in != 0) in_channel = i;
        }
      }
      if (in_channel == -2) continue;
      if (in_channel != -1) {
        if (selecter.CheckWrite(parent[0])) {
          size_t nsend = size_in - size_up_out;
          if (nsend != 0) {
            size_up_out += parent[0].Send(&(*sendrecv_data)[0] + size_up_out, nsend);
          }
        }
      } else {
        size_up_out = size_in;
      }
      size_t nfinished = size_up_out;
      for (int i = 0; i < nchilds; ++i) {
        if (in_channel != i) {
          if (selecter.CheckWrite(childs[i].sock)) {
            size_t nsend = size_in - childs[i].size_write;
            if (nsend != 0) {
              childs[i].size_write += childs[i].sock
                  .Send(&(*sendrecv_data)[0] + childs[i].size_write, nsend);
            }
          }
          nfinished = std::min(nfinished, childs[i].size_write);
        }
      }
      // check boundary condition
      if (nfinished >= total_size) {
        utils::Assert(nfinished == total_size, "Bcast: nfinished check");
        break;
      }
    }
  }

 private:
  inline void BcastRecvData(utils::TCPSocket &sock,
                            std::string *sendrecv_data,   
                            size_t &header_in,
                            size_t &size_in,
                            size_t &total_size) {
    if (header_in < sizeof(total_size)) {
      char *p = reinterpret_cast<char*>(&total_size);
      header_in += sock.Recv(p + size_in, sizeof(total_size) - header_in);
      if (header_in == sizeof(total_size)) {
        sendrecv_data->resize(total_size);
      }
    } else {
      size_t nread  = total_size - size_in;
      if (nread != 0) {
        size_in += sock
            .Recv(&(*sendrecv_data)[0] + size_in, nread);
      }
    }
  }
  
  // 128 MB
  const static size_t kBufferSize = 128;
  // an independent child record
  struct ChildRecord {
   public:
    // socket to get data from child
    utils::TCPSocket sock;
    // size of data readed from child
    size_t size_read;
    // size of data write into child
    size_t size_write;
    // pointer to buffer head
    char *buffer_head;
    // buffer size, in bytes
    size_t buffer_size;
    // initialize buffer
    inline void Init(size_t type_nbytes, size_t count) {
      utils::Assert(type_nbytes < kBufferSize, "too large type_nbytes");
      size_t n = (type_nbytes * count + 7)/ 8;
      buffer_.resize(std::min(kBufferSize, n));
      // make sure align to type_nbytes
      buffer_size = buffer_.size() * sizeof(uint64_t) / type_nbytes * type_nbytes;
      // set buffer head
      buffer_head = reinterpret_cast<char*>(BeginPtr(buffer_));
      // set write head
      size_write = size_read = 0;
    }
    // maximum number of bytes we are able to read
    // currently without corrupt the data
    inline void Read(size_t size_up_out) {
      size_t ngap = size_read - size_up_out;
      utils::Assert(ngap <= buffer_size, "AllReduce: boundary check");
      size_t offset = size_read % buffer_size;      
      size_t nmax = std::min(ngap, buffer_size - offset);
      size_t len = sock.Recv(buffer_head + offset, nmax);
      size_read += len;
    }

   private:
    // recv buffer to get data from child
    // aligned with 64 bits, will be able to perform 64 bits operations freely
    std::vector<uint64_t> buffer_;
  };
  // current rank
  int rank;                  
  // parent socket, can be of size 0 or 1
  std::vector<utils::TCPSocket> parent;
  // sockets of all childs, can be of size 0, 1, 2 or more
  std::vector<ChildRecord> childs;
  // select helper
  utils::SelectHelper selecter;
};

}  // namespace sync
}  // namespace xgboost
