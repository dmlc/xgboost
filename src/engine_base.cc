/*!
 * \file engine_base.cc
 * \brief Basic implementation of AllReduce
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <cstring>
#include "./engine_base.h"

namespace engine {
// constructor
AllReduceBase::AllReduceBase(void) {
  master_uri = "NULL";
  master_port = 9000;
  host_uri = "";
  slave_port = 9010;
  nport_trial = 1000;
  rank = 0;
  world_size = 1;
  this->SetParam("reduce_buffer", "256MB");
}

// initialization function
void AllReduceBase::Init(void) {
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

void AllReduceBase::Shutdown(void) {
  for (size_t i = 0; i < links.size(); ++i) {
    links[i].sock.Close();
  }
  links.clear();
  utils::TCPSocket::Finalize();
}
// set the parameters for AllReduce
void AllReduceBase::SetParam(const char *name, const char *val) {
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
AllReduceBase::ReturnType
AllReduceBase::TryAllReduce(void *sendrecvbuf_,
                            size_t type_nbytes,
                            size_t count,
                            ReduceFunction reducer) {
  if (links.size() == 0 || count == 0) return kSuccess;
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
    bool finished = true;
    utils::SelectHelper selecter;
    for (int i = 0; i < nlink; ++i) {
      if (i == parent_index) {
        if (size_down_in != total_size) {
          selecter.WatchRead(links[i].sock);
          // only watch for exception in live channels
          selecter.WatchException(links[i].sock);
          finished = false;
        }
        if (size_up_out != total_size) {
          selecter.WatchWrite(links[i].sock);
        }
      } else {
        if (links[i].size_read != total_size) {
          selecter.WatchRead(links[i].sock);
        }
        // size_write <= size_read 
        if (links[i].size_write != total_size) {
          selecter.WatchWrite(links[i].sock);
          // only watch for exception in live channels
          selecter.WatchException(links[i].sock);
          finished = false;
        }
      }

    }
    // finish runing allreduce
    if (finished) break;
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
    // can pass message down to childs
    for (int i = 0; i < nlink; ++i) {
      if (i != parent_index && selecter.CheckWrite(links[i].sock)) {
        if (!links[i].WriteFromArray(sendrecvbuf, size_down_in)) return kSockError;
      }
    }
  }
  return kSuccess;  
}
/*!
 * \brief broadcast data from root to all nodes, this function can fail,and will return the cause of failure
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param total_size the size of the data to be broadcasted
 * \param root the root worker id to broadcast the data
 * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
 * \sa ReturnType
 */
AllReduceBase::ReturnType
AllReduceBase::TryBroadcast(void *sendrecvbuf_, size_t total_size, int root) {
  if (links.size() == 0 || total_size == 0) return kSuccess;
  utils::Check(root < world_size, "Broadcast: root should be smaller than world size");
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
    bool finished = true;
    // select helper
    utils::SelectHelper selecter;
    for (int i = 0; i < nlink; ++i) {
      if (in_link == -2) {
        selecter.WatchRead(links[i].sock); finished = false;
      }
      if (i == in_link && links[i].size_read != total_size) {
        selecter.WatchRead(links[i].sock); finished = false;
      }
      if (in_link != -2 && i != in_link && links[i].size_write != total_size) {
        selecter.WatchWrite(links[i].sock); finished = false;
      }
      selecter.WatchException(links[i].sock);        
    }
    // finish running
    if (finished) break;
    // select
    selecter.Select();
    // exception handling
    for (int i = 0; i < nlink; ++i) {
        // recive OOB message from some link 
      if (selecter.CheckExcept(links[i].sock)) return kGetExcept;
    }
    if (in_link == -2) {
      // probe in-link
      for (int i = 0; i < nlink; ++i) {
        if (selecter.CheckRead(links[i].sock)) {
          if (!links[i].ReadToArray(sendrecvbuf_, total_size)) return kSockError;
          size_in = links[i].size_read;
          if (size_in != 0) {
            in_link = i; break;
          }
        }
      }
    } else {
      // read from in link
      if (in_link >= 0 && selecter.CheckRead(links[in_link].sock)) {
        if(!links[in_link].ReadToArray(sendrecvbuf_, total_size)) return kSockError;
        size_in = links[in_link].size_read;
      }
    }
    // send data to all out-link
    for (int i = 0; i < nlink; ++i) {
      if (i != in_link && selecter.CheckWrite(links[i].sock)) {
        if (!links[i].WriteFromArray(sendrecvbuf_, size_in)) return kSockError;
      }
    }
  }
  return kSuccess;
}
} // namespace engine
