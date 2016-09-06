/*!
 *  Copyright (c) 2014 by Contributors
 * \file allreduce_base.cc
 * \brief Basic implementation of AllReduce
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <map>
#include <cstdlib>
#include <cstring>
#include "./allreduce_base.h"

namespace rabit {
namespace engine {
// constructor
AllreduceBase::AllreduceBase(void) {
  tracker_uri = "NULL";
  tracker_port = 9000;
  host_uri = "";
  slave_port = 9010;
  nport_trial = 1000;
  rank = 0;
  world_size = -1;
  connect_retry = 5;
  hadoop_mode = 0;
  version_number = 0;
  // 32 K items
  reduce_ring_mincount = 32 << 10;
  // tracker URL
  task_id = "NULL";
  err_link = NULL;
  dmlc_role = "worker";
  this->SetParam("rabit_reduce_buffer", "256MB");
  // setup possible enviroment variable of intrest
  env_vars.push_back("rabit_task_id");
  env_vars.push_back("rabit_num_trial");
  env_vars.push_back("rabit_reduce_buffer");
  env_vars.push_back("rabit_reduce_ring_mincount");
  env_vars.push_back("rabit_tracker_uri");
  env_vars.push_back("rabit_tracker_port");
  // also include dmlc support direct variables
  env_vars.push_back("DMLC_TASK_ID");
  env_vars.push_back("DMLC_ROLE");
  env_vars.push_back("DMLC_NUM_ATTEMPT");
  env_vars.push_back("DMLC_TRACKER_URI");
  env_vars.push_back("DMLC_TRACKER_PORT");
  env_vars.push_back("DMLC_WORKER_CONNECT_RETRY");
}

// initialization function
void AllreduceBase::Init(int argc, char* argv[]) {
  // setup from enviroment variables
  // handler to get variables from env
  for (size_t i = 0; i < env_vars.size(); ++i) {
    const char *value = getenv(env_vars[i].c_str());
    if (value != NULL) {
      this->SetParam(env_vars[i].c_str(), value);
    }
  }
  // pass in arguments override env variable.
  for (int i = 0; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      this->SetParam(name, val);
    }
  }

  {
    // handling for hadoop
    const char *task_id = getenv("mapred_tip_id");
    if (task_id == NULL) {
      task_id = getenv("mapreduce_task_id");
    }
    if (hadoop_mode != 0) {
      utils::Check(task_id != NULL,
                   "hadoop_mode is set but cannot find mapred_task_id");
    }
    if (task_id != NULL) {
      this->SetParam("rabit_task_id", task_id);
      this->SetParam("rabit_hadoop_mode", "1");
    }
    const char *attempt_id = getenv("mapred_task_id");
    if (attempt_id != 0) {
      const char *att = strrchr(attempt_id, '_');
      int num_trial;
      if (att != NULL && sscanf(att + 1, "%d", &num_trial) == 1) {
        this->SetParam("rabit_num_trial", att + 1);
      }
    }
    // handling for hadoop
    const char *num_task = getenv("mapred_map_tasks");
    if (num_task == NULL) {
      num_task = getenv("mapreduce_job_maps");
    }
    if (hadoop_mode != 0) {
      utils::Check(num_task != NULL,
                   "hadoop_mode is set but cannot find mapred_map_tasks");
    }
    if (num_task != NULL) {
      this->SetParam("rabit_world_size", num_task);
    }
  }
  if (dmlc_role != "worker") {
    fprintf(stderr, "Rabit Module currently only work with dmlc worker"\
            ", quit this program by exit 0\n");
    exit(0);
  }
  // clear the setting before start reconnection
  this->rank = -1;
  //---------------------
  // start socket
  utils::Socket::Startup();
  utils::Assert(all_links.size() == 0, "can only call Init once");
  this->host_uri = utils::SockAddr::GetHostName();
  // get information from tracker
  this->ReConnectLinks();
}

void AllreduceBase::Shutdown(void) {
  for (size_t i = 0; i < all_links.size(); ++i) {
    all_links[i].sock.Close();
  }
  all_links.clear();
  tree_links.plinks.clear();

  if (tracker_uri == "NULL") return;
  // notify tracker rank i have shutdown
  utils::TCPSocket tracker = this->ConnectTracker();
  tracker.SendStr(std::string("shutdown"));
  tracker.Close();
  utils::TCPSocket::Finalize();
}
void AllreduceBase::TrackerPrint(const std::string &msg) {
  if (tracker_uri == "NULL") {
    utils::Printf("%s", msg.c_str()); return;
  }
  utils::TCPSocket tracker = this->ConnectTracker();
  tracker.SendStr(std::string("print"));
  tracker.SendStr(msg);
  tracker.Close();
}
// util to parse data with unit suffix
inline size_t ParseUnit(const char *name, const char *val) {
  char unit;
  unsigned long amt;  // NOLINT(*)
  int n = sscanf(val, "%lu%c", &amt, &unit);
  size_t amount = amt;
  if (n == 2) {
    switch (unit) {
      case 'B': return amount;
      case 'K': return amount << 10UL;
      case 'M': return amount << 20UL;
      case 'G': return amount << 30UL;
      default: utils::Error("invalid format for %s", name); return 0;
    }
  } else if (n == 1) {
    return amount;
  } else {
    utils::Error("invalid format for %s,"                               \
                 "shhould be {integer}{unit}, unit can be {B, KB, MB, GB}", name);
    return 0;
  }
}
/*!
 * \brief set parameters to the engine
 * \param name parameter name
 * \param val parameter value
 */
void AllreduceBase::SetParam(const char *name, const char *val) {
  if (!strcmp(name, "rabit_tracker_uri")) tracker_uri = val;
  if (!strcmp(name, "rabit_tracker_port")) tracker_port = atoi(val);
  if (!strcmp(name, "rabit_task_id")) task_id = val;
  if (!strcmp(name, "DMLC_TRACKER_URI")) tracker_uri = val;
  if (!strcmp(name, "DMLC_TRACKER_PORT")) tracker_port = atoi(val);
  if (!strcmp(name, "DMLC_TASK_ID")) task_id = val;
  if (!strcmp(name, "DMLC_ROLE")) dmlc_role = val;
  if (!strcmp(name, "rabit_world_size")) world_size = atoi(val);
  if (!strcmp(name, "rabit_hadoop_mode")) hadoop_mode = atoi(val);
  if (!strcmp(name, "rabit_reduce_ring_mincount")) {
    reduce_ring_mincount = ParseUnit(name, val);
  }
  if (!strcmp(name, "rabit_reduce_buffer")) {
    reduce_buffer_size = (ParseUnit(name, val) + 7) >> 3;
  }
  if (!strcmp(name, "DMLC_WORKER_CONNECT_RETRY")) {
    connect_retry = atoi(val);
  }
}
/*!
 * \brief initialize connection to the tracker
 * \return a socket that initializes the connection
 */
utils::TCPSocket AllreduceBase::ConnectTracker(void) const {
  int magic = kMagic;
  // get information from tracker
  utils::TCPSocket tracker;
  tracker.Create();

  int retry = 0;
  do {
    if (!tracker.Connect(utils::SockAddr(tracker_uri.c_str(), tracker_port))) {
      if (++retry >= connect_retry) {
        fprintf(stderr, "connect to (failed): [%s]\n", tracker_uri.c_str());
        utils::Socket::Error("Connect");
      } else {
        fprintf(stderr, "retry connect to ip(retry time %d): [%s]\n", retry, tracker_uri.c_str());
        #ifdef _MSC_VER
        Sleep(1);
        #else
        sleep(1);
        #endif
        continue;
      }
    }
    break;
  } while (1);

  using utils::Assert;
  Assert(tracker.SendAll(&magic, sizeof(magic)) == sizeof(magic),
         "ReConnectLink failure 1");
  Assert(tracker.RecvAll(&magic, sizeof(magic)) == sizeof(magic),
         "ReConnectLink failure 2");
  utils::Check(magic == kMagic, "sync::Invalid tracker message, init failure");
  Assert(tracker.SendAll(&rank, sizeof(rank)) == sizeof(rank),
                "ReConnectLink failure 3");
  Assert(tracker.SendAll(&world_size, sizeof(world_size)) == sizeof(world_size),
         "ReConnectLink failure 3");
  tracker.SendStr(task_id);
  return tracker;
}
/*!
 * \brief connect to the tracker to fix the the missing links
 *   this function is also used when the engine start up
 */
void AllreduceBase::ReConnectLinks(const char *cmd) {
  // single node mode
  if (tracker_uri == "NULL") {
    rank = 0; world_size = 1; return;
  }
  utils::TCPSocket tracker = this->ConnectTracker();
  tracker.SendStr(std::string(cmd));

  // the rank of previous link, next link in ring
  int prev_rank, next_rank;
  // the rank of neighbors
  std::map<int, int> tree_neighbors;
  using utils::Assert;
  // get new ranks
  int newrank, num_neighbors;
  Assert(tracker.RecvAll(&newrank, sizeof(newrank)) == sizeof(newrank),
           "ReConnectLink failure 4");
  Assert(tracker.RecvAll(&parent_rank, sizeof(parent_rank)) ==\
         sizeof(parent_rank), "ReConnectLink failure 4");
  Assert(tracker.RecvAll(&world_size, sizeof(world_size)) == sizeof(world_size),
         "ReConnectLink failure 4");
  Assert(rank == -1 || newrank == rank,
         "must keep rank to same if the node already have one");
  rank = newrank;
  Assert(tracker.RecvAll(&num_neighbors, sizeof(num_neighbors)) ==  \
         sizeof(num_neighbors), "ReConnectLink failure 4");
  for (int i = 0; i < num_neighbors; ++i) {
    int nrank;
    Assert(tracker.RecvAll(&nrank, sizeof(nrank)) == sizeof(nrank),
           "ReConnectLink failure 4");
    tree_neighbors[nrank] = 1;
  }
  Assert(tracker.RecvAll(&prev_rank, sizeof(prev_rank)) == sizeof(prev_rank),
         "ReConnectLink failure 4");
  Assert(tracker.RecvAll(&next_rank, sizeof(next_rank)) == sizeof(next_rank),
         "ReConnectLink failure 4");
  // create listening socket
  utils::TCPSocket sock_listen;
  sock_listen.Create();
  int port = sock_listen.TryBindHost(slave_port, slave_port + nport_trial);
  utils::Check(port != -1, "ReConnectLink fail to bind the ports specified");
  sock_listen.Listen();

  // get number of to connect and number of to accept nodes from tracker
  int num_conn, num_accept, num_error = 1;
  do {
    // send over good links
    std::vector<int> good_link;
    for (size_t i = 0; i < all_links.size(); ++i) {
      if (!all_links[i].sock.BadSocket()) {
        good_link.push_back(static_cast<int>(all_links[i].rank));
      } else {
        if (!all_links[i].sock.IsClosed()) all_links[i].sock.Close();
      }
    }
    int ngood = static_cast<int>(good_link.size());
    Assert(tracker.SendAll(&ngood, sizeof(ngood)) == sizeof(ngood),
           "ReConnectLink failure 5");
    for (size_t i = 0; i < good_link.size(); ++i) {
      Assert(tracker.SendAll(&good_link[i], sizeof(good_link[i])) == \
             sizeof(good_link[i]), "ReConnectLink failure 6");
    }
    Assert(tracker.RecvAll(&num_conn, sizeof(num_conn)) == sizeof(num_conn),
           "ReConnectLink failure 7");
    Assert(tracker.RecvAll(&num_accept, sizeof(num_accept)) ==  \
           sizeof(num_accept), "ReConnectLink failure 8");
    num_error = 0;
    for (int i = 0; i < num_conn; ++i) {
      LinkRecord r;
      int hport, hrank;
      std::string hname;
      tracker.RecvStr(&hname);
      Assert(tracker.RecvAll(&hport, sizeof(hport)) == sizeof(hport),
             "ReConnectLink failure 9");
      Assert(tracker.RecvAll(&hrank, sizeof(hrank)) == sizeof(hrank),
             "ReConnectLink failure 10");
      r.sock.Create();
      if (!r.sock.Connect(utils::SockAddr(hname.c_str(), hport))) {
        num_error += 1; r.sock.Close(); continue;
      }
      Assert(r.sock.SendAll(&rank, sizeof(rank)) == sizeof(rank),
             "ReConnectLink failure 12");
      Assert(r.sock.RecvAll(&r.rank, sizeof(r.rank)) == sizeof(r.rank),
             "ReConnectLink failure 13");
      utils::Check(hrank == r.rank,
                   "ReConnectLink failure, link rank inconsistent");
      bool match = false;
      for (size_t i = 0; i < all_links.size(); ++i) {
        if (all_links[i].rank == hrank) {
          Assert(all_links[i].sock.IsClosed(),
                 "Override a link that is active");
          all_links[i].sock = r.sock; match = true; break;
        }
      }
      if (!match) all_links.push_back(r);
    }
    Assert(tracker.SendAll(&num_error, sizeof(num_error)) == sizeof(num_error),
           "ReConnectLink failure 14");
  } while (num_error != 0);
  // send back socket listening port to tracker
  Assert(tracker.SendAll(&port, sizeof(port)) == sizeof(port),
         "ReConnectLink failure 14");
  // close connection to tracker
  tracker.Close();
  // listen to incoming links
  for (int i = 0; i < num_accept; ++i) {
    LinkRecord r;
    r.sock = sock_listen.Accept();
    Assert(r.sock.SendAll(&rank, sizeof(rank)) == sizeof(rank),
           "ReConnectLink failure 15");
    Assert(r.sock.RecvAll(&r.rank, sizeof(r.rank)) == sizeof(r.rank),
           "ReConnectLink failure 15");
    bool match = false;
    for (size_t i = 0; i < all_links.size(); ++i) {
      if (all_links[i].rank == r.rank) {
        utils::Assert(all_links[i].sock.IsClosed(),
                      "Override a link that is active");
        all_links[i].sock = r.sock; match = true; break;
      }
    }
    if (!match) all_links.push_back(r);
  }
  // close listening sockets
  sock_listen.Close();
  this->parent_index = -1;
  // setup tree links and ring structure
  tree_links.plinks.clear();
  for (size_t i = 0; i < all_links.size(); ++i) {
    utils::Assert(!all_links[i].sock.BadSocket(), "ReConnectLink: bad socket");
    // set the socket to non-blocking mode, enable TCP keepalive
    all_links[i].sock.SetNonBlock(true);
    all_links[i].sock.SetKeepAlive(true);
    if (tree_neighbors.count(all_links[i].rank) != 0) {
      if (all_links[i].rank == parent_rank) {
        parent_index = static_cast<int>(tree_links.plinks.size());
      }
      tree_links.plinks.push_back(&all_links[i]);
    }
    if (all_links[i].rank == prev_rank) ring_prev = &all_links[i];
    if (all_links[i].rank == next_rank) ring_next = &all_links[i];
  }
  Assert(parent_rank == -1 || parent_index != -1,
         "cannot find parent in the link");
  Assert(prev_rank == -1 || ring_prev != NULL,
         "cannot find prev ring in the link");
  Assert(next_rank == -1 || ring_next != NULL,
         "cannot find next ring in the link");
}
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
AllreduceBase::ReturnType
AllreduceBase::TryAllreduce(void *sendrecvbuf_,
                            size_t type_nbytes,
                            size_t count,
                            ReduceFunction reducer) {
  if (count > reduce_ring_mincount) {
    return this->TryAllreduceRing(sendrecvbuf_, type_nbytes, count, reducer);
  } else {
    return this->TryAllreduceTree(sendrecvbuf_, type_nbytes, count, reducer);
  }
}
/*!
 * \brief perform in-place allreduce, on sendrecvbuf,
 * this function implements tree-shape reduction
 *
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param type_nbytes the unit number of bytes the type have
 * \param count number of elements to be reduced
 * \param reducer reduce function
 * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
 * \sa ReturnType
 */
AllreduceBase::ReturnType
AllreduceBase::TryAllreduceTree(void *sendrecvbuf_,
                                size_t type_nbytes,
                                size_t count,
                                ReduceFunction reducer) {
  RefLinkVector &links = tree_links;
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
        if (size_up_out != total_size && size_up_out < size_up_reduce) {
          selecter.WatchWrite(links[i].sock);
        }
      } else {
        if (links[i].size_read != total_size) {
          selecter.WatchRead(links[i].sock);
        }
        // size_write <= size_read
        if (links[i].size_write != total_size) {
          if (links[i].size_write < size_down_in) {
            selecter.WatchWrite(links[i].sock);
          }
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
      if (selecter.CheckExcept(links[i].sock)) {
        return ReportError(&links[i], kGetExcept);
      }
    }
    // read data from childs
    for (int i = 0; i < nlink; ++i) {
      if (i != parent_index && selecter.CheckRead(links[i].sock)) {
        ReturnType ret = links[i].ReadToRingBuffer(size_up_out, total_size);
        if (ret != kSuccess) {
          return ReportError(&links[i], ret);
        }
      }
    }
    // this node have childs, peform reduce
    if (nlink > static_cast<int>(parent_index != -1)) {
      size_t buffer_size = 0;
      // do upstream reduce
      size_t max_reduce = total_size;
      for (int i = 0; i < nlink; ++i) {
        if (i != parent_index) {
          max_reduce = std::min(max_reduce, links[i].size_read);
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
        size_t nread = std::min(buffer_size - start,
                                max_reduce - size_up_reduce);
        utils::Assert(nread % type_nbytes == 0, "Allreduce: size check");
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
      if (size_up_out < size_up_reduce) {
        ssize_t len = links[parent_index].sock.
            Send(sendrecvbuf + size_up_out, size_up_reduce - size_up_out);
        if (len != -1) {
          size_up_out += static_cast<size_t>(len);
        } else {
          ReturnType ret = Errno2Return();
          if (ret != kSuccess) {
            return ReportError(&links[parent_index], ret);
          }
        }
      }
      // read data from parent
      if (selecter.CheckRead(links[parent_index].sock) &&
          total_size > size_down_in) {
        ssize_t len = links[parent_index].sock.
            Recv(sendrecvbuf + size_down_in, total_size - size_down_in);
        if (len == 0) {
          links[parent_index].sock.Close();
          return ReportError(&links[parent_index], kRecvZeroLen);
        }
        if (len != -1) {
          size_down_in += static_cast<size_t>(len);
          utils::Assert(size_down_in <= size_up_out,
                        "Allreduce: boundary error");
        } else {
          ReturnType ret = Errno2Return();
          if (ret != kSuccess) {
            return ReportError(&links[parent_index], ret);
          }
        }
      }
    } else {
      // this is root, can use reduce as most recent point
      size_down_in = size_up_out = size_up_reduce;
    }
    // can pass message down to childs
    for (int i = 0; i < nlink; ++i) {
      if (i != parent_index && links[i].size_write < size_down_in) {
        ReturnType ret = links[i].WriteFromArray(sendrecvbuf, size_down_in);
        if (ret != kSuccess) {
          return ReportError(&links[i], ret);
        }
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
AllreduceBase::ReturnType
AllreduceBase::TryBroadcast(void *sendrecvbuf_, size_t total_size, int root) {
  RefLinkVector &links = tree_links;
  if (links.size() == 0 || total_size == 0) return kSuccess;
  utils::Check(root < world_size,
               "Broadcast: root should be smaller than world size");
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
  while (true) {
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
        if (links[i].size_write < size_in) {
          selecter.WatchWrite(links[i].sock);
        }
        finished = false;
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
      if (selecter.CheckExcept(links[i].sock)) {
        return ReportError(&links[i], kGetExcept);
      }
    }
    if (in_link == -2) {
      // probe in-link
      for (int i = 0; i < nlink; ++i) {
        if (selecter.CheckRead(links[i].sock)) {
          ReturnType ret = links[i].ReadToArray(sendrecvbuf_, total_size);
          if (ret != kSuccess) {
            return ReportError(&links[i], ret);
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
        ReturnType ret = links[in_link].ReadToArray(sendrecvbuf_, total_size);
        if (ret != kSuccess) {
          return ReportError(&links[in_link], ret);
        }
        size_in = links[in_link].size_read;
      }
    }
    // send data to all out-link
    for (int i = 0; i < nlink; ++i) {
      if (i != in_link && links[i].size_write < size_in) {
        ReturnType ret = links[i].WriteFromArray(sendrecvbuf_, size_in);
        if (ret != kSuccess) {
          return ReportError(&links[i], ret);
        }
      }
    }
  }
  return kSuccess;
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
AllreduceBase::ReturnType
AllreduceBase::TryAllgatherRing(void *sendrecvbuf_, size_t total_size,
                                size_t slice_begin,
                                size_t slice_end,
                                size_t size_prev_slice) {
  // read from next link and send to prev one
  LinkRecord &prev = *ring_prev, &next = *ring_next;
  // need to reply on special rank structure
  utils::Assert(next.rank == (rank + 1) % world_size &&
                rank == (prev.rank + 1) % world_size,
                "need to assume rank structure");
  // send recv buffer
  char *sendrecvbuf = reinterpret_cast<char*>(sendrecvbuf_);
  const size_t stop_read = total_size + slice_begin;
  const size_t stop_write = total_size + slice_begin - size_prev_slice;
  size_t write_ptr = slice_begin;
  size_t read_ptr = slice_end;

  while (true) {
    // select helper
    bool finished = true;
    utils::SelectHelper selecter;
    if (read_ptr != stop_read) {
      selecter.WatchRead(next.sock);
      finished = false;
    }
    if (write_ptr != stop_write) {
      if (write_ptr < read_ptr) {
        selecter.WatchWrite(prev.sock);
      }
      finished  = false;
    }
    if (finished) break;
    selecter.Select();
    if (read_ptr != stop_read && selecter.CheckRead(next.sock)) {
      size_t size = stop_read - read_ptr;
      size_t start = read_ptr % total_size;
      if (start + size > total_size) {
        size = total_size - start;
      }
      ssize_t len = next.sock.Recv(sendrecvbuf + start, size);
      if (len != -1) {
        read_ptr += static_cast<size_t>(len);
      } else {
        ReturnType ret = Errno2Return();
        if (ret != kSuccess) return ReportError(&next, ret);
      }
    }
    if (write_ptr < read_ptr && write_ptr != stop_write) {
      size_t size = std::min(read_ptr, stop_write) - write_ptr;
      size_t start = write_ptr % total_size;
      if (start + size > total_size) {
        size = total_size - start;
      }
      ssize_t len = prev.sock.Send(sendrecvbuf + start, size);
      if (len != -1) {
        write_ptr += static_cast<size_t>(len);
      } else {
        ReturnType ret = Errno2Return();
        if (ret != kSuccess) return ReportError(&prev, ret);
      }
    }
  }
  return kSuccess;
}
/*!
 * \brief perform in-place allreduce, on sendrecvbuf, this function can fail,
 *  and will return the cause of failure
 *
 *  Ring-based algorithm
 *
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param type_nbytes the unit number of bytes the type have
 * \param count number of elements to be reduced
 * \param reducer reduce function
 * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
 * \sa ReturnType, TryAllreduce
 */
AllreduceBase::ReturnType
AllreduceBase::TryReduceScatterRing(void *sendrecvbuf_,
                                    size_t type_nbytes,
                                    size_t count,
                                    ReduceFunction reducer) {
  // read from next link and send to prev one
  LinkRecord &prev = *ring_prev, &next = *ring_next;
  // need to reply on special rank structure
  utils::Assert(next.rank == (rank + 1) % world_size &&
                rank == (prev.rank + 1) % world_size,
                "need to assume rank structure");
  // total size of message
  const size_t total_size = type_nbytes * count;
  size_t n = static_cast<size_t>(world_size);
  size_t step = (count + n - 1) / n;
  size_t r = static_cast<size_t>(next.rank);
  size_t write_ptr = std::min(r * step, count) * type_nbytes;
  size_t read_ptr = std::min((r + 1) * step, count) * type_nbytes;
  size_t reduce_ptr = read_ptr;
  // send recv buffer
  char *sendrecvbuf = reinterpret_cast<char*>(sendrecvbuf_);
  // position to stop reading
  const size_t stop_read = total_size + write_ptr;
  // position to stop writing
  size_t stop_write = total_size + std::min(rank * step, count) * type_nbytes;
  if (stop_write > stop_read) {
    stop_write -= total_size;
    utils::Assert(write_ptr <= stop_write, "write ptr boundary check");
  }
  // use ring buffer in next position
  next.InitBuffer(type_nbytes, step, reduce_buffer_size);
  // set size_read to read pointer for ring buffer to work properly
  next.size_read = read_ptr;

  while (true) {
    // select helper
    bool finished = true;
    utils::SelectHelper selecter;
    if (read_ptr != stop_read) {
      selecter.WatchRead(next.sock);
      finished = false;
    }
    if (write_ptr != stop_write) {
      if (write_ptr < reduce_ptr) {
        selecter.WatchWrite(prev.sock);
      }
      finished = false;
    }
    if (finished) break;
    selecter.Select();
    if (read_ptr != stop_read && selecter.CheckRead(next.sock)) {
      ReturnType ret = next.ReadToRingBuffer(reduce_ptr, stop_read);
      if (ret != kSuccess) {
        return ReportError(&next, ret);
      }
      // sync the rate
      read_ptr = next.size_read;
      utils::Assert(read_ptr <= stop_read, "[%d] read_ptr boundary check", rank);
      const size_t buffer_size = next.buffer_size;
      size_t max_reduce = (read_ptr  / type_nbytes) * type_nbytes;
      while (reduce_ptr < max_reduce) {
        size_t bstart = reduce_ptr % buffer_size;
        size_t nread = std::min(buffer_size - bstart,
                                max_reduce - reduce_ptr);
        size_t rstart = reduce_ptr % total_size;
        nread = std::min(nread, total_size - rstart);
        reducer(next.buffer_head + bstart,
                sendrecvbuf + rstart,
                static_cast<int>(nread / type_nbytes),
                MPI::Datatype(type_nbytes));
        reduce_ptr += nread;
      }
    }
    if (write_ptr < reduce_ptr && write_ptr != stop_write) {
      size_t size = std::min(reduce_ptr, stop_write) - write_ptr;
      size_t start = write_ptr % total_size;
      if (start + size > total_size) {
        size = total_size - start;
      }
      ssize_t len = prev.sock.Send(sendrecvbuf + start, size);
      if (len != -1) {
        write_ptr += static_cast<size_t>(len);
      } else {
        ReturnType ret = Errno2Return();
        if (ret != kSuccess) return ReportError(&prev, ret);
      }
    }
  }
  return kSuccess;
}
/*!
 * \brief perform in-place allreduce, on sendrecvbuf
 *  use a ring based algorithm
 *
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param type_nbytes the unit number of bytes the type have
 * \param count number of elements to be reduced
 * \param reducer reduce function
 * \return this function can return kSuccess, kSockError, kGetExcept, see ReturnType for details
 * \sa ReturnType
 */
AllreduceBase::ReturnType
AllreduceBase::TryAllreduceRing(void *sendrecvbuf_,
                                size_t type_nbytes,
                                size_t count,
                                ReduceFunction reducer) {
  ReturnType ret = TryReduceScatterRing(sendrecvbuf_, type_nbytes, count, reducer);
  if (ret != kSuccess) return ret;
  size_t n = static_cast<size_t>(world_size);
  size_t step = (count + n - 1) / n;
  size_t begin = std::min(rank * step, count) * type_nbytes;
  size_t end = std::min((rank + 1) * step, count) * type_nbytes;
  // previous rank
  int prank = ring_prev->rank;
  // get rank of previous
  return TryAllgatherRing
      (sendrecvbuf_, type_nbytes * count,
       begin, end,
       (std::min((prank + 1) * step, count) -
        std::min(prank * step, count)) * type_nbytes);
}
}  // namespace engine
}  // namespace rabit
