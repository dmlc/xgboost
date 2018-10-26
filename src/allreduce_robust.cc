/*!
 *  Copyright (c) 2014 by Contributors
 * \file allreduce_robust.cc
 * \brief Robust implementation of Allreduce
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <limits>
#include <utility>
#include "../include/rabit/internal/io.h"
#include "../include/rabit/internal/utils.h"
#include "../include/rabit/internal/engine.h"
#include "../include/rabit/internal/rabit-inl.h"
#include "./allreduce_robust.h"

namespace rabit {
namespace engine {
AllreduceRobust::AllreduceRobust(void) {
  num_local_replica = 0;
  num_global_replica = 5;
  default_local_replica = 2;
  seq_counter = 0;
  local_chkpt_version = 0;
  result_buffer_round = 1;
  global_lazycheck = NULL;
  use_local_model = -1;
  recover_counter = 0;
  env_vars.push_back("rabit_global_replica");
  env_vars.push_back("rabit_local_replica");
}
void AllreduceRobust::Init(int argc, char* argv[]) {
  AllreduceBase::Init(argc, argv);
  if (num_global_replica == 0) {
    result_buffer_round = -1;
  } else {
    result_buffer_round = std::max(world_size / num_global_replica, 1);
  }
}
/*! \brief shutdown the engine */
void AllreduceRobust::Shutdown(void) {
  // need to sync the exec before we shutdown, do a pesudo check point
  // execute checkpoint, note: when checkpoint existing, load will not happen
  utils::Assert(RecoverExec(NULL, 0, ActionSummary::kCheckPoint, ActionSummary::kSpecialOp),
                "Shutdown: check point must return true");
  // reset result buffer
  resbuf.Clear(); seq_counter = 0;
  // execute check ack step, load happens here
  utils::Assert(RecoverExec(NULL, 0, ActionSummary::kCheckAck, ActionSummary::kSpecialOp),
                "Shutdown: check ack must return true");
#ifdef __APPLE__
  // In OSX, one worker shutdowns and closes sockets while rest still run kCheckAck
  // This cause rest workers checkandrecover and hang inf, https://github.com/dmlc/xgboost/pull/3818
  // TODO: a fundamental fix for this
  sleep(2);
#endif
  AllreduceBase::Shutdown();
}
/*!
 * \brief set parameters to the engine
 * \param name parameter name
 * \param val parameter value
 */
void AllreduceRobust::SetParam(const char *name, const char *val) {
  AllreduceBase::SetParam(name, val);
  if (!strcmp(name, "rabit_global_replica")) num_global_replica = atoi(val);
  if (!strcmp(name, "rabit_local_replica")) {
    num_local_replica = atoi(val);
  }
}
/*!
 * \brief perform in-place allreduce, on sendrecvbuf
 *        this function is NOT thread-safe
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param type_nbytes the unit number of bytes the type have
 * \param count number of elements to be reduced
 * \param reducer reduce function
 * \param prepare_func Lazy preprocessing function, lazy prepare_fun(prepare_arg)
 *                     will be called by the function before performing Allreduce, to intialize the data in sendrecvbuf_.
 *                     If the result of Allreduce can be recovered directly, then prepare_func will NOT be called
 * \param prepare_arg argument used to passed into the lazy preprocessing function
 */
void AllreduceRobust::Allreduce(void *sendrecvbuf_,
                                size_t type_nbytes,
                                size_t count,
                                ReduceFunction reducer,
                                PreprocFunction prepare_fun,
                                void *prepare_arg) {
  // skip action in single node
  if (world_size == 1 || world_size == -1) {
    if (prepare_fun != NULL) prepare_fun(prepare_arg);
    return;
  }
  bool recovered = RecoverExec(sendrecvbuf_, type_nbytes * count, 0, seq_counter);
  // now we are free to remove the last result, if any
  if (resbuf.LastSeqNo() != -1 &&
      (result_buffer_round == -1 ||
       resbuf.LastSeqNo() % result_buffer_round != rank % result_buffer_round)) {
    resbuf.DropLast();
  }
  if (!recovered && prepare_fun != NULL) prepare_fun(prepare_arg);
  void *temp = resbuf.AllocTemp(type_nbytes, count);
  while (true) {
    if (recovered) {
      std::memcpy(temp, sendrecvbuf_, type_nbytes * count); break;
    } else {
      std::memcpy(temp, sendrecvbuf_, type_nbytes * count);
      if (CheckAndRecover(TryAllreduce(temp, type_nbytes, count, reducer))) {
        std::memcpy(sendrecvbuf_, temp, type_nbytes * count); break;
      } else {
        recovered = RecoverExec(sendrecvbuf_, type_nbytes * count, 0, seq_counter);
      }
    }
  }
  resbuf.PushTemp(seq_counter, type_nbytes, count);
  seq_counter += 1;
}
/*!
 * \brief broadcast data from root to all nodes
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param size the size of the data to be broadcasted
 * \param root the root worker id to broadcast the data
 */
void AllreduceRobust::Broadcast(void *sendrecvbuf_, size_t total_size, int root) {
  // skip action in single node
  if (world_size == 1 || world_size == -1) return;
  bool recovered = RecoverExec(sendrecvbuf_, total_size, 0, seq_counter);
  // now we are free to remove the last result, if any
  if (resbuf.LastSeqNo() != -1 &&
      (result_buffer_round == -1 ||
       resbuf.LastSeqNo() % result_buffer_round != rank % result_buffer_round)) {
    resbuf.DropLast();
  }
  void *temp = resbuf.AllocTemp(1, total_size);
  while (true) {
    if (recovered) {
      std::memcpy(temp, sendrecvbuf_, total_size); break;
    } else {
      if (CheckAndRecover(TryBroadcast(sendrecvbuf_, total_size, root))) {
        std::memcpy(temp, sendrecvbuf_, total_size); break;
      } else {
        recovered = RecoverExec(sendrecvbuf_, total_size, 0, seq_counter);
      }
    }
  }
  resbuf.PushTemp(seq_counter, 1, total_size);
  seq_counter += 1;
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
int AllreduceRobust::LoadCheckPoint(Serializable *global_model,
                                    Serializable *local_model) {
  // skip action in single node
  if (world_size == 1) return 0;
  this->LocalModelCheck(local_model != NULL);
  if (num_local_replica == 0) {
    utils::Check(local_model == NULL,
                 "need to set rabit_local_replica larger than 1 to checkpoint local_model");
  }
  // check if we succesful
  if (RecoverExec(NULL, 0, ActionSummary::kLoadCheck, ActionSummary::kSpecialOp)) {
    int nlocal = std::max(static_cast<int>(local_rptr[local_chkpt_version].size()) - 1, 0);
    if (local_model != NULL) {
      if (nlocal == num_local_replica + 1) {
        // load in local model
        utils::MemoryFixSizeBuffer fs(BeginPtr(local_chkpt[local_chkpt_version]),
                                      local_rptr[local_chkpt_version][1]);
        local_model->Load(&fs);
      } else {
        utils::Assert(nlocal == 0, "[%d] local model inconsistent, nlocal=%d", rank, nlocal);
      }
    }
    // reset result buffer
    resbuf.Clear(); seq_counter = 0;
    // load from buffer
    utils::MemoryBufferStream fs(&global_checkpoint);
    if (global_checkpoint.length() == 0) {
      version_number = 0;
    } else {
      utils::Assert(fs.Read(&version_number, sizeof(version_number)) != 0,
                    "read in version number");
      global_model->Load(&fs);
      utils::Assert(local_model == NULL || nlocal == num_local_replica + 1,
                    "local model inconsistent, nlocal=%d", nlocal);
    }
    // run another phase of check ack, if recovered from data
    utils::Assert(RecoverExec(NULL, 0, ActionSummary::kCheckAck, ActionSummary::kSpecialOp),
                  "check ack must return true");
    return version_number;
  } else {
    // reset result buffer
    resbuf.Clear(); seq_counter = 0; version_number = 0;
    // nothing loaded, a fresh start, everyone init model
    return version_number;
  }
}
/*!
 * \brief internal consistency check function,
 *  use check to ensure user always call CheckPoint/LoadCheckPoint
 *  with or without local but not both, this function will set the approperiate settings
 *  in the first call of LoadCheckPoint/CheckPoint
 *
 * \param with_local whether the user calls CheckPoint with local model
 */
void AllreduceRobust::LocalModelCheck(bool with_local) {
  if (use_local_model == -1) {
    if (with_local) {
      use_local_model = 1;
      if (num_local_replica == 0) {
        num_local_replica = default_local_replica;
      }
    } else {
      use_local_model = 0;
      num_local_replica = 0;
    }
  } else {
    utils::Check(use_local_model == static_cast<int>(with_local),
                 "Can only call Checkpoint/LoadCheckPoint always with"\
                 "or without local_model, but not mixed case");
  }
}
/*!
 * \brief internal implementation of checkpoint, support both lazy and normal way
 *
 * \param global_model pointer to the globally shared model/state
 *   when calling this function, the caller need to gauranttees that global_model
 *   is the same in all nodes
 * \param local_model pointer to local model, that is specific to current node/rank
 *   this can be NULL when no local state is needed
 * \param lazy_checkpt whether the action is lazy checkpoint
 *
 * \sa CheckPoint, LazyCheckPoint
 */
void AllreduceRobust::CheckPoint_(const Serializable *global_model,
                                  const Serializable *local_model,
                                  bool lazy_checkpt) {
  // never do check point in single machine mode
  if (world_size == 1) {
    version_number += 1; return;
  }
  this->LocalModelCheck(local_model != NULL);
  if (num_local_replica == 0) {
    utils::Check(local_model == NULL,
                 "need to set rabit_local_replica larger than 1 to checkpoint local_model");
  }
  if (num_local_replica != 0) {
    while (true) {
      if (RecoverExec(NULL, 0, 0, ActionSummary::kLocalCheckPoint)) break;
      // save model model to new version place
      int new_version = !local_chkpt_version;
      local_chkpt[new_version].clear();
      utils::MemoryBufferStream fs(&local_chkpt[new_version]);
      if (local_model != NULL) {
        local_model->Save(&fs);
      }
      local_rptr[new_version].clear();
      local_rptr[new_version].push_back(0);
      local_rptr[new_version].push_back(local_chkpt[new_version].length());
      if (CheckAndRecover(TryCheckinLocalState(&local_rptr[new_version],
                                               &local_chkpt[new_version]))) break;
    }
    // run the ack phase, can be true or false
    RecoverExec(NULL, 0, 0, ActionSummary::kLocalCheckAck);
    // switch pointer to new version
    local_chkpt_version = !local_chkpt_version;
  }
  // execute checkpoint, note: when checkpoint existing, load will not happen
  utils::Assert(RecoverExec(NULL, 0, ActionSummary::kCheckPoint, ActionSummary::kSpecialOp),
                "check point must return true");
  // this is the critical region where we will change all the stored models
  // increase version number
  version_number += 1;
  // save model
  if (lazy_checkpt) {
    global_lazycheck = global_model;
  } else {
    global_checkpoint.resize(0);
    utils::MemoryBufferStream fs(&global_checkpoint);
    fs.Write(&version_number, sizeof(version_number));
    global_model->Save(&fs);
    global_lazycheck = NULL;
  }
  // reset result buffer
  resbuf.Clear(); seq_counter = 0;
  // execute check ack step, load happens here
  utils::Assert(RecoverExec(NULL, 0, ActionSummary::kCheckAck, ActionSummary::kSpecialOp),
                "check ack must return true");
}
/*!
 * \brief reset the all the existing links by sending Out-of-Band message marker
 *  after this function finishes, all the messages received and sent before in all live links are discarded,
 *  This allows us to get a fresh start after error has happened
 *
 * \return this function can return kSuccess or kSockError
 *         when kSockError is returned, it simply means there are bad sockets in the links,
 *         and some link recovery proceduer is needed
 */
AllreduceRobust::ReturnType AllreduceRobust::TryResetLinks(void) {
  // number of links
  const int nlink = static_cast<int>(all_links.size());
  for (int i = 0; i < nlink; ++i) {
    all_links[i].InitBuffer(sizeof(int), 1 << 10, reduce_buffer_size);
    all_links[i].ResetSize();
  }
  // read and discard data from all channels until pass mark
  while (true) {
    for (int i = 0; i < nlink; ++i) {
      if (all_links[i].sock.BadSocket()) continue;
      if (all_links[i].size_write == 0) {
        char sig = kOOBReset;
        ssize_t len = all_links[i].sock.Send(&sig, sizeof(sig), MSG_OOB);
        // error will be filtered in next loop
        if (len == sizeof(sig)) all_links[i].size_write = 1;
      }
      if (all_links[i].size_write == 1) {
        char sig = kResetMark;
        ssize_t len = all_links[i].sock.Send(&sig, sizeof(sig));
        if (len == sizeof(sig)) all_links[i].size_write = 2;
      }
    }
    utils::PollHelper rsel;
    bool finished = true;
    for (int i = 0; i < nlink; ++i) {
      if (all_links[i].size_write != 2 && !all_links[i].sock.BadSocket()) {
        rsel.WatchWrite(all_links[i].sock); finished = false;
      }
    }
    if (finished) break;
    // wait to read from the channels to discard data
    rsel.Poll();
  }
  for (int i = 0; i < nlink; ++i) {
    if (!all_links[i].sock.BadSocket()) {
      utils::PollHelper::WaitExcept(all_links[i].sock);
    }
  }
  while (true) {
    utils::PollHelper rsel;
    bool finished = true;
    for (int i = 0; i < nlink; ++i) {
      if (all_links[i].size_read == 0 && !all_links[i].sock.BadSocket()) {
        rsel.WatchRead(all_links[i].sock); finished = false;
      }
    }
    if (finished) break;
    rsel.Poll();
    for (int i = 0; i < nlink; ++i) {
      if (all_links[i].sock.BadSocket()) continue;
      if (all_links[i].size_read == 0) {
        int atmark = all_links[i].sock.AtMark();
        if (atmark < 0) {
          utils::Assert(all_links[i].sock.BadSocket(), "must already gone bad");
        } else if (atmark > 0) {
          all_links[i].size_read = 1;
        } else {
          // no at mark, read and discard data
          ssize_t len = all_links[i].sock.Recv(all_links[i].buffer_head, all_links[i].buffer_size);
          if (all_links[i].sock.AtMark()) all_links[i].size_read = 1;
          // zero length, remote closed the connection, close socket
          if (len == 0) all_links[i].sock.Close();
        }
      }
    }
  }
  // start synchronization, use blocking I/O to avoid select
  for (int i = 0; i < nlink; ++i) {
    if (!all_links[i].sock.BadSocket()) {
      char oob_mark;
      all_links[i].sock.SetNonBlock(false);
      ssize_t len = all_links[i].sock.Recv(&oob_mark, sizeof(oob_mark), MSG_WAITALL);
      if (len == 0) {
        all_links[i].sock.Close(); continue;
      } else if (len > 0) {
        utils::Assert(oob_mark == kResetMark, "wrong oob msg");
        utils::Assert(all_links[i].sock.AtMark() != 1, "should already read past mark");
      } else {
        utils::Assert(errno != EAGAIN|| errno != EWOULDBLOCK, "BUG");
      }
      // send out ack
      char ack = kResetAck;
      while (true) {
        len = all_links[i].sock.Send(&ack, sizeof(ack));
        if (len == sizeof(ack)) break;
        if (len == -1) {
          if (errno != EAGAIN && errno != EWOULDBLOCK) break;
        }
      }
    }
  }
  // wait all ack
  for (int i = 0; i < nlink; ++i) {
    if (!all_links[i].sock.BadSocket()) {
      char ack;
      ssize_t len = all_links[i].sock.Recv(&ack, sizeof(ack), MSG_WAITALL);
      if (len == 0) {
        all_links[i].sock.Close(); continue;
      } else if (len > 0) {
        utils::Assert(ack == kResetAck, "wrong Ack MSG");
      } else {
        utils::Assert(errno != EAGAIN|| errno != EWOULDBLOCK, "BUG");
      }
      // set back to nonblock mode
      all_links[i].sock.SetNonBlock(true);
    }
  }
  for (int i = 0; i < nlink; ++i) {
    if (all_links[i].sock.BadSocket()) return kSockError;
  }
  return kSuccess;
}
/*!
 * \brief if err_type indicates an error
 *         recover links according to the error type reported
 *        if there is no error, return true
 * \param err_type the type of error happening in the system
 * \return true if err_type is kSuccess, false otherwise
 */
bool AllreduceRobust::CheckAndRecover(ReturnType err_type) {
  if (err_type == kSuccess) return true;
  utils::Assert(err_link != NULL, "must know the error source");
  recover_counter += 1;
  {
    // simple way, shutdown all links
    for (size_t i = 0; i < all_links.size(); ++i) {
      if (!all_links[i].sock.BadSocket()) all_links[i].sock.Close();
    }
    ReConnectLinks("recover");
    return false;
  }
  // this was old way
  // TryResetLinks still causes possible errors, so not use this one
  while (err_type != kSuccess) {
    switch (err_type.value) {
      case kGetExcept: err_type = TryResetLinks(); break;
      case kSockError: {
        TryResetLinks();
        ReConnectLinks();
        err_type = kSuccess;
        break;
      }
      default: utils::Assert(false, "RecoverLinks: cannot reach here");
    }
  }
  return false;
}
/*!
 * \brief message passing function, used to decide the
 *        shortest distance to the possible source of data
 * \param node_value a pair of have_data and size
 *           have_data whether current node have data
 *           size gives the size of data, if current node is kHaveData
 * \param dist_in the shorest to any data source distance in each direction
 * \param out_index the edge index of output link
 * \return the shorest distance result of out edge specified by out_index
 */
inline std::pair<int, size_t>
ShortestDist(const std::pair<bool, size_t> &node_value,
             const std::vector< std::pair<int, size_t> > &dist_in,
             size_t out_index) {
  if (node_value.first) {
    return std::make_pair(1, node_value.second);
  }
  size_t size = 0;
  int res = std::numeric_limits<int>::max();
  for (size_t i = 0; i < dist_in.size(); ++i) {
    if (i == out_index) continue;
    if (dist_in[i].first == std::numeric_limits<int>::max()) continue;
    if (dist_in[i].first + 1 < res) {
      res = dist_in[i].first + 1;
      size = dist_in[i].second;
    }
  }
  // add one hop

  return std::make_pair(res, size);
}
/*!
 * \brief message passing function, used to decide the
 *    data request from each edge, whether need to request data from certain edge
 * \param node_value a pair of request_data and best_link
 *           request_data stores whether current node need to request data
 *           best_link gives the best edge index to fetch the data
 * \param req_in the data request from incoming edges
 * \param out_index the edge index of output link
 * \return the request to the output edge
 */
inline char DataRequest(const std::pair<bool, int> &node_value,
                        const std::vector<char> &req_in,
                        size_t out_index) {
  // whether current node need to request data
  bool request_data = node_value.first;
  // which edge index is the best link to request data
  // can be -1, which means current node contains data
  const int best_link = node_value.second;
  if (static_cast<int>(out_index) == best_link) {
    if (request_data) return 1;
    for (size_t i = 0; i < req_in.size(); ++i) {
      if (i == out_index) continue;
      if (req_in[i] != 0) return 1;
    }
  }
  return 0;
}
/*!
 * \brief try to decide the recovery message passing request
 * \param role the current role of the node
 * \param p_size used to store the size of the message, for node in state kHaveData,
 *               this size must be set correctly before calling the function
 *               for others, this surves as output parameter
 *
 * \param p_recvlink used to store the link current node should recv data from, if necessary
 *          this can be -1, which means current node have the data
 * \param p_req_in used to store the resulting vector, indicating which link we should send the data to
 *
 * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
 * \sa ReturnType
 */
AllreduceRobust::ReturnType
AllreduceRobust::TryDecideRouting(AllreduceRobust::RecoverType role,
                                  size_t *p_size,
                                  int *p_recvlink,
                                  std::vector<bool> *p_req_in) {
  int best_link = -2;
  {
    // get the shortest distance to the request point
    std::vector<std::pair<int, size_t> > dist_in, dist_out;
    ReturnType succ = MsgPassing(std::make_pair(role == kHaveData, *p_size),
                                 &dist_in, &dist_out, ShortestDist);
    if (succ != kSuccess) return succ;
    if (role != kHaveData) {
      for (size_t i = 0; i < dist_in.size(); ++i) {
        if (dist_in[i].first != std::numeric_limits<int>::max()) {
          utils::Check(best_link == -2 || *p_size == dist_in[i].second,
                       "[%d] Allreduce size inconsistent, distin=%lu, size=%lu, reporting=%lu\n",
                       rank, dist_in[i].first, *p_size, dist_in[i].second);
          if (best_link == -2 || dist_in[i].first < dist_in[best_link].first) {
            best_link = static_cast<int>(i);
            *p_size = dist_in[i].second;
          }
        }
      }
      utils::Check(best_link != -2, "Too many nodes went down and we cannot recover..");
    } else {
      best_link = -1;
    }
  }
  // get the node request
  std::vector<char> req_in, req_out;
  ReturnType succ = MsgPassing(std::make_pair(role == kRequestData, best_link),
                               &req_in, &req_out, DataRequest);
  if (succ != kSuccess) return succ;
  // set p_req_in
  p_req_in->resize(req_in.size());
  for (size_t i = 0; i < req_in.size(); ++i) {
    // set p_req_in
    (*p_req_in)[i] = (req_in[i] != 0);
    if (req_out[i] != 0) {
      utils::Assert(req_in[i] == 0, "cannot get and receive request");
      utils::Assert(static_cast<int>(i) == best_link, "request result inconsistent");
    }
  }
  *p_recvlink = best_link;
  return kSuccess;
}
/*!
 * \brief try to finish the data recovery request,
 *        this function is used together with TryDecideRouting
 * \param role the current role of the node
 * \param sendrecvbuf_ the buffer to store the data to be sent/recived
 *          - if the role is kHaveData, this stores the data to be sent
 *          - if the role is kRequestData, this is the buffer to store the result
 *          - if the role is kPassData, this will not be used, and can be NULL
 * \param size the size of the data, obtained from TryDecideRouting
 * \param recv_link the link index to receive data, if necessary, obtained from TryDecideRouting
 * \param req_in the request of each link to send data, obtained from TryDecideRouting
 *
 * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
 * \sa ReturnType, TryDecideRouting
 */
AllreduceRobust::ReturnType
AllreduceRobust::TryRecoverData(RecoverType role,
                                void *sendrecvbuf_,
                                size_t size,
                                int recv_link,
                                const std::vector<bool> &req_in) {
  RefLinkVector &links = tree_links;
  // no need to run recovery for zero size messages
  if (links.size() == 0 || size == 0) return kSuccess;
  utils::Assert(req_in.size() == links.size(), "TryRecoverData");
  const int nlink = static_cast<int>(links.size());
  {
    bool req_data = role == kRequestData;
    for (int i = 0; i < nlink; ++i) {
      if (req_in[i]) {
        utils::Assert(i != recv_link, "TryDecideRouting");
        req_data = true;
      }
    }
    // do not need to provide data or receive data, directly exit
    if (!req_data) return kSuccess;
  }
  utils::Assert(recv_link >= 0 || role == kHaveData, "recv_link must be active");
  if (role == kPassData) {
    links[recv_link].InitBuffer(1, size, reduce_buffer_size);
  }
  for (int i = 0; i < nlink; ++i) {
    links[i].ResetSize();
  }
  while (true) {
    bool finished = true;
    utils::PollHelper watcher;
    for (int i = 0; i < nlink; ++i) {
      if (i == recv_link && links[i].size_read != size) {
        watcher.WatchRead(links[i].sock);
        finished = false;
      }
      if (req_in[i] && links[i].size_write != size) {
        if (role == kHaveData ||
            (links[recv_link].size_read != links[i].size_write)) {
          watcher.WatchWrite(links[i].sock);
        }
        finished = false;
      }
      watcher.WatchException(links[i].sock);
    }
    if (finished) break;
    watcher.Poll();
    // exception handling
    for (int i = 0; i < nlink; ++i) {
      if (watcher.CheckExcept(links[i].sock)) {
        return ReportError(&links[i], kGetExcept);
      }
    }
    if (role == kRequestData) {
      const int pid = recv_link;
      if (watcher.CheckRead(links[pid].sock)) {
        ReturnType ret = links[pid].ReadToArray(sendrecvbuf_, size);
        if (ret != kSuccess) {
          return ReportError(&links[pid], ret);
        }
      }
      for (int i = 0; i < nlink; ++i) {
        if (req_in[i] && links[i].size_write != links[pid].size_read) {
          ReturnType ret = links[i].WriteFromArray(sendrecvbuf_, links[pid].size_read);
          if (ret != kSuccess) {
            return ReportError(&links[i], ret);
          }
        }
      }
    }
    if (role == kHaveData) {
      for (int i = 0; i < nlink; ++i) {
        if (req_in[i] && links[i].size_write != size) {
          ReturnType ret = links[i].WriteFromArray(sendrecvbuf_, size);
          if (ret != kSuccess) {
            return ReportError(&links[i], ret);
          }
        }
      }
    }
    if (role == kPassData) {
      const int pid = recv_link;
      const size_t buffer_size = links[pid].buffer_size;
      if (watcher.CheckRead(links[pid].sock)) {
        size_t min_write = size;
        for (int i = 0; i < nlink; ++i) {
          if (req_in[i]) min_write = std::min(links[i].size_write, min_write);
        }
        utils::Assert(min_write <= links[pid].size_read, "boundary check");
        ReturnType ret = links[pid].ReadToRingBuffer(min_write, size);
        if (ret != kSuccess) {
          return ReportError(&links[pid], ret);
        }
      }
      for (int i = 0; i < nlink; ++i) {
        if (req_in[i] && links[pid].size_read != links[i].size_write) {
          size_t start = links[i].size_write % buffer_size;
          // send out data from ring buffer
          size_t nwrite = std::min(buffer_size - start, links[pid].size_read - links[i].size_write);
          ssize_t len = links[i].sock.Send(links[pid].buffer_head + start, nwrite);
          if (len != -1) {
            links[i].size_write += len;
          } else {
            ReturnType ret = Errno2Return();
            if (ret != kSuccess) return ReportError(&links[i], ret);
          }
        }
      }
    }
  }
  return kSuccess;
}
/*!
 * \brief try to load check point
 *
 *        This is a collaborative function called by all nodes
 *        only the nodes with requester set to true really needs to load the check point
 *        other nodes acts as collaborative roles to complete this request
 *
 * \param requester whether current node is the requester
 * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
 * \sa ReturnType
 */
AllreduceRobust::ReturnType AllreduceRobust::TryLoadCheckPoint(bool requester) {
  // check in local data
  RecoverType role =  requester ? kRequestData : kHaveData;
  ReturnType succ;
  if (num_local_replica != 0) {
    if (requester) {
      // clear existing history, if any, before load
      local_rptr[local_chkpt_version].clear();
      local_chkpt[local_chkpt_version].clear();
    }
    // recover local checkpoint
    succ = TryRecoverLocalState(&local_rptr[local_chkpt_version],
                                &local_chkpt[local_chkpt_version]);
    if (succ != kSuccess) return succ;
    int nlocal = std::max(static_cast<int>(local_rptr[local_chkpt_version].size()) - 1, 0);
    // check if everyone is OK
    unsigned state = 0;
    if (nlocal == num_local_replica + 1) {
      // complete recovery
      state = 1;
    } else if (nlocal == 0) {
      // get nothing
      state = 2;
    } else {
      // partially complete state
      state = 4;
    }
    succ = TryAllreduce(&state, sizeof(state), 1, op::Reducer<op::BitOR, unsigned>);
    if (succ != kSuccess) return succ;
    utils::Check(state == 1 || state == 2,
                 "LoadCheckPoint: too many nodes fails, cannot recover local state");
  }
  // do call save model if the checkpoint was lazy
  if (role == kHaveData && global_lazycheck != NULL) {
    global_checkpoint.resize(0);
    utils::MemoryBufferStream fs(&global_checkpoint);
    fs.Write(&version_number, sizeof(version_number));
    global_lazycheck->Save(&fs);
    global_lazycheck = NULL;
  }
  // recover global checkpoint
  size_t size = this->global_checkpoint.length();
  int recv_link;
  std::vector<bool> req_in;
  succ = TryDecideRouting(role, &size, &recv_link, &req_in);
  if (succ != kSuccess) return succ;
  if (role == kRequestData) {
    global_checkpoint.resize(size);
  }
  if (size == 0) return kSuccess;
  return TryRecoverData(role, BeginPtr(global_checkpoint), size, recv_link, req_in);
}
/*!
 * \brief try to get the result of operation specified by seqno
 *
 *        This is a collaborative function called by all nodes
 *        only the nodes with requester set to true really needs to get the result
 *        other nodes acts as collaborative roles to complete this request
 *
 * \param buf the buffer to store the result, this parameter is only used when current node is requester
 * \param size the total size of the buffer, this parameter is only used when current node is requester
 * \param seqno sequence number of the operation, this is unique index of a operation in current iteration
 * \param requester whether current node is the requester
 * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
 * \sa ReturnType
 */
AllreduceRobust::ReturnType
AllreduceRobust::TryGetResult(void *sendrecvbuf, size_t size, int seqno, bool requester) {
  // if minimum sequence requested is local check point ack,
  // this means all nodes have finished local check point, directly return
  if (seqno == ActionSummary::kLocalCheckAck) return kSuccess;
  if (seqno == ActionSummary::kLocalCheckPoint) {
    // new version of local model
    int new_version = !local_chkpt_version;
    int nlocal = std::max(static_cast<int>(local_rptr[new_version].size()) - 1, 0);
    // if we goes to this place, use must have already setup the state once
    utils::Assert(nlocal == 1 || nlocal == num_local_replica + 1,
                  "TryGetResult::Checkpoint");
    return TryRecoverLocalState(&local_rptr[new_version], &local_chkpt[new_version]);
  }
  // handles normal data recovery
  RecoverType role;
  if (!requester) {
    sendrecvbuf = resbuf.Query(seqno, &size);
    role = sendrecvbuf != NULL ? kHaveData : kPassData;
  } else {
    role = kRequestData;
  }
  int recv_link;
  std::vector<bool> req_in;
  // size of data
  size_t data_size = size;
  ReturnType succ = TryDecideRouting(role, &data_size, &recv_link, &req_in);
  if (succ != kSuccess) return succ;
  utils::Check(data_size != 0, "zero size check point is not allowed");
  if (role == kRequestData || role == kHaveData) {
    utils::Check(data_size == size,
                 "Allreduce Recovered data size do not match the specification of function call.\n"\
                 "Please check if calling sequence of recovered program is the " \
                 "same the original one in current VersionNumber");
  }
  return TryRecoverData(role, sendrecvbuf, data_size, recv_link, req_in);
}
/*!
 * \brief try to run recover execution for a request action described by flag and seqno,
 *        the function will keep blocking to run possible recovery operations before the specified action,
 *        until the requested result is received by a recovering procedure,
 *        or the function discovers that the requested action is not yet executed, and return false
 *
 * \param buf the buffer to store the result
 * \param size the total size of the buffer
 * \param flag flag information about the action \sa ActionSummary
 * \param seqno sequence number of the action, if it is special action with flag set,
 *              seqno needs to be set to ActionSummary::kSpecialOp
 *
 * \return if this function can return true or false
 *    - true means buf already set to the
 *           result by recovering procedure, the action is complete, no further action is needed
 *    - false means this is the lastest action that has not yet been executed, need to execute the action
 */
bool AllreduceRobust::RecoverExec(void *buf, size_t size, int flag, int seqno) {
  if (flag != 0) {
    utils::Assert(seqno == ActionSummary::kSpecialOp, "must only set seqno for normal operations");
  }
  // request
  ActionSummary req(flag, seqno);
  while (true) {
    this->ReportStatus();
    // action
    ActionSummary act = req;
    // get the reduced action
    if (!CheckAndRecover(TryAllreduce(&act, sizeof(act), 1, ActionSummary::Reducer))) continue;
    if (act.check_ack()) {
      if (act.check_point()) {
        // if we also have check_point, do check point first
        utils::Assert(!act.diff_seq(),
                      "check ack & check pt  cannot occur together with normal ops");
        // if we requested checkpoint, we are free to go
        if (req.check_point()) return true;
      } else if (act.load_check()) {
        // if there is only check_ack and load_check, do load_check
        if (!CheckAndRecover(TryLoadCheckPoint(req.load_check()))) continue;
        // if requested load check, then misson complete
        if (req.load_check()) return true;
      } else {
        // there is no check point and no load check, execute check ack
        if (req.check_ack()) return true;
      }
      // if execute to this point
      // this means the action requested has not been completed
      // try next round
    } else {
      if (act.check_point()) {
        if (act.diff_seq()) {
          utils::Assert(act.min_seqno() != ActionSummary::kSpecialOp, "min seq bug");
          bool requester = req.min_seqno() == act.min_seqno();
          if (!CheckAndRecover(TryGetResult(buf, size, act.min_seqno(), requester))) continue;
          if (requester) return true;
        } else  {
          // no difference in seq no, means we are free to check point
          if (req.check_point()) return true;
        }
      } else {
        // no check point
        if (act.load_check()) {
          // all the nodes called load_check, this is an incomplete action
          if (!act.diff_seq()) return false;
          // load check have higher priority, do load_check
          if (!CheckAndRecover(TryLoadCheckPoint(req.load_check()))) continue;
          // if requested load check, then misson complete
          if (req.load_check()) return true;
        } else {
          // no special flags, no checkpoint, check ack, load_check
          utils::Assert(act.min_seqno() != ActionSummary::kSpecialOp, "min seq bug");
          if (act.diff_seq()) {
            bool requester = req.min_seqno() == act.min_seqno();
            if (!CheckAndRecover(TryGetResult(buf, size, act.min_seqno(), requester))) continue;
            if (requester) return true;
          } else {
            // all the request is same,
            // this is most recent command that is yet to be executed
            return false;
          }
        }
      }
      // something is still incomplete try next round
    }
  }
  utils::Assert(false, "RecoverExec: should not reach here");
  return true;
}
/*!
 * \brief try to recover the local state, making each local state to be the result of itself
 *        plus replication of states in previous num_local_replica hops in the ring
 *
 * The input parameters must contain the valid local states available in current nodes,
 * This function try ist best to "complete" the missing parts of local_rptr and local_chkpt
 * If there is sufficient information in the ring, when the function returns, local_chkpt will
 * contain num_local_replica + 1 checkpoints (including the chkpt of this node)
 * If there is no sufficient information in the ring, this function the number of checkpoints
 * will be less than the specified value
 *
 * \param p_local_rptr the pointer to the segment pointers in the states array
 * \param p_local_chkpt the pointer to the storage of local check points
 * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
 * \sa ReturnType
 */
AllreduceRobust::ReturnType
AllreduceRobust::TryRecoverLocalState(std::vector<size_t> *p_local_rptr,
                                      std::string *p_local_chkpt) {
  // if there is no local replica, we can do nothing
  if (num_local_replica == 0) return kSuccess;
  std::vector<size_t> &rptr = *p_local_rptr;
  std::string &chkpt = *p_local_chkpt;
  if (rptr.size() == 0) {
    rptr.push_back(0);
    utils::Assert(chkpt.length() == 0, "local chkpt space inconsistent");
  }
  const int n = num_local_replica;
  {
    // backward passing, passing state in backward direction of the ring
    const int nlocal = static_cast<int>(rptr.size() - 1);
    utils::Assert(nlocal <= n + 1, "invalid local replica");
    std::vector<int> msg_back(n + 1);
    msg_back[0] = nlocal;
    // backward passing one hop the request
    ReturnType succ;
    succ = RingPassing(BeginPtr(msg_back),
                       1 * sizeof(int), (n+1) * sizeof(int),
                       0 * sizeof(int), n * sizeof(int),
                       ring_next, ring_prev);
    if (succ != kSuccess) return succ;
    int msg_forward[2];
    msg_forward[0] = nlocal;
    succ = RingPassing(msg_forward,
                       1 * sizeof(int), 2 * sizeof(int),
                       0 * sizeof(int), 1 * sizeof(int),
                       ring_prev, ring_next);
    if (succ != kSuccess) return succ;
    // calculate the number of things we can read from next link
    int nread_end = nlocal;
    for (int i = 1; i <= n; ++i) {
      nread_end = std::max(nread_end, msg_back[i] - i);
    }
    // gives the size of forward
    int nwrite_start = std::min(msg_forward[1] + 1, nread_end);
    // get the size of each segments
    std::vector<size_t> sizes(nread_end);
    for (int i = 0; i < nlocal; ++i) {
      sizes[i] = rptr[i + 1] - rptr[i];
    }
    // pass size through the link
    succ = RingPassing(BeginPtr(sizes),
                       nlocal * sizeof(size_t),
                       nread_end * sizeof(size_t),
                       nwrite_start * sizeof(size_t),
                       nread_end * sizeof(size_t),
                       ring_next, ring_prev);
    if (succ != kSuccess) return succ;
    // update rptr
    rptr.resize(nread_end + 1);
    for (int i = nlocal; i < nread_end; ++i) {
      rptr[i + 1] = rptr[i] + sizes[i];
    }
    chkpt.resize(rptr.back());
    // pass data through the link
    succ = RingPassing(BeginPtr(chkpt), rptr[nlocal], rptr[nread_end],
                       rptr[nwrite_start], rptr[nread_end],
                       ring_next, ring_prev);
    if (succ != kSuccess) {
      rptr.resize(nlocal + 1); chkpt.resize(rptr.back()); return succ;
    }
  }
  {
    // forward passing, passing state in forward direction of the ring
    const int nlocal = static_cast<int>(rptr.size() - 1);
    utils::Assert(nlocal <= n + 1, "invalid local replica");
    std::vector<int> msg_forward(n + 1);
    msg_forward[0] = nlocal;
    // backward passing one hop the request
    ReturnType succ;
    succ = RingPassing(BeginPtr(msg_forward),
                       1 * sizeof(int), (n+1) * sizeof(int),
                       0 * sizeof(int), n * sizeof(int),
                       ring_prev, ring_next);
    if (succ != kSuccess) return succ;
    int msg_back[2];
    msg_back[0] = nlocal;
    succ = RingPassing(msg_back,
                       1 * sizeof(int), 2 * sizeof(int),
                       0 * sizeof(int), 1 * sizeof(int),
                       ring_next, ring_prev);
    if (succ != kSuccess) return succ;
    // calculate the number of things we can read from next link
    int nread_end = nlocal, nwrite_end = 1;
    // have to have itself in order to get other data from prev link
    if (nlocal != 0) {
      for (int i = 1; i <= n; ++i) {
        if (msg_forward[i] == 0) break;
        nread_end = std::max(nread_end, i + 1);
        nwrite_end = i + 1;
      }
      if (nwrite_end > n) nwrite_end = n;
    } else  {
      nread_end = 0; nwrite_end = 0;
    }
    // gives the size of forward
    int nwrite_start = std::min(msg_back[1] - 1, nwrite_end);
    // next node miss the state of itself, cannot recover
    if (nwrite_start < 0) nwrite_start = nwrite_end = 0;
    // get the size of each segments
    std::vector<size_t> sizes(nread_end);
    for (int i = 0; i < nlocal; ++i) {
      sizes[i] = rptr[i + 1] - rptr[i];
    }
    // pass size through the link, check consistency
    succ = RingPassing(BeginPtr(sizes),
                       nlocal * sizeof(size_t),
                       nread_end * sizeof(size_t),
                       nwrite_start * sizeof(size_t),
                       nwrite_end * sizeof(size_t),
                       ring_prev, ring_next);
    if (succ != kSuccess) return succ;
    // update rptr
    rptr.resize(nread_end + 1);
    for (int i = nlocal; i < nread_end; ++i) {
      rptr[i + 1] = rptr[i] + sizes[i];
    }
    chkpt.resize(rptr.back());
    // pass data through the link
    succ = RingPassing(BeginPtr(chkpt), rptr[nlocal], rptr[nread_end],
                       rptr[nwrite_start], rptr[nwrite_end],
                       ring_prev, ring_next);
    if (succ != kSuccess) {
      rptr.resize(nlocal + 1); chkpt.resize(rptr.back()); return succ;
    }
  }
  return kSuccess;
}
/*!
 * \brief try to checkpoint local state, this function is called in normal executation phase
 *    of checkpoint that contains local state
 *  the input state must exactly one saved state(local state of current node),
 *  after complete, this function will get local state from previous num_local_replica nodes and put them
 *  into local_chkpt and local_rptr
 *
 *  It is also OK to call TryRecoverLocalState instead,
 *  TryRecoverLocalState makes less assumption about the input, and requires more communications
 *
 * \param p_local_rptr the pointer to the segment pointers in the states array
 * \param p_local_chkpt the pointer to the storage of local check points
 * \return this function can return kSuccess/kSockError/kGetExcept, see ReturnType for details
 * \sa ReturnType, TryRecoverLocalState
 */
AllreduceRobust::ReturnType
AllreduceRobust::TryCheckinLocalState(std::vector<size_t> *p_local_rptr,
                                      std::string *p_local_chkpt) {
  // if there is no local replica, we can do nothing
  if (num_local_replica == 0) return kSuccess;
  std::vector<size_t> &rptr = *p_local_rptr;
  std::string &chkpt = *p_local_chkpt;
  utils::Assert(rptr.size() == 2,
                "TryCheckinLocalState must have exactly 1 state");
  const int n = num_local_replica;
  std::vector<size_t> sizes(n + 1);
  sizes[0] = rptr[1] - rptr[0];
  ReturnType succ;
  // pass size through the link
  succ = RingPassing(BeginPtr(sizes),
                     1 * sizeof(size_t),
                     (n + 1) * sizeof(size_t),
                     0 * sizeof(size_t),
                     n * sizeof(size_t),
                     ring_prev, ring_next);
  if (succ != kSuccess) return succ;
  // update rptr
  rptr.resize(n + 2);
  for (int i = 1; i <= n; ++i) {
    rptr[i + 1] = rptr[i] + sizes[i];
  }
  chkpt.resize(rptr.back());
  // pass data through the link
  succ = RingPassing(BeginPtr(chkpt),
                     rptr[1], rptr[n + 1],
                     rptr[0], rptr[n],
                     ring_prev, ring_next);
  if (succ != kSuccess) {
    rptr.resize(2); chkpt.resize(rptr.back()); return succ;
  }
  return kSuccess;
}
/*!
 * \brief perform a ring passing to receive data from prev link, and sent data to next link
 *  this allows data to stream over a ring structure
 *  sendrecvbuf[0:read_ptr] are already provided by current node
 *  current node will recv sendrecvbuf[read_ptr:read_end] from prev link
 *  current node will send sendrecvbuf[write_ptr:write_end] to next link
 *  write_ptr will wait till the data is readed before sending the data
 *  this function requires read_end >= write_end
 *
 * \param sendrecvbuf_ the place to hold the incoming and outgoing data
 * \param read_ptr the initial read pointer
 * \param read_end the ending position to read
 * \param write_ptr the initial write pointer
 * \param write_end the ending position to write
 * \param read_link pointer to link to previous position in ring
 * \param write_link pointer to link of next position in ring
 */
AllreduceRobust::ReturnType
AllreduceRobust::RingPassing(void *sendrecvbuf_,
                             size_t read_ptr,
                             size_t read_end,
                             size_t write_ptr,
                             size_t write_end,
                             LinkRecord *read_link,
                             LinkRecord *write_link) {
  if (read_link == NULL || write_link == NULL || read_end == 0) return kSuccess;
  utils::Assert(write_end <= read_end,
                "RingPassing: boundary check1");
  utils::Assert(read_ptr <= read_end, "RingPassing: boundary check2");
  utils::Assert(write_ptr <= write_end, "RingPassing: boundary check3");
  // take reference
  LinkRecord &prev = *read_link, &next = *write_link;
  // send recv buffer
  char *buf = reinterpret_cast<char*>(sendrecvbuf_);
  while (true) {
    bool finished = true;
    utils::PollHelper watcher;
    if (read_ptr != read_end) {
      watcher.WatchRead(prev.sock);
      finished = false;
    }
    if (write_ptr < read_ptr && write_ptr != write_end) {
      watcher.WatchWrite(next.sock);
      finished = false;
    }
    watcher.WatchException(prev.sock);
    watcher.WatchException(next.sock);
    if (finished) break;
    watcher.Poll();
    if (watcher.CheckExcept(prev.sock)) return ReportError(&prev, kGetExcept);
    if (watcher.CheckExcept(next.sock)) return ReportError(&next, kGetExcept);
    if (read_ptr != read_end && watcher.CheckRead(prev.sock)) {
      ssize_t len = prev.sock.Recv(buf + read_ptr, read_end - read_ptr);
      if (len == 0) {
        prev.sock.Close(); return ReportError(&prev, kRecvZeroLen);
      }
      if (len != -1) {
        read_ptr += static_cast<size_t>(len);
      } else {
        ReturnType ret = Errno2Return();
        if (ret != kSuccess) return ReportError(&prev, ret);
      }
    }
    if (write_ptr != write_end && write_ptr < read_ptr) {
      size_t nsend = std::min(write_end - write_ptr, read_ptr - write_ptr);
      ssize_t len = next.sock.Send(buf + write_ptr, nsend);
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
}  // namespace engine
}  // namespace rabit
