/*!
 *  Copyright (c) 2014-2019 by Contributors
 * \file allreduce_robust.cc
 * \brief Robust implementation of Allreduce
 *
 * \author Tianqi Chen, Ignacio Cano, Tianyi Zhou
 */
#define NOMINMAX
#include <rabit/base.h>
#include <chrono>
#include <thread>
#include <limits>
#include <utility>
#include "rabit/internal/io.h"
#include "rabit/internal/timer.h"
#include "rabit/internal/utils.h"
#include "rabit/internal/engine.h"
#include "rabit/internal/rabit-inl.h"
#include "allreduce_robust.h"

#undef _assert

namespace rabit {
namespace engine {

AllreduceRobust::AllreduceRobust() {
  num_local_replica_ = 0;
  num_global_replica_ = 5;
  default_local_replica_ = 2;
  seq_counter = 0;
  cur_cache_seq_ = 0;
  local_chkpt_version_ = 0;
  result_buffer_round_ = 1;
  global_lazycheck_ = nullptr;
  use_local_model_ = -1;
  recover_counter_ = 0;
  checkpoint_loaded_ = false;
  env_vars.emplace_back("rabit_global_replica");
  env_vars.emplace_back("rabit_local_replica");
}
bool AllreduceRobust::Init(int argc, char* argv[]) {
  if (AllreduceBase::Init(argc, argv)) {
    // chenqin: alert user opted in experimental feature.
    if (rabit_bootstrap_cache) { utils::HandleLogInfo(
      "[EXPERIMENTAL] bootstrap cache has been enabled\n");
}
    checkpoint_loaded_ = false;
    if (num_global_replica_ == 0) {
      result_buffer_round_ = -1;
    } else {
      result_buffer_round_ = std::max(world_size / num_global_replica_, 1);
    }
    return true;
  } else {
    return false;
  }
}
/*! \brief shutdown the engine */
bool AllreduceRobust::Shutdown() {
  try {
    // need to sync the exec before we shutdown, do a pesudo check point
    // execute checkpoint, note: when checkpoint existing, load will not happen
    assert_(RecoverExec(nullptr, 0, ActionSummary::kCheckPoint, ActionSummary::kSpecialOp,
      cur_cache_seq_), "Shutdown: check point must return true");
    // reset result buffer
    resbuf_.Clear(); seq_counter = 0;
    cachebuf_.Clear(); cur_cache_seq_ = 0;
    lookupbuf_.Clear();
    // execute check ack step, load happens here
    assert_(RecoverExec(nullptr, 0, ActionSummary::kCheckAck,
      ActionSummary::kSpecialOp, cur_cache_seq_), "Shutdown: check ack must return true");
// travis ci only osx test hang
#if defined (__APPLE__)
    sleep(1);
#endif
    shutdown_timeout_ = true;
    if (rabit_timeout_task_.valid()) {
      rabit_timeout_task_.wait();
      assert_(rabit_timeout_task_.get(), "expect timeout task return\n");
    }
    return AllreduceBase::Shutdown();
  } catch (const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    return false;
  }
}

/*!
 * \brief set parameters to the engine
 * \param name parameter name
 * \param val parameter value
 */
void AllreduceRobust::SetParam(const char *name, const char *val) {
  AllreduceBase::SetParam(name, val);
  if (!strcmp(name, "rabit_global_replica")) num_global_replica_ = atoi(val);
  if (!strcmp(name, "rabit_local_replica")) {
    num_local_replica_ = atoi(val);
  }
}

int AllreduceRobust::SetBootstrapCache(const std::string &key, const void *buf,
  const size_t type_nbytes, const size_t count) {
  for (int i = 0 ; i < cur_cache_seq_; i++) {
    size_t nsize = 0;
    void* name = lookupbuf_.Query(i, &nsize);
    if (nsize == key.length() + 1
      && strcmp(static_cast<const char*>(name), key.c_str()) == 0) {
      break;
    }
  }
  // we should consider way to support duplicated signatures
  // https://github.com/dmlc/xgboost/issues/5012
  // _assert(index == -1, "immutable cache key already exists");
  assert_(type_nbytes*count > 0, "can't set empty cache");
  void* temp = cachebuf_.AllocTemp(type_nbytes, count);
  cachebuf_.PushTemp(cur_cache_seq_, type_nbytes, count);
  std::memcpy(temp, buf, type_nbytes*count);

  std::string k(key);
  void* name = lookupbuf_.AllocTemp(strlen(k.c_str()) + 1, 1);
  lookupbuf_.PushTemp(cur_cache_seq_, strlen(k.c_str()) + 1, 1);
  std::memcpy(name, key.c_str(), strlen(k.c_str()) + 1);
  cur_cache_seq_ += 1;
  return 0;
}

int AllreduceRobust::GetBootstrapCache(const std::string &key, void* buf,
  const size_t type_nbytes, const size_t count) {
  // as requester sync with rest of nodes on latest cache content
  if (!RecoverExec(nullptr, 0, ActionSummary::kLoadBootstrapCache,
    seq_counter, cur_cache_seq_)) return -1;

  int index = -1;
  for (int i = 0 ; i < cur_cache_seq_; i++) {
    size_t nsize = 0;
    void* name = lookupbuf_.Query(i, &nsize);
    if (nsize == strlen(key.c_str()) + 1
      && strcmp(reinterpret_cast<char*>(name), key.c_str()) == 0) {
      index = i;
      break;
    }
  }
  // cache doesn't exists
  if (index == -1) return -1;

  size_t siz = 0;
  void* temp = cachebuf_.Query(index, &siz);
  utils::Assert(cur_cache_seq_ > index, "cur_cache_seq is smaller than lookup cache seq index");
  utils::Assert(siz == type_nbytes*count, "cache size stored expected to be same as requested");
  utils::Assert(siz > 0, "cache size should be greater than 0");
  std::memcpy(buf, temp, type_nbytes*count);
  return 0;
}

/*!
 * \brief Allgather function, each node have a segment of data in the ring of sendrecvbuf,
 *  the data provided by current node k is [slice_begin, slice_end),
 *  the next node's segment must start with slice_end
 *  after the call of Allgather, sendrecvbuf_ contains all the contents including all segments
 *  use a ring based algorithm
 *
 * \param sendrecvbuf buffer for both sending and receiving data, it is a ring conceptually
 * \param total_size total size of data to be gathered
 * \param slice_begin beginning of the current slice
 * \param slice_end end of the current slice
 * \param size_prev_slice size of the previous slice i.e. slice of node (rank - 1) % world_size
 * \param _file caller file name used to generate unique cache key
 * \param _line caller line number used to generate unique cache key
 * \param _caller caller function name used to generate unique cache key
 */
void AllreduceRobust::Allgather(void *sendrecvbuf,
                                    size_t total_size,
                                    size_t slice_begin,
                                    size_t slice_end,
                                    size_t size_prev_slice,
                                    const char* _file,
                                    const int _line,
                                    const char* _caller) {
  if (world_size == 1 || world_size == -1) return;
  // genreate unique allgather signature
  std::string key = std::string(_file) + "::" + std::to_string(_line) + "::"
    + std::string(_caller) + "#" +std::to_string(total_size);

  // try fetch bootstrap allgather results from cache
  if (!checkpoint_loaded_ && rabit_bootstrap_cache &&
    GetBootstrapCache(key, sendrecvbuf, total_size, 1) != -1) return;

  double start = utils::GetTime();
  bool recovered = RecoverExec(sendrecvbuf, total_size, 0, seq_counter, cur_cache_seq_);

  if (resbuf_.LastSeqNo() != -1 &&
    (result_buffer_round_ == -1 ||
      resbuf_.LastSeqNo() % result_buffer_round_ != rank % result_buffer_round_)) {
    resbuf_.DropLast();
  }

  void *temp = resbuf_.AllocTemp(total_size, 1);
  while (true) {
    if (recovered) {
      std::memcpy(temp, sendrecvbuf, total_size); break;
    } else {
      std::memcpy(temp, sendrecvbuf, total_size);
      if (CheckAndRecover(TryAllgatherRing(temp, total_size,
                                           slice_begin, slice_end, size_prev_slice))) {
        std::memcpy(sendrecvbuf, temp, total_size); break;
      } else {
        recovered = RecoverExec(sendrecvbuf, total_size, 0, seq_counter, cur_cache_seq_);
      }
    }
  }
  double delta = utils::GetTime() - start;
  // log allgather latency
  if (rabit_debug) {
    utils::HandleLogInfo("[%d] allgather (%s) finished version %d, seq %d, take %f seconds\n",
      rank, key.c_str(), version_number, seq_counter, delta);
  }

  // if bootstrap allgather, store and fetch through cache
  if (checkpoint_loaded_ || !rabit_bootstrap_cache) {
    resbuf_.PushTemp(seq_counter, total_size, 1);
    seq_counter += 1;
  } else {
    SetBootstrapCache(key, sendrecvbuf, total_size, 1);
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
 * \param _file caller file name used to generate unique cache key
 * \param _line caller line number used to generate unique cache key
 * \param _caller caller function name used to generate unique cache key
 */
void AllreduceRobust::Allreduce(void *sendrecvbuf_,
                                size_t type_nbytes,
                                size_t count,
                                ReduceFunction reducer,
                                PreprocFunction prepare_fun,
                                void *prepare_arg,
                                const char* _file,
                                const int _line,
                                const char* _caller) {
  // skip action in single node
  if (world_size == 1 || world_size == -1) {
    if (prepare_fun != nullptr) prepare_fun(prepare_arg);
    return;
  }

  // genreate unique allreduce signature
  std::string key = std::string(_file) + "::" + std::to_string(_line) + "::"
    + std::string(_caller) + "#" +std::to_string(type_nbytes) + "x" + std::to_string(count);

  // try fetch bootstrap allreduce results from cache
  if (!checkpoint_loaded_ && rabit_bootstrap_cache &&
    GetBootstrapCache(key, sendrecvbuf_, type_nbytes, count) != -1) return;

  double start = utils::GetTime();
  bool recovered = RecoverExec(sendrecvbuf_, type_nbytes * count, 0, seq_counter, cur_cache_seq_);

  if (resbuf_.LastSeqNo() != -1 &&
    (result_buffer_round_ == -1 ||
      resbuf_.LastSeqNo() % result_buffer_round_ != rank % result_buffer_round_)) {
    resbuf_.DropLast();
  }

  if (!recovered && prepare_fun != nullptr) prepare_fun(prepare_arg);
  void *temp = resbuf_.AllocTemp(type_nbytes, count);
  while (true) {
    if (recovered) {
      std::memcpy(temp, sendrecvbuf_, type_nbytes * count); break;
    } else {
      std::memcpy(temp, sendrecvbuf_, type_nbytes * count);
      if (CheckAndRecover(TryAllreduce(temp, type_nbytes, count, reducer))) {
        std::memcpy(sendrecvbuf_, temp, type_nbytes * count); break;
      } else {
        recovered = RecoverExec(sendrecvbuf_, type_nbytes * count, 0, seq_counter, cur_cache_seq_);
      }
    }
  }
  double delta = utils::GetTime() - start;
  // log allreduce latency
  if (rabit_debug) {
    utils::HandleLogInfo("[%d] allreduce (%s) finished version %d, seq %d, take %f seconds\n",
      rank, key.c_str(), version_number, seq_counter, delta);
  }

  // if bootstrap allreduce, store and fetch through cache
  if (checkpoint_loaded_ || !rabit_bootstrap_cache) {
    resbuf_.PushTemp(seq_counter, type_nbytes, count);
    seq_counter += 1;
  } else {
    SetBootstrapCache(key, sendrecvbuf_, type_nbytes, count);
  }
}
/*!
 * \brief broadcast data from root to all nodes
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param size the size of the data to be broadcasted
 * \param root the root worker id to broadcast the data
 * \param _file caller file name used to generate unique cache key
 * \param _line caller line number used to generate unique cache key
 * \param _caller caller function name used to generate unique cache key
 */
void AllreduceRobust::Broadcast(void *sendrecvbuf_, size_t total_size, int root,
                                const char* _file,
                                const int _line,
                                const char* _caller) {
  // skip action in single node
  if (world_size == 1 || world_size == -1) return;
  // genreate unique cache signature
  std::string key = std::string(_file) + "::" + std::to_string(_line) + "::"
    + std::string(_caller) + "#" +std::to_string(total_size) + "@" + std::to_string(root);
  // try fetch bootstrap allreduce results from cache
  if (!checkpoint_loaded_ && rabit_bootstrap_cache &&
      GetBootstrapCache(key, sendrecvbuf_, total_size, 1) != -1) {
    return;
  }
  double start = utils::GetTime();
  bool recovered = RecoverExec(sendrecvbuf_, total_size, 0, seq_counter, cur_cache_seq_);
  // now we are free to remove the last result, if any
  if (resbuf_.LastSeqNo() != -1 &&
      (result_buffer_round_ == -1 ||
       resbuf_.LastSeqNo() % result_buffer_round_ != rank % result_buffer_round_)) {
    resbuf_.DropLast();
  }
  void *temp = resbuf_.AllocTemp(1, total_size);
  while (true) {
    if (recovered) {
      std::memcpy(temp, sendrecvbuf_, total_size); break;
    } else {
      if (CheckAndRecover(TryBroadcast(sendrecvbuf_, total_size, root))) {
        std::memcpy(temp, sendrecvbuf_, total_size); break;
      } else {
        recovered = RecoverExec(sendrecvbuf_, total_size, 0, seq_counter, cur_cache_seq_);
      }
    }
  }

  double delta = utils::GetTime() - start;
  // log broadcast latency
  if (rabit_debug) {
    utils::HandleLogInfo(
      "[%d] broadcast (%s) root %d finished version %d,seq %d, take %f seconds\n",
      rank, key.c_str(), root, version_number, seq_counter, delta);
  }
  // if bootstrap broadcast, store and fetch through cache
  if (checkpoint_loaded_ || !rabit_bootstrap_cache) {
    resbuf_.PushTemp(seq_counter, 1, total_size);
    seq_counter += 1;
  } else {
    SetBootstrapCache(key, sendrecvbuf_, total_size, 1);
  }
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
  checkpoint_loaded_ = true;
  // skip action in single node
  if (world_size == 1) return 0;
  this->LocalModelCheck(local_model != nullptr);
  if (num_local_replica_ == 0) {
    utils::Check(local_model == nullptr,
                 "need to set rabit_local_replica larger than 1 to checkpoint local_model");
  }
  double start = utils::GetTime();
  // check if we succeed
  if (RecoverExec(nullptr, 0, ActionSummary::kLoadCheck, ActionSummary::kSpecialOp, cur_cache_seq_)) {
    int nlocal = std::max(static_cast<int>(local_rptr_[local_chkpt_version_].size()) - 1, 0);
    if (local_model != nullptr) {
      if (nlocal == num_local_replica_ + 1) {
        // load in local model
        utils::MemoryFixSizeBuffer fs(BeginPtr(local_chkpt_[local_chkpt_version_]),
                                      local_rptr_[local_chkpt_version_][1]);
        local_model->Load(&fs);
      } else {
        assert_(nlocal == 0, "[%d] local model inconsistent, nlocal=%d", rank, nlocal);
      }
    }
    // reset result buffer
    resbuf_.Clear(); seq_counter = 0;
    // load from buffer
    utils::MemoryBufferStream fs(&global_checkpoint_);
    if (global_checkpoint_.length() == 0) {
      version_number = 0;
    } else {
      assert_(fs.Read(&version_number, sizeof(version_number)) != 0,
                    "read in version number");
      global_model->Load(&fs);
      assert_(local_model == nullptr || nlocal == num_local_replica_ + 1,
                    "local model inconsistent, nlocal=%d", nlocal);
    }
    // run another phase of check ack, if recovered from data
    assert_(RecoverExec(nullptr, 0, ActionSummary::kCheckAck,
      ActionSummary::kSpecialOp, cur_cache_seq_), "check ack must return true");

    if (!RecoverExec(nullptr, 0, ActionSummary::kLoadBootstrapCache, seq_counter, cur_cache_seq_)) {
      utils::Printf("no need to load cache\n");
    }
    double delta = utils::GetTime() - start;

    // log broadcast latency
    if (rabit_debug) {
      utils::HandleLogInfo("[%d] loadcheckpoint size %ld finished version %d, "
                         "seq %d, take %f seconds\n",
                         rank, global_checkpoint_.length(),
                         version_number, seq_counter, delta);
    }
    return version_number;
  } else {
    // log job fresh start
    if (rabit_debug) utils::HandleLogInfo("[%d] loadcheckpoint reset\n", rank);

    // reset result buffer
    resbuf_.Clear(); seq_counter = 0; version_number = 0;
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
  if (use_local_model_ == -1) {
    if (with_local) {
      use_local_model_ = 1;
      if (num_local_replica_ == 0) {
        num_local_replica_ = default_local_replica_;
      }
    } else {
      use_local_model_ = 0;
      num_local_replica_ = 0;
    }
  } else {
    utils::Check(use_local_model_ == static_cast<int>(with_local),
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
void AllreduceRobust::CheckPointImpl(const Serializable *global_model,
                                     const Serializable *local_model,
                                     bool lazy_checkpt) {
  // never do check point in single machine mode
  if (world_size == 1) {
    version_number += 1; return;
  }
  double start = utils::GetTime();
  this->LocalModelCheck(local_model != nullptr);
  if (num_local_replica_ == 0) {
    utils::Check(local_model == nullptr,
                 "need to set rabit_local_replica larger than 1 to checkpoint local_model");
  }
  if (num_local_replica_ != 0) {
    while (true) {
      if (RecoverExec(nullptr, 0, 0, ActionSummary::kLocalCheckPoint)) break;
      // save model to new version place
      int new_version = !local_chkpt_version_;

      local_chkpt_[new_version].clear();
      utils::MemoryBufferStream fs(&local_chkpt_[new_version]);
      if (local_model != nullptr) {
        local_model->Save(&fs);
      }
      local_rptr_[new_version].clear();
      local_rptr_[new_version].push_back(0);
      local_rptr_[new_version].push_back(local_chkpt_[new_version].length());
      if (CheckAndRecover(TryCheckinLocalState(&local_rptr_[new_version],
                                               &local_chkpt_[new_version]))) break;
    }
    // run the ack phase, can be true or false
    RecoverExec(nullptr, 0, 0, ActionSummary::kLocalCheckAck);
    // switch pointer to new version
    local_chkpt_version_ = !local_chkpt_version_;
  }
  // execute checkpoint, note: when checkpoint existing, load will not happen
  assert_(RecoverExec(nullptr, 0, ActionSummary::kCheckPoint,
                      ActionSummary::kSpecialOp, cur_cache_seq_),
          "check point must return true");
  // this is the critical region where we will change all the stored models
  // increase version number
  version_number += 1;
  // save model
  if (lazy_checkpt) {
    global_lazycheck_ = global_model;
  } else {
    global_checkpoint_.resize(0);
    utils::MemoryBufferStream fs(&global_checkpoint_);
    fs.Write(&version_number, sizeof(version_number));
    global_model->Save(&fs);
    global_lazycheck_ = nullptr;
  }
  double delta = utils::GetTime() - start;
  // log checkpoint latency
  if (rabit_debug) {
    utils::HandleLogInfo(
      "[%d] checkpoint finished version %d,seq %d, take %f seconds\n",
      rank, version_number, seq_counter, delta);
  }
  start = utils::GetTime();
  // reset result buffer, mark boostrap phase complete
  resbuf_.Clear(); seq_counter = 0;
  // execute check ack step, load happens here
  assert_(RecoverExec(nullptr, 0, ActionSummary::kCheckAck,
    ActionSummary::kSpecialOp, cur_cache_seq_), "check ack must return true");

  delta = utils::GetTime() - start;
  // log checkpoint ack latency
  if (rabit_debug) {
    utils::HandleLogInfo(
        "[%d] checkpoint ack finished version %d, take %f seconds\n", rank,
        version_number, delta);
  }
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
AllreduceRobust::ReturnType AllreduceRobust::TryResetLinks() {
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
          assert_(all_links[i].sock.BadSocket(), "must already gone bad");
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
        assert_(oob_mark == kResetMark, "wrong oob msg");
        assert_(all_links[i].sock.AtMark() != 1, "should already read past mark");
      } else {
        assert_(errno != EAGAIN|| errno != EWOULDBLOCK, "BUG");
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
        assert_(ack == kResetAck, "wrong Ack MSG");
      } else {
        assert_(errno != EAGAIN|| errno != EWOULDBLOCK, "BUG");
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
  shutdown_timeout_ = err_type == kSuccess;
  if (err_type == kSuccess) return true;

  assert_(err_link != nullptr, "must know the error link");
  recover_counter_ += 1;
  // async launch timeout task if enable_rabit_timeout is set
  if (rabit_timeout && !rabit_timeout_task_.valid()) {
    utils::Printf("[EXPERIMENTAL] timeout thread expires in %d second(s)\n", timeout_sec);
    rabit_timeout_task_ = std::async(std::launch::async, [=]() {
      if (rabit_debug) {
        utils::Printf("[%d] timeout thread %ld starts\n", rank,
                      std::this_thread::get_id());
      }
      int time = 0;
      // check if rabit recovered every 100ms
      while (time++ < 10 * timeout_sec) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (shutdown_timeout_.load()) {
          if (rabit_debug) {
            utils::Printf("[%d] timeout task thread %ld exits\n",
              rank, std::this_thread::get_id());
          }
          return true;
        }
      }
      error_("[%d] exit due to time out %d s\n", rank, timeout_sec);
      return false;
    });
  }
  // simple way, shutdown all links
  for (auto & all_link : all_links) {
    if (!all_link.sock.BadSocket()) all_link.sock.Close();
  }
  // smooth out traffic to tracker
  std::this_thread::sleep_for(std::chrono::milliseconds(10*rank));
  ReConnectLinks("recover");
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
      assert_(req_in[i] == 0, "cannot get and receive request");
      assert_(static_cast<int>(i) == best_link, "request result inconsistent");
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
  if (links.Size() == 0 || size == 0) return kSuccess;
  assert_(req_in.size() == links.Size(), "TryRecoverData");
  const int nlink = static_cast<int>(links.Size());
  {
    bool req_data = role == kRequestData;
    for (int i = 0; i < nlink; ++i) {
      if (req_in[i]) {
        assert_(i != recv_link, "TryDecideRouting");
        req_data = true;
      }
    }
    // do not need to provide data or receive data, directly exit
    if (!req_data) return kSuccess;
  }
  assert_(recv_link >= 0 || role == kHaveData, "recv_link must be active");
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
        assert_(min_write <= links[pid].size_read, "boundary check");
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
 * \brief try to fetch allreduce/broadcast results from rest of nodes
 *  as collaberative function called by all nodes, only requester node
 *  will pass seqno to rest of nodes and reconstruct/backfill sendrecvbuf_
 *  of specific seqno from other nodes.
 */
AllreduceRobust::ReturnType AllreduceRobust::TryRestoreCache(bool requester,
  const int min_seq, const int max_seq) {
  // clear requester and rebuild from those with most cache entries
  if (requester) {
    assert_(cur_cache_seq_ <= max_seq, "requester is expected to have fewer cache entries");
    cachebuf_.Clear();
    lookupbuf_.Clear();
    cur_cache_seq_ = 0;
  }
  RecoverType role = requester ? kRequestData : kHaveData;
  size_t size = 1;
  int recv_link;
  std::vector<bool> req_in;
  ReturnType ret = TryDecideRouting(role, &size, &recv_link, &req_in);
  if (ret != kSuccess) return ret;
  // only recover missing cache entries in requester
  // as tryrecoverdata is collective call, need to go through entire cache
  // and only work on those missing
  for (int i = 0; i < max_seq; i++) {
    // restore lookup map
    size_t cache_size = 0;
    void* key = lookupbuf_.Query(i, &cache_size);
    ret = TryRecoverData(role, &cache_size, sizeof(size_t), recv_link, req_in);
    if (ret != kSuccess) return ret;
    if (requester) {
      key = lookupbuf_.AllocTemp(cache_size, 1);
      lookupbuf_.PushTemp(i, cache_size, 1);
    }
    ret = TryRecoverData(role, key, cache_size, recv_link, req_in);
    if (ret != kSuccess) return ret;
    // restore cache content
    cache_size = 0;
    void* buf = cachebuf_.Query(i, &cache_size);
    ret = TryRecoverData(role, &cache_size, sizeof(size_t), recv_link, req_in);
    if (requester) {
      buf = cachebuf_.AllocTemp(cache_size, 1);
      cachebuf_.PushTemp(i, cache_size, 1);
      cur_cache_seq_ +=1;
    }
    ret = TryRecoverData(role, buf, cache_size, recv_link, req_in);
    if (ret != kSuccess) return ret;
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
  if (num_local_replica_ != 0) {
    if (requester) {
      // clear existing history, if any, before load
      local_rptr_[local_chkpt_version_].clear();
      local_chkpt_[local_chkpt_version_].clear();
    }
    // recover local checkpoint
    succ = TryRecoverLocalState(&local_rptr_[local_chkpt_version_],
                                &local_chkpt_[local_chkpt_version_]);
    if (succ != kSuccess) return succ;
    int nlocal = std::max(static_cast<int>(local_rptr_[local_chkpt_version_].size()) - 1, 0);
    // check if everyone is OK
    unsigned state = 0;
    if (nlocal == num_local_replica_ + 1) {
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
  if (role == kHaveData && global_lazycheck_ != nullptr) {
    global_checkpoint_.resize(0);
    utils::MemoryBufferStream fs(&global_checkpoint_);
    fs.Write(&version_number, sizeof(version_number));
    global_lazycheck_->Save(&fs);
    global_lazycheck_ = nullptr;
  }
  // recover global checkpoint
  size_t size = this->global_checkpoint_.length();
  int recv_link;
  std::vector<bool> req_in;
  succ = TryDecideRouting(role, &size, &recv_link, &req_in);
  if (succ != kSuccess) return succ;
  if (role == kRequestData) {
    global_checkpoint_.resize(size);
  }
  if (size == 0) return kSuccess;
  return TryRecoverData(role, BeginPtr(global_checkpoint_), size, recv_link, req_in);
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
    int new_version = !local_chkpt_version_;
    int nlocal = std::max(static_cast<int>(local_rptr_[new_version].size()) - 1, 0);
    // if we goes to this place, use must have already setup the state once
    assert_(nlocal == 1 || nlocal == num_local_replica_ + 1,
                  "TryGetResult::Checkpoint");
    return TryRecoverLocalState(&local_rptr_[new_version], &local_chkpt_[new_version]);
  }

  // handles normal data recovery
  RecoverType role;
  if (!requester) {
    sendrecvbuf = resbuf_.Query(seqno, &size);
    role = sendrecvbuf != nullptr ? kHaveData : kPassData;
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
bool AllreduceRobust::RecoverExec(void *buf, size_t size, int flag, int seqno,
                                  int cache_seqno, const char* caller) {
  // kLoadBootstrapCache should be treated similar as allreduce
  // when loadcheck/check/checkack runs in other nodes
  if (flag != 0 && flag != ActionSummary::kLoadBootstrapCache) {
    assert_(seqno == ActionSummary::kSpecialOp, "must only set seqno for normal operations");
  }

  std::string msg = std::string(caller) + " pass negative seqno "
    + std::to_string(seqno) + " flag " + std::to_string(flag)
    + " version " + std::to_string(version_number);
  assert_(seqno >=0, msg.c_str());

  ActionSummary req(flag, flag, seqno, cache_seqno);

  while (true) {
    this->ReportStatus();
    // copy to action and send to allreduce with other nodes
    ActionSummary act = req;
    // get the reduced action
    if (!CheckAndRecover(TryAllreduce(&act, sizeof(act), 1, ActionSummary::Reducer))) continue;

    if (act.CheckAck()) {
      if (act.CheckPoint()) {
        // if we also have check_point, do check point first
        assert_(!act.DiffSeq(),
                      "check ack & check pt  cannot occur together with normal ops");
        // if we requested checkpoint, we are free to go
        if (req.CheckPoint()) return true;
      } else if (act.LoadCheck()) {
        // if there is only check_ack and load_check, do load_check
        if (!CheckAndRecover(TryLoadCheckPoint(req.LoadCheck()))) continue;
        // if requested load check, then misson complete
        if (req.LoadCheck()) return true;
      } else {
        // there is no check point and no load check, execute check ack
        if (req.CheckAck()) return true;
      }
      // if execute to this point
      // this means the action requested has not been completed
      // try next round
    } else {
      if (act.CheckPoint()) {
        if (act.DiffSeq()) {
          assert_(act.Seqno() != ActionSummary::kSpecialOp, "min seq bug");
          // print checkpoint consensus flag if user turn on debug
          if (rabit_debug) {
            req.PrintFlags(rank, "checkpoint req");
            act.PrintFlags(rank, "checkpoint act");
          }
          /*
           * Chen Qin
           * at least one hit checkpoint_ code & at least one not hitting
           * compare with version_number of req.check_point() set true with rest
           * expect to be equal, means rest fall behind in sequence
           * use resbuf resbuf to recover
           * worker-0           worker-1
           * checkpoint(n-1)    checkpoint(n-1)
           * allreduce          allreduce (requester) |
           * broadcast                                V
           * checkpoint(n req)
           * after catch up to checkpoint n, diff_seq will be false
           * */
          // assume requester is falling behind
          bool requester = req.Seqno() == act.Seqno();
          // if not load cache
          if (!act.LoadCache()) {
            if (act.Seqno() > 0) {
              if (!requester) {
                assert_(req.CheckPoint(), "checkpoint node should be KHaveData role");
                buf = resbuf_.Query(act.Seqno(), &size);
                assert_(buf != nullptr, "buf should have data from resbuf");
                assert_(size > 0, "buf size should be greater than 0");
              }
              if (!CheckAndRecover(TryGetResult(buf, size, act.Seqno(), requester))) continue;
            }
          } else {
            // cache seq no should be smaller than kSpecialOp
            assert_(act.Seqno(SeqType::kCache) != ActionSummary::kSpecialOp,
              "checkpoint with kSpecialOp");
            int max_cache_seq = cur_cache_seq_;
            if (TryAllreduce(&max_cache_seq, sizeof(max_cache_seq), 1,
              op::Reducer<op::Max, unsigned>) != kSuccess) continue;

            if (TryRestoreCache(req.LoadCache(), act.Seqno(), max_cache_seq)
              != kSuccess) continue;
          }
          if (requester) return true;
        } else  {
          // no difference in seq no, means we are free to check point
          if (req.CheckPoint()) return true;
        }
      } else {
        // no check point
        if (act.LoadCheck()) {
          // all the nodes called load_check, this is an incomplete action
          if (!act.DiffSeq()) return false;
          // load check have higher priority, do load_check
          if (!CheckAndRecover(TryLoadCheckPoint(req.LoadCheck()))) continue;
          // if requested load check, then misson complete
          if (req.LoadCheck()) return true;
        } else {
          // run all nodes in a isolated cache restore logic
          if (act.LoadCache()) {
            // print checkpoint consensus flag if user turn on debug
            if (rabit_debug) {
              req.PrintFlags(rank, "loadcache req");
              act.PrintFlags(rank, "loadcache act");
            }
            // load cache should not running in parralel with other states
            assert_(!act.LoadCheck(),
              "load cache state expect no nodes doing load checkpoint");
            assert_(!act.CheckPoint() ,
              "load cache state expect no nodes doing checkpoint");
            assert_(!act.CheckAck(),
              "load cache state expect no nodes doing checkpoint ack");

            // if all nodes are requester in load cache, skip
            if (act.LoadCache(SeqType::kCache)) return false;

            // bootstrap cache always restore before loadcheckpoint
            // requester always have seq diff with non requester
            if (act.DiffSeq()) {
              // restore cache failed, retry from what's left
              if (TryRestoreCache(req.LoadCache(), act.Seqno(), act.Seqno(SeqType::kCache))
                != kSuccess) continue;
            }
            // if requested load cache, then mission complete
            if (req.LoadCache()) return true;
            continue;
          }

          // assert no req with load cache set goes into seq catch up
          assert_(!req.LoadCache(), "load cache not interacte with rest states");

          // no special flags, no checkpoint, check ack, load_check
          assert_(act.Seqno() != ActionSummary::kSpecialOp, "min seq bug");
          if (act.DiffSeq()) {
            bool requester = req.Seqno() == act.Seqno();
            if (!CheckAndRecover(TryGetResult(buf, size, act.Seqno(), requester))) continue;
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
  assert_(false, "RecoverExec: should not reach here");
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
  if (num_local_replica_ == 0) return kSuccess;
  std::vector<size_t> &rptr = *p_local_rptr;
  std::string &chkpt = *p_local_chkpt;
  if (rptr.size() == 0) {
    rptr.push_back(0);
    assert_(chkpt.length() == 0, "local chkpt space inconsistent");
  }
  const int n = num_local_replica_;
  {
    // backward passing, passing state in backward direction of the ring
    const int nlocal = static_cast<int>(rptr.size() - 1);
    assert_(nlocal <= n + 1, "invalid local replica");
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
    assert_(nlocal <= n + 1, "invalid local replica");
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
  if (num_local_replica_ == 0) return kSuccess;
  std::vector<size_t> &rptr = *p_local_rptr;
  std::string &chkpt = *p_local_chkpt;
  assert_(rptr.size() == 2,
                "TryCheckinLocalState must have exactly 1 state");
  const int n = num_local_replica_;
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
  if (read_link == nullptr || write_link == nullptr || read_end == 0) return kSuccess;
  assert_(write_end <= read_end,
                "RingPassing: boundary check1");
  assert_(read_ptr <= read_end, "RingPassing: boundary check2");
  assert_(write_ptr <= write_end, "RingPassing: boundary check3");
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
