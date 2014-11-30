#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include "./utils.h"
#include "./engine_robust.h"

namespace engine {
/*!
 * \brief perform in-place allreduce, on sendrecvbuf 
 *        this function is NOT thread-safe
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param type_nbytes the unit number of bytes the type have
 * \param count number of elements to be reduced
 * \param reducer reduce function
 */
void AllReduceRobust::AllReduce(void *sendrecvbuf_,
                                size_t type_nbytes,
                                size_t count,           
                                ReduceFunction reducer) {
  while (true) {
    ReturnType ret = TryAllReduce(sendrecvbuf_, type_nbytes, count, reducer);
    if (ret == kSuccess) return;
    if (ret == kSockError) {
      utils::Error("error occur during all reduce\n");
    }
    utils::LogPrintf("[%d] receive except signal, start reset link\n", rank);
    TryResetLinks();
  }
  // TODO
}
/*!
 * \brief broadcast data from root to all nodes
 * \param sendrecvbuf_ buffer for both sending and recving data
 * \param size the size of the data to be broadcasted
 * \param root the root worker id to broadcast the data
 */
void AllReduceRobust::Broadcast(void *sendrecvbuf_, size_t total_size, int root) {
  utils::Assert(TryBroadcast(sendrecvbuf_, total_size, root) == kSuccess,
                "AllReduce failed");
  // TODO
}
/*!
 * \brief load latest check point
 * \param p_model pointer to the model
 * \return true if there was stored checkpoint and load was successful
 *   false if there was no stored checkpoint, means we are start over gain
 */
bool AllReduceRobust::LoadCheckPoint(utils::ISerializable *p_model) {
  // TODO
  return false;
}
/*!
 * \brief checkpoint the model, meaning we finished a stage of execution
 * \param p_model pointer to the model
 */
void AllReduceRobust::CheckPoint(const utils::ISerializable &model) {
  // TODO
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
AllReduceRobust::ReturnType AllReduceRobust::TryResetLinks(void) {
  // number of links
  const int nlink = static_cast<int>(links.size());
  for (int i = 0; i < nlink; ++i) {
    links[i].InitBuffer(sizeof(int), 1 << 10, reduce_buffer_size);
    links[i].ResetSize();
  }
  
  // read and discard data from all channels until pass mark
  while (true) {
    for (int i = 0; i < nlink; ++i) {
      if (links[i].sock.BadSocket()) continue;
      if (links[i].size_write == 0) {
        char sig = kOOBReset;
        ssize_t len = links[i].sock.Send(&sig, sizeof(sig), MSG_OOB);
        // error will be filtered in next loop
        if (len == sizeof(sig)) links[i].size_write = 1;
      }
      if (links[i].size_write == 1) {
        char sig = kResetMark;
        ssize_t len = links[i].sock.Send(&sig, sizeof(sig));
        if (len == sizeof(sig)) links[i].size_write = 2;
      }
    }
    utils::SelectHelper rsel;
    bool finished = true;
    for (int i = 0; i < nlink; ++i) {
      if (links[i].size_write != 2 && !links[i].sock.BadSocket()) {
        rsel.WatchWrite(links[i].sock); finished = false;
      }
    }
    if (finished) break;
    // wait to read from the channels to discard data
    rsel.Select();
  }
  for (int i = 0; i < nlink; ++i) {
    if (!links[i].sock.BadSocket()) {
      utils::SelectHelper::WaitExcept(links[i].sock);
    }
  }
  while (true) {
    for (int i = 0; i < nlink; ++i) {
      if (links[i].size_read == 0) {
        int atmark = links[i].sock.AtMark();
        if (atmark < 0) {
          utils::Assert(links[i].sock.BadSocket(), "must already gone bad");
        } else if (atmark > 0) {
          links[i].size_read = 1;
        } else {
          // no at mark, read and discard data
          ssize_t len = links[i].sock.Recv(links[i].buffer_head, links[i].buffer_size);
          if (links[i].sock.AtMark()) links[i].size_read = 1;
          // zero length, remote closed the connection, close socket
          if (len == 0) links[i].sock.Close();
        }
      }
    }
    utils::SelectHelper rsel;
    bool finished = true;    
    for (int i = 0; i < nlink; ++i) {
      if (links[i].size_read == 0 && !links[i].sock.BadSocket()) {
        rsel.WatchRead(links[i].sock); finished = false;
      }
    }
    if (finished) break;
    rsel.Select();
  }

  // start synchronization, use blocking I/O to avoid select
  for (int i = 0; i < nlink; ++i) {
    if (!links[i].sock.BadSocket()) {
      char oob_mark;
      links[i].sock.SetNonBlock(false);
      ssize_t len = links[i].sock.Recv(&oob_mark, sizeof(oob_mark), MSG_WAITALL);
      if (len == 0) {
        links[i].sock.Close(); continue;
      } else if (len > 0) {
        utils::Assert(oob_mark == kResetMark, "wrong oob msg");
        utils::Assert(links[i].sock.AtMark() != 1, "should already read past mark");
      } else {
        utils::Assert(errno != EAGAIN|| errno != EWOULDBLOCK, "BUG");
      }
      // send out ack
      char ack = kResetAck;
      while (true) {
        len = links[i].sock.Send(&ack, sizeof(ack));
        if (len == sizeof(ack)) break;
        if (len == -1) {
          if (errno != EAGAIN && errno != EWOULDBLOCK) break;
        }
      }
    }
  }
  // wait all ack
  for (int i = 0; i < nlink; ++i) {
    if (!links[i].sock.BadSocket()) {
      char ack;
      ssize_t len = links[i].sock.Recv(&ack, sizeof(ack), MSG_WAITALL);
      if (len == 0) {
        links[i].sock.Close(); continue;
      } else if (len > 0) {
        utils::Assert(ack == kResetAck, "wrong Ack MSG");
      } else {
        utils::Assert(errno != EAGAIN|| errno != EWOULDBLOCK, "BUG");
      }
      // set back to nonblock mode
      links[i].sock.SetNonBlock(true);
    }
  }
  for (int i = 0; i < nlink; ++i) {
    if (links[i].sock.BadSocket()) return kSockError;
  }
  return kSuccess;
}

bool AllReduceRobust::RecoverExec(void *sendrecvbuf_, size_t size, int flag, int seqno) {
  if (flag != 0) {
    utils::Assert(seqno == ActionSummary::kMaxSeq, "must only set seqno for normal operations");      
  }
  ActionSummary act(flag, seqno);
  return true;
}
}  // namespace engine
