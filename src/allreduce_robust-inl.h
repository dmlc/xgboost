/*!
 *  Copyright (c) 2014 by Contributors
 * \file allreduce_robust-inl.h
 * \brief implementation of inline template function in AllreduceRobust
 *
 * \author Tianqi Chen
 */
#ifndef RABIT_ALLREDUCE_ROBUST_INL_H_
#define RABIT_ALLREDUCE_ROBUST_INL_H_
#include <vector>

namespace rabit {
namespace engine {
/*!
 * \brief run message passing algorithm on the allreduce tree
 *        the result is edge message stored in p_edge_in and p_edge_out
 * \param node_value the value associated with current node
 * \param p_edge_in used to store input message from each of the edge
 * \param p_edge_out used to store output message from each of the edge
 * \param func a function that defines the message passing rule
 *        Parameters of func:
 *           - node_value same as node_value in the main function
 *           - edge_in the array of input messages from each edge,
 *                     this includes the output edge, which should be excluded
 *           - out_index array the index of output edge, the function should
 *                       exclude the output edge when compute the message passing value
 *        Return of func:
 *           the function returns the output message based on the input message and node_value
 *
 * \tparam EdgeType type of edge message, must be simple struct
 * \tparam NodeType type of node value
 */
template<typename NodeType, typename EdgeType>
inline AllreduceRobust::ReturnType
AllreduceRobust::MsgPassing(const NodeType &node_value,
                            std::vector<EdgeType> *p_edge_in,
                            std::vector<EdgeType> *p_edge_out,
                            EdgeType(*func)
                            (const NodeType &node_value,
                             const std::vector<EdgeType> &edge_in,
                             size_t out_index)) {
  RefLinkVector &links = tree_links;
  if (links.size() == 0) return kSuccess;
  // number of links
  const int nlink = static_cast<int>(links.size());
  // initialize the pointers
  for (int i = 0; i < nlink; ++i) {
    links[i].ResetSize();
  }
  std::vector<EdgeType> &edge_in = *p_edge_in;
  std::vector<EdgeType> &edge_out = *p_edge_out;
  edge_in.resize(nlink);
  edge_out.resize(nlink);
  // stages in the process
  // 0: recv messages from childs
  // 1: send message to parent
  // 2: recv message from parent
  // 3: send message to childs
  int stage = 0;
  // if no childs, no need to, directly start passing message
  if (nlink == static_cast<int>(parent_index != -1)) {
    utils::Assert(parent_index == 0, "parent must be 0");
    edge_out[parent_index] = func(node_value, edge_in, parent_index);
    stage = 1;
  }
  // while we have not passed the messages out
  while (true) {
    // for node with no parent, directly do stage 3
    if (parent_index == -1) {
      utils::Assert(stage != 2 && stage != 1, "invalie stage id");
    }
    // poll helper
    utils::PollHelper watcher;
    bool done = (stage == 3);
    for (int i = 0; i < nlink; ++i) {
      watcher.WatchException(links[i].sock);
      switch (stage) {
        case 0:
          if (i != parent_index && links[i].size_read != sizeof(EdgeType)) {
            watcher.WatchRead(links[i].sock);
          }
          break;
        case 1:
          if (i == parent_index) {
            watcher.WatchWrite(links[i].sock);
          }
          break;
        case 2:
          if (i == parent_index) {
            watcher.WatchRead(links[i].sock);
          }
          break;
        case 3:
          if (i != parent_index && links[i].size_write != sizeof(EdgeType)) {
            watcher.WatchWrite(links[i].sock);
            done = false;
          }
          break;
        default: utils::Error("invalid stage");
      }
    }
    // finish all the stages, and write out message
    if (done) break;
    watcher.Poll();
    // exception handling
    for (int i = 0; i < nlink; ++i) {
      // recive OOB message from some link
      if (watcher.CheckExcept(links[i].sock)) {
        return ReportError(&links[i], kGetExcept);
      }
    }
    if (stage == 0) {
      bool finished = true;
      // read data from childs
      for (int i = 0; i < nlink; ++i) {
        if (i != parent_index) {
          if (watcher.CheckRead(links[i].sock)) {
            ReturnType ret = links[i].ReadToArray(&edge_in[i], sizeof(EdgeType));
            if (ret != kSuccess) return ReportError(&links[i], ret);
          }
          if (links[i].size_read != sizeof(EdgeType)) finished = false;
        }
      }
      // if no parent, jump to stage 3, otherwise do stage 1
      if (finished) {
        if (parent_index != -1) {
          edge_out[parent_index] = func(node_value, edge_in, parent_index);
          stage = 1;
        } else {
          for (int i = 0; i < nlink; ++i) {
            edge_out[i] = func(node_value, edge_in, i);
          }
          stage = 3;
        }
      }
    }
    if (stage == 1) {
      const int pid = this->parent_index;
      utils::Assert(pid != -1, "MsgPassing invalid stage");
      ReturnType ret = links[pid].WriteFromArray(&edge_out[pid], sizeof(EdgeType));
      if (ret != kSuccess) return ReportError(&links[pid], ret);
      if (links[pid].size_write == sizeof(EdgeType)) stage = 2;
    }
    if (stage == 2) {
      const int pid = this->parent_index;
      utils::Assert(pid != -1, "MsgPassing invalid stage");
      ReturnType ret = links[pid].ReadToArray(&edge_in[pid], sizeof(EdgeType));
      if (ret != kSuccess) return ReportError(&links[pid], ret);
      if (links[pid].size_read == sizeof(EdgeType)) {
        for (int i = 0; i < nlink; ++i) {
          if (i != pid) edge_out[i] = func(node_value, edge_in, i);
        }
        stage = 3;
      }
    }
    if (stage == 3) {
      for (int i = 0; i < nlink; ++i) {
        if (i != parent_index && links[i].size_write != sizeof(EdgeType)) {
          ReturnType ret = links[i].WriteFromArray(&edge_out[i], sizeof(EdgeType));
          if (ret != kSuccess) return ReportError(&links[i], ret);
        }
      }
    }
  }
  return kSuccess;
}
}  // namespace engine
}  // namespace rabit
#endif  // RABIT_ALLREDUCE_ROBUST_INL_H_
