#ifndef ALLREDUCE_MOCK_H
#define ALLREDUCE_MOCK_H
/*!
 * \file mock.h
 * \brief This file defines a mock object to test the system
 * \author Tianqi Chen, Nacho, Tianyi
 */
#include "./allreduce.h"
#include "./config.h"
 #include <map>


/*! \brief namespace of mock */
namespace test {

class Mock {


public:

  Mock(const int& rank, char *config) : rank(rank) {
    Init(config);
  }

  template<typename OP>
  inline void AllReduce(float *sendrecvbuf, size_t count) {
    utils::Assert(verify(allReduce), "[%d] error when calling allReduce", rank);
    sync::AllReduce<OP>(sendrecvbuf, count);
  }

  inline bool LoadCheckPoint(utils::ISerializable *p_model) {
    utils::Assert(verify(loadCheckpoint), "[%d] error when loading checkpoint", rank);
    return sync::LoadCheckPoint(p_model);
  }

  inline void CheckPoint(const utils::ISerializable &model) {
    utils::Assert(verify(checkpoint), "[%d] error when checkpointing", rank);
    sync::CheckPoint(model);
  }

  inline void Broadcast(std::string *sendrecv_data, int root) {
    utils::Assert(verify(broadcast), "[%d] error when broadcasting", rank);
    sync::Bcast(sendrecv_data, root);

  }

private:

  inline void Init(char* config) {
    utils::ConfigIterator itr(config);
    while (itr.Next()) {
      char round[4], node_rank[4];
      sscanf(itr.name(), "%[^_]_%s", round, node_rank);
      int i_round = atoi(round);
      if (i_round == 1) {
        int i_node_rank = atoi(node_rank);
        if (i_node_rank == rank) {
          printf("[%d] round %d, value %s\n", rank, i_round, itr.val());
          if (strcmp("allreduce", itr.val())) record(allReduce);
          else if (strcmp("broadcast", itr.val())) record(broadcast);
          else if (strcmp("loadcheckpoint", itr.val())) record(loadCheckpoint);
          else if (strcmp("checkpoint", itr.val())) record(checkpoint);
        }
      }
    }
  }

  inline void record(std::map<int,bool>& m) {
    m[rank] = false;
  }

  inline bool verify(std::map<int,bool>& m) {
    bool result = true;
    if (m.find(rank) != m.end()) {
      result = m[rank];
    }
    return result;
  }

  int rank;
  std::map<int,bool> allReduce;
  std::map<int,bool> broadcast;
  std::map<int,bool> loadCheckpoint;
  std::map<int,bool> checkpoint;
};

}

#endif  // ALLREDUCE_MOCK_H
