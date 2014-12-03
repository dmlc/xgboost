#ifndef RABIT_MOCK_H
#define RABIT_MOCK_H
/*!
 * \file mock.h
 * \brief This file defines a mock object to test the system
 * \author Ignacio Cano
 */
#include "./rabit.h"
#include "./config.h"
#include <map>
#include <sstream>
#include <fstream>

namespace rabit {
/*! \brief namespace of mock */
namespace test {

class Mock {


public:

  explicit Mock(const int& rank, char *config, char* round_dir) : rank(rank) {
    Init(config, round_dir);
  }

  template<typename OP>
  inline void Allreduce(float *sendrecvbuf, size_t count) {
    utils::Assert(verify(allReduce), "[%d] error when calling allReduce", rank);
    rabit::Allreduce<OP>(sendrecvbuf, count);
  }

  inline bool LoadCheckPoint(utils::ISerializable *p_model) {
    utils::Assert(verify(loadCheckpoint), "[%d] error when loading checkpoint", rank);
    return rabit::LoadCheckPoint(p_model);
  }

  inline void CheckPoint(const utils::ISerializable &model) {
    utils::Assert(verify(checkpoint), "[%d] error when checkpointing", rank);
    rabit::CheckPoint(model);
  }

  inline void Broadcast(std::string *sendrecv_data, int root) {
    utils::Assert(verify(broadcast), "[%d] error when broadcasting", rank);
    rabit::Broadcast(sendrecv_data, root);

  }

private:

  inline void Init(char* config, char* round_dir) {
    std::stringstream ss;
    ss << round_dir << "node" << rank << ".round";
    const char* round_file = ss.str().c_str();
    std::ifstream ifs(round_file);
    int current_round = 1;
    if (!ifs.good()) {
      // file does not exists, it's the first time, so save the current round to 1
      std::ofstream ofs(round_file);
      ofs << current_round;
      ofs.close();
    } else {
      // file does exists, read the previous round, increment by one, and save it back
      ifs >> current_round;
      current_round++;
      ifs.close();
      std::ofstream ofs(round_file);
      ofs << current_round;
      ofs.close();
    }
    printf("[%d] in round %d\n", rank, current_round);
    utils::ConfigIterator itr(config);
    while (itr.Next()) {
      char round[4], node_rank[4];
      sscanf(itr.name(), "%[^_]_%s", round, node_rank);
      int i_node_rank = atoi(node_rank);
      // if it's something for me
      if (i_node_rank == rank) {
        int i_round = atoi(round);
        // in my current round
        if (i_round == current_round) {
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

}  // namespace test
}  // namespace rabit

#endif  // RABIT_MOCK_H
