#ifndef ALLREDUCE_MOCK_H
#define ALLREDUCE_MOCK_H
/*!
 * \file mock.h
 * \brief This file defines a mock object to test the system
 * \author Tianqi Chen, Nacho, Tianyi
 */
#include "./engine.h"
#include "./utils.h"
#include <queue>
 #include <map>


/*! \brief namespace of mock */
namespace test {

class Mock {

  typedef std::map<int,std::queue<int> > Map;

public:

  Mock() : record(true) {}

  inline void Replay() {
    record = false;
  }

  // record methods

  inline void OnAllReduce(int rank, int code) {
    utils::Check(record, "Not in record state");
    Map::iterator it = allReduce.find(rank);
    if (it == allReduce.end()) {
      std::queue<int> aQueue;
      allReduce[rank] = aQueue;
    }
    allReduce[rank].push(code);
  }

  inline void OnBroadcast() {
    utils::Check(record, "Not in record state");
  }

  inline void OnLoadCheckpoint() {
    utils::Check(record, "Not in record state");
  }

  inline void OnCheckpoint() {
    utils::Check(record, "Not in record state");
  }


  // replay methods

  inline int AllReduce(int rank) {
    utils::Check(!record, "Not in replay state");
    utils::Check(allReduce.find(rank) != allReduce.end(), "Not recorded");
    int result = 0;
    if (!allReduce[rank].empty()) {
      result = allReduce[rank].front();
      allReduce[rank].pop();
    }
    return result;
  }

  inline int Broadcast(int rank) {
    utils::Check(!record, "Not in replay state");
    return 0;
  }

  inline int LoadCheckpoint(int rank) {
    utils::Check(!record, "Not in replay state");
    return 0;
  }

  inline int Checkpoint(int rank) {
    utils::Check(!record, "Not in replay state");
    return 0;
  }


private:

  // flag to indicate if the mock is in record state
  bool record;

  Map allReduce;
  Map broadcast;
  Map loadCheckpoint;
  Map checkpoint;
};

}

#endif  // ALLREDUCE_MOCK_H
