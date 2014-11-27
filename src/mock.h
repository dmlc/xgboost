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

  typedef std::map<int,std::queue<bool> > Map;

public:

  Mock() : record(true) {}

  inline void Replay() {
    record = false;
  }

  // record methods
  inline void OnAllReduce(int rank, bool success) {
    onRecord(allReduce, rank, success);
  }

  inline void OnBroadcast(int rank, bool success) {
    onRecord(broadcast, rank, success);
  }

  inline void OnLoadCheckPoint(int rank, bool success) {
    onRecord(loadCheckpoint, rank, success);
  }

  inline void OnCheckPoint(int rank, bool success) {
    onRecord(checkpoint, rank, success);
  }


  // replay methods
  inline bool AllReduce(int rank) {
    return onReplay(allReduce, rank);
  }

  inline bool Broadcast(int rank) {
    return onReplay(broadcast, rank);
  }

  inline bool LoadCheckPoint(int rank) {
    return onReplay(loadCheckpoint, rank);  
  }

  inline bool CheckPoint(int rank) {
    return onReplay(checkpoint, rank);  
  }


private:

  inline void onRecord(Map& m, int rank, bool success) {
    utils::Check(record, "Not in record state");
    Map::iterator it = m.find(rank);
    if (it == m.end()) {
      std::queue<bool> aQueue;
      m[rank] = aQueue;
    }
    m[rank].push(success);
  }

  inline bool onReplay(Map& m, int rank) {
    utils::Check(!record, "Not in replay state");
    utils::Check(m.find(rank) != m.end(), "Not recorded");
    bool result = true;
    if (!m[rank].empty()) {
      result = m[rank].front();
      m[rank].pop();
    }
    return result;
  }

  // flag to indicate if the mock is in record state
  bool record;

  Map allReduce;
  Map broadcast;
  Map loadCheckpoint;
  Map checkpoint;
};

}

#endif  // ALLREDUCE_MOCK_H
