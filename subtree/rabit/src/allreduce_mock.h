/*!
 * \file allreduce_mock.h
 * \brief Mock test module of AllReduce engine,
 * insert failures in certain call point, to test if the engine is robust to failure
 * 
 * \author Ignacio Cano, Tianqi Chen
 */
#ifndef RABIT_ALLREDUCE_MOCK_H
#define RABIT_ALLREDUCE_MOCK_H
#include <vector>
#include <map>
#include "../include/rabit/engine.h"
#include "./allreduce_robust.h"

namespace rabit {
namespace engine {
class AllreduceMock : public AllreduceRobust {
 public:
  // constructor
  AllreduceMock(void) {
    num_trial = 0;
  }
  // destructor
  virtual ~AllreduceMock(void) {}
  virtual void SetParam(const char *name, const char *val) {
    AllreduceRobust::SetParam(name, val);
    // additional parameters
    if (!strcmp(name, "rabit_num_trial")) num_trial = atoi(val);
    if (!strcmp(name, "mock")) {
      MockKey k;
      utils::Check(sscanf(val, "%d,%d,%d,%d",
                          &k.rank, &k.version, &k.seqno, &k.ntrial) == 4,
                   "invalid mock parameter");
      mock_map[k] = 1;
    }
  }
  virtual void Allreduce(void *sendrecvbuf_,
                         size_t type_nbytes,
                         size_t count,
                         ReduceFunction reducer,
                         PreprocFunction prepare_fun,
                         void *prepare_arg) {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial), "AllReduce");
    AllreduceRobust::Allreduce(sendrecvbuf_, type_nbytes,
                               count, reducer, prepare_fun, prepare_arg);
  }
  virtual void Broadcast(void *sendrecvbuf_, size_t total_size, int root) {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial), "Broadcast");
    AllreduceRobust::Broadcast(sendrecvbuf_, total_size, root);
  }
  virtual void CheckPoint(const ISerializable *global_model,
                          const ISerializable *local_model) {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial), "CheckPoint");
    AllreduceRobust::CheckPoint(global_model, local_model);
  }

  virtual void LazyCheckPoint(const ISerializable *global_model) {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial), "LazyCheckPoint");
    AllreduceRobust::LazyCheckPoint(global_model);
  }
  
 private:
  // key to identify the mock stage
  struct MockKey {
    int rank;
    int version;
    int seqno;
    int ntrial;
    MockKey(void) {}
    MockKey(int rank, int version, int seqno, int ntrial) 
        : rank(rank), version(version), seqno(seqno), ntrial(ntrial) {}
    inline bool operator==(const MockKey &b) const {
      return rank == b.rank && 
          version == b.version &&
          seqno == b.seqno &&
          ntrial == b.ntrial;
    }
    inline bool operator<(const MockKey &b) const {
      if (rank != b.rank) return rank < b.rank;
      if (version != b.version) return version < b.version;
      if (seqno != b.seqno) return seqno < b.seqno;
      return ntrial < b.ntrial;
    }
  };
  // number of failure trials
  int num_trial;
  // record all mock actions
  std::map<MockKey, int> mock_map;
  // used to generate all kinds of exceptions
  inline void Verify(const MockKey &key, const char *name) {
    if (mock_map.count(key) != 0) {
      num_trial += 1;
      fprintf(stderr, "[%d]@@@Hit Mock Error:%s\n", rank, name);
      exit(-2);
    }
  }
};
}  // namespace engine
}  // namespace rabit
#endif // RABIT_ALLREDUCE_MOCK_H
