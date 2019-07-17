/*!
 * Copyright by Contributors
 * \file allreduce_mock.h
 * \brief Mock test module of AllReduce engine,
 * insert failures in certain call point, to test if the engine is robust to failure
 *
 * \author Ignacio Cano, Tianqi Chen
 */
#ifndef RABIT_ALLREDUCE_MOCK_H_
#define RABIT_ALLREDUCE_MOCK_H_
#include <vector>
#include <map>
#include <sstream>
#include "../include/rabit/internal/engine.h"
#include "../include/rabit/internal/timer.h"
#include "./allreduce_robust.h"

namespace rabit {
namespace engine {
class AllreduceMock : public AllreduceRobust {
 public:
  // constructor
  AllreduceMock(void) {
    num_trial = 0;
    force_local = 0;
    report_stats = 0;
    tsum_allreduce = 0.0;
  }
  // destructor
  virtual ~AllreduceMock(void) {}
  virtual void SetParam(const char *name, const char *val) {
    AllreduceRobust::SetParam(name, val);
    // additional parameters
    if (!strcmp(name, "rabit_num_trial")) num_trial = atoi(val);
    if (!strcmp(name, "DMLC_NUM_ATTEMPT")) num_trial = atoi(val);
    if (!strcmp(name, "report_stats")) report_stats = atoi(val);
    if (!strcmp(name, "force_local")) force_local = atoi(val);
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
    double tstart = utils::GetTime();
    AllreduceRobust::Allreduce(sendrecvbuf_, type_nbytes,
                               count, reducer, prepare_fun, prepare_arg);
    tsum_allreduce += utils::GetTime() - tstart;
  }
  virtual void Broadcast(void *sendrecvbuf_, size_t total_size, int root) {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial), "Broadcast");
    AllreduceRobust::Broadcast(sendrecvbuf_, total_size, root);
  }
  virtual int LoadCheckPoint(Serializable *global_model,
                             Serializable *local_model) {
    tsum_allreduce = 0.0;
    time_checkpoint = utils::GetTime();
    if (force_local == 0) {
      return AllreduceRobust::LoadCheckPoint(global_model, local_model);
    } else {
      DummySerializer dum;
      ComboSerializer com(global_model, local_model);
      return AllreduceRobust::LoadCheckPoint(&dum, &com);
    }
  }
  virtual void CheckPoint(const Serializable *global_model,
                          const Serializable *local_model) {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial), "CheckPoint");
    double tstart = utils::GetTime();
    double tbet_chkpt = tstart - time_checkpoint;
    if (force_local == 0) {
      AllreduceRobust::CheckPoint(global_model, local_model);
    } else {
      DummySerializer dum;
      ComboSerializer com(global_model, local_model);
      AllreduceRobust::CheckPoint(&dum, &com);
    }
    time_checkpoint = utils::GetTime();
    double tcost = utils::GetTime() - tstart;
    if (report_stats != 0 && rank == 0) {
      std::stringstream ss;
      ss << "[v" << version_number << "] global_size=" << global_checkpoint.length()
         << ",local_size=" << (local_chkpt[0].length() + local_chkpt[1].length())
         << ",check_tcost="<< tcost <<" sec"
         << ",allreduce_tcost=" << tsum_allreduce << " sec"
         << ",between_chpt=" << tbet_chkpt << "sec\n";
      this->TrackerPrint(ss.str());
    }
    tsum_allreduce = 0.0;
  }

  virtual void LazyCheckPoint(const Serializable *global_model) {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial), "LazyCheckPoint");
    AllreduceRobust::LazyCheckPoint(global_model);
  }

 protected:
  // force checkpoint to local
  int force_local;
  // whether report statistics
  int report_stats;
  // sum of allreduce
  double tsum_allreduce;
  double time_checkpoint;

 private:
  struct DummySerializer : public Serializable {
    virtual void Load(Stream *fi) {
    }
    virtual void Save(Stream *fo) const {
    }
  };
  struct ComboSerializer : public Serializable {
    Serializable *lhs;
    Serializable *rhs;
    const Serializable *c_lhs;
    const Serializable *c_rhs;
    ComboSerializer(Serializable *lhs, Serializable *rhs)
        : lhs(lhs), rhs(rhs), c_lhs(lhs), c_rhs(rhs) {
    }
    ComboSerializer(const Serializable *lhs, const Serializable *rhs)
        : lhs(NULL), rhs(NULL), c_lhs(lhs), c_rhs(rhs) {
    }
    virtual void Load(Stream *fi) {
      if (lhs != NULL) lhs->Load(fi);
      if (rhs != NULL) rhs->Load(fi);
    }
    virtual void Save(Stream *fo) const {
      if (c_lhs != NULL) c_lhs->Save(fo);
      if (c_rhs != NULL) c_rhs->Save(fo);
    }
  };
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
#endif  // RABIT_ALLREDUCE_MOCK_H_
