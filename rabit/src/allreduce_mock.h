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
#include "rabit/internal/engine.h"
#include "rabit/internal/timer.h"
#include "allreduce_robust.h"

namespace rabit {
namespace engine {
class AllreduceMock : public AllreduceRobust {
 public:
  // constructor
  AllreduceMock() {
    num_trial_ = 0;
    force_local_ = 0;
    report_stats_ = 0;
    tsum_allreduce_ = 0.0;
    tsum_allgather_ = 0.0;
  }
  // destructor
  ~AllreduceMock() override = default;
  void SetParam(const char *name, const char *val) override {
    AllreduceRobust::SetParam(name, val);
    // additional parameters
    if (!strcmp(name, "rabit_num_trial")) num_trial_ = atoi(val);
    if (!strcmp(name, "DMLC_NUM_ATTEMPT")) num_trial_ = atoi(val);
    if (!strcmp(name, "report_stats")) report_stats_ = atoi(val);
    if (!strcmp(name, "force_local")) force_local_ = atoi(val);
    if (!strcmp(name, "mock")) {
      MockKey k;
      utils::Check(sscanf(val, "%d,%d,%d,%d",
                          &k.rank, &k.version, &k.seqno, &k.ntrial) == 4,
                   "invalid mock parameter");
      mock_map_[k] = 1;
    }
  }
  void Allreduce(void *sendrecvbuf_, size_t type_nbytes, size_t count,
                 ReduceFunction reducer, PreprocFunction prepare_fun,
                 void *prepare_arg, const char *_file = _FILE,
                 const int _line = _LINE,
                 const char *_caller = _CALLER) override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "AllReduce");
    double tstart = utils::GetTime();
    AllreduceRobust::Allreduce(sendrecvbuf_, type_nbytes,
                               count, reducer, prepare_fun, prepare_arg,
                               _file, _line, _caller);
    tsum_allreduce_ += utils::GetTime() - tstart;
  }
  void Allgather(void *sendrecvbuf, size_t total_size, size_t slice_begin,
                 size_t slice_end, size_t size_prev_slice,
                 const char *_file = _FILE, const int _line = _LINE,
                 const char *_caller = _CALLER) override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "Allgather");
    double tstart = utils::GetTime();
    AllreduceRobust::Allgather(sendrecvbuf, total_size,
                                   slice_begin, slice_end,
                                   size_prev_slice, _file, _line, _caller);
    tsum_allgather_ += utils::GetTime() - tstart;
  }
  void Broadcast(void *sendrecvbuf_, size_t total_size, int root,
                 const char *_file = _FILE, const int _line = _LINE,
                 const char *_caller = _CALLER) override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "Broadcast");
    AllreduceRobust::Broadcast(sendrecvbuf_, total_size, root, _file, _line, _caller);
  }
  int LoadCheckPoint(Serializable *global_model,
                     Serializable *local_model) override {
    tsum_allreduce_ = 0.0;
    tsum_allgather_ = 0.0;
    time_checkpoint_ = utils::GetTime();
    if (force_local_ == 0) {
      return AllreduceRobust::LoadCheckPoint(global_model, local_model);
    } else {
      DummySerializer dum;
      ComboSerializer com(global_model, local_model);
      return AllreduceRobust::LoadCheckPoint(&dum, &com);
    }
  }
  void CheckPoint(const Serializable *global_model,
                  const Serializable *local_model) override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "CheckPoint");
    double tstart = utils::GetTime();
    double tbet_chkpt = tstart - time_checkpoint_;
    if (force_local_ == 0) {
      AllreduceRobust::CheckPoint(global_model, local_model);
    } else {
      DummySerializer dum;
      ComboSerializer com(global_model, local_model);
      AllreduceRobust::CheckPoint(&dum, &com);
    }
    time_checkpoint_ = utils::GetTime();
    double tcost = utils::GetTime() - tstart;
    if (report_stats_ != 0 && rank == 0) {
      std::stringstream ss;
      ss << "[v" << version_number << "] global_size=" << global_checkpoint_.length()
         << ",local_size=" << (local_chkpt_[0].length() + local_chkpt_[1].length())
         << ",check_tcost="<< tcost <<" sec"
         << ",allreduce_tcost=" << tsum_allreduce_ << " sec"
         << ",allgather_tcost=" << tsum_allgather_ << " sec"
         << ",between_chpt=" << tbet_chkpt << "sec\n";
      this->TrackerPrint(ss.str());
    }
    tsum_allreduce_ = 0.0;
    tsum_allgather_ = 0.0;
  }

  void LazyCheckPoint(const Serializable *global_model) override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "LazyCheckPoint");
    AllreduceRobust::LazyCheckPoint(global_model);
  }

 protected:
  // force checkpoint to local
  int force_local_;
  // whether report statistics
  int report_stats_;
  // sum of allreduce
  double tsum_allreduce_;
  // sum of allgather
  double tsum_allgather_;
  double time_checkpoint_;

 private:
  struct DummySerializer : public Serializable {
    void Load(Stream *fi) override {}
    void Save(Stream *fo) const override {}
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
        : lhs(nullptr), rhs(nullptr), c_lhs(lhs), c_rhs(rhs) {
    }
    void Load(Stream *fi) override {
      if (lhs != nullptr) lhs->Load(fi);
      if (rhs != nullptr) rhs->Load(fi);
    }
    void Save(Stream *fo) const override {
      if (c_lhs != nullptr) c_lhs->Save(fo);
      if (c_rhs != nullptr) c_rhs->Save(fo);
    }
  };
  // key to identify the mock stage
  struct MockKey {
    int rank;
    int version;
    int seqno;
    int ntrial;
    MockKey() = default;
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
  int num_trial_;
  // record all mock actions
  std::map<MockKey, int> mock_map_;
  // used to generate all kinds of exceptions
  inline void Verify(const MockKey &key, const char *name) {
    if (mock_map_.count(key) != 0) {
      num_trial_ += 1;
      // data processing frameworks runs on shared process
      error_("[%d]@@@Hit Mock Error:%s ", rank, name);
    }
  }
};
}  // namespace engine
}  // namespace rabit
#endif  // RABIT_ALLREDUCE_MOCK_H_
