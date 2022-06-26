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
#include <dmlc/timer.h>
#include "rabit/internal/engine.h"
#include "allreduce_base.h"

namespace rabit {
namespace engine {
class AllreduceMock : public AllreduceBase {
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
    AllreduceBase::SetParam(name, val);
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
                 void *prepare_arg) override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "AllReduce");
    double tstart = dmlc::GetTime();
    AllreduceBase::Allreduce(sendrecvbuf_, type_nbytes, count, reducer,
                             prepare_fun, prepare_arg);
    tsum_allreduce_ += dmlc::GetTime() - tstart;
  }
  void Allgather(void *sendrecvbuf, size_t total_size, size_t slice_begin,
                 size_t slice_end, size_t size_prev_slice) override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "Allgather");
    double tstart = dmlc::GetTime();
    AllreduceBase::Allgather(sendrecvbuf, total_size, slice_begin, slice_end,
                             size_prev_slice);
    tsum_allgather_ += dmlc::GetTime() - tstart;
  }
  void Broadcast(void *sendrecvbuf_, size_t total_size, int root) override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "Broadcast");
    AllreduceBase::Broadcast(sendrecvbuf_, total_size, root);
  }
  int LoadCheckPoint() override {
    tsum_allreduce_ = 0.0;
    tsum_allgather_ = 0.0;
    time_checkpoint_ = dmlc::GetTime();
    if (force_local_ == 0) {
      return AllreduceBase::LoadCheckPoint();
    } else {
      return AllreduceBase::LoadCheckPoint();
    }
  }
  void CheckPoint() override {
    this->Verify(MockKey(rank, version_number, seq_counter, num_trial_), "CheckPoint");
    double tstart = dmlc::GetTime();
    double tbet_chkpt = tstart - time_checkpoint_;
    AllreduceBase::CheckPoint();
    time_checkpoint_ = dmlc::GetTime();
    double tcost = dmlc::GetTime() - tstart;
    if (report_stats_ != 0 && rank == 0) {
      std::stringstream ss;
      ss << "[v" << version_number << "] global_size="
         << ",check_tcost="<< tcost <<" sec"
         << ",allreduce_tcost=" << tsum_allreduce_ << " sec"
         << ",allgather_tcost=" << tsum_allgather_ << " sec"
         << ",between_chpt=" << tbet_chkpt << "sec\n";
      this->TrackerPrint(ss.str());
    }
    tsum_allreduce_ = 0.0;
    tsum_allgather_ = 0.0;
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
      throw dmlc::Error(std::to_string(rank) + "@@@Hit Mock Error: " + name);
    }
  }
};
}  // namespace engine
}  // namespace rabit
#endif  // RABIT_ALLREDUCE_MOCK_H_
