#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
#include <ctime>
#include <utils/group_data.h>
#include <utils/random.h>
#include <utils/omp.h>
#include <utils/utils.h>

using namespace xgboost::utils;
using namespace xgboost;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: <nkey> <ndata> pnthread]\n");
    return 0;
  }
  if (argc > 3) {
    omp_set_num_threads(atoi(argv[3]));
  }
  random::Seed(0);
  unsigned nkey = static_cast<unsigned>(atoi(argv[1]));
  size_t ndata = static_cast<size_t>(atol(argv[2]));
  
  std::vector<unsigned> keys;
  std::vector< std::pair<unsigned, unsigned> > raw;
  raw.reserve(ndata); keys.reserve(ndata);
  for (size_t i = 0; i < ndata; ++i) {
    unsigned key = random::NextUInt32(nkey);
    utils::Check(key < nkey, "key exceed bound\n");
    raw.push_back(std::make_pair(key, i));
    keys.push_back(key);
  }
  printf("loading finish, start working\n");
  time_t start_t = time(NULL);
  int nthread;
  #pragma omp parallel
  {
    nthread = omp_get_num_threads();
  }
  std::vector<size_t> rptr;
  std::vector<unsigned> data;
  ParallelGroupBuilder<unsigned> builder(&rptr, &data);
  builder.InitBudget(0, nthread);

  size_t nstep = (raw.size() +nthread-1)/ nthread;
  #pragma omp parallel
  {
    int tid = omp_get_thread_num(); 
    size_t begin = tid * nstep;
    size_t end = std::min((tid + 1) * nstep, raw.size());
    for (size_t i = begin; i < end; ++i) {
      builder.AddBudget(raw[i].first, tid);
    }
  }
  double first_cost = time(NULL) - start_t;
  builder.InitStorage();  

  #pragma omp parallel
  {
    int tid = omp_get_thread_num(); 
    size_t begin = tid * nstep;
    size_t end = std::min((tid + 1)* nstep, raw.size());
    for (size_t i = begin; i < end; ++i) {
      builder.Push(raw[i].first, raw[i].second, tid);
    }
  }

  double second_cost = time(NULL) - start_t;
  printf("all finish, phase1=%g sec, phase2=%g sec\n", first_cost, second_cost);
  Check(rptr.size() <= nkey+1, "nkey exceed bound");
  Check(rptr.back() == ndata, "data shape inconsistent");
  for (size_t i = 0; i < rptr.size()-1; ++ i) {
    Check(rptr[i] <= rptr[i+1], "rptr error");
    for (size_t j = rptr[i]; j < rptr[i+1]; ++j) {
      unsigned pos = data[j];
      Check(pos < keys.size(), "invalid pos");
      Check(keys[pos] == i, "invalid key entry");
    }
  }
  printf("all check pass\n");
  return 0;
}
