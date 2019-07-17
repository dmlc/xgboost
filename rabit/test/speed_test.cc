// This program is used to test the speed of rabit API
#include <rabit/rabit.h>
#include <rabit/internal/timer.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>

using namespace rabit;

double max_tdiff, sum_tdiff, bcast_tdiff, tot_tdiff;

inline void TestMax(size_t n) {
  int rank = rabit::GetRank();
  std::vector<float> ndata(n);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % 111;
  }
  double tstart = utils::GetTime();
  rabit::Allreduce<op::Max>(&ndata[0], ndata.size());
  max_tdiff += utils::GetTime() - tstart;
}

inline void TestSum(size_t n) {
  int rank = rabit::GetRank();
  const int z = 131;
  std::vector<float> ndata(n);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % z;
  }
  double tstart = utils::GetTime();
  rabit::Allreduce<op::Sum>(&ndata[0], ndata.size());
  sum_tdiff += utils::GetTime() - tstart;
}

inline void TestBcast(size_t n, int root) {
  int rank = rabit::GetRank();
  std::string s; s.resize(n);
  for (size_t i = 0; i < n; ++i) {
    s[i] = char(i % 126 + 1);
  }
  std::string res;
  res.resize(n);
  if (root == rank) {
    res = s;
  }
  double tstart = utils::GetTime();
  rabit::Broadcast(&res[0], res.length(), root);
  bcast_tdiff += utils::GetTime() - tstart;
}

inline void PrintStats(const char *name, double tdiff, int n, int nrep, size_t size) {
  int nproc = rabit::GetWorldSize();
  double tsum = tdiff;
  rabit::Allreduce<op::Sum>(&tsum, 1);
  double tavg = tsum / nproc;
  double tsqr = tdiff - tavg;
  tsqr *= tsqr;
  rabit::Allreduce<op::Sum>(&tsqr, 1);
  double tstd = sqrt(tsqr / nproc);
  if (rabit::GetRank() == 0) {
    rabit::TrackerPrintf("%s: mean=%g, std=%g sec\n", name, tavg, tstd);
    double ndata = n;
    ndata *= nrep * size;
    if (n != 0) {
      rabit::TrackerPrintf("%s-speed: %g MB/sec\n", name, (ndata / tavg) / 1024 / 1024 );
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: <ndata> <nrepeat>\n");
    return 0;
  }
  srand(0);
  int n = atoi(argv[1]);
  int nrep = atoi(argv[2]);
  utils::Check(nrep >= 1, "need to at least repeat running once");
  rabit::Init(argc, argv);
  //int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  std::string name = rabit::GetProcessorName();
  max_tdiff = sum_tdiff = bcast_tdiff = 0;
  double tstart = utils::GetTime();
  for (int i = 0; i < nrep; ++i) {
    TestMax(n);
    TestSum(n);
    TestBcast(n, rand() % nproc);
  }
  tot_tdiff = utils::GetTime() - tstart;
  // use allreduce to get the sum and std of time
  PrintStats("max_tdiff", max_tdiff, n, nrep, sizeof(float));
  PrintStats("sum_tdiff", sum_tdiff, n, nrep, sizeof(float));
  PrintStats("bcast_tdiff", bcast_tdiff, n, nrep, sizeof(char));
  PrintStats("tot_tdiff", tot_tdiff, 0, nrep, sizeof(float));
  rabit::Finalize();
  return 0;
}
