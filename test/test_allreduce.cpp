#include <allreduce.h>
#include <utils.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mock.h>


using namespace sync;

inline void TestMax(size_t n) {
  int rank = sync::GetRank();
  int nproc = sync::GetWorldSize();
  
  std::vector<float> ndata(n);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % 111;
  }
  sync::AllReduce<op::Max>(&ndata[0], ndata.size());  
  for (size_t i = 0; i < ndata.size(); ++i) {
    float rmax = (i * 1) % 111;
    for (int r = 0; r < nproc; ++r) {
      rmax = std::max(rmax, (float)((i * (r+1)) % 111));
    }
    utils::Check(rmax == ndata[i], "[%d] TestMax check failure", rank);
  }
}

inline void TestSum(size_t n) {
  int rank = sync::GetRank();
  int nproc = sync::GetWorldSize();
  const int z = 131;

  std::vector<float> ndata(n);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % z;
  }
  sync::AllReduce<op::Sum>(&ndata[0], ndata.size());  
  for (size_t i = 0; i < ndata.size(); ++i) {
    float rsum = 0.0f;
    for (int r = 0; r < nproc; ++r) {
      rsum += (float)((i * (r+1)) % z);
    }
    utils::Check(fabsf(rsum - ndata[i]) < 1e-5 ,
                 "[%d] TestSum check failure, local=%g, allreduce=%g", rank, rsum, ndata[i]);
  }
}

inline void TestBcast(size_t n, int root) {
  int rank = sync::GetRank();
  std::string s; s.resize(n);      
  for (size_t i = 0; i < n; ++i) {
    s[i] = char(i % 126 + 1);
  }
  std::string res;
  if (root == rank) {
    res = s;
    sync::Bcast(&res, root);
  } else {
    sync::Bcast(&res, root);
  }
  utils::Check(res == s, "[%d] TestBcast fail", rank);
}

// ugly stuff, just to see if it works
inline void record(test::Mock& mock, int rank) {
  switch(rank) {
    case 0:
      mock.OnAllReduce(0, -1);
      break;
    case 1: 
      mock.OnAllReduce(1, -1);
      break;
    case 2:
      mock.OnAllReduce(2, 0);
      break;
  }
}

// to be removed, should be added in engine tcp
inline void replay(test::Mock& mock, int rank) {
  printf("[%d] All reduce %d\n", rank, mock.AllReduce(rank));
  printf("[%d] All reduce %d\n", rank, mock.AllReduce(rank));
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: <ndata>\n");
    return 0;
  }
  int n = atoi(argv[1]);
  sync::Init(argc, argv);
  int rank = sync::GetRank();
  std::string name = sync::GetProcessorName();

  #ifdef TEST
    test::Mock mock;
    record(mock, rank);
    mock.Replay();
    replay(mock, rank);
  #endif

  printf("[%d] start at %s\n", rank, name.c_str());
  TestMax(n);
  printf("[%d] TestMax pass\n", rank);
  TestSum(n);
  printf("[%d] TestSum pass\n", rank);
  sync::Finalize();
  printf("[%d] all check pass\n", rank);
  return 0;
}
