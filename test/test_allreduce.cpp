#include <rabit.h>
#include <utils.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mock.h>

using namespace rabit;

inline void TestMax(test::Mock &mock, size_t n) {
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  
  std::vector<float> ndata(n);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % 111;
  }
  mock.Allreduce<op::Max>(&ndata[0], ndata.size());  
  for (size_t i = 0; i < ndata.size(); ++i) {
    float rmax = (i * 1) % 111;
    for (int r = 0; r < nproc; ++r) {
      rmax = std::max(rmax, (float)((i * (r+1)) % 111));
    }
    utils::Check(rmax == ndata[i], "[%d] TestMax check failure", rank);
  }
}

inline void TestSum(test::Mock &mock, size_t n) {
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  const int z = 131;

  std::vector<float> ndata(n);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % z;
  }
  mock.Allreduce<op::Sum>(&ndata[0], ndata.size());  
  for (size_t i = 0; i < ndata.size(); ++i) {
    float rsum = 0.0f;
    for (int r = 0; r < nproc; ++r) {
      rsum += (float)((i * (r+1)) % z);
    }
    utils::Check(fabsf(rsum - ndata[i]) < 1e-5 ,
                 "[%d] TestSum check failure, local=%g, allreduce=%g", rank, rsum, ndata[i]);
  }
}

inline void TestBcast(test::Mock &mock, size_t n, int root) {
  int rank = rabit::GetRank();
  std::string s; s.resize(n);      
  for (size_t i = 0; i < n; ++i) {
    s[i] = char(i % 126 + 1);
  }
  std::string res;
  if (root == rank) {
    res = s;
    mock.Broadcast(&res, root);
  } else {
    mock.Broadcast(&res, root);
  }
  utils::Check(res == s, "[%d] TestBcast fail", rank);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: <ndata> <config>\n");
    return 0;
  }
  int n = atoi(argv[1]);
  rabit::Init(argc, argv);
  int rank = rabit::GetRank();
  int nproc = rabit::GetWorldSize();
  std::string name = rabit::GetProcessorName();

  test::Mock mock(rank, argv[2], argv[3]);

  utils::LogPrintf("[%d] start at %s\n", rank, name.c_str());
  TestMax(mock, n);
  utils::LogPrintf("[%d] !!!TestMax pass\n", rank);
  TestSum(mock, n);
  utils::LogPrintf("[%d] !!!TestSum pass\n", rank);
  int step = std::max(nproc / 3, 1);
  for (int i = 0; i < nproc; i += step) {    
    TestBcast(mock, n, i);
  }
  utils::LogPrintf("[%d] !!!TestBcast pass\n", rank);
  rabit::Finalize();
  printf("[%d] all check pass\n", rank);
  return 0;
}
