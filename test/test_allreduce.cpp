#include <sync/sync.h>
#include <utils/utils.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace xgboost;

inline void TestMax(size_t n) {
  int rank = sync::GetRank();
  int nproc = sync::GetWorldSize();
  
  std::vector<float> ndata(n);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = (i * (rank+1)) % 111;
  }
  sync::AllReduce(&ndata[0], ndata.size(), sync::kMax);  
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
  sync::AllReduce(&ndata[0], ndata.size(), sync::kSum);  
  for (size_t i = 0; i < ndata.size(); ++i) {
    float rsum = 0.0f;
    for (int r = 0; r < nproc; ++r) {
      rsum += (float)((i * (r+1)) % z);
    }
    utils::Check(fabsf(rsum - ndata[i]) < 1e-5 ,
                 "[%d] TestSum check failure, local=%g, allreduce=%g", rank, rsum, ndata[i]);
  }
}

struct Rec {
  double rmax;
  double rmin;
  double rsum;
  Rec() {}
  Rec(double r) {
    rmax = rmin = rsum = r;
  }
  inline void Reduce(const Rec &b) {
    rmax = std::max(b.rmax, rmax);
    rmin = std::max(b.rmin, rmin);
    rsum += b.rsum;
  }
  inline void CheckSameAs(const Rec &b) {
    if (rmax != b.rmax || rmin != b.rmin || fabs(rsum - b.rsum) > 1e-6) {
      utils::Error("[%d] TestReducer check failure", sync::GetRank());
    }
  }
};

inline void TestReducer(int n) {
  int rank = sync::GetRank();
  int nproc = sync::GetWorldSize();
  const int z = 131;
  sync::Reducer<Rec> red;
  std::vector<Rec> ndata(n);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = Rec((i * (rank+1)) % z);
  }
  red.AllReduce(&ndata[0], ndata.size());  
                
  for (size_t i = 0; i < ndata.size(); ++i) {
    Rec rec((i * 1) % z);
    for (int r = 1; r < nproc; ++r) {
      rec.Reduce(Rec((i * (r+1)) % z));
    }
    rec.CheckSameAs(ndata[i]);
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

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: <ndata>\n");
    return 0;
  }
  int n = atoi(argv[1]);
  sync::Init(argc, argv);
  int rank = sync::GetRank();
  //int nproc = sync::GetWorldSize();
  std::string name = sync::GetProcessorName();
  printf("[%d] start at %s\n", rank, name.c_str());
  TestMax(n);
  TestSum(n);
  TestReducer(n);
  sync::Finalize();
  printf("[%d] all check pass\n", rank);
  return 0;
}
