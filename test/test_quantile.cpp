#include <vector>
#include <utils/quantile.h>
#include <ctime>
using namespace xgboost;


struct Entry {
  double x, w, rmin;
  inline bool operator<(const Entry &e) const {
    return x < e.x;
  }
};

inline void MakeQuantile(std::vector<Entry> &dat) {
  std::sort(dat.begin(), dat.end());
  size_t top = 0;
  double wsum = 0.0;
  for (size_t i = 0; i < dat.size();) {
    size_t j = i + 1;
    for (;j < dat.size() && dat[i].x == dat[j].x; ++j) {
      dat[i].w += dat[j].w;
    }
    dat[top] = dat[i];
    dat[top].rmin = wsum;
    wsum += dat[top].w;
    ++top;
    i = j;
  }
  dat.resize(top);
}

template<typename Summary>
inline void verifyWQ(std::vector<Entry> &dat, Summary out) {
 MakeQuantile(dat);
 size_t j = 0;
 double err = 0.0;
 const double eps = 1e-4;
 for (size_t i = 0; i < out.size; ++i) {
   while (j < dat.size() && dat[j].x < out.data[i].value) ++j;
   utils::Assert(j < dat.size() && fabs(dat[j].x - out.data[i].value) < eps, "bug");
   err = std::min(dat[j].rmin - out.data[i].rmin, err);
   err = std::min(out.data[i].rmax - dat[j].rmin + dat[j].w, err);
   err = std::min(dat[j].w - out.data[i].wmin, err);
 }
 if (err < 0.0) err = -err;
 printf("verify correctness, max-constraint-violation=%g (0 means perfect, coubld be nonzero due to floating point)\n", err);
}

template<typename Sketch, typename RType>
inline typename Sketch::SummaryContainer test(std::vector<Entry> &dat) {
  Sketch sketch;
  size_t n;
  double wsum = 0.0;
  float eps;
  utils::Check(scanf("%lu%f", &n, &eps) == 2, "needs to start with n eps");
  sketch.Init(n, eps);
  Entry e;
  while (scanf("%lf%lf", &e.x, &e.w) == 2) {
    dat.push_back(e);
    wsum += e.w;
  }
  clock_t start = clock();
  for (size_t i = 0; i < dat.size(); ++i) {
    sketch.Push(dat[i].x, dat[i].w);
  }
  double tcost = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
  typename Sketch::SummaryContainer out;
  sketch.GetSummary(&out); 
  double maxerr = static_cast<double>(out.MaxError());
  out.Print();
  printf("-------------------------\n");
  printf("timecost=%g sec\n", tcost);
  printf("MaxError=%g/%g = %g\n", maxerr, wsum, maxerr / wsum);
  printf("maxlevel = %lu, usedlevel=%lu, limit_size=%lu\n", sketch.nlevel, sketch.level.size(), sketch.limit_size);
  return out;
}

int main(int argc, char *argv[]) {
  const char *method = "wq";
  if (argc > 1) method = argv[1];
  std::vector<Entry> dat;
  if (!strcmp(method, "wq")) {
    verifyWQ(dat, test<utils::WQuantileSketch<float, float>, float>(dat));
  }
  if (!strcmp(method, "wx")) {
    verifyWQ(dat, test<utils::WXQuantileSketch<float, float>, float>(dat));
  }
  if (!strcmp(method, "gk")) {
    test<utils::GKQuantileSketch<float, unsigned>, unsigned>(dat);
  }
  return 0;
}
