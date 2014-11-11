#include <vector>
#include <utils/quantile.h>
using namespace xgboost;

int main(int argc, char *argv[]) {  
  utils::WQuantileSketch<float, float> sketch;
  size_t n;
  double wsum = 0.0;
  float eps, x, w;
  utils::Check(scanf("%lu%f", &n, &eps) == 2, "needs to start with n eps");
  sketch.Init(n, eps);
  printf("nlevel = %lu, limit_size=%lu\n", sketch.nlevel, sketch.limit_size);
  while (scanf("%f%f", &x, &w) == 2) {
    sketch.Push(x, w);
    wsum += w;
  }
  sketch.CheckValid(0.1);
  utils::WQuantileSketch<float, float>::SummaryContainer out;
  sketch.GetSummary(&out);
  printf("MaxError=%f/%f = %g\n", out.MaxError(), wsum, out.MaxError() / wsum);
  out.Print();
  return 0;
}
