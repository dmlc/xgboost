#include <sync/sync.h>

using namespace xgboost;

int main(int argc, char *argv[]) {
  sync::Init(argc, argv);
  int rank = sync::GetRank();
  std::string name = sync::GetProcessorName().c_str();
  printf("start %s rank=%d\n", name.c_str(), rank);

  std::vector<float> ndata(16);
  for (size_t i = 0; i < ndata.size(); ++i) {
    ndata[i] = i + rank;
  }
  sync::AllReduce(&ndata[0], ndata.size(), sync::kMax);
  sync::Finalize();
  for (size_t i = 0; i < ndata.size(); ++i) {
    printf("%lu: %f\n", i, ndata[i]);
  }
  printf("all end\n");
  return 0;
}
