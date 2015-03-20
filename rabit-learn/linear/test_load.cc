#include <rabit.h>
#include "../io/io.h"

int main(int argc, char *argv[]) {
  using namespace rabit::io;
  if (argc < 4) {
    // intialize rabit engine
    rabit::Init(argc, argv);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Usage: <data_in> npart rank\n");
    }
    rabit::Finalize();
    return 0;
  }
  rabit::Init(argc, argv);
  int n = 0;
  InputSplit *in = CreateInputSplit(argv[1],
                                    atoi(argv[2]),
                                    atoi(argv[3]));
  std::string line;
  while (in->NextLine(&line)) {
    if (n % 100 == 0) {
      rabit::TrackerPrintf("[%d] finishes loading %d lines\n",
                           rabit::GetRank(), n);
    }
    n++;
  }
  delete in;
  rabit::TrackerPrintf("[%d] ALL finishes loading %d lines\n",
                       rabit::GetRank(), n);
  rabit::Finalize();
  return 0;
}
