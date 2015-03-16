#include <rabit.h>
#include "../io/io.h"

int main(int argc, char *argv[]) {
  using namespace rabit::io;
  if (argc < 2) {
    // intialize rabit engine
    rabit::Init(argc, argv);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Usage: <data_in> param=val\n");
    }
    rabit::Finalize();
    return 0;
  }
  int n = 0;
  InputSplit *in = CreateInputSplit(argv[1],
                                    rabit::GetRank(),
                                    rabit::GetWorldSize());
  std::string line;
  while (in->NextLine(&line)) {
    if (n % 100 == 0) {
      rabit::TrackerPrintf("[%d] finishes loading %d lines\n",
                           rabit::GetRank(), n);
    }
    n++;
  }
  delete in;
  rabit::TrackerPrintf("[%d] finishes loading %d lines\n",
                       rabit::GetRank(), n);
  return 0;
}
