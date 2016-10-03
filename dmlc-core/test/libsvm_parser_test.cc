// test reading speed from a InputSplit
#include <cstdlib>
#include <cstdio>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include "../src/data/libsvm_parser.h"

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("Usage: <libsvm> partid npart nthread\n");
    return 0;
  }
  using namespace dmlc;
  InputSplit *split = InputSplit::Create(argv[1],
                                         atoi(argv[2]),
                                         atoi(argv[3]),
                                         "text");
  int nthread = atoi(argv[4]);
  data::LibSVMParser<unsigned> parser(split, nthread);
  double tstart = GetTime();
  size_t bytes_read = 0;
  size_t bytes_expect = 10UL << 20UL;
  size_t num_ex = 0;
  while (parser.Next()) {
    bytes_read  = parser.BytesRead();
    num_ex += parser.Value().size;
    double tdiff = GetTime() - tstart;
    if (bytes_read >= bytes_expect) {
      printf("%lu examples, %lu MB read, %g MB/sec\n",
             num_ex, bytes_read >> 20UL,
             (bytes_read >> 20UL) / tdiff);
      bytes_expect += 10UL << 20UL;
    }
  }
  return 0;
}
