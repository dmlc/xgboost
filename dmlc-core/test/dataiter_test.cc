#include <dmlc/data.h>
#include <dmlc/timer.h>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: filename partid npart [format]\n");
    return 0;
  }
  char libsvm[10] = "libsvm";
  char* format;
  if (argc > 4) {
    format = argv[4];
  } else {
    format = libsvm;
  }

  using namespace dmlc;
  RowBlockIter<index_t> *iter =
      RowBlockIter<index_t>::Create(
          argv[1], atoi(argv[2]), atoi(argv[3]), format);
  double tstart = GetTime();
  size_t bytes_read = 0;
  while (iter->Next()) {
    const RowBlock<index_t> &batch = iter->Value();
    bytes_read += batch.MemCostBytes();
    double tdiff = GetTime() - tstart;
    LOG(INFO) << (bytes_read >> 20UL) <<
        " MB read " << ((bytes_read >> 20UL) / tdiff)<< " MB/sec";
  }
  return 0;
}
