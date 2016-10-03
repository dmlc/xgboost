// test reading speed from a InputSplit
#include <cstdlib>
#include <cstdio>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include "../src/data/csv_parser.h"

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("Usage: <libsvm> partid npart nthread [dump csv]\n");
    return 0;
  }
  FILE *fo = NULL;
  if (argc > 5) {
    if (!strcmp(argv[5], "stdout")) {
      fo = stdout;
    } else {
      fo = fopen(argv[5], "w");
    }
  }
  using namespace dmlc;
  std::unique_ptr<dmlc::Parser<unsigned> > parser(
      dmlc::Parser<unsigned>::Create(argv[1],
                                     atoi(argv[2]),
                                     atoi(argv[3]),
                                     "csv"));
  double tstart = GetTime();
  size_t bytes_read = 0;
  size_t bytes_expect = 10UL << 20UL;
  size_t num_ex = 0;
  while (parser->Next());
  parser->BeforeFirst();
  while (parser->Next()) {
    bytes_read  = parser->BytesRead();
    num_ex += parser->Value().size;
    if (fo != NULL){
      const dmlc::RowBlock<unsigned>& batch = parser->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        for (size_t j = 0; j < batch[i].length; ++j) {
          fprintf(fo, "%g", batch[i].value[j]);
          if (j + 1 == batch[i].length) {
            fprintf(fo, "\n");
          } else {
            fprintf(fo, ",");
          }
        }
      }
    }
    double tdiff = GetTime() - tstart;
    if (bytes_read >= bytes_expect) {
      printf("%lu examples, %lu MB read, %g MB/sec\n",
             num_ex, bytes_read >> 20UL,
             (bytes_read >> 20UL) / tdiff);
      bytes_expect += 10UL << 20UL;
    }
  }
  if (fo != NULL && fo != stdout) {
    fclose(fo);
  }
  return 0;
}
