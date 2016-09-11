// test reading speed from a Stream
#include <cstdlib>
#include <cstdio>
#include <dmlc/io.h>
#include <dmlc/timer.h>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: uri buffersize [skip-proc]\n");
    return 0;
  }
  int skip_proc = 0;
  if (argc > 3) {
    skip_proc = atoi(argv[3]);
  }
  size_t sz = atol(argv[2]);
  std::string buffer; buffer.resize(sz);
  using namespace dmlc;
  Stream *fi = Stream::Create(argv[1], "r", true);
  CHECK(fi != NULL) << "cannot open " << argv[1];
  double tstart = GetTime();
  size_t size;
  size_t bytes_read = 0;
  size_t bytes_expect = 10UL << 20UL;
  while ((size = fi->Read(BeginPtr(buffer), sz)) != 0) {
    int cnt = 0;
    if (skip_proc == 0) {
      //#pragma omp parallel for reduction(+:cnt)
      for (size_t i = 0; i < size; ++i) {
        if (buffer[i] == '\n' || buffer[i] == '\r') {
          buffer[i] = '\0'; ++ cnt;
        }
      }    
    }
    bytes_read += size;
    double tdiff = GetTime() - tstart;
    if (bytes_read >= bytes_expect) {
      printf("%lu MB read, %g MB/sec, cnt=%d\n",
             bytes_read >> 20UL,
             (bytes_read >> 20UL) / tdiff, cnt);
      bytes_expect += 10UL << 20UL;
    }
  }
  delete fi;
  return 0;
}


