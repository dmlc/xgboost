// test reading speed from a InputSplit
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <dmlc/io.h>
#include <dmlc/base.h>

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("Usage: <libsvm> partid npart\n");
    return 0;
  }
  using namespace dmlc;
  InputSplit *split = InputSplit::Create(argv[1],
                                         atoi(argv[2]),
                                         atoi(argv[3]),
                                         "text");
  InputSplit::Blob blb;
  while (split->NextChunk(&blb)) {
    std::cout << std::string((char*)blb.dptr, blb.size);
  }
  delete split;
  return 0;
}

