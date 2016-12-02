#include "./helpers.h"

std::string TempFileName() {
  return std::tmpnam(nullptr);
}

long GetFileSize(const std::string filename) {
  struct stat st;
  stat(filename.c_str(), &st);
  return st.st_size;
}
