#include "./helpers.h"

std::string TempFileName() {
  return std::tmpnam(nullptr);
}

bool FileExists(const std::string name) {
  struct stat st;
  return stat(name.c_str(), &st) == 0; 
}

long GetFileSize(const std::string filename) {
  struct stat st;
  stat(filename.c_str(), &st);
  return st.st_size;
}

std::string CreateSimpleTestData() {
  std::string tmp_file = TempFileName();
  std::ofstream fo;
  fo.open(tmp_file);
  fo << "0 0:0 1:10 2:20\n";
  fo << "1 0:0 3:30 4:40\n";
  fo.close();
  return tmp_file;
}
