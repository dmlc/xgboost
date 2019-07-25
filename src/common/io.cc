/*!
 * Copyright (c) by Contributors 2019
 */
#if defined(__unix__)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // defined(__unix__)
#include <cstdio>
#include <string>

#include "xgboost/logging.h"

namespace xgboost {
namespace common {

std::string LoadSequentialFile(std::string fname) {
  auto OpenErr = [&fname]() {
                   std::string msg;
                   msg = "Opening " + fname + " failed: ";
                   msg += strerror(errno);
                   LOG(FATAL) << msg;
                 };
  auto ReadErr = [&fname]() {
                   std::string msg {"Error in reading file: "};
                   msg += fname;
                   msg += ": ";
                   msg += strerror(errno);
                   LOG(FATAL) << msg;
                 };

  std::string buffer;
#if defined(__unix__)
  struct stat fs;
  if (stat(fname.c_str(), &fs) != 0) {
    OpenErr();
  }

  size_t f_size_bytes = fs.st_size;
  buffer.resize(f_size_bytes+1);
  int32_t fd = open(fname.c_str(), O_RDONLY);
  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
  ssize_t bytes_read = read(fd, &buffer[0], f_size_bytes);
  if (bytes_read < 0) {
    close(fd);
    ReadErr();
  }
  close(fd);
#else
  FILE *f = fopen(fname.c_str(), "r");
  if (f == NULL) {
    std::string msg;
    OpenErr();
  }
  fseek(f, 0, SEEK_END);
  auto fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  buffer.resize(fsize + 1);
  fread(&buffer[0], 1, fsize, f);
  fclose(f);
#endif  // defined(__unix__)
  return buffer;
}

}  // namespace common
}  // namespace xgboost
