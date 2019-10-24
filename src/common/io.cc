/*!
 * Copyright (c) by Contributors 2019
 */
#if defined(__unix__)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // defined(__unix__)

#include <dmlc/base.h>

#include <cstdio>
#include <string>
#include <fstream>
#include <chrono>

#if DMLC_ENABLE_STD_THREAD
#include <thread>
#endif  //  DMLC_ENABLE_STD_THREAD

#include "xgboost/base.h"
#include "xgboost/logging.h"
#include "io.h"

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

// File lock
void FileLock::lock() {  // NOLINT
  std::ifstream fin(path_);
  while (fin) {
    sleep(10);
    fin.open(path_);
  }
  fin.close();
  std::ofstream fout { path_ };
  CHECK(fout) << "Failed to acquire file lock: " << path_;
}

bool FileLock::try_lock() const {  // NOLINT
  std::ifstream fin(path_);
  return !fin;
}

void FileLock::unlock() noexcept(true) {  // NOLINT
  std::ifstream fin(path_);
  if (fin) {
    std::remove(path_.c_str());
  }
}

void WaitForLock(FileLock const& lock) {
#if DMLC_ENABLE_STD_THREAD
  while (!lock.try_lock()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
#else
      LOG(FATAL) << "External memory is not enabled in mingw";
#endif  //  DMLC_ENABLE_STD_THREAD
}

}  // namespace common
}  // namespace xgboost
