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
#include <fstream>

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

namespace {
void LockFileImpl(std::string const& path) {
  std::ifstream fin (path);
  while (fin) {
    sleep(10);
    fin.open(path);
  }
  fin.close();
  std::ofstream fout { path };
  CHECK(fout) << "Failed to acquire file lock: " << path;
  std::cout << "Acquired file lock:" << path << std::endl;
}

bool TryLockFileImpl(std::string const& path) {
  std::ifstream fin(path);
  return !fin;
}

void UnlockFileImpl(std::string const& path) noexcept(true) {
  std::ifstream fin(path);
  if (fin) {
    std::remove(path.c_str());
  }
}
}  // anonymous namespace

// Read file lock
void ReadFileLock::lock() {
  LockFileImpl(path_);
}

bool ReadFileLock::try_lock() {
  return TryLockFileImpl(path_);
}

void ReadFileLock::unlock() noexcept(true) {
  UnlockFileImpl(path_);
}

// Write file lock
void WriteFileLock::lock() {
  LockFileImpl(path_);
}

bool WriteFileLock::try_lock() {
  return TryLockFileImpl(path_);
}

void WriteFileLock::unlock() noexcept(true) {
  UnlockFileImpl(path_);
}

// File lock
void FileLock::lock() {
  read_lock_.lock();
  write_lock_.lock();
}

bool FileLock::try_lock() {
  return read_lock_.try_lock() && write_lock_.try_lock();
}

void FileLock::unlock() noexcept(true) {
  read_lock_.unlock();
  write_lock_.unlock();
}

}  // namespace common
}  // namespace xgboost
