/*!
 * Copyright 2017 by Contributors
 * \file memory.h
 * \brief Utility for memory
 * \author Philip Cho
 */
#ifndef XGBOOST_COMMON_MEMORY_H_
#define XGBOOST_COMMON_MEMORY_H_

#ifndef _WIN32
#include <unistd.h>
#else
#define NOMINMAX
#include <windows.h>
#endif

namespace xgboost {
namespace common {

#ifndef _WIN32
inline size_t GetSystemMemory() {
  size_t pages = sysconf(_SC_PHYS_PAGES);
  size_t page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}
#else
inline size_t GetSystemMemory() {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
}
#endif

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_MEMORY_H_
