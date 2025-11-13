/**
 * Copyright 2021-2025, XGBoost Contributors
 */
#include "cache_manager.h"
#include "xgboost/logging.h"

#include <cstdint>     // for uint64_t

#if defined(__x86_64__)
void RunCpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd) {
#if defined(_MSC_VER)
    __cpuidex(static_cast<int*>(abcd), eax, ecx);
#else
    uint32_t ebx = 0, edx = 0;
    __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
    abcd[0] = eax;
    abcd[1] = ebx;
    abcd[2] = ecx;
    abcd[3] = edx;
#endif
}

#define __extract_bitmask_value(val, mask, shift) (((val) & (mask)) >> shift)

#define _CPUID_GET_TYPE(__eax) __extract_bitmask_value(__eax /*4:0*/, 0x1fU, 0)

#define _CPUID_GET_LEVEL(__eax) __extract_bitmask_value(__eax /*7:5*/, 0xe0U, 5)

#define _CPUID_GET_SETS(__ecx) ((__ecx) + 1)

#define _CPUID_GET_LINE_SIZE(__ebx) (__extract_bitmask_value(__ebx /*11:0*/, 0x7ffU, 0) + 1)

#define _CPUID_GET_PARTITIONS(__ebx) (__extract_bitmask_value(__ebx /*21:11*/, 0x3ff800U, 11) + 1)

#define _CPUID_GET_WAYS(__ebx) (__extract_bitmask_value(__ebx /*31:22*/, 0xffc00000U, 22) + 1)

#define _CPUID_CACHE_INFO 0x4U

bool GetCacheInfo(int cache_num, int * type, int * level, int64_t * sets,
                  int * line_size, int * partitions, int * ways) {
  static uint32_t abcd[4];
  RunCpuid(_CPUID_CACHE_INFO, cache_num, abcd);

  bool under_hypervisor = (abcd[2] >> 31) & 1;

  const uint32_t eax = abcd[0];
  const uint32_t ebx = abcd[1];
  const uint32_t ecx = abcd[2];
  // const uint32_t edx = abcd[3];  // Not used
  *type              = _CPUID_GET_TYPE(eax);
  *level             = _CPUID_GET_LEVEL(eax);
  *sets              = _CPUID_GET_SETS(ecx);
  *line_size         = _CPUID_GET_LINE_SIZE(ebx);
  *partitions        = _CPUID_GET_PARTITIONS(ebx);
  *ways              = _CPUID_GET_WAYS(ebx);

  bool trust_cpuid = !under_hypervisor;
  return trust_cpuid;
}

constexpr int kCpuidTypeNull = 0;
constexpr int kCpuidTypeData = 1;
constexpr int kCpuidTypeInst = 2;
constexpr int kCpuidTypeUnif = 3;

bool DetectDataCaches(int cache_sizes_len, int64_t* cache_sizes) {
  int cache_num = 0, cache_sizes_idx = 0;
  while (cache_sizes_idx < cache_sizes_len) {
    int type, level, line_size, partitions, ways;
    int64_t sets, size;
    bool trust_cpuid =
      GetCacheInfo(cache_num++, &type, &level, &sets, &line_size, &partitions, &ways);
    if (!trust_cpuid) return trust_cpuid;

    if (type == kCpuidTypeNull) break;
    if (type == kCpuidTypeInst) continue;

    size                           = ways * partitions * line_size * sets;
    cache_sizes[cache_sizes_idx++] = size;
  }
  return true;
}
#endif  // defined(__x86_64__)

void SetDefaultCaches(int64_t* cache_sizes) {
  cache_sizes[0] = 32 * 1024;    // L1
  cache_sizes[1] = 1024 * 1024;  // L2
  cache_sizes[2] = 0;            // L3 place holder
  cache_sizes[3] = 0;            // L4 place holder
}

namespace xgboost::common {

CacheManager::CacheManager() {
#if defined(__x86_64__)
  bool trust_cpuid = DetectDataCaches(kMaxCacheSize, cache_size_.data());
  if (!trust_cpuid) SetDefaultCaches(cache_size_.data());
#else
  SetDefaultCaches(cache_size_.data());
#endif  // defined(__x86_64__)
}

}  // namespace xgboost::common

