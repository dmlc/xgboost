#include <stdio.h>

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

int main() {
  ::sycl::queue qu(::sycl::default_selector_v);

  int num_iters = 1;
  for (int i = 0; i < num_iters; ++i) {
    auto ze_device = ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(qu.get_device());
    uint32_t cache_count = 0;
    ze_result_t status = zeDeviceGetCacheProperties(ze_device, &cache_count, nullptr);
    if (status == ZE_RESULT_SUCCESS) {
      ze_device_cache_properties_t cache;
      status = zeDeviceGetCacheProperties(ze_device, &cache_count, &cache);
      if (status == ZE_RESULT_SUCCESS) {
        fprintf(stdout, "Detected L2 Size = %d\n", int(cache.cacheSize));
      }
    }
  }

  return 0;
}