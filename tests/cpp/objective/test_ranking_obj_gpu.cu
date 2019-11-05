#include "test_ranking_obj.cc"

#include <nvml.h>

TEST(Objective, PrintDriverVersion) {
#if defined(__CUDACC__)
  EXPECT_EQ(nvmlInit(), NVML_SUCCESS);
  char ver_buf[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE] = {0};
  EXPECT_EQ(nvmlSystemGetDriverVersion(ver_buf, sizeof(ver_buf)), NVML_SUCCESS);
  std::cout << "Driver version is: " << ver_buf << std::endl;
#endif
}
