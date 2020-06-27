/*!
 * Copyright (c) by XGBoost Contributors 2019
 */
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <atomic>
#include <type_traits>
#include <fstream>
#include <cstdint>
#include "../../../src/common/io.h"

namespace xgboost {
namespace common {
TEST(IO, FileExtension) {
  std::string filename {u8"model.json"};
  auto ext = FileExtension(filename);
  ASSERT_EQ(ext, u8"json");
}

TEST(IO, FixedSizeStream) {
  std::string buffer {"This is the content of stream"};
  {
    MemoryFixSizeBuffer stream(static_cast<void *>(&buffer[0]), buffer.size());
    PeekableInStream peekable(&stream);
    FixedSizeStream fixed(&peekable);

    std::string out_buffer;
    fixed.Take(&out_buffer);
    ASSERT_EQ(buffer, out_buffer);
  }

  {
    std::string huge_buffer;
    for (size_t i = 0; i < 512; i++) {
      huge_buffer += buffer;
    }

    MemoryFixSizeBuffer stream(static_cast<void*>(&huge_buffer[0]), huge_buffer.size());
    PeekableInStream peekable(&stream);
    FixedSizeStream fixed(&peekable);

    std::string out_buffer;
    fixed.Take(&out_buffer);
    ASSERT_EQ(huge_buffer, out_buffer);
  }
}


#if SIZE_MAX == 0xFFFFFFFFFFFFFFFF  // Only run this test on 64-bit system
TEST(IO, LoadSequentialFile) {
  const size_t nbyte = static_cast<size_t>(2896075944LL);  // About 2.69 GB
  static_assert(sizeof(size_t) == 8, "Assumption failed: size_t was assumed to be 8-bytes long");
  static_assert(std::is_same<size_t, std::string::size_type>::value,
                "Assumption failed: size_type of std::string was assumed to be 8-bytes long");

  dmlc::TemporaryDirectory tempdir;
  std::string path = "/dev/shm/xgboost_test_io_big_file.txt";
  {
    std::ofstream f(path);
    if (!f) {  // /dev/shm not present
      LOG(INFO) << "No /dev/shm; using dmlc::TemporaryDirectory instead";
      path = tempdir.path + "/xgboost_test_io_big_file.txt";
      f = std::ofstream(path);
    }
    CHECK(f);
    std::string str(nbyte, 'a');
    CHECK_EQ(str.size(), nbyte);
    f << str;
  }
  {
    std::string str = LoadSequentialFile(path);
    CHECK_EQ(str.size(), nbyte);
    dmlc::OMPException omp_exc;
    std::atomic<bool> success{true};
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(nbyte); ++i) {
      omp_exc.Run([&] {
        if (str[i] != 'a' && success.load(std::memory_order_acquire)) {
          success.store(false, std::memory_order_release);
          LOG(FATAL) << "Big file got corrupted. Expected: str[" << i << "] = 'a', "
            << "Actual: str[" << i << "] = '"
            << (str[i] ? std::string(1, str[i]) : std::string("\\0")) << "'";
        }
      });
    }
    omp_exc.Rethrow();
    CHECK(success.load(std::memory_order_acquire));
  }
}
#endif  // SIZE_MAX == 0xFFFFFFFFFFFFFFFF

}  // namespace common
}  // namespace xgboost
