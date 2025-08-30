/**
 * Copyright 2025, XGBoost Contributors
 */
#include "filesystem.h"

#include <xgboost/windefs.h>

#include <filesystem>  // for path, temp_directory_path

#if !defined(xgboost_IS_WIN)

#include <cstdlib>  // for mkdtemp

#include "../../src/common/error_msg.h"  // for SystemError

#else

#include <random>  // for uniform_int_distribution

#include "xgboost/string_view.h"  // for StringView

#endif  // !defined(xgboost_IS_WIN)

namespace xgboost::common {
TemporaryDirectory::TemporaryDirectory(std::string prefix) : prefix_{std::move(prefix)} {
  namespace fs = std::filesystem;

  auto tmp = fs::temp_directory_path();

#if defined(xgboost_IS_WIN)
  std::default_random_engine rng;
  auto make_name = [&rng, this] {
    constexpr std::size_t kPathMax = 6;
    constexpr StringView kAlphabet{"abcdefghijklmnopqrstuvwxyz"};
    static_assert(kAlphabet.size() == 26);
    std::uniform_int_distribution dist{0, 25};
    char path[kPathMax + 1];
    std::memset(path, 0, sizeof(path));
    for (std::size_t i = 0; i < kPathMax; ++i) {
      auto k = dist(rng);
      path[i] = kAlphabet[k];
    }
    auto res = std::string{path};
    CHECK_EQ(res.size(), kPathMax);
    return this->prefix_ + "tmpdir." + std::string{path};
  };
  auto dirname = tmp / make_name();
  std::int32_t retry = 0;
  while (fs::exists(dirname) && retry < 64) {
    dirname = tmp / make_name();
    ++retry;
  }
  if (retry >= 64) {
    LOG(FATAL) << "Failed to create temporary directory.";
  }
  this->path_ = dirname.string();
  CHECK(fs::create_directory(this->path_));
#else
  auto dirtemplate = (tmp / (this->prefix_ + "tmpdir.XXXXXX")).string();
  // https://man7.org/linux/man-pages/man3/mkdtemp.3.html
  char* tmpdir = mkdtemp(dirtemplate.data());
  if (!tmpdir) {
    LOG(FATAL) << error::SystemError().message();
  }
  this->path_ = tmpdir;
#endif
  LOG(DEBUG) << "TmpDir:" << this->path_;
  CHECK(fs::exists(this->path_));
}

TemporaryDirectory::~TemporaryDirectory() noexcept(false) {
  std::filesystem::remove_all(this->path_);
}
}  // namespace xgboost::common
