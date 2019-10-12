/*!
 * Copyright 2019 XGBoost contributors
 */
#include <gtest/gtest.h>

#include <dmlc/filesystem.h>
#include <dmlc/io.h>

#include <xgboost/json.h>
#include <xgboost/base.h>

#include "../../../src/common/version.h"

namespace xgboost {
TEST(Version, IO) {
  Json j_ver { Object() };
  Version::Save(&j_ver);
  auto triplet { Version::Load(j_ver) };
  ASSERT_TRUE(Version::Same(triplet));

  dmlc::TemporaryDirectory tempdir;
  const std::string fname = tempdir.path + "/version";

  {
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(fname.c_str(), "w"));
    Version::Save(fo.get());
  }

  {
    std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(fname.c_str(), "r"));
    auto triplet { Version::Load(fi.get())};;
    ASSERT_TRUE(Version::Same(triplet));
  }
}
}  // namespace xgboost
