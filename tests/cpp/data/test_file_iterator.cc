/*!
 * Copyright 2021 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>

#include <memory>

#include "../../../src/data/file_iterator.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/data/adapter.h"
#include "../helpers.h"

namespace xgboost {
namespace data {
TEST(FileIterator, Basic) {
  auto check_n_features = [](FileIterator *iter) {
    size_t n_features = 0;
    iter->Reset();
    while (iter->Next()) {
      auto proxy = MakeProxy(iter->Proxy());
      auto csr = dmlc::get<std::shared_ptr<CSRArrayAdapter>>(proxy->Adapter());
      n_features = std::max(n_features, csr->NumColumns());
    }
    ASSERT_EQ(n_features, 5);
  };

  dmlc::TemporaryDirectory tmpdir;
  {
    auto zpath = tmpdir.path + "/0-based.svm";
    CreateBigTestData(zpath, 3 * 64, true);
    zpath += "?indexing_mode=0";
    FileIterator iter{zpath, 0, 1, "libsvm"};
    check_n_features(&iter);
  }

  {
    auto opath = tmpdir.path + "/1-based.svm";
    CreateBigTestData(opath, 3 * 64, false);
    opath += "?indexing_mode=1";
    FileIterator iter{opath, 0, 1, "libsvm"};
    check_n_features(&iter);
  }
}
}  // namespace data
}  // namespace xgboost
