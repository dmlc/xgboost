/**
 * Copyright 2021-2023 XGBoost contributors
 */
#include <gtest/gtest.h>

#include <any>  // for any_cast
#include <memory>

#include "../../../src/data/adapter.h"
#include "../../../src/data/file_iterator.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"

namespace xgboost::data {
TEST(FileIterator, Basic) {
  auto check_n_features = [](FileIterator *iter) {
    size_t n_features = 0;
    iter->Reset();
    while (iter->Next()) {
      auto proxy = MakeProxy(iter->Proxy());
      auto csr = std::any_cast<std::shared_ptr<CSRArrayAdapter>>(proxy->Adapter());
      n_features = std::max(n_features, csr->NumColumns());
    }
    ASSERT_EQ(n_features, 5);
  };

  dmlc::TemporaryDirectory tmpdir;
  {
    auto zpath = tmpdir.path + "/0-based.svm";
    CreateBigTestData(zpath, 3 * 64, true);
    zpath += "?indexing_mode=0&format=libsvm";
    FileIterator iter{zpath, 0, 1};
    check_n_features(&iter);
  }

  {
    auto opath = tmpdir.path + "/1-based.svm";
    CreateBigTestData(opath, 3 * 64, false);
    opath += "?indexing_mode=1&format=libsvm";
    FileIterator iter{opath, 0, 1};
    check_n_features(&iter);
  }
}
}  // namespace xgboost::data
