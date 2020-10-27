#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <fstream>
#include "../../../src/data/parser.h"

namespace xgboost {
namespace data {
TEST(Parser, Format) {
  dmlc::TemporaryDirectory tmpdir;
  auto path = tmpdir.path + "/data";
  auto uri = tmpdir.path + "/data?format=csv";
  std::ofstream fout(path);
  fout << "1,2,3,4,5";
  fout.flush();

  std::unique_ptr<dmlc::Parser<uint32_t, float>> parser{
      CreateParser<uint32_t, float>(uri, 0, 1, "auto")};
  auto csv = dynamic_cast<CSVParser<uint32_t, float>*>(parser.get());
  ASSERT_TRUE(csv);

  // default to libsvm
  parser.reset(CreateParser<uint32_t, float>(path, 0, 1, "auto"));
  auto svm = dynamic_cast<LibSVMParser<uint32_t, float>*>(parser.get());
  ASSERT_TRUE(svm);

  uri = path + "?format=libfm";
  parser.reset(CreateParser<uint32_t, float>(uri, 0, 1, "auto"));
  auto fm = dynamic_cast<LibFMParser<uint32_t, float>*>(parser.get());
  ASSERT_TRUE(fm);
}
}  // namespace data
}  // namespace xgboost
