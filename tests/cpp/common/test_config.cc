/*!
 * Copyright 2019 by Contributors
 */
#include <fstream>
#include <string>
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include "../../../src/common/config.h"
#include "../helpers.h"

namespace xgboost {
namespace common {

TEST(ConfigParser, NormalizeConfigEOL) {
  // Test whether strings with NL are loaded correctly.
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/my.conf";
  /* Old Mac OS uses \r for line ending */
  {
    std::string const input = "foo\rbar\rdog\r";
    std::string const output = "foo\nbar\ndog\n";
    {
      std::ofstream fp(
          tmp_file,
          std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
      fp << input;
    }
    {
      ConfigParser parser(tmp_file);
      auto content = parser.LoadConfigFile(tmp_file);
      content = parser.NormalizeConfigEOL(content);
      ASSERT_EQ(content, output);
    }
  }
  /* Windows uses \r\n for line ending */
  {
    std::string const input = "foo\r\nbar\r\ndog\r\n";
    std::string const output = "foo\n\nbar\n\ndog\n\n";
    {
      std::ofstream fp(tmp_file,
                       std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
      fp << input;
    }
    {
      ConfigParser parser(tmp_file);
      auto content = parser.LoadConfigFile(tmp_file);
      content = parser.NormalizeConfigEOL(content);
      ASSERT_EQ(content, output);
    }
  }
}

TEST(ConfigParser, TrimWhitespace) {
  ASSERT_EQ(ConfigParser::TrimWhitespace("foo bar"), "foo bar");
  ASSERT_EQ(ConfigParser::TrimWhitespace("  foo bar"), "foo bar");
  ASSERT_EQ(ConfigParser::TrimWhitespace("foo bar  "), "foo bar");
  ASSERT_EQ(ConfigParser::TrimWhitespace("foo bar\t"), "foo bar");
  ASSERT_EQ(ConfigParser::TrimWhitespace("   foo bar  "), "foo bar");
  ASSERT_EQ(ConfigParser::TrimWhitespace("\t\t  foo bar  \t"), "foo bar");
  ASSERT_EQ(ConfigParser::TrimWhitespace("\tabc\t"), "abc");
  ASSERT_EQ(ConfigParser::TrimWhitespace("\r abc\t"), "abc");
}

TEST(ConfigParser, ParseKeyValuePair) {
  // Create dummy configuration file
  dmlc::TemporaryDirectory tempdir;
  const std::string tmp_file = tempdir.path + "/my.conf";
  {
    std::ofstream fp(tmp_file);
    fp << "";
  }

  ConfigParser parser(tmp_file);

  std::string key, value;
  // 1. Empty lines or comments
  ASSERT_FALSE(parser.ParseKeyValuePair("# Mary had a little lamb",
                                        &key, &value));
  ASSERT_FALSE(parser.ParseKeyValuePair("#tree_method = gpu_hist",
                                        &key, &value));
  ASSERT_FALSE(parser.ParseKeyValuePair(
                 "# minimum sum of instance weight(hessian) needed in a child",
                 &key, &value));
  ASSERT_FALSE(parser.ParseKeyValuePair("", &key, &value));

  // 2. Key-value pairs
  ASSERT_TRUE(parser.ParseKeyValuePair("booster = gbtree", &key, &value));
  ASSERT_EQ(key, "booster");
  ASSERT_EQ(value, "gbtree");
  ASSERT_TRUE(parser.ParseKeyValuePair("gpu_id = 2", &key, &value));
  ASSERT_EQ(key, "gpu_id");
  ASSERT_EQ(value, "2");
  ASSERT_TRUE(parser.ParseKeyValuePair("monotone_constraints = (1,0,-1)",
                                       &key, &value));
  ASSERT_EQ(key, "monotone_constraints");
  ASSERT_EQ(value, "(1,0,-1)");
  // whitespace should not matter
  ASSERT_TRUE(parser.ParseKeyValuePair("  objective=binary:logistic",
                                       &key, &value));
  ASSERT_EQ(key, "objective");
  ASSERT_EQ(value, "binary:logistic");
  ASSERT_TRUE(parser.ParseKeyValuePair("tree_method\t=\thist  ", &key, &value));
  ASSERT_EQ(key, "tree_method");
  ASSERT_EQ(value, "hist");

  // 3. Use of forward and backward slashes in value
  ASSERT_TRUE(parser.ParseKeyValuePair("test:data = test/data.libsvm",
                                       &key, &value));
  ASSERT_EQ(key, "test:data");
  ASSERT_EQ(value, "test/data.libsvm");
  ASSERT_TRUE(parser.ParseKeyValuePair("data = C:\\data.libsvm", &key, &value));
  ASSERT_EQ(key, "data");
  ASSERT_EQ(value, "C:\\data.libsvm");

  // 4. One-line comment
  ASSERT_TRUE(parser.ParseKeyValuePair("learning_rate = 0.3   # small step",
                                       &key, &value));
  ASSERT_EQ(key, "learning_rate");
  ASSERT_EQ(value, "0.3");
  // Note: '#' in path won't be accepted correctly unless the whole path is
  // wrapped with quotes. This is important for external memory.
  ASSERT_TRUE(parser.ParseKeyValuePair("data = dmatrix.libsvm#dtrain.cache",
                                       &key, &value));
  ASSERT_EQ(key, "data");
  ASSERT_EQ(value, "dmatrix.libsvm");  // cache was silently ignored

  // 5. Wrapping key/value with quotes
  // Any key or value containing '#' needs to be wrapped with quotes
  ASSERT_TRUE(parser.ParseKeyValuePair("data = \"dmatrix.libsvm#dtrain.cache\"",
                                       &key, &value));
  ASSERT_EQ(key, "data");
  ASSERT_EQ(value, "dmatrix.libsvm#dtrain.cache");  // cache is now kept
  ASSERT_TRUE(parser.ParseKeyValuePair(
                "data = \"C:\\Administrator\\train_file.txt#trainbincache\"",
                &key, &value));
  ASSERT_EQ(key, "data");
  ASSERT_EQ(value, "C:\\Administrator\\train_file.txt#trainbincache");
  ASSERT_TRUE(parser.ParseKeyValuePair("\'month#day\' = \"November#2019\"",
                                       &key, &value));
  ASSERT_EQ(key, "month#day");
  ASSERT_EQ(value, "November#2019");
  // Likewise, key or value containing a space needs to be quoted
  ASSERT_TRUE(parser.ParseKeyValuePair("\"my data\" = \' so precious!  \'",
                                       &key, &value));
  ASSERT_EQ(key, "my data");
  ASSERT_EQ(value, " so precious!  ");
  ASSERT_TRUE(parser.ParseKeyValuePair("interaction_constraints = "
                                       "\"[[0, 2], [1, 3, 4], [5, 6]]\"",
                                       &key, &value));
  ASSERT_EQ(key, "interaction_constraints");
  ASSERT_EQ(value, "[[0, 2], [1, 3, 4], [5, 6]]");

  // 6. Unicode
  ASSERT_TRUE(parser.ParseKeyValuePair("클래스상속 = 类继承", &key, &value));
  ASSERT_EQ(key, "클래스상속");
  ASSERT_EQ(value, "类继承");

  // 7. Ill-formed data should throw exception
  for (const char* str : {"data = C:\\My Documents\\cat.csv", "cow=",
                          "C# = 100%", "= woof ",
                          "interaction_constraints = [[0, 2], [1]]",
                          "data = \"train.txt#cache",
                          "data = \'train.txt#cache", "foo = \'bar\""}) {
    ASSERT_THROW(parser.ParseKeyValuePair(str, &key, &value), dmlc::Error);
  }
}

}  // namespace common
}  // namespace xgboost
