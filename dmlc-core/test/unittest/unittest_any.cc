// Copyright by Contributors

#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <dmlc/any.h>
#include <dmlc/json.h>
#include <gtest/gtest.h>


TEST(Any, basics) {
  std::unordered_map<std::string, dmlc::any> dict;
  dict["1"] = 1;
  dict["vec"] = std::vector<int>{1,2,3};
  dict["shapex"] = std::string("xtyz");
  std::unordered_map<std::string, dmlc::any> dict2(std::move(dict));
  dmlc::get<int>(dict2["1"]) += 1;

  CHECK_EQ(dmlc::get<int>(dict2["1"]), 2);
  CHECK_EQ(dmlc::get<std::vector<int> >(dict2["vec"])[1], 2);
}

TEST(Any, cover) {
  dmlc::any a = std::string("abc");
  dmlc::any b = 1;

  CHECK_EQ(dmlc::get<std::string>(a), "abc");
  a = std::move(b);
  CHECK(b.empty());
  CHECK_EQ(dmlc::get<int>(a), 1);

  std::shared_ptr<int> x = std::make_shared<int>(10);
  {
    dmlc::any aa(x);
    CHECK_EQ(*dmlc::get<std::shared_ptr<int> >(aa), 10);
  }
  // aa must be destructed.
  CHECK(x.unique());
}

DMLC_JSON_ENABLE_ANY(std::vector<int>, IntVector);
DMLC_JSON_ENABLE_ANY(int, Int);

TEST(Any, json) {
  std::unordered_map<std::string, dmlc::any> x;
  x["vec"] = std::vector<int>{1, 2, 3};
  x["int"] = 300;

  std::ostringstream os;
  {
    std::unordered_map<std::string, dmlc::any> temp(x);
    dmlc::JSONWriter writer(&os);
    writer.Write(temp);
    temp.clear();
  }
  std::string json = os.str();
  LOG(INFO) << json;
  std::istringstream is(json);
  dmlc::JSONReader reader(&is);
  std::unordered_map<std::string, dmlc::any> copy_data;
  reader.Read(&copy_data);

  ASSERT_EQ(dmlc::get<std::vector<int> >(x["vec"]),
            dmlc::get<std::vector<int> >(copy_data["vec"]));
  ASSERT_EQ(dmlc::get<int>(x["int"]),
            dmlc::get<int>(copy_data["int"]));
}
