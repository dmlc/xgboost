#include <dmlc/json.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <sstream>
#include <cstring>
#include <unordered_map>
#include <iostream>

using namespace std;
namespace json {
template<typename T>
inline void TestSaveLoad(T data) {
  std::ostringstream os;
  {
    T temp(data);
    dmlc::JSONWriter writer(&os);
    writer.Write(temp);
    temp.clear();
  }
  std::string json = os.str();
  LOG(INFO) << json;
  std::istringstream is(json);
  dmlc::JSONReader reader(&is);
  T copy_data;
  reader.Read(&copy_data);
  ASSERT_EQ(data, copy_data);
}

class MyClass {
 public:
  MyClass() {}
  MyClass(std::string data) : data_{data}, value_(0) {}
  inline void Save(dmlc::JSONWriter *writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("value", value_);
    writer->WriteObjectKeyValue("data", data_);
    writer->EndObject();
  }
  inline void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("data", &data_);
    helper.DeclareOptionalField("value", &value_);
    helper.ReadAllFields(reader);
  }
  inline bool operator==(const MyClass &other) const {
    return value_ == other.value_;
  }

 private:
  std::vector<std::string> data_;
  int value_;
};
}

DMLC_JSON_ENABLE_ANY(std::vector<std::string>, StrVector);

// test json module
TEST(JSON, basics) {
  using namespace json;
  int n = 10;
  std::vector<int> a;
  for (int i = 0; i < n; ++i) {
    a.push_back(i);
  }
  TestSaveLoad(a);

  std::vector<std::string> b;
  for (int i = 0; i < n; ++i) {
    std::string ss(i, 'a' + (i % 26));
    b.push_back(ss);
  }
  TestSaveLoad(b);

  std::vector<std::vector<int> > temp {{1,2,3}, {1,2}, {1,2,3,4}};
  TestSaveLoad(temp);

  std::vector<std::vector<int> > temp2 {{}, {}, {1,2,3,4}};
  TestSaveLoad(temp2);

  TestSaveLoad(
      std::map<std::string, int>  {{"hellkow", 1}, {"world", 2}});

  TestSaveLoad(
      std::unordered_map<std::string, int>  {{"hellkow", 1}, {"world", 2}});
  TestSaveLoad(std::list<std::string>  {"hjhjm", "asasa"});
  TestSaveLoad(std::list<int>(a.begin(), a.end()));
  TestSaveLoad(std::list<MyClass> {MyClass("abc"), MyClass("def")});
}


TEST(JSON, any) {
  dmlc::any x = std::vector<std::string>{"a", "b", "c"};

  std::ostringstream os;
  {
    dmlc::any temp(x);
    dmlc::JSONWriter writer(&os);
    writer.Write(temp);
  }

  std::string json = os.str();
  LOG(INFO) << json;
  std::istringstream is(json);
  dmlc::JSONReader reader(&is);
  dmlc::any copy_data;
  reader.Read(&copy_data);

  ASSERT_EQ(dmlc::get<std::vector<std::string> >(x),
            dmlc::get<std::vector<std::string> >(copy_data));
}
