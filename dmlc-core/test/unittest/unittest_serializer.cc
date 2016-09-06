#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <sstream>
#include <cstring>
#include <unordered_map>
#include <iostream>

using namespace std;

template<typename T>
inline void TestSaveLoad(T data) {
  std::string blob;
  dmlc::MemoryStringStream fs(&blob);
  {
    T temp(data);
    static_cast<dmlc::Stream*>(&fs)->Write(temp);
    temp.clear();
  }
  fs.Seek(0);
  T copy_data;
  CHECK(static_cast<dmlc::Stream*>(&fs)->Read(&copy_data));
  ASSERT_EQ(data, copy_data);
}

class MyClass {
 public:
  MyClass() {}
  MyClass(std::string data) : data_(data) {}
  inline void Save(dmlc::Stream *strm) const {
    strm->Write(this->data_);
  }
  inline bool Load(dmlc::Stream *strm) {
    return strm->Read(&data_);
  }
  inline bool operator==(const MyClass &other) const {
    return data_ == other.data_;
  }

 private:
  std::string data_;
};
// need to declare the traits property of my class to dmlc
namespace dmlc { DMLC_DECLARE_TRAITS(has_saveload, MyClass, true); }

struct Param {
  int a;
  int b;
  Param() {}
  Param(int a, int b) : a(a), b(b) {}
  inline bool operator==(const Param &other) const {
    return a == other.a && b == other.b;
  }
};
// need to declare the traits property of my class to dmlc
namespace dmlc { DMLC_DECLARE_TRAITS(is_pod, Param, true); }

// test serializer
TEST(Serializer, basics) {
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
  TestSaveLoad(
      std::map<int, std::string>  {{1, "hellkow"}, {2, "world"}});
  TestSaveLoad(
      std::unordered_map<int, std::string>  {{1, "hellkow"}, {2, "world"}});
  TestSaveLoad(
      std::unordered_multimap<int, std::string>  {{1, "hellkow"}, {1, "world"}, {2, "111"}});
  TestSaveLoad(std::set<std::string>  {"hjhjm", "asasa"});
  TestSaveLoad(std::unordered_set<std::string>  {"hjhjm", "asasa"});
  TestSaveLoad(std::list<std::string>  {"hjhjm", "asasa"});
  TestSaveLoad(std::list<int>(a.begin(), a.end()));
  TestSaveLoad(std::list<MyClass> {MyClass("abc"), MyClass("def")});
  TestSaveLoad(std::list<Param> {Param(3, 4), Param(5, 6)});

}
