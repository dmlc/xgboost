#include <gtest/gtest.h>

#include <random>
#include <fstream>

#include "../../../src/common/json_experimental.h"
#include "../../../src/common/json_writer_experimental.h"
#include "../../../src/common/json_reader_experimental.h"
#include "../helpers.h"

namespace xgboost {
namespace experimental {

TEST(Json_Experimental, Basic) {
  {
    Document doc;
    {
      auto value{doc.CreateMember("ok")};
      value.SetInteger(12);
    }

    std::string str = doc.Dump<JsonWriter>();

    ASSERT_NE(str.find(R"s("ok")s"), std::string::npos);
    ASSERT_NE(str.find('{'), std::string::npos);
    ASSERT_NE(str.find('}'), std::string::npos);
    ASSERT_NE(str.find(':'), std::string::npos);
    ASSERT_EQ(doc.Length(), 1);
    ASSERT_NE(str.find(u8"12"), std::string::npos);
  }

  {
    Document doc;
    {
      auto value{doc.CreateMember("ok")};
      value.SetFloat(12);
    }

    std::string str = doc.Dump<JsonWriter>();
    ASSERT_NE(str.find(R"s("ok")s"), std::string::npos);
    ASSERT_NE(str.find('{'), std::string::npos);
    ASSERT_NE(str.find('}'), std::string::npos);
    ASSERT_NE(str.find(':'), std::string::npos);
    ASSERT_EQ(doc.Length(), 1);
    ASSERT_NE(str.find(u8"1.2E1"), std::string::npos);
  }
}

TEST(Json_Experimental, ObjectGeneral) {
  Document doc;
  {
    auto level_0_0 = doc.CreateMember("0-0");
    level_0_0.SetObject();
    auto level_1_0 = level_0_0.CreateMember("1-0");
    level_1_0.SetFloat(12.3);
  }

  {
    auto level_0_1 = doc.CreateMember("0-1");
    level_0_1.SetObject();
    auto level_1_1 = level_0_1.CreateMember("1-1");
    level_1_1.SetInteger(1);
  }

  ASSERT_EQ(doc.Dump<JsonWriter>(), R"s({"0-0":{"1-0":1.23E1},"0-1":{"1-1":1}})s");

  auto it = doc.GetObject().FindMemberByKey("0-0");
  ASSERT_NE(it, doc.GetObject().cend());
  Json v = *it;
  ASSERT_TRUE(v.IsObject());
}

TEST(Json_Experimental, NestedObjects) {
  Document doc;
  {
    auto value{doc.CreateMember("ok")};
    value.SetObject();

    auto mem_value = value.CreateMember("member-key");
    mem_value.SetFloat(12.3);
  }
  auto str = doc.Dump<JsonWriter>();

  ASSERT_NE(str.find("member-key"), std::string::npos);
  ASSERT_NE(str.find("1.23"), std::string::npos);
}

TEST(Json_Experimental, MultipleObjects) {
  Document doc;
  {
    auto a_0{doc.CreateMember("a_0")};
    a_0.SetInteger(34);
  }

  {
    auto a_1{doc.CreateMember("a_1")};
    a_1.SetFloat(3.14);
  }
  auto str = doc.Dump<JsonWriter>();
}

TEST(Json_Experimental, Array) {
  Document doc;
  {
    auto value = doc.CreateMember("array");
    value.SetArray(16);
    auto a = value.GetArrayElem(0);
    a.SetFloat(3.14159);

    Json b = value.GetArrayElem(15);
    b.SetInteger(4);

    value.EndArray();
  }

  std::string str = doc.Dump<JsonWriter>();
  ASSERT_NE(str.find('['), std::string::npos);
  ASSERT_NE(str.find(']'), std::string::npos);
  ASSERT_NE(str.find("array"), std::string::npos);
  ASSERT_NE(str.find("3.14159E0"), std::string::npos);
  ASSERT_NE(str.find(",4"), std::string::npos);
}

TEST(Json_Experimental, NestedArray) {
  Document doc;
  {
    auto value = doc.CreateMember("array");
    value.SetArray(2);
    auto r_0 = value.GetArrayElem(0);
    auto r_1 = value.GetArrayElem(1);

    r_0.SetArray(2);
    r_1.SetArray(2);
    r_0.GetArrayElem(0).SetFloat(12.3);
    r_0.GetArrayElem(1).SetFloat(13.3);
    r_1.GetArrayElem(0).SetFloat(70);
    r_1.GetArrayElem(1).SetFloat(3);
  }

  auto str = doc.Dump<JsonWriter>();
  ASSERT_EQ(str, R"s({"array":[[1.23E1,1.33E1],[7E1,3E0]]})s");
}

TEST(Json_Experimental, String) {
  Document doc;
  {
    auto value{doc.CreateMember("string-member")};
    ASSERT_TRUE(value.SetString("hello world.").IsString());
  }

  auto str = doc.Dump<JsonWriter>();
  ASSERT_EQ(str, R"str({"string-member":"hello world."})str");
}

TEST(Json_Experimental, Parse_String) {
  {
    std::string json_str = R"({"str":"guess what"})";
    Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});

    auto str = doc.Dump<JsonWriter>();
    ASSERT_EQ(str, json_str);
  }
  {
    std::string json_str = R"({"str":"guess\twhat"})";

    Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});

    auto str = doc.Dump<JsonWriter>();
    ASSERT_EQ(str, R"str({"str":"guess\twhat"})str");
  }
}

TEST(Json_Experimental, Parse_Object) {
  {
    // parse empty object
    std::string json_str = R"({})";
    Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});
    ASSERT_EQ(doc.Errc(), jError::kSuccess);

    auto str = doc.Dump<JsonWriter>();
    ASSERT_EQ(str, R"({})");
  }
  {
    std::string json_str = R"({"rank":1})";
    Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});
    ASSERT_EQ(doc.Errc(), jError::kSuccess);

    auto str = doc.Dump<JsonWriter>();
    ASSERT_EQ(str, R"({"rank":1})");
  }
  {
    std::string json_str = R"({"rank":1.23})";
    Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});
    ASSERT_EQ(doc.Errc(), jError::kSuccess);

    auto str = doc.Dump<JsonWriter>();
    ASSERT_EQ(str, R"({"rank":1.23E0})");
  }
}

TEST(Json_Experimental, Parse_NestedObject) {
  {
    std::string json_str = R"({"ok":{"member-key":1.23E1}})";
    Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});
    ASSERT_EQ(doc.Errc(), jError::kSuccess);

    auto str = doc.Dump<JsonWriter>();
    ASSERT_EQ(str, R"({"ok":{"member-key":1.23E1}})");
  }
}

TEST(Json_Experimental, Parse_Array) {
  std::string json_str = R"s({"arr":[12,2,3,3,4]})s";
  Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});
  ASSERT_EQ(doc.Errc(), jError::kSuccess);

  auto str = doc.Dump<JsonWriter>();
  ASSERT_EQ(str, json_str);
}

TEST(Json_Experimental, Parse_ObjectOfArrays) {
  std::string json_str = R"s({"arr_0":[12,2,3,3,4],"arr_1":[0E0,2,3,3,4]})s";
  Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});
  ASSERT_EQ(doc.Errc(), jError::kSuccess);

  auto str = doc.Dump<JsonWriter>();
  ASSERT_EQ(str, json_str);
}

TEST(Json_Experimental, Parse_ArrayOfObjects) {
  std::string json_str = R"s({"key":[{"o_0":0},{"o_1":1}]})s";
  Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});
  ASSERT_EQ(doc.Errc(), jError::kSuccess);

  auto str = doc.Dump<JsonWriter>();
  ASSERT_EQ(str, json_str);
}

TEST(Json_Experimental, Parse_NestedArray) {
  std::string json_str = R"s({"arr":[[1,2],[3,4],[5,6]]})s";
  Document doc = Document::Load<JsonRecursiveReader>(StringRef{json_str});
  ASSERT_EQ(doc.Errc(), jError::kSuccess);

  auto str = doc.Dump<JsonWriter>();
  ASSERT_EQ(str, json_str);
}

template <typename D>
void TestRoundTrip(D dist) {
  size_t constexpr kElems = 1024;

  std::random_device dev;
  std::vector<float> numbers(kElems);

  for (auto &v : numbers) {
    v = dist(dev);
  }

  std::string str;

  Document doc;
  {
    auto j_numbers = doc.CreateMember("numbers");
    j_numbers.SetArray(kElems);

    for (size_t i = 0; i < numbers.size(); ++i) {
      j_numbers.GetArrayElem(i).SetFloat(numbers[i]);
    }

    j_numbers.EndArray();
    doc.GetObject().EndObject();
  }
  str = doc.Dump<JsonWriter>();
  {
    auto loaded = Document::Load<JsonRecursiveReader>(StringRef {str});
    auto j_numbers = *loaded.GetObject().FindMemberByKey("numbers");
    ASSERT_TRUE(j_numbers.IsArray());

    for (size_t i = 0; i < j_numbers.Length(); ++i) {
      auto v = j_numbers.GetArrayElem(i);
      ASSERT_EQ(v.GetFloat(), numbers[i]);
    }
  }
}

TEST(Json_Experimental, RoundTrip) {
  {
    float f_max = std::numeric_limits<float>::max();
    float f_min = std::numeric_limits<float>::min();
    std::uniform_real_distribution<float> dist(f_min, f_max);
    TestRoundTrip(dist);
  }

  {
    float i_max = std::numeric_limits<int64_t>::max();
    float i_min = std::numeric_limits<int64_t>::min();
    std::uniform_int_distribution<int64_t> dist(i_min, i_max);
    TestRoundTrip(dist);
  }
}

}  // namespace experimental
}  // namespace xgboost
