/**
 * Copyright (c) 2019-2023, XGBoost Contributors
 */
#include <gtest/gtest.h>

#include <fstream>
#include <map>

#include "../../../src/common/charconv.h"
#include "../../../src/common/io.h"
#include "../../../src/common/threading_utils.h"  // for ParallelFor
#include "../filesystem.h"                        // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "dmlc/logging.h"
#include "xgboost/json.h"
#include "xgboost/json_io.h"
#include "xgboost/logging.h"

namespace xgboost {

std::string GetModelStr() {
  std::string model_json = R"json(
{
  "model_parameter": {
    "base_score": "0.5",
    "num_class": "0",
    "num_feature": "10"
  },
  "train_parameter": {
    "debug_verbose": "0",
    "disable_default_eval_metric": "0",
    "nthread": "0",
    "seed": "0",
    "seed_per_iteration": "0",
    "test_flag": "",
    "tree_method": "gpu_hist"
  },
  "configuration": {
    "booster": "gbtree",
    "gpu_id": "0",
    "num_class": "0",
    "num_feature": "10",
    "objective": "reg:linear",
    "predictor": "gpu_predictor",
    "tree_method": "gpu_hist",
    "updater": "grow_gpu_hist"
  },
  "objective": "reg:linear",
  "booster": "gbtree",
  "gbm": {
    "GBTreeModelParam": {
      "num_feature": "10",
      "num_output_group": "1",
      "num_roots": "1",
      "size_leaf_vector": "0"
    },
    "trees": [{
        "TreeParam": {
          "num_feature": "10",
          "num_roots": "1",
          "size_leaf_vector": "0"
        },
        "num_nodes": "9",
        "nodes": [
          {
            "depth": 0,
            "gain": 31.8892,
            "hess": 10,
            "left": 1,
            "missing": 1,
            "nodeid": 0,
            "right": 2,
            "split_condition": 0.580717,
            "split_index": 2
          },
          {
            "depth": 1,
            "gain": 1.5625,
            "hess": 3,
            "left": 5,
            "missing": 5,
            "nodeid": 2,
            "right": 6,
            "split_condition": 0.160345,
            "split_index": 0
          },
          {
            "depth": 2,
            "gain": 0.25,
            "hess": 2,
            "left": 7,
            "missing": 7,
            "nodeid": 6,
            "right": 8,
            "split_condition": 0.62788,
            "split_index": 0
          },
          {
            "hess": 1,
            "leaf": 0.375,
            "nodeid": 8
          },
          {
            "hess": 1,
            "leaf": 0.075,
            "nodeid": 7
          },
          {
            "hess": 1,
            "leaf": -0.075,
            "nodeid": 5
          },
          {
            "depth": 3,
            "gain": 10.4866,
            "hess": 7,
            "left": 3,
            "missing": 3,
            "nodeid": 1,
            "right": 4,
            "split_condition": 0.238748,
            "split_index": 1
          },
          {
            "hess": 6,
            "leaf": 1.54286,
            "nodeid": 4
          },
          {
            "hess": 1,
            "leaf": 0.225,
            "nodeid": 3
          }
        ],
        "leaf_vector": []
      }],
    "tree_info": [0]
  }
}
)json";
  return model_json;
}

TEST(Json, TestParseObject) {
  std::string str = R"obj({"TreeParam" : {"num_feature": "10"}})obj";
  auto json = Json::Load(StringView{str.c_str(), str.size()});
}

TEST(Json, ParseNumber) {
  {
    std::string str = "31.8892";
    auto json = Json::Load(StringView{str.c_str(), str.size()});
    ASSERT_EQ(get<JsonNumber>(json), 31.8892f);
  }
  {
    std::string str = "-31.8892";
    auto json = Json::Load(StringView{str.c_str(), str.size()});
    ASSERT_EQ(get<JsonNumber>(json), -31.8892f);
  }
  {
    std::string str = "2e4";
    auto json = Json::Load(StringView{str.c_str(), str.size()});
    ASSERT_EQ(get<JsonNumber>(json), 2e4f);
  }
  {
    std::string str = "2e-4";
    auto json = Json::Load(StringView{str.c_str(), str.size()});
    ASSERT_EQ(get<JsonNumber>(json), 2e-4f);
  }
  {
    std::string str = "-2e-4";
    auto json = Json::Load(StringView{str.c_str(), str.size()});
    ASSERT_EQ(get<JsonNumber>(json), -2e-4f);
  }
  {
    std::string str = "-0.0";
    auto json = Json::Load(StringView{str.c_str(), str.size()});
    ASSERT_TRUE(std::signbit(get<JsonNumber>(json)));
    ASSERT_EQ(get<JsonNumber>(json), -0);
  }
  {
    std::string str = "-5.37645816802978516e-01";
    auto json = Json::Load(StringView{str.c_str(), str.size()});
    ASSERT_TRUE(std::signbit(get<JsonNumber>(json)));
    // Larger than fast path limit.
    ASSERT_EQ(get<JsonNumber>(json), -5.37645816802978516e-01);
  }
  {
    std::string str = "9.86623668670654297e+00";
    auto json = Json::Load(StringView{str.c_str(), str.size()});
    ASSERT_FALSE(std::signbit(get<JsonNumber>(json)));
    ASSERT_EQ(get<JsonNumber>(json), 9.86623668670654297e+00);
  }
}

TEST(Json, ParseArray) {
  std::string str = R"json(
{
    "nodes": [
        {
	    "depth": 3,
	    "gain": 10.4866,
	    "hess": 7,
	    "left": 3,
	    "missing": 3,
	    "nodeid": 1,
	    "right": 4,
	    "split_condition": 0.238748,
	    "split_index": 1
        },
        {
	    "hess": 6,
	    "leaf": 1.54286,
	    "nodeid": 4
        },
        {
	    "hess": 1,
	    "leaf": 0.225,
	    "nodeid": 3
        }
    ]
}
)json";
  auto json = Json::Load(StringView{str.c_str(), str.size()});
  json = json["nodes"];
  std::vector<Json> arr = get<JsonArray>(json);
  ASSERT_EQ(arr.size(), 3ul);
  Json v0 = arr[0];
  ASSERT_EQ(get<Integer>(v0["depth"]), 3);
  ASSERT_NEAR(get<Number>(v0["gain"]), 10.4866, kRtEps);

  {
    std::string str =
        "[5.04713470458984375e+02,9.86623668670654297e+00,4.94847229003906250e+"
        "02,2.13924217224121094e+00,7.72699451446533203e+00,2."
        "30380615234375000e+02,2.64466613769531250e+02]";
    auto json = Json::Load(StringView{str.c_str(), str.size()});

    auto const& vec = get<Array const>(json);
    ASSERT_EQ(get<Number const>(vec[0]), 5.04713470458984375e+02);
    ASSERT_EQ(get<Number const>(vec[1]), 9.86623668670654297e+00);
    ASSERT_EQ(get<Number const>(vec[2]), 4.94847229003906250e+02);
    ASSERT_EQ(get<Number const>(vec[3]), 2.13924217224121094e+00);
    ASSERT_EQ(get<Number const>(vec[4]), 7.72699451446533203e+00);
    ASSERT_EQ(get<Number const>(vec[5]), 2.30380615234375000e+02);
    ASSERT_EQ(get<Number const>(vec[6]), 2.64466613769531250e+02);
  }
}

TEST(Json, Null) {
  Json json {JsonNull()};
  std::string ss;
  Json::Dump(json, &ss);
  ASSERT_EQ(ss, "null");

  std::string null_input {R"null({"key":  null })null"};

  json = Json::Load({null_input.c_str(), null_input.size()});
  ASSERT_TRUE(IsA<Null>(json["key"]));

  std::string dumped;
  Json::Dump(json, &dumped, std::ios::binary);
  ASSERT_TRUE(IsA<Null>(Json::Load(StringView{dumped}, std::ios::binary)["key"]));
}

TEST(Json, EmptyObject) {
  std::string str = R"json(
{
  "rank": 1,
  "statistic": {

  }
}
)json";
  std::stringstream iss(str);
  auto json = Json::Load(StringView{str.c_str(), str.size()});
  ASSERT_TRUE(IsA<Object>(json["statistic"]));

  str = R"json({"Config": {},"Model": {}})json"; // NOLINT
  json = Json::Load(StringView{str.c_str(), str.size()});
  ASSERT_TRUE(IsA<Object>(json["Model"]));
}

TEST(Json, EmptyArray) {
  std::string str = R"json(
{
  "leaf_vector": []
}
)json";
  std::istringstream iss(str);
  auto json = Json::Load(StringView{str.c_str(), str.size()});
  auto arr = get<JsonArray>(json["leaf_vector"]);
  ASSERT_EQ(arr.size(), 0ul);
}

TEST(Json, Boolean) {
  std::string str = R"json(
{
  "left_child": true,
  "right_child": false
}
)json";
  Json j {Json::Load(StringView{str.c_str(), str.size()})};
  ASSERT_EQ(get<JsonBoolean>(j["left_child"]), true);
  ASSERT_EQ(get<JsonBoolean>(j["right_child"]), false);

  std::string dumped;
  Json::Dump(j, &dumped, std::ios::binary);
  ASSERT_TRUE(get<Boolean const>(Json::Load(StringView{dumped}, std::ios::binary)["left_child"]));
}

TEST(Json, Indexing) {
  auto str = GetModelStr();
  JsonReader reader(StringView{str.c_str(), str.size()});
  Json j {Json::Load(&reader)};
  auto& value_1 = j["model_parameter"];
  auto& value = value_1["base_score"];
  std::string result = Cast<JsonString>(&value.GetValue())->GetString();

  ASSERT_EQ(result, "0.5");
}

TEST(Json, AssigningObjects) {
  {
    Json json;
    json = JsonObject();
    json["Okay"] = JsonArray();
    ASSERT_EQ(get<JsonArray>(json["Okay"]).size(), 0ul);
  }

  {
    std::map<std::string, Json> objects;
    Json json_objects { JsonObject() };
    std::vector<Json> arr_0 (1, Json(3.3f));
    json_objects["tree_parameters"] = JsonArray(arr_0);
    std::vector<Json> json_arr = get<JsonArray>(json_objects["tree_parameters"]);
    ASSERT_NEAR(get<JsonNumber>(json_arr[0]), 3.3f, kRtEps);
  }

  {
    Json json_object { JsonObject() };
    auto str = JsonString("1");
    auto& k = json_object["1"];
    k  = std::move(str);
    ASSERT_TRUE(str.GetString().empty());  // NOLINT
    auto& m = json_object["1"];
    std::string value = get<JsonString>(m);
    ASSERT_EQ(value, "1");
    ASSERT_EQ(get<JsonString>(json_object["1"]), "1");
  }
}

TEST(Json, AssigningArray) {
  Json json;
  json = JsonArray();
  std::vector<Json> tmp_0 {Json(Number(1.0f)), Json(Number(2.0f))};
  json = tmp_0;
  std::vector<Json> tmp_1 {Json(Number(3.0f))};
  get<Array>(json) = tmp_1;
  std::vector<Json> res = get<Array>(json);
  ASSERT_EQ(get<Number>(res[0]), 3);
}

TEST(Json, AssigningNumber) {
  {
    // right value
    Json json = Json{ Number(4.0f) };
    get<Number>(json) = 15;
    ASSERT_EQ(get<Number>(json), 15);
  }

  {
    // left value ref
    Json json = Json{ Number(4.0f) };
    Number::Float& ref = get<Number>(json);
    ref = 15;
    ASSERT_EQ(get<Number>(json), 15);
  }

  {
    // left value
    Json json = Json{ Number(4.0f) };
    double value = get<Number>(json);
    ASSERT_EQ(value, 4);
    value = 15;  // NOLINT
    ASSERT_EQ(get<Number>(json), 4);
  }

  {
    Json value {Number(std::numeric_limits<float>::quiet_NaN())};
    ASSERT_TRUE(IsA<Number>(value));
  }
}

TEST(Json, AssigningString) {
  {
    // right value
    Json json = Json{ String("str") };
    get<String>(json) = "modified";
    ASSERT_EQ(get<String>(json), "modified");
  }

  {
    // left value ref
    Json json = Json{ String("str") };
    std::string& ref = get<String>(json);
    ref = "modified";
    ASSERT_EQ(get<String>(json), "modified");
  }

  {
    // left value
    Json json = Json{ String("str") };
    std::string value = get<String>(json);
    value = "modified";
    ASSERT_EQ(get<String>(json), "str");
  }
}

TEST(Json, LoadDump) {
  std::string ori_buffer = GetModelStr();
  Json origin {Json::Load(StringView{ori_buffer.c_str(), ori_buffer.size()})};

  dmlc::TemporaryDirectory tempdir;
  auto const& path = tempdir.path + "test_model_dump";

  std::string out;
  Json::Dump(origin, &out);

  std::ofstream fout(path);
  ASSERT_TRUE(fout);
  fout << out << std::flush;

  std::string new_buffer = common::LoadSequentialFile(path);

  Json load_back {Json::Load(StringView(new_buffer.c_str(), new_buffer.size()))};
  ASSERT_EQ(load_back, origin);
}

TEST(Json, Invalid) {
  {
    std::string str = "}";
    bool has_thrown = false;
    try {
      Json load{Json::Load(StringView(str.c_str(), str.size()))};
    } catch (dmlc::Error const &e) {
      std::string msg = e.what();
      ASSERT_NE(msg.find("Unknown"), std::string::npos);
      has_thrown = true;
    };
    ASSERT_TRUE(has_thrown);
  }
  {
    std::string str = R"json({foo)json";
    bool has_thrown = false;
    try {
      Json load{Json::Load(StringView(str.c_str(), str.size()))};
    } catch (dmlc::Error const &e) {
      std::string msg = e.what();
      ASSERT_NE(msg.find("position: 1"), std::string::npos);
      has_thrown = true;
    };
    ASSERT_TRUE(has_thrown);
  }
  {
    std::string str = R"json({"foo")json";
    bool has_thrown = false;
    try {
      Json load{Json::Load(StringView(str.c_str(), str.size()))};
    } catch (dmlc::Error const& e) {
      std::string msg = e.what();
      // EOF is printed as 255 on s390x
      ASSERT_TRUE(msg.find("EOF") != std::string::npos || msg.find("255") != std::string::npos);
      has_thrown = true;
    };
    ASSERT_TRUE(has_thrown);
  }
}

// For now Json is quite ignorance about unicode.
TEST(Json, CopyUnicode) {
  std::string json_str = R"json(
{"m": ["\ud834\udd1e", "\u20ac", "\u0416", "\u00f6"]}
)json";
  Json loaded {Json::Load(StringView{json_str.c_str(), json_str.size()})};

  std::string dumped_string;
  Json::Dump(loaded, &dumped_string);

  ASSERT_NE(dumped_string.find("\\u20ac"), std::string::npos);
}

TEST(Json, WrongCasts) {
  {
    Json json = Json{ String{"str"} };
    ASSERT_ANY_THROW(get<Number>(json));
  }
  {
    Json json = Json{ Array{ std::vector<Json>{ Json{ Number{1.0f} } } } };
    ASSERT_ANY_THROW(get<Number>(json));
  }
  {
    Json json = Json{Object{{{"key", Json{String{"value"}}}}}};
    ASSERT_ANY_THROW(get<Number>(json));
  }
}

TEST(Json, Integer) {
  for (int64_t i = 1; i < 10000; i *= 10) {
    auto ten = Json{Integer{i}};
    std::string str;
    Json::Dump(ten, &str);
    ASSERT_EQ(str, std::to_string(i));
  }
}

TEST(Json, IntVSFloat) {
  // If integer is parsed as float, calling `get<Integer>()' will throw.
  {
    std::string str = R"json(
{
  "number": 123.4,
  "integer": 123
})json";

    Json obj = Json::Load({str.c_str(), str.size()});
    JsonNumber::Float number = get<Number>(obj["number"]);
    ASSERT_NEAR(number, 123.4f, kRtEps);
    JsonInteger::Int integer = get<Integer>(obj["integer"]);
    ASSERT_EQ(integer, 123);
  }

  {
    std::string str = R"json(
{"data": [2503595760, false], "shape": [10]}
)json";
    Json obj = Json::Load({str.c_str(), str.size()});
    auto array = get<Array>(obj["data"]);
    auto ptr = get<Integer>(array[0]);
    ASSERT_EQ(ptr, 2503595760);
  }
}

namespace {
void TestRroundTrip(std::ios::openmode mode) {
  uint32_t i = 0;
  SimpleLCG rng;
  SimpleRealUniformDistribution<float> dist(1.0f, 4096.0f);

  while (i <= std::numeric_limits<uint32_t>::max()) {
    float f;
    std::memcpy(&f, &i, sizeof(f));

    Json jf{f};
    std::string str;
    Json::Dump(jf, &str, mode);
    auto loaded = Json::Load(StringView{str}, mode);
    if (XGBOOST_EXPECT(std::isnan(f), false)) {
      ASSERT_TRUE(std::isnan(get<Number const>(loaded)));
    } else {
      ASSERT_EQ(get<Number const>(loaded), f);
    }

    auto t = i;
    i += static_cast<uint32_t>(dist(&rng));
    if (i < t) {
      break;
    }
  }
}
}  // namespace

TEST(Json, RoundTrip) {
  TestRroundTrip(std::ios::out);
  TestRroundTrip(std::ios::binary);
}

TEST(Json, DISABLED_RoundTripExhaustive) {
  auto test = [](uint32_t i) {
    float f;
    std::memcpy(&f, &i, sizeof(f));

    Json jf{f};
    std::string str;
    Json::Dump(jf, &str);
    auto loaded = Json::Load({str.c_str(), str.size()});
    if (XGBOOST_EXPECT(std::isnan(f), false)) {
      EXPECT_TRUE(std::isnan(get<Number const>(loaded)));
    } else {
      EXPECT_EQ(get<Number const>(loaded), f);
    }
  };
  int64_t int32_max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
  Context ctx;
  common::ParallelFor(int32_max, ctx.Threads(), [&](auto i) { test(static_cast<uint32_t>(i)); });
}

TEST(Json, TypedArray) {
  size_t n = 16;
  F32Array f32{n};
  std::iota(f32.GetArray().begin(), f32.GetArray().end(), -8);
  U8Array u8{n};
  std::iota(u8.GetArray().begin(), u8.GetArray().end(), 0);
  I32Array i32{n};
  std::iota(i32.GetArray().begin(), i32.GetArray().end(), -8);
  I64Array i64{n};
  std::iota(i64.GetArray().begin(), i64.GetArray().end(), -8);

  Json json{Object{}};
  json["u8"] = std::move(u8);
  ASSERT_TRUE(IsA<U8Array>(json["u8"]));
  json["f32"] = std::move(f32);
  ASSERT_TRUE(IsA<F32Array>(json["f32"]));
  json["i32"] = std::move(i32);
  ASSERT_TRUE(IsA<I32Array>(json["i32"]));
  json["i64"] = std::move(i64);
  ASSERT_TRUE(IsA<I64Array>(json["i64"]));

  std::string str;
  Json::Dump(json, &str);
  {
    auto loaded = Json::Load(StringView{str});
    // for text output there's no typed array.
    ASSERT_TRUE(IsA<Array>(loaded["u8"]));
    auto const& arr = loaded["f32"];
    for (int32_t i = -8; i < 8; ++i) {
      ASSERT_EQ(get<Number>(arr[i + 8]), i);
    }
  }

  std::string binary;
  Json::Dump(json, &binary, std::ios::binary);
  {
    auto loaded = Json::Load(StringView{binary}, std::ios::binary);
    ASSERT_TRUE(IsA<U8Array>(loaded["u8"]));
    auto const& arr = get<F32Array>(loaded["f32"]);
    for (int32_t i = -8; i < 8; ++i) {
      ASSERT_EQ(arr[i + 8], i);
    }
  }
}

TEST(UBJson, Basic) {
  auto run_test = [](StringView str) {
    auto json = Json::Load(str);
    std::vector<char> stream;
    UBJWriter writer{&stream};
    Json::Dump(json, &writer);
    {
      std::ofstream fout{"test.ubj", std::ios::binary | std::ios::out};
      fout.write(stream.data(), stream.size());
    }

    auto data = common::LoadSequentialFile("test.ubj");
    UBJReader reader{StringView{data}};
    json = reader.Load();
    return json;
  };
  {
    // empty
    auto ret = run_test(R"({})");
    std::stringstream ss;
    ss << ret;
    ASSERT_EQ(ss.str(), "{}");
  }
  {
    auto ret = run_test(R"({"":[]})");
    std::stringstream ss;
    ss << ret;
    ASSERT_EQ(ss.str(), R"({"":[]})");
  }
  {
    // basic
    auto ret = run_test(R"({"test": [2.71, 3.14, Infinity]})");
    ASSERT_TRUE(std::isinf(get<Number>(get<Array>(ret["test"])[2])));
    ASSERT_FLOAT_EQ(3.14, get<Number>(get<Array>(ret["test"])[1]));
    ASSERT_FLOAT_EQ(2.71, get<Number>(get<Array>(ret["test"])[0]));
  }
}

TEST(Json, TypeCheck) {
  Json config{Object{}};
  config["foo"] = String{"bar"};
  auto test = [&]() { TypeCheck<Number, Integer, Array, I32Array>(config["foo"], "foo"); };
  ASSERT_THROW({ test(); }, dmlc::Error);
  try {
    test();
  } catch (dmlc::Error const& e) {
    auto err = std::string{e.what()};
    ASSERT_NE(err.find("Number"), std::string::npos);
    ASSERT_NE(err.find("I32Array"), std::string::npos);
    ASSERT_NE(err.find("foo"), std::string::npos);
  }
}
}  // namespace xgboost
