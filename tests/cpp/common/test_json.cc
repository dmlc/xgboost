/*!
 * Copyright (c) by Contributors 2019
 */
#include <gtest/gtest.h>
#include <dmlc/filesystem.h>
#include <fstream>
#include <map>

#include "xgboost/json.h"
#include "xgboost/logging.h"
#include "xgboost/json_io.h"
#include "../../../src/common/io.h"

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
    "dsplit": "auto",
    "nthread": "0",
    "seed": "0",
    "seed_per_iteration": "0",
    "test_flag": "",
    "tree_method": "gpu_hist"
  },
  "configuration": {
    "booster": "gbtree",
    "n_gpus": "1",
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
  std::string str = "31.8892";
  auto json = Json::Load(StringView{str.c_str(), str.size()});
  ASSERT_NEAR(get<JsonNumber>(json), 31.8892, kRtEps);
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
  auto json = Json::Load(StringView{str.c_str(), str.size()}, true);
  json = json["nodes"];
  std::vector<Json> arr = get<JsonArray>(json);
  ASSERT_EQ(arr.size(), 3);
  Json v0 = arr[0];
  ASSERT_EQ(get<JsonNumber>(v0["depth"]), 3);
}

TEST(Json, EmptyArray) {
  std::string str = R"json(
{
  "leaf_vector": []
}
)json";
  std::istringstream iss(str);
  auto json = Json::Load(StringView{str.c_str(), str.size()}, true);
  auto arr = get<JsonArray>(json["leaf_vector"]);
  ASSERT_EQ(arr.size(), 0);
}

TEST(Json, Boolean) {
  std::string str = R"json(
{
  "left_child": true,
  "right_child": false
}
)json";
  Json j {Json::Load(StringView{str.c_str(), str.size()}, true)};
  ASSERT_EQ(get<JsonBoolean>(j["left_child"]), true);
  ASSERT_EQ(get<JsonBoolean>(j["right_child"]), false);
}

TEST(Json, Indexing) {
  auto str = GetModelStr();
  JsonReader reader(StringView{str.c_str(), str.size()}, true);
  Json j {Json::Load(&reader)};
  auto& value_1 = j["model_parameter"];
  auto& value = value_1["base_score"];
  std::string result = Cast<JsonString>(&value.GetValue())->getString();

  ASSERT_EQ(result, "0.5");
}

TEST(Json, AssigningObjects) {
  {
    Json json;
    json = JsonObject();
    json["Okay"] = JsonArray();
    ASSERT_EQ(get<JsonArray>(json["Okay"]).size(), 0);
  }

  {
    std::map<std::string, Json> objects;
    Json json_objects { JsonObject() };
    std::vector<Json> arr_0 (1, Json(3.3));
    json_objects["tree_parameters"] = JsonArray(arr_0);
    std::vector<Json> json_arr = get<JsonArray>(json_objects["tree_parameters"]);
    ASSERT_NEAR(get<JsonNumber>(json_arr[0]), 3.3, kRtEps);
  }

  {
    Json json_object { JsonObject() };
    auto str = JsonString("1");
    auto& k = json_object["1"];
    k  = str;
    auto& m = json_object["1"];
    std::string value = get<JsonString>(m);
    ASSERT_EQ(value, "1");
    ASSERT_EQ(get<JsonString>(json_object["1"]), "1");
  }
}

TEST(Json, AssigningArray) {
  Json json;
  json = JsonArray();
  std::vector<Json> tmp_0 {Json(Number(1)), Json(Number(2))};
  json = tmp_0;
  std::vector<Json> tmp_1 {Json(Number(3))};
  get<Array>(json) = tmp_1;
  std::vector<Json> res = get<Array>(json);
  ASSERT_EQ(get<Number>(res[0]), 3);
}

TEST(Json, AssigningNumber) {
  {
    // right value
    Json json = Json{ Number(4) };
    get<Number>(json) = 15;
    ASSERT_EQ(get<Number>(json), 15);
  }

  {
    // left value ref
    Json json = Json{ Number(4) };
    Number::Float& ref = get<Number>(json);
    ref = 15;
    ASSERT_EQ(get<Number>(json), 15);
  }

  {
    // left value
    Json json = Json{ Number(4) };
    double value = get<Number>(json);
    ASSERT_EQ(value, 4);
    value = 15;  // NOLINT
    ASSERT_EQ(get<Number>(json), 4);
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
  std::string buffer = GetModelStr();
  Json origin {Json::Load(StringView{buffer.c_str(), buffer.size()}, true)};

  dmlc::TemporaryDirectory tempdir;
  auto const& path = tempdir.path + "test_model_dump";

  std::ofstream fout (path);
  Json::Dump(origin, &fout);
  fout.close();

  buffer = common::LoadSequentialFile(path);
  Json load_back {Json::Load(StringView(buffer.c_str(), buffer.size()), true)};

  ASSERT_EQ(load_back, origin);
}

// For now Json is quite ignorance about unicode.
TEST(Json, CopyUnicode) {
  std::string json_str = R"json(
{"m": ["\ud834\udd1e", "\u20ac", "\u0416", "\u00f6"]}
)json";
  Json loaded {Json::Load(StringView{json_str.c_str(), json_str.size()}, true)};

  std::stringstream ss_1;
  Json::Dump(loaded, &ss_1);

  std::string dumped_string = ss_1.str();
  ASSERT_NE(dumped_string.find("\\u20ac"), std::string::npos);
}

TEST(Json, WrongCasts) {
  {
    Json json = Json{ String{"str"} };
    ASSERT_ANY_THROW(get<Number>(json));
  }
  {
    Json json = Json{ Array{ std::vector<Json>{ Json{ Number{1} } } } };
    ASSERT_ANY_THROW(get<Number>(json));
  }
  {
    Json json = Json{ Object{std::map<std::string, Json>{
          {"key", Json{String{"value"}}}} } };
    ASSERT_ANY_THROW(get<Number>(json));
  }
}

}  // namespace xgboost
