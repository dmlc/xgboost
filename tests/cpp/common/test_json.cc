#include "../../../src/common/json.h"
#include <fstream>
#include <map>

#include <gtest/gtest.h>

namespace xgboost {
namespace json {

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
  std::string str = "{\"TreeParam\" : {\"num_feature\": \"10\"}}";
  std::istringstream iss(str);
  auto json = json::Json::Load(&iss);
}

TEST(Json, ParseNumber) {
  std::string str = "31.8892";
  std::istringstream iss(str);
  auto json = json::Json::Load(&iss);
  ASSERT_EQ(Get<JsonNumber>(json).GetDouble(), 31.8892);
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
  std::istringstream iss(str);
  auto json = json::Json::Load(&iss);
  json = json["nodes"];
  auto arr = Get<JsonArray>(json).GetArray();
  ASSERT_EQ(arr.size(), 3);
  auto v0 = arr[0];
  ASSERT_EQ(Get<JsonNumber>(v0["depth"]).GetDouble(), 3);
}

TEST(Json, EmptyArray) {
  std::string str = R"json(
{
  "leaf_vector": []
}
)json";
  std::istringstream iss(str);
  auto json = json::Json::Load(&iss);
  auto arr = Get<JsonArray>(json["leaf_vector"]).GetArray();
  ASSERT_EQ(arr.size(), 0);
}

TEST(Json, Boolean) {
  std::string str = R"json(
{
  "left_child": true,
  "right_child": false
}
)json";
  std::stringstream ss(str);
  Json j {json::Json::Load(&ss)};
  ASSERT_EQ(
      json::Get<JsonBoolean>(j["left_child"]).GetBoolean(), true);
  ASSERT_EQ(
      json::Get<JsonBoolean>(j["right_child"]).GetBoolean(), false);
}

TEST(Json, Indexing) {
  std::stringstream ss(GetModelStr());
  Json j {json::Json::Load(&ss)};

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
    ASSERT_EQ(Get<JsonArray>(json["Okay"]).GetArray().size(), 0);
  }

  {
    std::map<std::string, Json> objects;
    Json json_objects {JsonObject()};
    std::vector<Json> arr_0 (1, Json(3.3));
    json_objects["tree_parameters"] = JsonArray(arr_0);
    std::vector<Json> json_arr = Get<JsonArray>(json_objects["tree_parameters"]).GetArray();
    ASSERT_EQ(Get<JsonNumber>(json_arr[0]).GetDouble(), 3.3);
  }

  {
    Json json_object { JsonObject() };
    auto str = JsonString("1");
    auto& k = json_object["1"];
    k  = str;
    auto& m = json_object["1"];
    std::string value = Get<JsonString>(m).GetString();
    ASSERT_EQ(value, "1");
    ASSERT_EQ(Get<JsonString>(json_object["1"]).GetString(), "1");
  }
}

TEST(Json, LoadDump) {
  std::stringstream ss(GetModelStr());
  Json origin {json::Json::Load(&ss)};

  std::ofstream fout ("/tmp/model_dump.json");
  json::Json::Dump(origin, &fout);
  fout.close();

  std::ifstream fin ("/tmp/model_dump.json");
  Json load_back {json::Json::Load(&fin)};
  fin.close();

  ASSERT_EQ(load_back, origin);

  load_back["new_entry"] = json::Number(23.2);
  ASSERT_FALSE(load_back == origin);
}

// For now Json is quite ignorance about unicode.
TEST(Json, CopyUnicode) {
  std::string json_str = R"json(
{"m": ["\ud834\udd1e", "\u20ac", "\u0416", "\u00f6"]}
)json";
  std::stringstream ss_0(json_str);
  Json loaded {json::Json::Load(&ss_0)};

  std::stringstream ss_1;
  Json::Dump(loaded, &ss_1);

  std::string dumped_string = ss_1.str();
  ASSERT_NE(dumped_string.find("\\u20ac"), std::string::npos);
}

}  // namespace json
}  // namespace xgboost
