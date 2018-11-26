#include <sstream>
#include <memory>
#include <limits>
#include <queue>
#include <dmlc/logging.h>
#include "../../../src/common/json.h"
#include "gtest/gtest.h"

namespace {

struct MockBinaryTree {
  struct Node {
    int nodeid;
    float leaf_value;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
  };
  std::unique_ptr<Node> head;
};

std::unique_ptr<MockBinaryTree::Node>
BuildFromKVStore_(const xgboost::serializer::NestedKVStore& kvstore) {
  std::unique_ptr<MockBinaryTree::Node> node(new MockBinaryTree::Node);

  using xgboost::serializer::Get;
  using xgboost::serializer::Object;
  using xgboost::serializer::Integer;
  using xgboost::serializer::Number;

  CHECK_EQ(kvstore.GetValue().TypeStr(), "Object");
  const auto& map = Get<Object>(kvstore).GetObject();
  CHECK_EQ(map.count("nodeid"), 1);
  CHECK_EQ(map.at("nodeid").GetValue().TypeStr(), "Integer");
  node->nodeid = static_cast<int>(Get<Integer>(map.at("nodeid")).GetInteger());
  if (map.count("leaf_value") > 0) {
    CHECK_EQ(map.at("leaf_value").GetValue().TypeStr(), "Number");
    node->leaf_value = Get<Number>(map.at("leaf_value")).GetFloat();
    node->left = nullptr;
    node->right = nullptr;
  } else {
    CHECK_EQ(map.count("left"), 1);
    CHECK_EQ(map.count("right"), 1);
    node->leaf_value = std::numeric_limits<float>::quiet_NaN();
    node->left = BuildFromKVStore_(map.at("left"));
    node->right = BuildFromKVStore_(map.at("right"));
  }
  return node;
}

std::unique_ptr<MockBinaryTree>
BuildFromKVStore(const xgboost::serializer::NestedKVStore& kvstore) {
  std::unique_ptr<MockBinaryTree> tree(new MockBinaryTree);
  tree->head = BuildFromKVStore_(kvstore);
  return tree;
}

}  // namespace anonymous

namespace xgboost {
namespace serializer {

TEST(NestedKVStoreJSON, JSONToKVStoreUnitTest) {
  {
    // JSON uses double quotes, so this must fail
    std::istringstream iss(R"JSON({ 'hello': 'world' })JSON");
    ASSERT_THROW(LoadKVStoreFromJSON(&iss), dmlc::Error);
  }
  {
    // JSON uses string keys exclusively, so this must fail
    std::istringstream iss(R"JSON({ 132: 342 })JSON");
    ASSERT_THROW(LoadKVStoreFromJSON(&iss), dmlc::Error);
  }
  {
    // Extra closing brackets
    std::istringstream iss(R"JSON({ "hello": "world" } } })JSON");
    ASSERT_THROW(LoadKVStoreFromJSON(&iss), dmlc::Error);
  }
  {
    // Missing closing bracket
    std::istringstream iss(R"JSON({ "hello": "world")JSON");
    ASSERT_THROW(LoadKVStoreFromJSON(&iss), dmlc::Error);
  }
  {
    // Stray double quote
    std::istringstream iss(R"JSON({ "hello": "world"")JSON");
    ASSERT_THROW(LoadKVStoreFromJSON(&iss), dmlc::Error);
  }
  {
    // Missing closing double quote
    std::istringstream iss(R"JSON({ "hello": "world)JSON");
    ASSERT_THROW(LoadKVStoreFromJSON(&iss), dmlc::Error);
  }
  {
    // Simple example, with a single key-value pair (string value)
    std::istringstream iss(R"JSON({ "hello": "world" })JSON");
    NestedKVStore kvstore = LoadKVStoreFromJSON(&iss);
    ASSERT_EQ(kvstore.GetValue().TypeStr(), "Object");
    Object obj = Get<Object>(kvstore);
    const auto& map = obj.GetObject();
    ASSERT_EQ(map.size(), 1);
    ASSERT_EQ(map.count("hello"), 1);
    NestedKVStore val = map.at("hello");
    ASSERT_EQ(val.GetValue().TypeStr(), "String");
    const std::string& val_str = Get<String>(val).GetString();
    ASSERT_EQ(val_str, "world");
  }
  {
    // Simple example, with a single key-value pair (float value)
    std::istringstream iss(R"JSON({ "pi": 3.1415 })JSON");
    NestedKVStore kvstore = LoadKVStoreFromJSON(&iss);
    ASSERT_EQ(kvstore.GetValue().TypeStr(), "Object");
    Object obj = Get<Object>(kvstore);
    const auto& map = obj.GetObject();
    ASSERT_EQ(map.size(), 1);
    ASSERT_EQ(map.count("pi"), 1);
    NestedKVStore val = map.at("pi");
    ASSERT_EQ(val.GetValue().TypeStr(), "Number");
    ASSERT_EQ(Get<Number>(val).GetFloat(), 3.1415f);
  }
  {
    // Simple example, with a single key-value pair (integer value)
    std::istringstream iss(R"JSON({ "area-code": 206 })JSON");
    NestedKVStore kvstore = LoadKVStoreFromJSON(&iss);
    ASSERT_EQ(kvstore.GetValue().TypeStr(), "Object");
    Object obj = Get<Object>(kvstore);
    const auto& map = obj.GetObject();
    ASSERT_EQ(map.size(), 1);
    ASSERT_EQ(map.count("area-code"), 1);
    NestedKVStore val = map.at("area-code");
    ASSERT_EQ(val.GetValue().TypeStr(), "Integer");
    ASSERT_EQ(Get<Integer>(val).GetInteger(), static_cast<int64_t>(206LL));
  }
  {
    // Simple example, with top-level list
    std::istringstream iss(R"JSON([1, 2, 3, 4])JSON");
    NestedKVStore kvstore = LoadKVStoreFromJSON(&iss);
    ASSERT_EQ(kvstore.GetValue().TypeStr(), "Array");
    const auto& array = Get<Array>(kvstore).GetArray();
    ASSERT_EQ(array.size(), 4);
    int64_t true_val = 1LL;
    for (const auto& e : array) {
      int64_t val = Get<Integer>(e).GetInteger();
      ASSERT_EQ(val, true_val);
      ++true_val;
    }
  }
  {
    // Nested lists
    std::istringstream iss(R"JSON([[1, 2], [3, 4]])JSON");
    NestedKVStore kvstore = LoadKVStoreFromJSON(&iss);
    ASSERT_EQ(kvstore.GetValue().TypeStr(), "Array");
    const auto& array = Get<Array>(kvstore).GetArray();
    ASSERT_EQ(array.size(), 2);

    ASSERT_EQ(array[0].GetValue().TypeStr(), "Array");
    const auto& array2 = Get<Array>(array[0]).GetArray();
    ASSERT_EQ(array2.size(), 2);

    ASSERT_EQ(array[1].GetValue().TypeStr(), "Array");
    const auto& array3 = Get<Array>(array[1]).GetArray();
    ASSERT_EQ(array3.size(), 2);

    int64_t true_val = 1LL;
    for (const auto& e : array2) {
      int64_t val = Get<Integer>(e).GetInteger();
      ASSERT_EQ(val, true_val);
      ++true_val;
    }
    for (const auto& e : array3) {
      int64_t val = Get<Integer>(e).GetInteger();
      ASSERT_EQ(val, true_val);
      ++true_val;
    }
  }
  {
    // Slightly bigger example, with multiple keys
    std::istringstream iss(R"JSON(
      {
        "hello": "world",
        "t": true,
        "f": false,
        "n": null,
        "i": 123,
        "pi": 3.1415,
        "a": [1, 2, 3, 4],
        "b": [{"foo" : "bar"}],
        "foobar": { "foo" : "bar", "cat" : "dog" }
      }
    )JSON");
    NestedKVStore kvstore = LoadKVStoreFromJSON(&iss);
    ASSERT_EQ(kvstore.GetValue().TypeStr(), "Object");
    Object obj = Get<Object>(kvstore);
    const auto& map = obj.GetObject();
    ASSERT_EQ(map.size(), 9);

    ASSERT_EQ(map.count("hello"), 1);
    NestedKVStore val = map.at("hello");
    ASSERT_EQ(val.GetValue().TypeStr(), "String");
    ASSERT_EQ(Get<String>(val).GetString(), "world");

    ASSERT_EQ(map.count("t"), 1);
    val = map.at("t");
    ASSERT_EQ(val.GetValue().TypeStr(), "Boolean");
    ASSERT_TRUE(Get<Boolean>(val).GetBoolean());

    ASSERT_EQ(map.count("f"), 1);
    val = map.at("f");
    ASSERT_EQ(val.GetValue().TypeStr(), "Boolean");
    ASSERT_FALSE(Get<Boolean>(val).GetBoolean());

    ASSERT_EQ(map.count("n"), 1);
    val = map.at("n");
    ASSERT_EQ(val.GetValue().TypeStr(), "Null");
    ASSERT_NO_THROW(Get<Null>(val));

    ASSERT_EQ(map.count("i"), 1);
    val = map.at("i");
    ASSERT_EQ(val.GetValue().TypeStr(), "Integer");
    ASSERT_EQ(Get<Integer>(val).GetInteger(), 123);

    ASSERT_EQ(map.count("pi"), 1);
    val = map.at("pi");
    ASSERT_EQ(val.GetValue().TypeStr(), "Number");
    ASSERT_EQ(Get<Number>(val).GetFloat(), 3.1415f);

    ASSERT_EQ(map.count("a"), 1);
    val = map.at("a");
    ASSERT_EQ(val.GetValue().TypeStr(), "Array");
    const auto& array2 = Get<Array>(val).GetArray();
    ASSERT_EQ(array2.size(), 4);
    // values of array elements are 1, 2, 3, 4
    int true_val = 1;
    for (const auto& e : array2) {
      ASSERT_EQ(e.GetValue().TypeStr(), "Integer");
      ASSERT_EQ(Get<Integer>(e).GetInteger(), true_val);
      ++true_val;
    }

    ASSERT_EQ(map.count("b"), 1);
    val = map.at("b");
    ASSERT_EQ(val.GetValue().TypeStr(), "Array");
    const auto& array3 = Get<Array>(val).GetArray();
    ASSERT_EQ(array3.size(), 1);
    ASSERT_EQ(array3[0].GetValue().TypeStr(), "Object");
    const auto& dict = Get<Object>(array3[0]).GetObject();
    ASSERT_EQ(dict.size(), 1);
    ASSERT_EQ(dict.count("foo"), 1);
    NestedKVStore val2 = dict.at("foo");
    ASSERT_EQ(val2.GetValue().TypeStr(), "String");
    ASSERT_EQ(Get<String>(val2).GetString(), "bar");

    ASSERT_EQ(map.count("foobar"), 1);
    val = map.at("foobar");
    ASSERT_EQ(val.GetValue().TypeStr(), "Object");
    const auto& dict2 = Get<Object>(val).GetObject();
    ASSERT_EQ(dict2.size(), 2);
    ASSERT_EQ(dict2.count("foo"), 1);
    val2 = dict2.at("foo");
    ASSERT_EQ(val2.GetValue().TypeStr(), "String");
    ASSERT_EQ(Get<String>(val2).GetString(), "bar");
    ASSERT_EQ(dict2.count("cat"), 1);
    val2 = dict2.at("cat");
    ASSERT_EQ(val2.GetValue().TypeStr(), "String");
    ASSERT_EQ(Get<String>(val2).GetString(), "dog");
  }
}

TEST(NestedKVStoreJSON, MockBinaryTreeTest) {
  std::istringstream iss(R"JSON(
    {
      "nodeid": 0,
      "left": {
        "nodeid": 1,
        "left" : {
          "nodeid": 3,
          "leaf_value": -1.0
        },
        "right" : {
          "nodeid": 4,
          "left" : {
            "nodeid": 9,
            "leaf_value": 1.0
          },
          "right" : {
            "nodeid": 10,
            "leaf_value": 1.5
          }
        }
      },
      "right": {
        "nodeid": 2,
        "left" : {
          "nodeid": 5,
          "leaf_value": -1.0
        },
        "right" : {
          "nodeid": 6,
          "leaf_value": 0.5
        }
      }
    }
  )JSON");
  NestedKVStore kvstore = LoadKVStoreFromJSON(&iss);
  std::unique_ptr<MockBinaryTree> tree = BuildFromKVStore(kvstore);

  // Traverse the tree in breadth-first order
  std::queue<const MockBinaryTree::Node*> Q;
  std::vector<std::string> traversal;
  Q.push(tree->head.get());
  while (!Q.empty()) {
    const MockBinaryTree::Node* ptr = Q.front();
    Q.pop();
    {
      std::ostringstream oss;
      oss << "nodeid = " << ptr->nodeid;
      traversal.push_back(oss.str());
    }
    if (ptr->left) {
      Q.push(ptr->left.get());
      Q.push(ptr->right.get());
    } else {
      {
        std::ostringstream oss;
        oss << "leaf_value = " << ptr->leaf_value;
        traversal.push_back(oss.str());
      }
    }
  }
  const std::vector<std::string> traversal_expected{
    "nodeid = 0", "nodeid = 1", "nodeid = 2", "nodeid = 3",
    "leaf_value = -1", "nodeid = 4", "nodeid = 5", "leaf_value = -1",
    "nodeid = 6", "leaf_value = 0.5", "nodeid = 9", "leaf_value = 1",
    "nodeid = 10", "leaf_value = 1.5"
  };
  ASSERT_EQ(traversal, traversal_expected);
}

}  // namespace serializer
}  // namespace xgboost
