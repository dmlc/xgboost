/**
 * Copyright 2015-2023 by Contributors
 * \file tree_model.cc
 * \brief model structure for tree
 */
#include <dmlc/json.h>
#include <dmlc/registry.h>
#include <xgboost/json.h>
#include <xgboost/tree_model.h>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>

#include "../common/categorical.h"
#include "../common/common.h"
#include "../predictor/predict_fn.h"
#include "io_utils.h"  // GetElem
#include "param.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/logging.h"

namespace xgboost {
// register tree parameter
DMLC_REGISTER_PARAMETER(TreeParam);

namespace tree {
DMLC_REGISTER_PARAMETER(TrainParam);
}

/*!
 * \brief Base class for dump model implementation, modeling closely after code generator.
 */
class TreeGenerator {
 protected:
  static int32_t constexpr kFloatMaxPrecision =
      std::numeric_limits<bst_float>::max_digits10;
  FeatureMap const& fmap_;
  std::stringstream ss_;
  bool const with_stats_;

  template <typename Float>
  static std::string ToStr(Float value) {
    static_assert(std::is_floating_point<Float>::value,
                  "Use std::to_string instead for non-floating point values.");
    std::stringstream ss;
    ss << std::setprecision(kFloatMaxPrecision) << value;
    return ss.str();
  }

  static std::string Tabs(uint32_t n) {
    std::string res;
    for (uint32_t i = 0; i < n; ++i) {
      res += '\t';
    }
    return res;
  }
  /* \brief Find the first occurrence of key in input and replace it with corresponding
   *        value.
   */
  static std::string Match(std::string const& input,
                           std::map<std::string, std::string> const& replacements) {
    std::string result = input;
    for (auto const& kv : replacements) {
      auto pos = result.find(kv.first);
      CHECK_NE(pos, std::string::npos);
      result.replace(pos, kv.first.length(), kv.second);
    }
    return result;
  }

  virtual std::string Indicator(RegTree const& /*tree*/,
                                int32_t /*nid*/, uint32_t /*depth*/) const {
    return "";
  }
  virtual std::string Categorical(RegTree const&, int32_t, uint32_t) const = 0;
  virtual std::string Integer(RegTree const& /*tree*/,
                                int32_t /*nid*/, uint32_t /*depth*/) const {
    return "";
  }
  virtual std::string Quantitive(RegTree const& /*tree*/,
                                int32_t /*nid*/, uint32_t /*depth*/) const {
    return "";
  }
  virtual std::string NodeStat(RegTree const& /*tree*/, int32_t /*nid*/) const {
    return "";
  }

  virtual std::string PlainNode(RegTree const& /*tree*/,
                                int32_t /*nid*/, uint32_t /*depth*/) const = 0;

  virtual std::string SplitNode(RegTree const& tree, int32_t nid, uint32_t depth) {
    auto const split_index = tree[nid].SplitIndex();
    std::string result;
    auto is_categorical = tree.GetSplitTypes()[nid] == FeatureType::kCategorical;
    if (split_index < fmap_.Size()) {
      auto check_categorical = [&]() {
        CHECK(is_categorical)
            << fmap_.Name(split_index)
            << " in feature map is numerical but tree node is categorical.";
      };
      auto check_numerical = [&]() {
        auto is_numerical = !is_categorical;
        CHECK(is_numerical)
            << fmap_.Name(split_index)
            << " in feature map is categorical but tree node is numerical.";
      };

      switch (fmap_.TypeOf(split_index)) {
      case FeatureMap::kCategorical: {
        check_categorical();
        result = this->Categorical(tree, nid, depth);
        break;
      }
      case FeatureMap::kIndicator: {
        check_numerical();
        result = this->Indicator(tree, nid, depth);
        break;
      }
      case FeatureMap::kInteger: {
        check_numerical();
        result = this->Integer(tree, nid, depth);
        break;
      }
      case FeatureMap::kFloat:
      case FeatureMap::kQuantitive: {
        check_numerical();
        result = this->Quantitive(tree, nid, depth);
        break;
      }
      default:
        LOG(FATAL) << "Unknown feature map type.";
      }
    } else {
      if (is_categorical) {
        result = this->Categorical(tree, nid, depth);
      } else {
        result = this->PlainNode(tree, nid, depth);
      }
    }
    return result;
  }

  virtual std::string LeafNode(RegTree const& tree, int32_t nid, uint32_t depth) const = 0;
  virtual std::string BuildTree(RegTree const& tree, int32_t nid, uint32_t depth) = 0;

 public:
  TreeGenerator(FeatureMap const& _fmap, bool with_stats) :
      fmap_{_fmap}, with_stats_{with_stats} {}
  virtual ~TreeGenerator() = default;

  virtual void BuildTree(RegTree const& tree) {
    ss_ << this->BuildTree(tree, 0, 0);
  }

  std::string Str() const {
    return ss_.str();
  }

  static TreeGenerator* Create(std::string const& attrs, FeatureMap const& fmap,
                               bool with_stats);
};

struct TreeGenReg : public dmlc::FunctionRegEntryBase<
  TreeGenReg,
  std::function<TreeGenerator* (
      FeatureMap const& fmap, std::string attrs, bool with_stats)> > {
};
}  // namespace xgboost


namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::TreeGenReg);
}  // namespace dmlc

namespace xgboost {

TreeGenerator* TreeGenerator::Create(std::string const& attrs, FeatureMap const& fmap,
                                     bool with_stats) {
  auto pos = attrs.find(':');
  std::string name;
  std::string params;
  if (pos != std::string::npos) {
    name = attrs.substr(0, pos);
    params = attrs.substr(pos+1, attrs.length() - pos - 1);
    // Eliminate all occurrences of single quote string.
    size_t pos = std::string::npos;
    while ((pos = params.find('\'')) != std::string::npos) {
      params.replace(pos, 1, "\"");
    }
  } else {
    name = attrs;
  }
  auto *e = ::dmlc::Registry< ::xgboost::TreeGenReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown Model Builder:" << name;
  }
  auto p_io_builder = (e->body)(fmap, params, with_stats);
  return p_io_builder;
}

#define XGBOOST_REGISTER_TREE_IO(UniqueId, Name)                        \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::TreeGenReg&                   \
  __make_ ## TreeGenReg ## _ ## UniqueId ## __ =                        \
                  ::dmlc::Registry< ::xgboost::TreeGenReg>::Get()->__REGISTER__(Name)

std::vector<bst_cat_t> GetSplitCategories(RegTree const &tree, int32_t nidx) {
  auto const &csr = tree.GetCategoriesMatrix();
  auto seg = csr.node_ptr[nidx];
  auto split = common::KCatBitField{csr.categories.subspan(seg.beg, seg.size)};

  std::vector<bst_cat_t> cats;
  for (size_t i = 0; i < split.Size(); ++i) {
    if (split.Check(i)) {
      cats.push_back(static_cast<bst_cat_t>(i));
    }
  }
  return cats;
}

std::string PrintCatsAsSet(std::vector<bst_cat_t> const &cats) {
  std::stringstream ss;
  ss << "{";
  for (size_t i = 0; i < cats.size(); ++i) {
    ss << cats[i];
    if (i != cats.size() - 1) {
      ss << ",";
    }
  }
  ss << "}";
  return ss.str();
}

class TextGenerator : public TreeGenerator {
  using SuperT = TreeGenerator;

 public:
  TextGenerator(FeatureMap const& fmap, bool with_stats) :
      TreeGenerator(fmap, with_stats) {}

  std::string LeafNode(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    static std::string kLeafTemplate = "{tabs}{nid}:leaf={leaf}{stats}";
    static std::string kStatTemplate = ",cover={cover}";
    std::string result = SuperT::Match(
        kLeafTemplate,
        {{"{tabs}",  SuperT::Tabs(depth)},
         {"{nid}",   std::to_string(nid)},
         {"{leaf}",  SuperT::ToStr(tree[nid].LeafValue())},
         {"{stats}", with_stats_ ?
          SuperT::Match(kStatTemplate,
                        {{"{cover}", SuperT::ToStr(tree.Stat(nid).sum_hess)}}) : ""}});
    return result;
  }

  std::string Indicator(RegTree const& tree, int32_t nid, uint32_t) const override {
    static std::string const kIndicatorTemplate = "{nid}:[{fname}] yes={yes},no={no}";
    int32_t nyes = tree[nid].DefaultLeft() ?
                   tree[nid].RightChild() : tree[nid].LeftChild();
    auto split_index = tree[nid].SplitIndex();
    std::string result = SuperT::Match(
        kIndicatorTemplate,
        {{"{nid}",   std::to_string(nid)},
         {"{fname}", fmap_.Name(split_index)},
         {"{yes}",   std::to_string(nyes)},
         {"{no}",    std::to_string(tree[nid].DefaultChild())}});
    return result;
  }

  std::string SplitNodeImpl(
      RegTree const& tree, int32_t nid, std::string const& template_str,
      std::string cond, uint32_t depth) const {
    auto split_index = tree[nid].SplitIndex();
    std::string const result = SuperT::Match(
        template_str,
        {{"{tabs}",    SuperT::Tabs(depth)},
         {"{nid}",     std::to_string(nid)},
         {"{fname}",   split_index < fmap_.Size() ? fmap_.Name(split_index) :
                                                    std::to_string(split_index)},
         {"{cond}",    cond},
         {"{left}",    std::to_string(tree[nid].LeftChild())},
         {"{right}",   std::to_string(tree[nid].RightChild())},
         {"{missing}", std::to_string(tree[nid].DefaultChild())}});
    return result;
  }

  std::string Integer(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    static std::string const kIntegerTemplate =
        "{tabs}{nid}:[{fname}<{cond}] yes={left},no={right},missing={missing}";
    auto cond = tree[nid].SplitCond();
    const bst_float floored = std::floor(cond);
    const int32_t integer_threshold
        = (floored == cond) ? static_cast<int>(floored)
        : static_cast<int>(floored) + 1;
    return SplitNodeImpl(tree, nid, kIntegerTemplate,
                         std::to_string(integer_threshold), depth);
  }

  std::string Quantitive(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    static std::string const kQuantitiveTemplate =
        "{tabs}{nid}:[{fname}<{cond}] yes={left},no={right},missing={missing}";
    auto cond = tree[nid].SplitCond();
    return SplitNodeImpl(tree, nid, kQuantitiveTemplate, SuperT::ToStr(cond), depth);
  }

  std::string PlainNode(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    auto cond = tree[nid].SplitCond();
    static std::string const kNodeTemplate =
        "{tabs}{nid}:[f{fname}<{cond}] yes={left},no={right},missing={missing}";
    return SplitNodeImpl(tree, nid, kNodeTemplate, SuperT::ToStr(cond), depth);
  }

  std::string Categorical(RegTree const &tree, int32_t nid,
                       uint32_t depth) const override {
    auto cats = GetSplitCategories(tree, nid);
    std::string cats_str = PrintCatsAsSet(cats);
    static std::string const kNodeTemplate =
        "{tabs}{nid}:[{fname}:{cond}] yes={right},no={left},missing={missing}";
    std::string const result =
        SplitNodeImpl(tree, nid, kNodeTemplate, cats_str, depth);
    return result;
  }

  std::string NodeStat(RegTree const& tree, int32_t nid) const override {
    static std::string const kStatTemplate = ",gain={loss_chg},cover={sum_hess}";
    std::string const result = SuperT::Match(
        kStatTemplate,
        {{"{loss_chg}", SuperT::ToStr(tree.Stat(nid).loss_chg)},
         {"{sum_hess}", SuperT::ToStr(tree.Stat(nid).sum_hess)}});
    return result;
  }

  std::string BuildTree(RegTree const& tree, int32_t nid, uint32_t depth) override {
    if (tree[nid].IsLeaf()) {
      return this->LeafNode(tree, nid, depth);
    }
    static std::string const kNodeTemplate = "{parent}{stat}\n{left}\n{right}";
    auto result = SuperT::Match(
        kNodeTemplate,
        {{"{parent}", this->SplitNode(tree, nid, depth)},
         {"{stat}",   with_stats_ ? this->NodeStat(tree, nid) : ""},
         {"{left}",   this->BuildTree(tree, tree[nid].LeftChild(), depth+1)},
         {"{right}",  this->BuildTree(tree, tree[nid].RightChild(), depth+1)}});
    return result;
  }

  void BuildTree(RegTree const& tree) override {
    static std::string const& kTreeTemplate = "{nodes}\n";
    auto result = SuperT::Match(
        kTreeTemplate,
        {{"{nodes}", this->BuildTree(tree, 0, 0)}});
    ss_ << result;
  }
};

XGBOOST_REGISTER_TREE_IO(TextGenerator, "text")
    .describe("Dump text representation of tree")
    .set_body([](FeatureMap const& fmap, std::string const& /*attrs*/, bool with_stats) {
      return new TextGenerator(fmap, with_stats);
    });

class JsonGenerator : public TreeGenerator {
  using SuperT = TreeGenerator;

 public:
  JsonGenerator(FeatureMap const& fmap, bool with_stats) :
      TreeGenerator(fmap, with_stats) {}

  std::string Indent(uint32_t depth) const {
    std::string result;
    for (uint32_t i = 0; i < depth + 1; ++i) {
      result += "  ";
    }
    return result;
  }

  std::string LeafNode(RegTree const& tree, int32_t nid, uint32_t) const override {
    static std::string const kLeafTemplate =
        R"L({ "nodeid": {nid}, "leaf": {leaf} {stat}})L";
    static std::string const kStatTemplate =
        R"S(, "cover": {sum_hess} )S";
    std::string result = SuperT::Match(
        kLeafTemplate,
        {{"{nid}",  std::to_string(nid)},
         {"{leaf}", SuperT::ToStr(tree[nid].LeafValue())},
         {"{stat}", with_stats_ ? SuperT::Match(
             kStatTemplate,
             {{"{sum_hess}",
               SuperT::ToStr(tree.Stat(nid).sum_hess)}})  : ""}});
    return result;
  }

  std::string Indicator(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    int32_t nyes = tree[nid].DefaultLeft() ?
                   tree[nid].RightChild() : tree[nid].LeftChild();
    static std::string const kIndicatorTemplate =
        R"ID( "nodeid": {nid}, "depth": {depth}, "split": "{fname}", "yes": {yes}, "no": {no})ID";
    auto split_index = tree[nid].SplitIndex();
    auto result = SuperT::Match(
        kIndicatorTemplate,
        {{"{nid}",   std::to_string(nid)},
         {"{depth}", std::to_string(depth)},
         {"{fname}", fmap_.Name(split_index)},
         {"{yes}",   std::to_string(nyes)},
         {"{no}",    std::to_string(tree[nid].DefaultChild())}});
    return result;
  }

  std::string Categorical(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    auto cats = GetSplitCategories(tree, nid);
    static std::string const kCategoryTemplate =
        R"I( "nodeid": {nid}, "depth": {depth}, "split": "{fname}", )I"
        R"I("split_condition": {cond}, "yes": {right}, "no": {left}, )I"
        R"I("missing": {missing})I";
    std::string cats_ptr = "[";
    for (size_t i = 0; i < cats.size(); ++i) {
      cats_ptr += std::to_string(cats[i]);
      if (i != cats.size() - 1) {
        cats_ptr += ", ";
      }
    }
    cats_ptr += "]";
    auto results = SplitNodeImpl(tree, nid, kCategoryTemplate, cats_ptr, depth);
    return results;
  }

  std::string SplitNodeImpl(RegTree const &tree, int32_t nid,
                            std::string const &template_str, std::string cond,
                            uint32_t depth) const {
    auto split_index = tree[nid].SplitIndex();
    std::string const result = SuperT::Match(
        template_str,
        {{"{nid}",     std::to_string(nid)},
         {"{depth}",   std::to_string(depth)},
         {"{fname}",   split_index < fmap_.Size() ? fmap_.Name(split_index) :
                                                    std::to_string(split_index)},
         {"{cond}",    cond},
         {"{left}",    std::to_string(tree[nid].LeftChild())},
         {"{right}",   std::to_string(tree[nid].RightChild())},
         {"{missing}", std::to_string(tree[nid].DefaultChild())}});
    return result;
  }

  std::string Integer(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    auto cond = tree[nid].SplitCond();
    const bst_float floored = std::floor(cond);
    const int32_t integer_threshold
        = (floored == cond) ? static_cast<int32_t>(floored)
        : static_cast<int32_t>(floored) + 1;
    static std::string const kIntegerTemplate =
        R"I( "nodeid": {nid}, "depth": {depth}, "split": "{fname}", )I"
        R"I("split_condition": {cond}, "yes": {left}, "no": {right}, )I"
        R"I("missing": {missing})I";
    return SplitNodeImpl(tree, nid, kIntegerTemplate,
                         std::to_string(integer_threshold), depth);
  }

  std::string Quantitive(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    static std::string const kQuantitiveTemplate =
        R"I( "nodeid": {nid}, "depth": {depth}, "split": "{fname}", )I"
        R"I("split_condition": {cond}, "yes": {left}, "no": {right}, )I"
        R"I("missing": {missing})I";
    bst_float cond = tree[nid].SplitCond();
    return SplitNodeImpl(tree, nid, kQuantitiveTemplate, SuperT::ToStr(cond), depth);
  }

  std::string PlainNode(RegTree const& tree, int32_t nid, uint32_t depth) const override {
    auto cond = tree[nid].SplitCond();
    static std::string const kNodeTemplate =
        R"I( "nodeid": {nid}, "depth": {depth}, "split": "{fname}", )I"
        R"I("split_condition": {cond}, "yes": {left}, "no": {right}, )I"
        R"I("missing": {missing})I";
    return SplitNodeImpl(tree, nid, kNodeTemplate, SuperT::ToStr(cond), depth);
  }

  std::string NodeStat(RegTree const& tree, int32_t nid) const override {
    static std::string kStatTemplate =
        R"S(, "gain": {loss_chg}, "cover": {sum_hess})S";
    auto result = SuperT::Match(
        kStatTemplate,
        {{"{loss_chg}", SuperT::ToStr(tree.Stat(nid).loss_chg)},
         {"{sum_hess}", SuperT::ToStr(tree.Stat(nid).sum_hess)}});
    return result;
  }

  std::string SplitNode(RegTree const& tree, int32_t nid, uint32_t depth) override {
    std::string properties = SuperT::SplitNode(tree, nid, depth);
    static std::string const kSplitNodeTemplate =
        "{{properties} {stat}, \"children\": [{left}, {right}\n{indent}]}";
    auto result = SuperT::Match(
        kSplitNodeTemplate,
        {{"{properties}", properties},
         {"{stat}",   with_stats_ ? this->NodeStat(tree, nid) : ""},
         {"{left}",   this->BuildTree(tree, tree[nid].LeftChild(), depth+1)},
         {"{right}",  this->BuildTree(tree, tree[nid].RightChild(), depth+1)},
         {"{indent}", this->Indent(depth)}});
    return result;
  }

  std::string BuildTree(RegTree const& tree, int32_t nid, uint32_t depth) override {
    static std::string const kNodeTemplate = "{newline}{indent}{nodes}";
    auto result = SuperT::Match(
        kNodeTemplate,
        {{"{newline}", depth == 0 ? "" : "\n"},
         {"{indent}", Indent(depth)},
         {"{nodes}",  tree[nid].IsLeaf() ? this->LeafNode(tree, nid, depth) :
                                           this->SplitNode(tree, nid, depth)}});
    return result;
  }
};

XGBOOST_REGISTER_TREE_IO(JsonGenerator, "json")
    .describe("Dump json representation of tree")
    .set_body([](FeatureMap const& fmap, std::string const& /*attrs*/, bool with_stats) {
      return new JsonGenerator(fmap, with_stats);
    });

struct GraphvizParam : public XGBoostParameter<GraphvizParam> {
  std::string yes_color;
  std::string no_color;
  std::string rankdir;
  std::string condition_node_params;
  std::string leaf_node_params;
  std::string graph_attrs;

  DMLC_DECLARE_PARAMETER(GraphvizParam){
    DMLC_DECLARE_FIELD(yes_color)
        .set_default("#0000FF")
        .describe("Edge color when meets the node condition.");
    DMLC_DECLARE_FIELD(no_color)
        .set_default("#FF0000")
        .describe("Edge color when doesn't meet the node condition.");
    DMLC_DECLARE_FIELD(rankdir)
        .set_default("TB")
        .describe("Passed to graphiz via graph_attr.");
    DMLC_DECLARE_FIELD(condition_node_params)
        .set_default("")
        .describe("Conditional node configuration");
    DMLC_DECLARE_FIELD(leaf_node_params)
        .set_default("")
        .describe("Leaf node configuration");
    DMLC_DECLARE_FIELD(graph_attrs)
        .set_default("")
        .describe("Any other extra attributes for graphviz `graph_attr`.");
  }
};

DMLC_REGISTER_PARAMETER(GraphvizParam);

class GraphvizGenerator : public TreeGenerator {
  using SuperT = TreeGenerator;
  GraphvizParam param_;

 public:
  GraphvizGenerator(FeatureMap const& fmap, std::string const& attrs, bool with_stats) :
      TreeGenerator(fmap, with_stats) {
    param_.UpdateAllowUnknown(std::map<std::string, std::string>{});
    using KwArg = std::map<std::string, std::map<std::string, std::string>>;
    KwArg kwargs;
    if (attrs.length() != 0) {
      std::istringstream iss(attrs);
      try {
        dmlc::JSONReader reader(&iss);
        reader.Read(&kwargs);
      } catch(dmlc::Error const& e) {
        LOG(FATAL) << "Failed to parse graphviz parameters:\n\t"
                   << attrs << "\n"
                   << "With error:\n"
                   << e.what();
      }
    }
    // This turns out to be tricky, as `dmlc::Parameter::Load(JSONReader*)` doesn't
    // support loading nested json objects.
    if (kwargs.find("condition_node_params") != kwargs.cend()) {
      auto const& cnp = kwargs["condition_node_params"];
      for (auto const& kv : cnp) {
        param_.condition_node_params += kv.first + '=' + "\"" + kv.second + "\" ";
      }
      kwargs.erase("condition_node_params");
    }
    if (kwargs.find("leaf_node_params") != kwargs.cend()) {
      auto const& lnp = kwargs["leaf_node_params"];
      for (auto const& kv : lnp) {
        param_.leaf_node_params += kv.first + '=' + "\"" + kv.second + "\" ";
      }
      kwargs.erase("leaf_node_params");
    }

    if (kwargs.find("edge") != kwargs.cend()) {
      if (kwargs["edge"].find("yes_color") != kwargs["edge"].cend()) {
        param_.yes_color = kwargs["edge"]["yes_color"];
      }
      if (kwargs["edge"].find("no_color") != kwargs["edge"].cend()) {
        param_.no_color = kwargs["edge"]["no_color"];
      }
      kwargs.erase("edge");
    }
    auto const& extra = kwargs["graph_attrs"];
    static std::string const kGraphTemplate = "    graph [ {key}=\"{value}\" ]\n";
    for (auto const& kv : extra) {
      param_.graph_attrs += SuperT::Match(kGraphTemplate,
                                     {{"{key}", kv.first},
                                      {"{value}", kv.second}});
    }

    kwargs.erase("graph_attrs");
    if (kwargs.size() != 0) {
      std::stringstream ss;
      ss << "The following parameters for graphviz are not recognized:\n";
      for (auto kv : kwargs) {
        ss << kv.first << ", ";
      }
      LOG(WARNING) << ss.str();
    }
  }

 protected:
  template <bool is_categorical>
  std::string BuildEdge(RegTree const &tree, bst_node_t nid, int32_t child, bool left) const {
    static std::string const kEdgeTemplate =
        "    {nid} -> {child} [label=\"{branch}\" color=\"{color}\"]\n";
    // Is this the default child for missing value?
    bool is_missing = tree[nid].DefaultChild() == child;
    std::string branch;
    if (is_categorical) {
      branch = std::string{left ? "no" : "yes"} + std::string{is_missing ? ", missing" : ""};
    } else {
      branch = std::string{left ? "yes" : "no"} + std::string{is_missing ? ", missing" : ""};
    }
    std::string buffer =
        SuperT::Match(kEdgeTemplate,
                {{"{nid}", std::to_string(nid)},
                 {"{child}", std::to_string(child)},
                 {"{color}", is_missing ? param_.yes_color : param_.no_color},
                 {"{branch}", branch}});
    return buffer;
  }

  // Only indicator is different, so we combine all different node types into this
  // function.
  std::string PlainNode(RegTree const& tree, int32_t nid, uint32_t) const override {
    auto split = tree[nid].SplitIndex();
    auto cond = tree[nid].SplitCond();
    static std::string const kNodeTemplate =
        "    {nid} [ label=\"{fname}{<}{cond}\" {params}]\n";

    // Indicator only has fname.
    bool has_less = (split >= fmap_.Size()) || fmap_.TypeOf(split) != FeatureMap::kIndicator;
    std::string result = SuperT::Match(kNodeTemplate, {
        {"{nid}",    std::to_string(nid)},
        {"{fname}",  split < fmap_.Size() ? fmap_.Name(split) :
                                           'f' + std::to_string(split)},
        {"{<}",      has_less ? "<" : ""},
        {"{cond}",   has_less ? SuperT::ToStr(cond) : ""},
        {"{params}", param_.condition_node_params}});

    result += BuildEdge<false>(tree, nid, tree[nid].LeftChild(), true);
    result += BuildEdge<false>(tree, nid, tree[nid].RightChild(), false);

    return result;
  };

  std::string Categorical(RegTree const& tree, int32_t nid, uint32_t) const override {
    static std::string const kLabelTemplate =
        "    {nid} [ label=\"{fname}:{cond}\" {params}]\n";
    auto cats = GetSplitCategories(tree, nid);
    auto cats_str = PrintCatsAsSet(cats);
    auto split = tree[nid].SplitIndex();
    std::string result = SuperT::Match(
        kLabelTemplate,
        {{"{nid}", std::to_string(nid)},
         {"{fname}", split < fmap_.Size() ? fmap_.Name(split)
                                          : 'f' + std::to_string(split)},
         {"{cond}", cats_str},
         {"{params}", param_.condition_node_params}});

    result += BuildEdge<true>(tree, nid, tree[nid].LeftChild(), true);
    result += BuildEdge<true>(tree, nid, tree[nid].RightChild(), false);

    return result;
  }

  std::string LeafNode(RegTree const& tree, int32_t nid, uint32_t) const override {
    static std::string const kLeafTemplate =
        "    {nid} [ label=\"leaf={leaf-value}\" {params}]\n";
    auto result = SuperT::Match(kLeafTemplate, {
        {"{nid}",        std::to_string(nid)},
        {"{leaf-value}", ToStr(tree[nid].LeafValue())},
        {"{params}",     param_.leaf_node_params}});
    return result;
  };

  std::string BuildTree(RegTree const& tree, int32_t nid, uint32_t depth) override {
    if (tree[nid].IsLeaf()) {
      return this->LeafNode(tree, nid, depth);
    }
    static std::string const kNodeTemplate = "{parent}\n{left}\n{right}";
    auto node = tree.GetSplitTypes()[nid] == FeatureType::kCategorical
                    ? this->Categorical(tree, nid, depth)
                    : this->PlainNode(tree, nid, depth);
    auto result = SuperT::Match(
        kNodeTemplate,
        {{"{parent}", node},
         {"{left}",   this->BuildTree(tree, tree[nid].LeftChild(), depth+1)},
         {"{right}",  this->BuildTree(tree, tree[nid].RightChild(), depth+1)}});
    return result;
  }

  void BuildTree(RegTree const& tree) override {
    static std::string const kTreeTemplate =
        "digraph {\n"
        "    graph [ rankdir={rankdir} ]\n"
        "{graph_attrs}\n"
        "{nodes}}";
    auto result = SuperT::Match(
        kTreeTemplate,
        {{"{rankdir}",     param_.rankdir},
         {"{graph_attrs}", param_.graph_attrs},
         {"{nodes}",       this->BuildTree(tree, 0, 0)}});
    ss_ << result;
  };
};

XGBOOST_REGISTER_TREE_IO(GraphvizGenerator, "dot")
.describe("Dump graphviz representation of tree")
.set_body([](FeatureMap const& fmap, std::string attrs, bool with_stats) {
            return new GraphvizGenerator(fmap, attrs, with_stats);
          });

constexpr bst_node_t RegTree::kRoot;

std::string RegTree::DumpModel(const FeatureMap& fmap, bool with_stats, std::string format) const {
  CHECK(!IsMultiTarget());
  std::unique_ptr<TreeGenerator> builder{TreeGenerator::Create(format, fmap, with_stats)};
  builder->BuildTree(*this);

  std::string result = builder->Str();
  return result;
}

bool RegTree::Equal(const RegTree& b) const {
  CHECK(!IsMultiTarget());
  if (NumExtraNodes() != b.NumExtraNodes()) {
    return false;
  }
  auto const& self = *this;
  bool ret { true };
  this->WalkTree([&self, &b, &ret](bst_node_t nidx) {
    if (!(self.nodes_.at(nidx) == b.nodes_.at(nidx))) {
      ret = false;
      return false;
    }
    return true;
  });
  return ret;
}

bst_node_t RegTree::GetNumLeaves() const {
  CHECK(!IsMultiTarget());
  bst_node_t leaves { 0 };
  auto const& self = *this;
  this->WalkTree([&leaves, &self](bst_node_t nidx) {
                   if (self[nidx].IsLeaf()) {
                     leaves++;
                   }
                   return true;
                 });
  return leaves;
}

bst_node_t RegTree::GetNumSplitNodes() const {
  CHECK(!IsMultiTarget());
  bst_node_t splits { 0 };
  auto const& self = *this;
  this->WalkTree([&splits, &self](bst_node_t nidx) {
                   if (!self[nidx].IsLeaf()) {
                     splits++;
                   }
                   return true;
                 });
  return splits;
}

void RegTree::ExpandNode(bst_node_t nid, unsigned split_index, bst_float split_value,
                         bool default_left, bst_float base_weight,
                         bst_float left_leaf_weight,
                         bst_float right_leaf_weight, bst_float loss_change,
                         float sum_hess, float left_sum, float right_sum,
                         bst_node_t leaf_right_child) {
  CHECK(!IsMultiTarget());
  int pleft = this->AllocNode();
  int pright = this->AllocNode();
  auto &node = nodes_[nid];
  CHECK(node.IsLeaf());
  node.SetLeftChild(pleft);
  node.SetRightChild(pright);
  nodes_[node.LeftChild()].SetParent(nid, true);
  nodes_[node.RightChild()].SetParent(nid, false);
  node.SetSplit(split_index, split_value, default_left);

  nodes_[pleft].SetLeaf(left_leaf_weight, leaf_right_child);
  nodes_[pright].SetLeaf(right_leaf_weight, leaf_right_child);

  this->Stat(nid) = {loss_change, sum_hess, base_weight};
  this->Stat(pleft) = {0.0f, left_sum, left_leaf_weight};
  this->Stat(pright) = {0.0f, right_sum, right_leaf_weight};

  this->split_types_.at(nid) = FeatureType::kNumerical;
}

void RegTree::ExpandNode(bst_node_t nidx, bst_feature_t split_index, float split_cond,
                         bool default_left, linalg::VectorView<float const> base_weight,
                         linalg::VectorView<float const> left_weight,
                         linalg::VectorView<float const> right_weight) {
  CHECK(IsMultiTarget());
  CHECK_LT(split_index, this->param_.num_feature);
  CHECK(this->p_mt_tree_);
  CHECK_GT(param_.size_leaf_vector, 1);

  this->p_mt_tree_->Expand(nidx, split_index, split_cond, default_left, base_weight, left_weight,
                           right_weight);

  split_types_.resize(this->Size(), FeatureType::kNumerical);
  split_categories_segments_.resize(this->Size());
  this->split_types_.at(nidx) = FeatureType::kNumerical;

  this->param_.num_nodes = this->p_mt_tree_->Size();
}

void RegTree::ExpandCategorical(bst_node_t nid, bst_feature_t split_index,
                                common::Span<const uint32_t> split_cat, bool default_left,
                                bst_float base_weight, bst_float left_leaf_weight,
                                bst_float right_leaf_weight, bst_float loss_change, float sum_hess,
                                float left_sum, float right_sum) {
  CHECK(!IsMultiTarget());
  this->ExpandNode(nid, split_index, std::numeric_limits<float>::quiet_NaN(),
                   default_left, base_weight,
                   left_leaf_weight, right_leaf_weight, loss_change, sum_hess,
                   left_sum, right_sum);

  size_t orig_size = split_categories_.size();
  this->split_categories_.resize(orig_size + split_cat.size());
  std::copy(split_cat.data(), split_cat.data() + split_cat.size(),
            split_categories_.begin() + orig_size);
  this->split_types_.at(nid) = FeatureType::kCategorical;
  this->split_categories_segments_.at(nid).beg = orig_size;
  this->split_categories_segments_.at(nid).size = split_cat.size();
}

void RegTree::Load(dmlc::Stream* fi) {
  CHECK_EQ(fi->Read(&param_, sizeof(TreeParam)), sizeof(TreeParam));
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    param_ = param_.ByteSwap();
  }
  nodes_.resize(param_.num_nodes);
  stats_.resize(param_.num_nodes);
  CHECK_NE(param_.num_nodes, 0);
  CHECK_EQ(fi->Read(dmlc::BeginPtr(nodes_), sizeof(Node) * nodes_.size()),
           sizeof(Node) * nodes_.size());
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    for (Node& node : nodes_) {
      node = node.ByteSwap();
    }
  }
  CHECK_EQ(fi->Read(dmlc::BeginPtr(stats_), sizeof(RTreeNodeStat) * stats_.size()),
           sizeof(RTreeNodeStat) * stats_.size());
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    for (RTreeNodeStat& stat : stats_) {
      stat = stat.ByteSwap();
    }
  }
  // chg deleted nodes
  deleted_nodes_.resize(0);
  for (int i = 1; i < param_.num_nodes; ++i) {
    if (nodes_[i].IsDeleted()) {
      deleted_nodes_.push_back(i);
    }
  }
  CHECK_EQ(static_cast<int>(deleted_nodes_.size()), param_.num_deleted);

  split_types_.resize(param_.num_nodes, FeatureType::kNumerical);
  split_categories_segments_.resize(param_.num_nodes);
}

void RegTree::Save(dmlc::Stream* fo) const {
  CHECK_EQ(param_.num_nodes, static_cast<int>(nodes_.size()));
  CHECK_EQ(param_.num_nodes, static_cast<int>(stats_.size()));
  CHECK_EQ(param_.deprecated_num_roots, 1);
  CHECK_NE(param_.num_nodes, 0);
  CHECK(!IsMultiTarget())
      << "Please use JSON/UBJSON for saving models with multi-target trees.";
  CHECK(!HasCategoricalSplit())
      << "Please use JSON/UBJSON for saving models with categorical splits.";

  if (DMLC_IO_NO_ENDIAN_SWAP) {
    fo->Write(&param_, sizeof(TreeParam));
  } else {
    TreeParam x = param_.ByteSwap();
    fo->Write(&x, sizeof(x));
  }

  if (DMLC_IO_NO_ENDIAN_SWAP) {
    fo->Write(dmlc::BeginPtr(nodes_), sizeof(Node) * nodes_.size());
  } else {
    for (const Node& node : nodes_) {
      Node x = node.ByteSwap();
      fo->Write(&x, sizeof(x));
    }
  }
  if (DMLC_IO_NO_ENDIAN_SWAP) {
    fo->Write(dmlc::BeginPtr(stats_), sizeof(RTreeNodeStat) * nodes_.size());
  } else {
    for (const RTreeNodeStat& stat : stats_) {
      RTreeNodeStat x = stat.ByteSwap();
      fo->Write(&x, sizeof(x));
    }
  }
}

template <bool typed>
void RegTree::LoadCategoricalSplit(Json const& in) {
  auto const& categories_segments = get<I64ArrayT<typed>>(in["categories_segments"]);
  auto const& categories_sizes = get<I64ArrayT<typed>>(in["categories_sizes"]);
  auto const& categories_nodes = get<I32ArrayT<typed>>(in["categories_nodes"]);
  auto const& categories = get<I32ArrayT<typed>>(in["categories"]);

  auto split_type = get<U8ArrayT<typed>>(in["split_type"]);
  bst_node_t n_nodes = split_type.size();
  std::size_t cnt = 0;
  bst_node_t last_cat_node = -1;
  if (!categories_nodes.empty()) {
    last_cat_node = GetElem<Integer>(categories_nodes, cnt);
  }
  // `categories_segments' is only available for categorical nodes to prevent overhead for
  // numerical node. As a result, we need to track the categorical nodes we have processed
  // so far.
  split_types_.resize(n_nodes, FeatureType::kNumerical);
  split_categories_segments_.resize(n_nodes);
  for (bst_node_t nidx = 0; nidx < n_nodes; ++nidx) {
    split_types_[nidx] = static_cast<FeatureType>(GetElem<Integer>(split_type, nidx));
    if (nidx == last_cat_node) {
      auto j_begin = GetElem<Integer>(categories_segments, cnt);
      auto j_end = GetElem<Integer>(categories_sizes, cnt) + j_begin;
      bst_cat_t max_cat{std::numeric_limits<bst_cat_t>::min()};
      CHECK_GT(j_end - j_begin, 0) << nidx;

      for (auto j = j_begin; j < j_end; ++j) {
        auto const& category = GetElem<Integer>(categories, j);
        auto cat = common::AsCat(category);
        max_cat = std::max(max_cat, cat);
      }
      // Have at least 1 category in split.
      CHECK_NE(std::numeric_limits<bst_cat_t>::min(), max_cat);
      size_t n_cats = max_cat + 1;  // cat 0
      size_t size = common::KCatBitField::ComputeStorageSize(n_cats);
      std::vector<uint32_t> cat_bits_storage(size, 0);
      common::CatBitField cat_bits{common::Span<uint32_t>(cat_bits_storage)};
      for (auto j = j_begin; j < j_end; ++j) {
        cat_bits.Set(common::AsCat(GetElem<Integer>(categories, j)));
      }

      auto begin = split_categories_.size();
      split_categories_.resize(begin + cat_bits_storage.size());
      std::copy(cat_bits_storage.begin(), cat_bits_storage.end(),
                split_categories_.begin() + begin);
      split_categories_segments_[nidx].beg = begin;
      split_categories_segments_[nidx].size = cat_bits_storage.size();

      ++cnt;
      if (cnt == categories_nodes.size()) {
        last_cat_node = -1;  // Don't break, we still need to initialize the remaining nodes.
      } else {
        last_cat_node = GetElem<Integer>(categories_nodes, cnt);
      }
    } else {
      split_categories_segments_[nidx].beg = categories.size();
      split_categories_segments_[nidx].size = 0;
    }
  }
}

template void RegTree::LoadCategoricalSplit<true>(Json const& in);
template void RegTree::LoadCategoricalSplit<false>(Json const& in);

void RegTree::SaveCategoricalSplit(Json* p_out) const {
  auto& out = *p_out;
  CHECK_EQ(this->split_types_.size(), this->Size());
  CHECK_EQ(this->GetSplitCategoriesPtr().size(), this->Size());

  I64Array categories_segments;
  I64Array categories_sizes;
  I32Array categories;        // bst_cat_t = int32_t
  I32Array categories_nodes;  // bst_note_t = int32_t
  U8Array split_type(split_types_.size());

  for (size_t i = 0; i < nodes_.size(); ++i) {
    split_type.Set(i, static_cast<std::underlying_type_t<FeatureType>>(this->NodeSplitType(i)));
    if (this->split_types_[i] == FeatureType::kCategorical) {
      categories_nodes.GetArray().emplace_back(i);
      auto begin = categories.Size();
      categories_segments.GetArray().emplace_back(begin);
      auto segment = split_categories_segments_[i];
      auto node_categories = this->GetSplitCategories().subspan(segment.beg, segment.size);
      common::KCatBitField const cat_bits(node_categories);
      for (size_t i = 0; i < cat_bits.Size(); ++i) {
        if (cat_bits.Check(i)) {
          categories.GetArray().emplace_back(i);
        }
      }
      size_t size = categories.Size() - begin;
      categories_sizes.GetArray().emplace_back(size);
      CHECK_NE(size, 0);
    }
  }

  out["split_type"] = std::move(split_type);
  out["categories_segments"] = std::move(categories_segments);
  out["categories_sizes"] = std::move(categories_sizes);
  out["categories_nodes"] = std::move(categories_nodes);
  out["categories"] = std::move(categories);
}

template <bool typed, bool feature_is_64>
void LoadModelImpl(Json const& in, TreeParam const& param, std::vector<RTreeNodeStat>* p_stats,
                   std::vector<RegTree::Node>* p_nodes) {
  namespace tf = tree_field;
  auto& stats = *p_stats;
  auto& nodes = *p_nodes;

  auto n_nodes = param.num_nodes;
  CHECK_NE(n_nodes, 0);
  // stats
  auto const& loss_changes = get<FloatArrayT<typed>>(in[tf::kLossChg]);
  CHECK_EQ(loss_changes.size(), n_nodes);
  auto const& sum_hessian = get<FloatArrayT<typed>>(in[tf::kSumHess]);
  CHECK_EQ(sum_hessian.size(), n_nodes);
  auto const& base_weights = get<FloatArrayT<typed>>(in[tf::kBaseWeight]);
  CHECK_EQ(base_weights.size(), n_nodes);
  // nodes
  auto const& lefts = get<I32ArrayT<typed>>(in[tf::kLeft]);
  CHECK_EQ(lefts.size(), n_nodes);
  auto const& rights = get<I32ArrayT<typed>>(in[tf::kRight]);
  CHECK_EQ(rights.size(), n_nodes);
  auto const& parents = get<I32ArrayT<typed>>(in[tf::kParent]);
  CHECK_EQ(parents.size(), n_nodes);
  auto const& indices = get<IndexArrayT<typed, feature_is_64>>(in[tf::kSplitIdx]);
  CHECK_EQ(indices.size(), n_nodes);
  auto const& conds = get<FloatArrayT<typed>>(in[tf::kSplitCond]);
  CHECK_EQ(conds.size(), n_nodes);
  auto const& default_left = get<U8ArrayT<typed>>(in[tf::kDftLeft]);
  CHECK_EQ(default_left.size(), n_nodes);

  // Initialization
  stats = std::remove_reference_t<decltype(stats)>(n_nodes);
  nodes = std::remove_reference_t<decltype(nodes)>(n_nodes);

  static_assert(std::is_integral<decltype(GetElem<Integer>(lefts, 0))>::value);
  static_assert(std::is_floating_point<decltype(GetElem<Number>(loss_changes, 0))>::value);

  // Set node
  for (int32_t i = 0; i < n_nodes; ++i) {
    auto& s = stats[i];
    s.loss_chg = GetElem<Number>(loss_changes, i);
    s.sum_hess = GetElem<Number>(sum_hessian, i);
    s.base_weight = GetElem<Number>(base_weights, i);

    auto& n = nodes[i];
    bst_node_t left = GetElem<Integer>(lefts, i);
    bst_node_t right = GetElem<Integer>(rights, i);
    bst_node_t parent = GetElem<Integer>(parents, i);
    bst_feature_t ind = GetElem<Integer>(indices, i);
    float cond{GetElem<Number>(conds, i)};
    bool dft_left{GetElem<Boolean>(default_left, i)};
    n = RegTree::Node{left, right, parent, ind, cond, dft_left};
  }
}

void RegTree::LoadModel(Json const& in) {
  namespace tf = tree_field;

  bool typed = IsA<I32Array>(in[tf::kParent]);
  auto const& in_obj = get<Object const>(in);
  // basic properties
  FromJson(in["tree_param"], &param_);
  // categorical splits
  bool has_cat = in_obj.find("split_type") != in_obj.cend();
  if (has_cat) {
    if (typed) {
      this->LoadCategoricalSplit<true>(in);
    } else {
      this->LoadCategoricalSplit<false>(in);
    }
  }
  // multi-target
  if (param_.size_leaf_vector > 1) {
    this->p_mt_tree_.reset(new MultiTargetTree{&param_});
    this->GetMultiTargetTree()->LoadModel(in);
    return;
  }

  bool feature_is_64 = IsA<I64Array>(in["split_indices"]);
  if (typed && feature_is_64) {
    LoadModelImpl<true, true>(in, param_, &stats_, &nodes_);
  } else if (typed && !feature_is_64) {
    LoadModelImpl<true, false>(in, param_, &stats_, &nodes_);
  } else if (!typed && feature_is_64) {
    LoadModelImpl<false, true>(in, param_, &stats_, &nodes_);
  } else {
    LoadModelImpl<false, false>(in, param_, &stats_, &nodes_);
  }

  if (!has_cat) {
    this->split_categories_segments_.resize(this->param_.num_nodes);
    this->split_types_.resize(this->param_.num_nodes);
    std::fill(split_types_.begin(), split_types_.end(), FeatureType::kNumerical);
  }

  deleted_nodes_.clear();
  for (bst_node_t i = 1; i < param_.num_nodes; ++i) {
    if (nodes_[i].IsDeleted()) {
      deleted_nodes_.push_back(i);
    }
  }
  // easier access to [] operator
  auto& self = *this;
  for (auto nid = 1; nid < param_.num_nodes; ++nid) {
    auto parent = self[nid].Parent();
    CHECK_NE(parent, RegTree::kInvalidNodeId);
    self[nid].SetParent(self[nid].Parent(), self[parent].LeftChild() == nid);
  }
  CHECK_EQ(static_cast<bst_node_t>(deleted_nodes_.size()), param_.num_deleted);
  CHECK_EQ(this->split_categories_segments_.size(), param_.num_nodes);
}

void RegTree::SaveModel(Json* p_out) const {
  auto& out = *p_out;
  // basic properties
  out["tree_param"] = ToJson(param_);
  // categorical splits
  this->SaveCategoricalSplit(p_out);
  // multi-target
  if (this->IsMultiTarget()) {
    CHECK_GT(param_.size_leaf_vector, 1);
    this->GetMultiTargetTree()->SaveModel(p_out);
    return;
  }
  /*  Here we are treating leaf node and internal node equally.  Some information like
   *  child node id doesn't make sense for leaf node but we will have to save them to
   *  avoid creating a huge map.  One difficulty is XGBoost has deleted node created by
   *  pruner, and this pruner can be used inside another updater so leaf are not necessary
   *  at the end of node array.
   */
  CHECK_EQ(param_.num_nodes, static_cast<int>(nodes_.size()));
  CHECK_EQ(param_.num_nodes, static_cast<int>(stats_.size()));

  CHECK_EQ(get<String>(out["tree_param"]["num_nodes"]), std::to_string(param_.num_nodes));
  auto n_nodes = param_.num_nodes;

  // stats
  F32Array loss_changes(n_nodes);
  F32Array sum_hessian(n_nodes);
  F32Array base_weights(n_nodes);

  // nodes
  I32Array lefts(n_nodes);
  I32Array rights(n_nodes);
  I32Array parents(n_nodes);

  F32Array conds(n_nodes);
  U8Array default_left(n_nodes);
  CHECK_EQ(this->split_types_.size(), param_.num_nodes);

  namespace tf = tree_field;

  auto save_tree = [&](auto* p_indices_array) {
    auto& indices_array = *p_indices_array;
    for (bst_node_t i = 0; i < n_nodes; ++i) {
      auto const& s = stats_[i];
      loss_changes.Set(i, s.loss_chg);
      sum_hessian.Set(i, s.sum_hess);
      base_weights.Set(i, s.base_weight);

      auto const& n = nodes_[i];
      lefts.Set(i, n.LeftChild());
      rights.Set(i, n.RightChild());
      parents.Set(i, n.Parent());
      indices_array.Set(i, n.SplitIndex());
      conds.Set(i, n.SplitCond());
      default_left.Set(i, static_cast<uint8_t>(!!n.DefaultLeft()));
    }
  };
  if (this->param_.num_feature > static_cast<bst_feature_t>(std::numeric_limits<int32_t>::max())) {
    I64Array indices_64(n_nodes);
    save_tree(&indices_64);
    out[tf::kSplitIdx] = std::move(indices_64);
  } else {
    I32Array indices_32(n_nodes);
    save_tree(&indices_32);
    out[tf::kSplitIdx] = std::move(indices_32);
  }

  out[tf::kLossChg] = std::move(loss_changes);
  out[tf::kSumHess] = std::move(sum_hessian);
  out[tf::kBaseWeight] = std::move(base_weights);

  out[tf::kLeft] = std::move(lefts);
  out[tf::kRight] = std::move(rights);
  out[tf::kParent] = std::move(parents);

  out[tf::kSplitCond] = std::move(conds);
  out[tf::kDftLeft] = std::move(default_left);
}

void RegTree::CalculateContributionsApprox(const RegTree::FVec &feat,
                                           std::vector<float>* mean_values,
                                           bst_float *out_contribs) const {
  CHECK_GT(mean_values->size(), 0U);
  // this follows the idea of http://blog.datadive.net/interpreting-random-forests/
  unsigned split_index = 0;
  // update bias value
  bst_float node_value = (*mean_values)[0];
  out_contribs[feat.Size()] += node_value;
  if ((*this)[0].IsLeaf()) {
    // nothing to do anymore
    return;
  }

  bst_node_t nid = 0;
  auto cats = this->GetCategoriesMatrix();

  while (!(*this)[nid].IsLeaf()) {
    split_index = (*this)[nid].SplitIndex();
    nid = predictor::GetNextNode<true, true>((*this)[nid], nid,
                                             feat.GetFvalue(split_index),
                                             feat.IsMissing(split_index), cats);
    bst_float new_value = (*mean_values)[nid];
    // update feature weight
    out_contribs[split_index] += new_value - node_value;
    node_value = new_value;
  }
  bst_float leaf_value = (*this)[nid].LeafValue();
  // update leaf feature weight
  out_contribs[split_index] += leaf_value - node_value;
}
}  // namespace xgboost
