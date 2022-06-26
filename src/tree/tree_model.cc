/*!
 * Copyright 2015-2022 by Contributors
 * \file tree_model.cc
 * \brief model structure for tree
 */
#include <dmlc/registry.h>
#include <dmlc/json.h>

#include <xgboost/tree_model.h>
#include <xgboost/logging.h>
#include <xgboost/json.h>

#include <sstream>
#include <limits>
#include <cmath>
#include <iomanip>
#include <stack>

#include "param.h"
#include "../common/common.h"
#include "../common/categorical.h"
#include "../predictor/predict_fn.h"

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

std::string RegTree::DumpModel(const FeatureMap& fmap,
                               bool with_stats,
                               std::string format) const {
  std::unique_ptr<TreeGenerator> builder {
    TreeGenerator::Create(format, fmap, with_stats)
  };
  builder->BuildTree(*this);

  std::string result = builder->Str();
  return result;
}

bool RegTree::Equal(const RegTree& b) const {
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

void RegTree::ExpandCategorical(bst_node_t nid, unsigned split_index,
                                common::Span<const uint32_t> split_cat, bool default_left,
                                bst_float base_weight, bst_float left_leaf_weight,
                                bst_float right_leaf_weight, bst_float loss_change, float sum_hess,
                                float left_sum, float right_sum) {
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
  CHECK_EQ(fi->Read(&param, sizeof(TreeParam)), sizeof(TreeParam));
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    param = param.ByteSwap();
  }
  nodes_.resize(param.num_nodes);
  stats_.resize(param.num_nodes);
  CHECK_NE(param.num_nodes, 0);
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
  for (int i = 1; i < param.num_nodes; ++i) {
    if (nodes_[i].IsDeleted()) {
      deleted_nodes_.push_back(i);
    }
  }
  CHECK_EQ(static_cast<int>(deleted_nodes_.size()), param.num_deleted);

  split_types_.resize(param.num_nodes, FeatureType::kNumerical);
  split_categories_segments_.resize(param.num_nodes);
}

void RegTree::Save(dmlc::Stream* fo) const {
  CHECK_EQ(param.num_nodes, static_cast<int>(nodes_.size()));
  CHECK_EQ(param.num_nodes, static_cast<int>(stats_.size()));
  CHECK_EQ(param.deprecated_num_roots, 1);
  CHECK_NE(param.num_nodes, 0);
  CHECK(!HasCategoricalSplit())
      << "Please use JSON/UBJSON for saving models with categorical splits.";

  if (DMLC_IO_NO_ENDIAN_SWAP) {
    fo->Write(&param, sizeof(TreeParam));
  } else {
    TreeParam x = param.ByteSwap();
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
// typed array, not boolean
template <typename JT, typename T>
std::enable_if_t<!std::is_same<T, Json>::value && !std::is_same<JT, Boolean>::value, T> GetElem(
    std::vector<T> const& arr, size_t i) {
  return arr[i];
}
// typed array boolean
template <typename JT, typename T>
std::enable_if_t<!std::is_same<T, Json>::value && std::is_same<T, uint8_t>::value &&
                     std::is_same<JT, Boolean>::value,
                 bool>
GetElem(std::vector<T> const& arr, size_t i) {
  return arr[i] == 1;
}
// json array
template <typename JT, typename T>
std::enable_if_t<
    std::is_same<T, Json>::value,
    std::conditional_t<std::is_same<JT, Integer>::value, int64_t,
                       std::conditional_t<std::is_same<Boolean, JT>::value, bool, float>>>
GetElem(std::vector<T> const& arr, size_t i) {
  if (std::is_same<JT, Boolean>::value && !IsA<Boolean>(arr[i])) {
    return get<Integer const>(arr[i]) == 1;
  }
  return get<JT const>(arr[i]);
}

template <bool typed>
void RegTree::LoadCategoricalSplit(Json const& in) {
  using I64ArrayT = std::conditional_t<typed, I64Array const, Array const>;
  using I32ArrayT = std::conditional_t<typed, I32Array const, Array const>;

  auto const& categories_segments = get<I64ArrayT>(in["categories_segments"]);
  auto const& categories_sizes = get<I64ArrayT>(in["categories_sizes"]);
  auto const& categories_nodes = get<I32ArrayT>(in["categories_nodes"]);
  auto const& categories = get<I32ArrayT>(in["categories"]);

  size_t cnt = 0;
  bst_node_t last_cat_node = -1;
  if (!categories_nodes.empty()) {
    last_cat_node = GetElem<Integer>(categories_nodes, cnt);
  }
  for (bst_node_t nidx = 0; nidx < param.num_nodes; ++nidx) {
    if (nidx == last_cat_node) {
      auto j_begin = GetElem<Integer>(categories_segments, cnt);
      auto j_end = GetElem<Integer>(categories_sizes, cnt) + j_begin;
      bst_cat_t max_cat{std::numeric_limits<bst_cat_t>::min()};
      CHECK_NE(j_end - j_begin, 0) << nidx;

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
  CHECK_EQ(this->split_types_.size(), param.num_nodes);
  CHECK_EQ(this->GetSplitCategoriesPtr().size(), param.num_nodes);

  I64Array categories_segments;
  I64Array categories_sizes;
  I32Array categories;        // bst_cat_t = int32_t
  I32Array categories_nodes;  // bst_note_t = int32_t

  for (size_t i = 0; i < nodes_.size(); ++i) {
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

  out["categories_segments"] = std::move(categories_segments);
  out["categories_sizes"] = std::move(categories_sizes);
  out["categories_nodes"] = std::move(categories_nodes);
  out["categories"] = std::move(categories);
}

template <bool typed, bool feature_is_64,
          typename FloatArrayT = std::conditional_t<typed, F32Array const, Array const>,
          typename U8ArrayT = std::conditional_t<typed, U8Array const, Array const>,
          typename I32ArrayT = std::conditional_t<typed, I32Array const, Array const>,
          typename I64ArrayT = std::conditional_t<typed, I64Array const, Array const>,
          typename IndexArrayT = std::conditional_t<feature_is_64, I64ArrayT, I32ArrayT>>
bool LoadModelImpl(Json const& in, TreeParam* param, std::vector<RTreeNodeStat>* p_stats,
                   std::vector<FeatureType>* p_split_types, std::vector<RegTree::Node>* p_nodes,
                   std::vector<RegTree::Segment>* p_split_categories_segments) {
  auto& stats = *p_stats;
  auto& split_types = *p_split_types;
  auto& nodes = *p_nodes;
  auto& split_categories_segments = *p_split_categories_segments;

  FromJson(in["tree_param"], param);
  auto n_nodes = param->num_nodes;
  CHECK_NE(n_nodes, 0);
  // stats
  auto const& loss_changes = get<FloatArrayT>(in["loss_changes"]);
  CHECK_EQ(loss_changes.size(), n_nodes);
  auto const& sum_hessian = get<FloatArrayT>(in["sum_hessian"]);
  CHECK_EQ(sum_hessian.size(), n_nodes);
  auto const& base_weights = get<FloatArrayT>(in["base_weights"]);
  CHECK_EQ(base_weights.size(), n_nodes);
  // nodes
  auto const& lefts = get<I32ArrayT>(in["left_children"]);
  CHECK_EQ(lefts.size(), n_nodes);
  auto const& rights = get<I32ArrayT>(in["right_children"]);
  CHECK_EQ(rights.size(), n_nodes);
  auto const& parents = get<I32ArrayT>(in["parents"]);
  CHECK_EQ(parents.size(), n_nodes);
  auto const& indices = get<IndexArrayT>(in["split_indices"]);
  CHECK_EQ(indices.size(), n_nodes);
  auto const& conds = get<FloatArrayT>(in["split_conditions"]);
  CHECK_EQ(conds.size(), n_nodes);
  auto const& default_left = get<U8ArrayT>(in["default_left"]);
  CHECK_EQ(default_left.size(), n_nodes);

  bool has_cat = get<Object const>(in).find("split_type") != get<Object const>(in).cend();
  std::remove_const_t<std::remove_reference_t<decltype(get<U8ArrayT const>(in["split_type"]))>>
      split_type;
  if (has_cat) {
    split_type = get<U8ArrayT const>(in["split_type"]);
  }
  stats = std::remove_reference_t<decltype(stats)>(n_nodes);
  nodes = std::remove_reference_t<decltype(nodes)>(n_nodes);
  split_types = std::remove_reference_t<decltype(split_types)>(n_nodes);
  split_categories_segments = std::remove_reference_t<decltype(split_categories_segments)>(n_nodes);

  static_assert(std::is_integral<decltype(GetElem<Integer>(lefts, 0))>::value, "");
  static_assert(std::is_floating_point<decltype(GetElem<Number>(loss_changes, 0))>::value, "");
  CHECK_EQ(n_nodes, split_categories_segments.size());

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

    if (has_cat) {
      split_types[i] = static_cast<FeatureType>(GetElem<Integer>(split_type, i));
    }
  }

  return has_cat;
}

void RegTree::LoadModel(Json const& in) {
  bool has_cat{false};
  bool typed = IsA<F32Array>(in["loss_changes"]);
  bool feature_is_64 = IsA<I64Array>(in["split_indices"]);
  if (typed && feature_is_64) {
    has_cat = LoadModelImpl<true, true>(in, &param, &stats_, &split_types_, &nodes_,
                                        &split_categories_segments_);
  } else if (typed && !feature_is_64) {
    has_cat = LoadModelImpl<true, false>(in, &param, &stats_, &split_types_, &nodes_,
                                         &split_categories_segments_);
  } else if (!typed && feature_is_64) {
    has_cat = LoadModelImpl<false, true>(in, &param, &stats_, &split_types_, &nodes_,
                                         &split_categories_segments_);
  } else {
    has_cat = LoadModelImpl<false, false>(in, &param, &stats_, &split_types_, &nodes_,
                                          &split_categories_segments_);
  }

  if (has_cat) {
    if (typed) {
      this->LoadCategoricalSplit<true>(in);
    } else {
      this->LoadCategoricalSplit<false>(in);
    }
  } else {
    this->split_categories_segments_.resize(this->param.num_nodes);
    std::fill(split_types_.begin(), split_types_.end(), FeatureType::kNumerical);
  }

  deleted_nodes_.clear();
  for (bst_node_t i = 1; i < param.num_nodes; ++i) {
    if (nodes_[i].IsDeleted()) {
      deleted_nodes_.push_back(i);
    }
  }
  // easier access to [] operator
  auto& self = *this;
  for (auto nid = 1; nid < param.num_nodes; ++nid) {
    auto parent = self[nid].Parent();
    CHECK_NE(parent, RegTree::kInvalidNodeId);
    self[nid].SetParent(self[nid].Parent(), self[parent].LeftChild() == nid);
  }
  CHECK_EQ(static_cast<bst_node_t>(deleted_nodes_.size()), param.num_deleted);
  CHECK_EQ(this->split_categories_segments_.size(), param.num_nodes);
}

void RegTree::SaveModel(Json* p_out) const {
  /*  Here we are treating leaf node and internal node equally.  Some information like
   *  child node id doesn't make sense for leaf node but we will have to save them to
   *  avoid creating a huge map.  One difficulty is XGBoost has deleted node created by
   *  pruner, and this pruner can be used inside another updater so leaf are not necessary
   *  at the end of node array.
   */
  auto& out = *p_out;
  CHECK_EQ(param.num_nodes, static_cast<int>(nodes_.size()));
  CHECK_EQ(param.num_nodes, static_cast<int>(stats_.size()));
  out["tree_param"] = ToJson(param);
  CHECK_EQ(get<String>(out["tree_param"]["num_nodes"]), std::to_string(param.num_nodes));
  auto n_nodes = param.num_nodes;

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
  U8Array split_type(n_nodes);
  CHECK_EQ(this->split_types_.size(), param.num_nodes);

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

      split_type.Set(i, static_cast<uint8_t>(this->NodeSplitType(i)));
    }
  };
  if (this->param.num_feature > static_cast<bst_feature_t>(std::numeric_limits<int32_t>::max())) {
    I64Array indices_64(n_nodes);
    save_tree(&indices_64);
    out["split_indices"] = std::move(indices_64);
  } else {
    I32Array indices_32(n_nodes);
    save_tree(&indices_32);
    out["split_indices"] = std::move(indices_32);
  }

  this->SaveCategoricalSplit(&out);

  out["split_type"] = std::move(split_type);
  out["loss_changes"] = std::move(loss_changes);
  out["sum_hessian"] = std::move(sum_hessian);
  out["base_weights"] = std::move(base_weights);

  out["left_children"] = std::move(lefts);
  out["right_children"] = std::move(rights);
  out["parents"] = std::move(parents);

  out["split_conditions"] = std::move(conds);
  out["default_left"] = std::move(default_left);
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

// Used by TreeShap
// data we keep about our decision path
// note that pweight is included for convenience and is not tied with the other attributes
// the pweight of the i'th path element is the permutation weight of paths with i-1 ones in them
struct PathElement {
  int feature_index;
  bst_float zero_fraction;
  bst_float one_fraction;
  bst_float pweight;
  PathElement() = default;
  PathElement(int i, bst_float z, bst_float o, bst_float w) :
    feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}
};

// extend our decision path with a fraction of one and zero extensions
void ExtendPath(PathElement *unique_path, unsigned unique_depth,
                bst_float zero_fraction, bst_float one_fraction,
                int feature_index) {
  unique_path[unique_depth].feature_index = feature_index;
  unique_path[unique_depth].zero_fraction = zero_fraction;
  unique_path[unique_depth].one_fraction = one_fraction;
  unique_path[unique_depth].pweight = (unique_depth == 0 ? 1.0f : 0.0f);
  for (int i = unique_depth - 1; i >= 0; i--) {
    unique_path[i+1].pweight += one_fraction * unique_path[i].pweight * (i + 1)
                                / static_cast<bst_float>(unique_depth + 1);
    unique_path[i].pweight = zero_fraction * unique_path[i].pweight * (unique_depth - i)
                             / static_cast<bst_float>(unique_depth + 1);
  }
}

// undo a previous extension of the decision path
void UnwindPath(PathElement *unique_path, unsigned unique_depth,
                unsigned path_index) {
  const bst_float one_fraction = unique_path[path_index].one_fraction;
  const bst_float zero_fraction = unique_path[path_index].zero_fraction;
  bst_float next_one_portion = unique_path[unique_depth].pweight;

  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const bst_float tmp = unique_path[i].pweight;
      unique_path[i].pweight = next_one_portion * (unique_depth + 1)
                               / static_cast<bst_float>((i + 1) * one_fraction);
      next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (unique_depth - i)
                               / static_cast<bst_float>(unique_depth + 1);
    } else {
      unique_path[i].pweight = (unique_path[i].pweight * (unique_depth + 1))
                               / static_cast<bst_float>(zero_fraction * (unique_depth - i));
    }
  }

  for (auto i = path_index; i < unique_depth; ++i) {
    unique_path[i].feature_index = unique_path[i+1].feature_index;
    unique_path[i].zero_fraction = unique_path[i+1].zero_fraction;
    unique_path[i].one_fraction = unique_path[i+1].one_fraction;
  }
}

// determine what the total permutation weight would be if
// we unwound a previous extension in the decision path
bst_float UnwoundPathSum(const PathElement *unique_path, unsigned unique_depth,
                         unsigned path_index) {
  const bst_float one_fraction = unique_path[path_index].one_fraction;
  const bst_float zero_fraction = unique_path[path_index].zero_fraction;
  bst_float next_one_portion = unique_path[unique_depth].pweight;
  bst_float total = 0;
  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const bst_float tmp = next_one_portion * (unique_depth + 1)
                            / static_cast<bst_float>((i + 1) * one_fraction);
      total += tmp;
      next_one_portion = unique_path[i].pweight - tmp * zero_fraction * ((unique_depth - i)
                         / static_cast<bst_float>(unique_depth + 1));
    } else if (zero_fraction != 0) {
      total += (unique_path[i].pweight / zero_fraction) / ((unique_depth - i)
               / static_cast<bst_float>(unique_depth + 1));
    } else {
      CHECK_EQ(unique_path[i].pweight, 0)
        << "Unique path " << i << " must have zero weight";
    }
  }
  return total;
}

// recursive computation of SHAP values for a decision tree
void RegTree::TreeShap(const RegTree::FVec &feat, bst_float *phi,
                       bst_node_t node_index, unsigned unique_depth,
                       PathElement *parent_unique_path,
                       bst_float parent_zero_fraction,
                       bst_float parent_one_fraction, int parent_feature_index,
                       int condition, unsigned condition_feature,
                       bst_float condition_fraction) const {
  const auto node = (*this)[node_index];

  // stop if we have no weight coming down to us
  if (condition_fraction == 0) return;

  // extend the unique path
  PathElement *unique_path = parent_unique_path + unique_depth + 1;
  std::copy(parent_unique_path, parent_unique_path + unique_depth + 1, unique_path);

  if (condition == 0 || condition_feature != static_cast<unsigned>(parent_feature_index)) {
    ExtendPath(unique_path, unique_depth, parent_zero_fraction,
               parent_one_fraction, parent_feature_index);
  }
  const unsigned split_index = node.SplitIndex();

  // leaf node
  if (node.IsLeaf()) {
    for (unsigned i = 1; i <= unique_depth; ++i) {
      const bst_float w = UnwoundPathSum(unique_path, unique_depth, i);
      const PathElement &el = unique_path[i];
      phi[el.feature_index] += w * (el.one_fraction - el.zero_fraction)
                                 * node.LeafValue() * condition_fraction;
    }

  // internal node
  } else {
    // find which branch is "hot" (meaning x would follow it)
    auto const &cats = this->GetCategoriesMatrix();
    bst_node_t hot_index = predictor::GetNextNode<true, true>(
        node, node_index, feat.GetFvalue(split_index),
        feat.IsMissing(split_index), cats);

    const auto cold_index =
        (hot_index == node.LeftChild() ? node.RightChild() : node.LeftChild());
    const bst_float w = this->Stat(node_index).sum_hess;
    const bst_float hot_zero_fraction = this->Stat(hot_index).sum_hess / w;
    const bst_float cold_zero_fraction = this->Stat(cold_index).sum_hess / w;
    bst_float incoming_zero_fraction = 1;
    bst_float incoming_one_fraction = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    unsigned path_index = 0;
    for (; path_index <= unique_depth; ++path_index) {
      if (static_cast<unsigned>(unique_path[path_index].feature_index) == split_index) break;
    }
    if (path_index != unique_depth + 1) {
      incoming_zero_fraction = unique_path[path_index].zero_fraction;
      incoming_one_fraction = unique_path[path_index].one_fraction;
      UnwindPath(unique_path, unique_depth, path_index);
      unique_depth -= 1;
    }

    // divide up the condition_fraction among the recursive calls
    bst_float hot_condition_fraction = condition_fraction;
    bst_float cold_condition_fraction = condition_fraction;
    if (condition > 0 && split_index == condition_feature) {
      cold_condition_fraction = 0;
      unique_depth -= 1;
    } else if (condition < 0 && split_index == condition_feature) {
      hot_condition_fraction *= hot_zero_fraction;
      cold_condition_fraction *= cold_zero_fraction;
      unique_depth -= 1;
    }

    TreeShap(feat, phi, hot_index, unique_depth + 1, unique_path,
             hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
             split_index, condition, condition_feature, hot_condition_fraction);

    TreeShap(feat, phi, cold_index, unique_depth + 1, unique_path,
             cold_zero_fraction * incoming_zero_fraction, 0,
             split_index, condition, condition_feature, cold_condition_fraction);
  }
}

void RegTree::CalculateContributions(const RegTree::FVec &feat,
                                     std::vector<float>* mean_values,
                                     bst_float *out_contribs,
                                     int condition,
                                     unsigned condition_feature) const {
  // find the expected value of the tree's predictions
  if (condition == 0) {
    bst_float node_value = (*mean_values)[0];
    out_contribs[feat.Size()] += node_value;
  }

  // Preallocate space for the unique path data
  const int maxd = this->MaxDepth(0) + 2;
  std::vector<PathElement> unique_path_data((maxd * (maxd + 1)) / 2);

  TreeShap(feat, out_contribs, 0, 0, unique_path_data.data(),
           1, 1, -1, condition, condition_feature, 1);
}
}  // namespace xgboost
