/*!
 * Copyright 2015-2019 by Contributors
 * \file tree_model.cc
 * \brief model structure for tree
 */
#include <dmlc/registry.h>
#include <dmlc/json.h>

#include <xgboost/tree_model.h>
#include <xgboost/logging.h>
#include <sstream>
#include <limits>
#include <cmath>
#include <iomanip>

#include "param.h"

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

  static std::string Match(std::string const& input,
                           std::map<std::string, std::string> replacements) {
    std::string result = input;
    for (auto const& kv : replacements) {
      auto pos = result.find(kv.first);
      CHECK_NE(pos, std::string::npos);
      result.replace(pos, kv.first.length(), kv.second);
    }
    return result;
  }

  virtual std::string Indicator(RegTree const& tree, int32_t nid, uint32_t depth) {
    return "";
  }
  virtual std::string Integer(RegTree const& tree, int32_t nid, uint32_t depth) {
    return "";
  }
  virtual std::string Quantitive(RegTree const& tree, int32_t nid, uint32_t depth) {
    return "";
  }

  virtual std::string PlainNode(RegTree const& tree, int32_t nid, uint32_t depth) = 0;
  virtual std::string NodeStat(RegTree const& tree, int32_t nid) {
    return "";
  }

  virtual std::string SplitNode(RegTree const& tree, int32_t nid, uint32_t depth) {
    auto const split_index = tree[nid].SplitIndex();
    std::string result;
    if (split_index < fmap_.Size()) {
      switch (fmap_.type(split_index)) {
        case FeatureMap::kIndicator: {
          result = this->Indicator(tree, nid, depth);
          break;
        }
        case FeatureMap::kInteger: {
          result = this->Integer(tree, nid, depth);
          break;
        }
        case FeatureMap::kFloat:
        case FeatureMap::kQuantitive: {
          result = this->Quantitive(tree, nid, depth);
          break;
        }
        default:
          LOG(FATAL) << "Unknown feature map type.";
      }
    } else {
      result = this->PlainNode(tree, nid, depth);
    }
    return result;
  }
  virtual std::string LeafNode(RegTree const& tree, int32_t nid, uint32_t depth) = 0;
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

  static TreeGenerator* Create(std::string attrs, FeatureMap const& fmap,
                               bool with_stats);
};

struct TreeIOReg : public dmlc::FunctionRegEntryBase<
  TreeIOReg,
  std::function<TreeGenerator* (
      FeatureMap const& fmap, std::string attrs, bool with_stats)> > {
};
}  // namespace xgboost


namespace dmlc {
DMLC_REGISTRY_ENABLE(::xgboost::TreeIOReg);
}  // namespace dmlc

namespace xgboost {

TreeGenerator* TreeGenerator::Create(std::string attrs, FeatureMap const& fmap,
                                     bool with_stats) {
  auto pos = attrs.find(':');
  std::string name;
  std::string params;
  if (pos != std::string::npos) {
    name = attrs.substr(0, pos);
    params = attrs.substr(pos+1, attrs.length() - pos - 1);
    // Eliminate all occurances of single quote string.
    size_t pos = std::string::npos;
    while ((pos = params.find('\'')) != std::string::npos) {
      params.replace(pos, 1, "\"");
    }
  } else {
    name = attrs;
  }
  auto *e = ::dmlc::Registry< ::xgboost::TreeIOReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown Model Builder:" << name;
  }
  auto p_io_builder = (e->body)(fmap, params, with_stats);
  return p_io_builder;
}

#define XGBOOST_REGISTER_TREE_IO(UniqueId, Name)                        \
  static DMLC_ATTRIBUTE_UNUSED ::xgboost::TreeIOReg&                    \
  __make_ ## TreeIOReg ## _ ## UniqueId ## __ =                         \
      ::dmlc::Registry< ::xgboost::TreeIOReg>::Get()->__REGISTER__(Name)


class TextGenerator : public TreeGenerator {
  using SuperT = TreeGenerator;

 public:
  TextGenerator(FeatureMap const& fmap, std::string attrs, bool with_stats) :
      TreeGenerator(fmap, with_stats) {}

  std::string LeafNode(RegTree const& tree, int32_t nid, uint32_t depth) override {
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

  std::string Indicator(RegTree const& tree, int32_t nid, uint32_t depth) override {
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
      std::string cond, uint32_t depth) {
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

  std::string Integer(RegTree const& tree, int32_t nid, uint32_t depth) override {
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

  std::string Quantitive(RegTree const& tree, int32_t nid, uint32_t depth) override {
    static std::string const kQuantitiveTemplate =
        "{tabs}{nid}:[{fname}<{cond}] yes={left},no={right},missing={missing}";
    auto cond = tree[nid].SplitCond();
    return SplitNodeImpl(tree, nid, kQuantitiveTemplate, SuperT::ToStr(cond), depth);
  }

  std::string PlainNode(RegTree const& tree, int32_t nid, uint32_t depth) override {
    auto cond = tree[nid].SplitCond();
    static std::string const kNodeTemplate =
        "{tabs}{nid}:[f{fname}<{cond}] yes={left},no={right},missing={missing}";
    return SplitNodeImpl(tree, nid, kNodeTemplate, SuperT::ToStr(cond), depth);
  }

  std::string NodeStat(RegTree const& tree, int32_t nid) override {
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
.set_body([](FeatureMap const& fmap, std::string attrs, bool with_stats) {
            return new TextGenerator(fmap, attrs, with_stats);
          });

class JsonGenerator : public TreeGenerator {
  using SuperT = TreeGenerator;

 public:
  JsonGenerator(FeatureMap const& fmap, std::string attrs, bool with_stats) :
      TreeGenerator(fmap, with_stats) {}

  std::string Indent(uint32_t depth) {
    std::string result;
    for (uint32_t i = 0; i < depth + 1; ++i) {
      result += "  ";
    }
    return result;
  }

  std::string LeafNode(RegTree const& tree, int32_t nid, uint32_t depth) override {
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

  std::string Indicator(RegTree const& tree, int32_t nid, uint32_t depth) override {
    int32_t nyes = tree[nid].DefaultLeft() ?
                   tree[nid].RightChild() : tree[nid].LeftChild();
    static std::string const kIndicatorTemplate =
        R"ID( "nodeid": {nid}, "depth": {depth}, "split": "{fname}", "yes": {yes}, "no": {no}})ID";
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

  std::string SplitNodeImpl(RegTree const& tree, int32_t nid,
                            std::string const& template_str, std::string cond, uint32_t depth) {
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

  std::string Integer(RegTree const& tree, int32_t nid, uint32_t depth) override {
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

  std::string Quantitive(RegTree const& tree, int32_t nid, uint32_t depth) override {
    static std::string const kQuantitiveTemplate =
        R"I( "nodeid": {nid}, "depth": {depth}, "split": "{fname}", )I"
        R"I("split_condition": {cond}, "yes": {left}, "no": {right}, )I"
        R"I("missing": {missing})I";
    bst_float cond = tree[nid].SplitCond();
    return SplitNodeImpl(tree, nid, kQuantitiveTemplate, SuperT::ToStr(cond), depth);
  }

  std::string PlainNode(RegTree const& tree, int32_t nid, uint32_t depth) override {
    auto cond = tree[nid].SplitCond();
    static std::string const kNodeTemplate =
        R"I( "nodeid": {nid}, "depth": {depth}, "split": {fname}, )I"
        R"I("split_condition": {cond}, "yes": {left}, "no": {right}, )I"
        R"I("missing": {missing})I";
    return SplitNodeImpl(tree, nid, kNodeTemplate, SuperT::ToStr(cond), depth);
  }

  std::string NodeStat(RegTree const& tree, int32_t nid) override {
    static std::string kStatTemplate =
        R"S(, "gain": {loss_chg}, "cover": {sum_hess})S";
    auto result = SuperT::Match(
        kStatTemplate,
        {{"{loss_chg}", SuperT::ToStr(tree.Stat(nid).loss_chg)},
         {"{sum_hess}", SuperT::ToStr(tree.Stat(nid).sum_hess)}});
    return result;
  }

  std::string SplitNode(RegTree const& tree, int32_t nid, uint32_t depth) override {
    std::string property = SuperT::SplitNode(tree, nid, depth);
    static std::string const kSplitNodeTemplate =
        "{{properties} {stat}, \"children\": [{left}, {right}\n{indent}]}";
    auto result = SuperT::Match(
        kSplitNodeTemplate,
        {{"{properties}", property},
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
.set_body([](FeatureMap const& fmap, std::string attrs, bool with_stats) {
            return new JsonGenerator(fmap, attrs, with_stats);
          });

struct GraphvizParam : public dmlc::Parameter<GraphvizParam> {
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
  std::stringstream& ss_;
  GraphvizParam param_;

 public:
  GraphvizGenerator(FeatureMap const& fmap, std::string attrs, bool with_stats) :
      TreeGenerator(fmap, with_stats), ss_{SuperT::ss_} {
    param_.InitAllowUnknown(std::map<std::string, std::string>{});
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
  // Only indicator is different, so we combine all different node
  // types into this function.
  std::string PlainNode(RegTree const& tree, int32_t nid, uint32_t depth) override {
    auto split = tree[nid].SplitIndex();
    auto cond = tree[nid].SplitCond();
    static std::string const kNodeTemplate = "    {nid} [ label=\"{label}\" {params}]\n";
    static std::string const kLabelTemplate = R"({fname}{<}{cond})";

    // Indicator only has fname.
    bool has_less = split > fmap_.Size() || (split < fmap_.Size() && fmap_.type(split));
    auto label = SuperT::Match(kLabelTemplate, {
        {"{fname}", split < fmap_.Size() ? fmap_.Name(split) :
                                           'f' + std::to_string(split)},
        {"{<}",     has_less ? "<" : ""},
        {"{cond}",  has_less ? SuperT::ToStr(cond) : ""}});
    std::string result = SuperT::Match(kNodeTemplate, {
        {"{nid}",    std::to_string(nid)},
        {"{label}",  label},
        {"{params}", param_.condition_node_params}});

    static std::string const kEdgeTemplate =
        "    {nid} -> {child} [label=\"{is_missing}\" color=\"{color}\"]\n";
    auto MatchFn = SuperT::Match;  // mingw failed to capture protected fn.
    auto BuildEdge =
        [&tree, nid, MatchFn, this](int32_t child) {
          bool is_missing = tree[nid].DefaultChild() == child;
          std::string buffer = MatchFn(kEdgeTemplate, {
              {"{nid}",        std::to_string(nid)},
              {"{child}",      std::to_string(child)},
              {"{color}",      is_missing ? param_.yes_color : param_.no_color},
              {"{is_missing}", is_missing ? "yes, missing": "no"}});
          return buffer;
        };
    result += BuildEdge(tree[nid].LeftChild());
    result += BuildEdge(tree[nid].RightChild());
    return result;
  };

  std::string LeafNode(RegTree const& tree, int32_t nid, uint32_t depth) override {
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
    auto result = SuperT::Match(
        kNodeTemplate,
        {{"{parent}", this->PlainNode(tree, nid, depth)},
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

std::string RegTree::DumpModel(const FeatureMap& fmap,
                               bool with_stats,
                               std::string format) const {
  std::unique_ptr<TreeGenerator> builder {
    TreeGenerator::Create(format, fmap, with_stats)
  };
  for (int32_t i = 0; i < param.num_roots; ++i) {
    builder->BuildTree(*this);
  }

  std::string result = builder->Str();
  return result;
}

void RegTree::FillNodeMeanValues() {
  size_t num_nodes = this->param.num_nodes;
  if (this->node_mean_values_.size() == num_nodes) {
    return;
  }
  this->node_mean_values_.resize(num_nodes);
  for (int root_id = 0; root_id < param.num_roots; ++root_id) {
    this->FillNodeMeanValue(root_id);
  }
}

bst_float RegTree::FillNodeMeanValue(int nid) {
  bst_float result;
  auto& node = (*this)[nid];
  if (node.IsLeaf()) {
    result = node.LeafValue();
  } else {
    result  = this->FillNodeMeanValue(node.LeftChild()) * this->Stat(node.LeftChild()).sum_hess;
    result += this->FillNodeMeanValue(node.RightChild()) * this->Stat(node.RightChild()).sum_hess;
    result /= this->Stat(nid).sum_hess;
  }
  this->node_mean_values_[nid] = result;
  return result;
}

void RegTree::CalculateContributionsApprox(const RegTree::FVec &feat,
                                           unsigned root_id,
                                           bst_float *out_contribs) const {
  CHECK_GT(this->node_mean_values_.size(), 0U);
  // this follows the idea of http://blog.datadive.net/interpreting-random-forests/
  unsigned split_index = 0;
  auto pid = static_cast<int>(root_id);
  // update bias value
  bst_float node_value = this->node_mean_values_[pid];
  out_contribs[feat.Size()] += node_value;
  if ((*this)[pid].IsLeaf()) {
    // nothing to do anymore
    return;
  }
  while (!(*this)[pid].IsLeaf()) {
    split_index = (*this)[pid].SplitIndex();
    pid = this->GetNext(pid, feat.Fvalue(split_index), feat.IsMissing(split_index));
    bst_float new_value = this->node_mean_values_[pid];
    // update feature weight
    out_contribs[split_index] += new_value - node_value;
    node_value = new_value;
  }
  bst_float leaf_value = (*this)[pid].LeafValue();
  // update leaf feature weight
  out_contribs[split_index] += leaf_value - node_value;
}

// Used by TreeShap
// data we keep about our decision path
// note that pweight is included for convenience and is not tied with the other attributes
// the pweight of the i'th path element is the permuation weight of paths with i-1 ones in them
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

// determine what the total permuation weight would be if
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
                       unsigned node_index, unsigned unique_depth,
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
    unsigned hot_index = 0;
    if (feat.IsMissing(split_index)) {
      hot_index = node.DefaultChild();
    } else if (feat.Fvalue(split_index) < node.SplitCond()) {
      hot_index = node.LeftChild();
    } else {
      hot_index = node.RightChild();
    }
    const unsigned cold_index = (static_cast<int>(hot_index) == node.LeftChild() ?
                                 node.RightChild() : node.LeftChild());
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
                                     unsigned root_id, bst_float *out_contribs,
                                     int condition,
                                     unsigned condition_feature) const {
  // find the expected value of the tree's predictions
  if (condition == 0) {
    bst_float node_value = this->node_mean_values_[static_cast<int>(root_id)];
    out_contribs[feat.Size()] += node_value;
  }

  // Preallocate space for the unique path data
  const int maxd = this->MaxDepth(root_id) + 2;
  auto *unique_path_data = new PathElement[(maxd * (maxd + 1)) / 2];

  TreeShap(feat, out_contribs, root_id, 0, unique_path_data,
           1, 1, -1, condition, condition_feature, 1);
  delete[] unique_path_data;
}
}  // namespace xgboost
