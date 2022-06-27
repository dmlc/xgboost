/*!
 * Copyright by Contributors 2019
 */
#include <iostream>
#include <vector>
#include <utility>

#include "xgboost/json.h"
#include "param.h"

namespace std {
std::istream &operator>>(std::istream &is, std::vector<int> &t) {
  t.clear();
  // get (
  while (true) {
    char ch = is.peek();
    if (isdigit(ch)) {
      int idx;
      if (is >> idx) {
        t.emplace_back(idx);
      }
      return is;
    }
    is.get();
    if (ch == '(') {
      break;
    }
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  int idx;
  std::vector<int> tmp;
  while (true) {
    char ch = is.peek();
    if (isspace(ch)) {
      is.get();
    } else {
      break;
    }
  }
  if (is.peek() == ')') {
    is.get();
    return is;
  }
  while (is >> idx) {
    tmp.push_back(idx);
    char ch;
    do {
      ch = is.get();
    } while (isspace(ch));
    if (ch == 'L') {
      ch = is.get();
    }
    if (ch == ',') {
      while (true) {
        ch = is.peek();
        if (isspace(ch)) {
          is.get();
          continue;
        }
        if (ch == ')') {
          is.get();
          break;
        }
        break;
      }
      if (ch == ')') {
        break;
      }
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  t = std::move(tmp);
  return is;
}
}  // namespace std

namespace xgboost {
void ParseInteractionConstraint(
    std::string const &constraint_str,
    std::vector<std::vector<bst_feature_t>> *p_out) {
  auto &out = *p_out;
  auto j_inc = Json::Load({constraint_str.c_str(), constraint_str.size()});
  auto const &all = get<Array>(j_inc);
  out.resize(all.size());
  for (size_t i = 0; i < all.size(); ++i) {
    auto const &set = get<Array const>(all[i]);
    for (auto const &v : set) {
      if (XGBOOST_EXPECT(IsA<Integer const>(v), true)) {
        auto u = static_cast<bst_feature_t>(get<Integer const>(v));
        out[i].emplace_back(u);
      } else if (IsA<Number>(v)) {
        double d = get<Number const>(v);
        CHECK_EQ(std::floor(d), d)
            << "Found floating point number in interaction constraints";
        out[i].emplace_back(static_cast<uint32_t>(d));
      } else {
        LOG(FATAL) << "Unknown value type for interaction constraint:"
                   << v.GetValue().TypeStr();
      }
    }
  }
}
}  // namespace xgboost
