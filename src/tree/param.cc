/*!
 * Copyright by Contributors 2019
 */
#include <iostream>
#include <vector>
#include <utility>

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
