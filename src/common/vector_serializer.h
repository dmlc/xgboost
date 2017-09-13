#include <iostream>
#include <cstddef>
#include <vector>

// define string serializer for vector, to get the arguments
namespace std {
inline std::ostream &operator<<(std::ostream &os, const std::vector<int> &t) {
  os << '(';
  for (std::vector<int>::const_iterator it = t.begin(); it != t.end(); ++it) {
    if (it != t.begin())
      os << ',';
    os << *it;
  }
  // python style tuple
  if (t.size() == 1)
    os << ',';
  os << ')';
  return os;
}

inline std::istream &operator>>(std::istream &is, std::vector<int> &t) {
  // get (
  while (true) {
    char ch = is.peek();
    if (isdigit(ch)) {
      int idx;
      if (is >> idx) {
        t.assign(&idx, &idx + 1);
      }
      return is;
    }
    is.get();
    if (ch == '(')
      break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  int idx;
  std::vector<int> tmp;
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
      if (ch == ')')
        break;
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  t.assign(tmp.begin(), tmp.end());
  return is;
}
}  // namespace std
