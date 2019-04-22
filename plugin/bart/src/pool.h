#ifndef BART_POOL_H_
#define BART_POOL_H_

#include <cstddef>
#include <vector>

namespace xgboost {
/*
 * \brief A pool of un-ordered memory that never got deallocated.
 */
template <typename Type>
class PersistentPool {
  std::vector<Type> _values;
  size_t _n_values;

 public:
  PersistentPool() : _n_values{0} {}
  PersistentPool(size_t n_values)
      : _values(n_values),
        _n_values{n_values} {}
  PersistentPool(size_t n_values, Type value) :
      _values(n_values), _n_values(n_values) {
    for (size_t i = 0; i < n_values; ++i) {
      _values[i] = std::move(value);
    }
  }

  void push(Type value) {
    if (_n_values == _values.size()) {
      _values.push_back(value);
    } else {
      _values[_n_values] = value;
    }
    _n_values ++;
  }
  void erase(size_t position) {
    CHECK_LT(position, _values.size());
    // no need for memory management
    std::swap(_values[position],
              _values[_n_values-1]);
    _n_values--;
    CHECK_GE(_n_values, 0);
  }

  size_t size() const { return _n_values; }
  std::vector<Type> const& data() const { return _values; }
  std::vector<Type>&       data()       { return _values; }

  Type& operator[](size_t position) {
    CHECK_GT(_n_values, position);
    return _values[position];
  }
  Type const& operator[](size_t position) const {
    CHECK_GT(_n_values, position);
    return _values[position];
  }
};

}      // namespace xgboost
#endif  // BART_POOL_H_
