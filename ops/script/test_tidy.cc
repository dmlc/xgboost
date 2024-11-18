#include <iostream>
#include <vector>

struct Foo {
  int bar_;
};

int main() {
  std::vector<Foo> values;
  values.push_back(Foo());
}
