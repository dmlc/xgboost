#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <cstring>
using namespace std;
#include "./gbm.h"
#include "./gbtree-inl.hpp"
#include "./gblinear-inl.hpp"

namespace xgboost {
namespace gbm {
IGradBooster* CreateGradBooster(const char *name) {
  if (!strcmp("gbtree", name)) return new GBTree();
  if (!strcmp("gblinear", name)) return new GBLinear();
  utils::Error("unknown booster type: %s", name);
  return NULL;
}
}  // namespace gbm
}  // namespace xgboost

