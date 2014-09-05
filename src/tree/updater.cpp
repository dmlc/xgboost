#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <cstring>
#include "./updater.h"
#include "./updater_prune-inl.hpp"
#include "./updater_refresh-inl.hpp"
#include "./updater_colmaker-inl.hpp"

namespace xgboost {
namespace tree {
IUpdater* CreateUpdater(const char *name) {
  using namespace std;
  if (!strcmp(name, "prune")) return new TreePruner();
  if (!strcmp(name, "refresh")) return new TreeRefresher<GradStats>();
  if (!strcmp(name, "grow_colmaker")) return new ColMaker<GradStats>();
  utils::Error("unknown updater:%s", name);
  return NULL;
}

}  // namespace tree
}  // namespace xgboost
