/*!
 * Copyright 2014 by Contributors
 * \file updater_sync.cc
 * \brief synchronize the tree in all distributed nodes
 */
#include <xgboost/tree_updater.h>
#include <vector>
#include <string>
#include <limits>
#include "../common/sync.h"
#include "../common/io.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_sync);

/*!
 * \brief syncher that synchronize the tree in all distributed nodes
 * can implement various strategies, so far it is always set to node 0's tree
 */
class TreeSyncher: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {}

  void Update(HostDeviceVector<GradientPair> *gpair,
              DMatrix* dmat,
              const std::vector<RegTree*> &trees) override {
    if (rabit::GetWorldSize() == 1) return;
    std::string s_model;
    common::MemoryBufferStream fs(&s_model);
    int rank = rabit::GetRank();
    if (rank == 0) {
      for (auto tree : trees) {
        tree->Save(&fs);
      }
    }
    fs.Seek(0);
    rabit::Broadcast(&s_model, 0);
    for (auto tree : trees) {
      tree->Load(&fs);
    }
  }
};

XGBOOST_REGISTER_TREE_UPDATER(TreeSyncher, "sync")
.describe("Syncher that synchronize the tree in all distributed nodes.")
.set_body([]() {
    return new TreeSyncher();
  });
}  // namespace tree
}  // namespace xgboost
