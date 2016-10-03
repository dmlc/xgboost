#include <cstdio>
#include <functional>
#include <dmlc/registry.h>

namespace tree {
struct Tree {
  virtual void Print() = 0;
  virtual ~Tree() {}
};

struct BinaryTree : public Tree {
  virtual void Print() {
    printf("I am binary tree\n");
  }
};

struct AVLTree : public Tree {
  virtual void Print() {
    printf("I am AVL tree\n");
  }
};
// registry to get the trees
struct TreeFactory
    : public dmlc::FunctionRegEntryBase<TreeFactory, std::function<Tree*()> > {
};

#define REGISTER_TREE(Name)                                             \
  DMLC_REGISTRY_REGISTER(::tree::TreeFactory, TreeFactory, Name)        \
  .set_body([]() { return new Name(); } )

DMLC_REGISTRY_FILE_TAG(my_tree);

}  // namespace tree


// usually this sits on a seperate file
namespace dmlc {
DMLC_REGISTRY_ENABLE(tree::TreeFactory);
}

namespace tree {
// Register the trees, can be in seperate files
REGISTER_TREE(BinaryTree)
.describe("This is a binary tree.");

REGISTER_TREE(AVLTree);

DMLC_REGISTRY_LINK_TAG(my_tree);
}

int main(int argc, char *argv[]) {
  // construct a binary tree
  tree::Tree *binary = dmlc::Registry<tree::TreeFactory>::Find("BinaryTree")->body();
  binary->Print();
  // construct a binary tree
  tree::Tree *avl = dmlc::Registry<tree::TreeFactory>::Find("AVLTree")->body();
  avl->Print();
  delete binary; delete avl;
  return 0;
}
