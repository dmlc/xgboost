#include "tree/updater.h"
#include "gbm/gbm.h"
#include "utils/omp.h"
#include "utils/utils.h"
#include "utils/random.h"
#include "learner/objective.h"
#include "learner/learner-inl.hpp"

// pass compile flag

using namespace xgboost;
int main(void){
  FMatrixS fmat;  
  tree::RegTree tree;
  tree::TrainParam param;
  std::vector<bst_gpair> gpair;
  std::vector<unsigned> roots;
  tree::IUpdater<FMatrixS> *up = tree::CreateUpdater<FMatrixS>("prune"); 
  gbm::IGradBooster<FMatrixS> *gbm = new gbm::GBTree<FMatrixS>();
  std::vector<tree::RegTree*> trees;
  learner::IObjFunction *func = learner::CreateObjFunction("reg:linear");
  learner::BoostLearner<FMatrixS> *learner= new learner::BoostLearner<FMatrixS>();
  up->Update(gpair, fmat, roots, trees);

  return 0;
}
