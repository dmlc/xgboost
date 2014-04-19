#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include <string>
#include <cstring>
#include "../base/xgboost_learner.h"
#include "../utils/xgboost_fmap.h"
#include "../utils/xgboost_random.h"
#include "../utils/xgboost_config.h"
#include "../base/xgboost_learner.h"
#include "../base/xgboost_boost_task.h"
#include "xgboost_rank.h"
#include "../regression/xgboost_reg.h"
#include "../regression/xgboost_reg_main.cpp"
#include "../base/xgboost_data_instance.h"

int main(int argc, char *argv[]) {    
  xgboost::random::Seed(0);
  xgboost::base::BoostTask rank_tsk;
  rank_tsk.SetLearner(new xgboost::rank::RankBoostLearner);
  return rank_tsk.Run(argc, argv);
}
