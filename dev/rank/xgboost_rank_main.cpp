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

int main(int argc, char *argv[]) {
	
	xgboost::random::Seed(0);
	xgboost::base::BoostTask tsk;
	xgboost::utils::ConfigIterator itr(argv[1]);
	int learner_index = 0;
	while (itr.Next()){
		if (!strcmp(itr.name(), "learning_task")){
			learner_index = atoi(itr.val());
		}
	}
	xgboost::rank::RankBoostLearner* rank_learner = new xgboost::rank::RankBoostLearner;
	xgboost::base::BoostLearner *parent = static_cast<xgboost::base::BoostLearner*>(rank_learner);
	tsk.SetLearner(parent);
	return tsk.Run(argc, argv);
}
