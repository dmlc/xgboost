#include"xgboost_reg_train.h"
#include"xgboost_reg_test.h"
using namespace xgboost::regression;

int main(int argc, char *argv[]){
//	char* config_path = argv[1];
//	bool silent = ( atoi(argv[2]) == 1 );
	char* config_path = "c:\\cygwin64\\home\\chen\\github\\xgboost\\demo\\regression\\reg.conf";
	bool silent = false;
	RegBoostTrain train;
	RegBoostTest test;
	train.train(config_path,false);
	test.test(config_path,false);
}