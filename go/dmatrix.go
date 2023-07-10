package xgboost

/*
#include <xgboost/c_api.h>
*/
import "C"

type DMatrix struct {
	handle C.DMatrixHandle
}
