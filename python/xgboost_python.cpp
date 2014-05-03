#include "xgboost_python.h"
#include "../regrank/xgboost_regrank.h"
#include "../regrank/xgboost_regrank_data.h"

namespace xgboost{
    namespace python{
        class DMatrix: public regrank::DMatrix{
        public:
            // whether column is initialized
            bool init_col_;
        public:
            DMatrix(void){
                init_col_ = false;
            }            
            ~DMatrix(void){}
        public:            
            inline void Load(const char *fname, bool silent){
                this->CacheLoad(fname, silent);
                init_col_ = this->data.HaveColAccess();
            }
        };
    };
};

using namespace xgboost::python;

extern "C"{
    void* XGDMatrixCreate(void){
        return new DMatrix();
    }
    void XGDMatrixFree(void *handle){
        delete static_cast<DMatrix*>(handle);
    }
    void XGDMatrixLoad(void *handle, const char *fname, int silent){
        static_cast<DMatrix*>(handle)->Load(fname, silent!=0);
    }
    void XGDMatrixSaveBinary(void *handle, const char *fname, int silent){
        static_cast<DMatrix*>(handle)->SaveBinary(fname, silent!=0);
    }
};

