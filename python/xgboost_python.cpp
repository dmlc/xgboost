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
            inline void AddRow( const XGEntry *data, size_t len ){
                xgboost::booster::FMatrixS &mat = this->data;
                mat.row_data_.resize( mat.row_ptr_.back() + len );
                memcpy( &mat.row_data_[mat.row_ptr_.back()], data, sizeof(XGEntry)*len );
                mat.row_ptr_.push_back( mat.row_ptr_.back() + len );
            }
            inline void ParseCSR( const size_t *indptr,
                                  const unsigned *indices,
                                  const float *data,
                                  size_t nindptr,
                                  size_t nelem ){
                xgboost::booster::FMatrixS &mat = this->data;
                mat.row_ptr_.resize( nindptr );
                memcpy( &mat.row_ptr_[0], indptr, sizeof(size_t)*nindptr );
                mat.row_data_.resize( nelem );
                for( size_t i = 0; i < nelem; ++ i ){
                    mat.row_data_[i] = XGEntry(indices[i], data[i]);
                }
            }
            inline void SetLabel( const float *label, size_t len ){
                this->info.labels.resize( len );
                memcpy( &(this->info).labels[0], label, sizeof(float)*len );
            }
            inline const float* GetLabel( size_t* len ) const{
                *len = this->info.labels.size();
                return &(this->info.labels[0]);
            }
            inline void InitTrain(void){
                if(!this->data.HaveColAccess()) this->data.InitData();
                utils::Assert( this->data.NumRow() == this->info.labels.size(), "DMatrix: number of labels must match number of rows in matrix");
            }
        };
    };
};

using namespace xgboost::python;

extern "C"{
    void* XGDMatrixCreate( void ){
        return new DMatrix();
    }
    void XGDMatrixFree( void *handle ){
        delete static_cast<DMatrix*>(handle);
    }
    void XGDMatrixLoad( void *handle, const char *fname, int silent ){
        static_cast<DMatrix*>(handle)->Load(fname, silent!=0);
    }
    void XGDMatrixSaveBinary( void *handle, const char *fname, int silent ){
        static_cast<DMatrix*>(handle)->SaveBinary(fname, silent!=0);
    }
    void XGDMatrixAddRow( void *handle, const XGEntry *data, size_t len ){
        static_cast<DMatrix*>(handle)->AddRow(data, len);
    }
    void XGDMatrixParseCSR( void *handle, 
                            const size_t *indptr,
                            const unsigned *indices,
                            const float *data,
                            size_t nindptr,
                            size_t nelem ){
        static_cast<DMatrix*>(handle)->ParseCSR(indptr, indices, data, nindptr, nelem);
    }
    void XGDMatrixSetLabel( void *handle, const float *label, size_t len ){
        static_cast<DMatrix*>(handle)->SetLabel(label,len);        
    }
    const float* XGDMatrixGetLabel( const void *handle, size_t* len ){
        return static_cast<const DMatrix*>(handle)->GetLabel(len);
    }
};

