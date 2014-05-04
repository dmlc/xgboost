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
            inline void Clear( void ){
                this->data.Clear();
                this->info.labels.clear();
                this->info.weights.clear();
                this->info.group_ptr.clear();
            }
            inline size_t NumRow( void ) const{
                return this->data.NumRow();
            }
            inline void AddRow( const XGEntry *data, size_t len ){
                xgboost::booster::FMatrixS &mat = this->data;
                mat.row_data_.resize( mat.row_ptr_.back() + len );
                memcpy( &mat.row_data_[mat.row_ptr_.back()], data, sizeof(XGEntry)*len );
                mat.row_ptr_.push_back( mat.row_ptr_.back() + len );
            }
            inline const XGEntry* GetRow(unsigned ridx, size_t* len) const{
                const xgboost::booster::FMatrixS &mat = this->data;

                *len = mat.row_ptr_[ridx+1] - mat.row_ptr_[ridx];
                return &mat.row_data_[ mat.row_ptr_[ridx] ];
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
    void XGDMatrixClear(void *handle){
        static_cast<DMatrix*>(handle)->Clear();
    }
    void XGDMatrixAddRow( void *handle, const XGEntry *data, size_t len ){
        static_cast<DMatrix*>(handle)->AddRow(data, len);
    }
    size_t XGDMatrixNumRow(const void *handle){
        return static_cast<const DMatrix*>(handle)->NumRow();
    }
    const XGEntry* XGDMatrixGetRow(void *handle, unsigned ridx, size_t* len){
        return static_cast<DMatrix*>(handle)->GetRow(ridx, len);
    }
};

