#ifndef _XGBOOST_REGDATA_H_
#define _XGBOOST_REGDATA_H_

/*!
* \file xgboost_regdata.h
* \brief input data structure for regression and binary classification task.
*     Format:
*        The data should contain each data instance in each line.
*		  The format of line data is as below:
*        label <nonzero feature dimension> [feature index:feature value]+
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/
#include <cstdio>
#include <vector>
#include "../booster/xgboost_data.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost{
    namespace regression{
        /*! \brief data matrix for regression content */
        struct DMatrix{
        public:
            /*! \brief maximum feature dimension */
            unsigned num_feature;
            /*! \brief feature data content */
            booster::FMatrixS data;
            /*! \brief label of each instance */
            std::vector<float> labels;
        public:
            /*! \brief default constructor */
            DMatrix( void ){}

            /*! \brief get the number of instances */
            inline size_t Size() const{
                return labels.size();
            }
            /*! 
            * \brief load from text file 
            * \param fname name of text data
            * \param silent whether print information or not
            */            
            inline void LoadText( const char* fname, bool silent = false ){
                data.Clear();
                FILE* file = utils::FopenCheck( fname, "r" );
                float label; bool init = true;
                char tmp[ 1024 ];
                std::vector<booster::bst_uint> findex;
                std::vector<booster::bst_float> fvalue;

                while( fscanf( file, "%s", tmp ) == 1 ){
                    unsigned index; float value;
                    if( sscanf( tmp, "%u:%f", &index, &value ) == 2 ){
                        findex.push_back( index ); fvalue.push_back( value );
                    }else{
                        if( !init ){
                            labels.push_back( label );
                            data.AddRow( findex, fvalue );
                        }
                        findex.clear(); fvalue.clear();
                        utils::Assert( sscanf( tmp, "%f", &label ) == 1, "invalid format" );
                        init = false;
                    }
                }

                labels.push_back( label );
                data.AddRow( findex, fvalue );

                this->UpdateInfo();
                if( !silent ){
                    printf("%ux%u matrix with %lu entries is loaded from %s\n", 
                        (unsigned)labels.size(), num_feature, (unsigned long)data.NumEntry(), fname );
                }
                fclose(file);
            }
            /*! 
            * \brief load from binary file 
            * \param fname name of binary data
            * \param silent whether print information or not
            * \return whether loading is success
            */
            inline bool LoadBinary( const char* fname, bool silent = false ){
                FILE *fp = fopen64( fname, "rb" );
                if( fp == NULL ) return false;                
                utils::FileStream fs( fp );
                data.LoadBinary( fs );
                labels.resize( data.NumRow() );
                utils::Assert( fs.Read( &labels[0], sizeof(float) * data.NumRow() ) != 0, "DMatrix LoadBinary" );
                fs.Close();
                this->UpdateInfo();
                if( !silent ){
                    printf("%ux%u matrix with %lu entries is loaded from %s\n", 
                        (unsigned)labels.size(), num_feature, (unsigned long)data.NumEntry(), fname );
                }
                return true;
            }
            /*! 
            * \brief save to binary file
            * \param fname name of binary data
            * \param silent whether print information or not
            */
            inline void SaveBinary( const char* fname, bool silent = false ){
                utils::FileStream fs( utils::FopenCheck( fname, "wb" ) );
                data.SaveBinary( fs );
                fs.Write( &labels[0], sizeof(float) * data.NumRow() );
                fs.Close();
                if( !silent ){
                    printf("%ux%u matrix with %lu entries is saved to %s\n", 
                        (unsigned)labels.size(), num_feature, (unsigned long)data.NumEntry(), fname );
                }
            }
            /*! 
            * \brief cache load data given a file name, the function will first check if fname + '.xgbuffer' exists,
            *        if binary buffer exists, it will reads from binary buffer, otherwise, it will load from text file,
            *        and try to create a buffer file 
            * \param fname name of binary data
            * \param silent whether print information or not
            * \return whether loading is success
            */            
            inline void CacheLoad( const char *fname, bool silent = false ){
                char bname[ 1024 ];
                sprintf( bname, "%s.buffer", fname );
                if( !this->LoadBinary( bname, silent ) ){
                    this->LoadText( fname, silent );
                    this->SaveBinary( fname, silent );
                }                
            }
        private:
            /*! \brief update num_feature info */
            inline void UpdateInfo( void ){
                this->num_feature = 0;
                for( size_t i = 0; i < data.NumRow(); i ++ ){
                    booster::FMatrixS::Line sp = data[i];
                    for( unsigned j = 0; j < sp.len; j ++ ){
                        if( num_feature <= sp.findex[j] ){
                            num_feature = sp.findex[j] + 1;
                        }
                    }
                }
            }
        };
    };
};
#endif
