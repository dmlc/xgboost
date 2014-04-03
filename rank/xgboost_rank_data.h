#ifndef XGBOOST_RANK_DATA_H
#define XGBOOST_RANK_DATA_H

/*!
 * \file xgboost_rank_data.h
 * \brief input data structure for rank task.
 *     Format:
 *        The data should contains groups of rank data, a group here may refer to
 *        the rank list of a query, or the browsing history of a user, etc.
 *        Each group first contains the size of the group in a single line,
 * 	  then following is the line data with the same format with the regression data:
 *        label <nonzero feature dimension> [feature index:feature value]+
 * \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <cstdio>
#include <vector>
#include "../booster/xgboost_data.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost {
namespace rank {
/*! \brief data matrix for regression content */
struct RMatrix {
public:
    /*! \brief maximum feature dimension */
    unsigned num_feature;
    /*! \brief feature data content */
    booster::FMatrixS data;
    /*! \brief label of each instance */
    std::vector<float> labels;
    /*! \brief The index of begin and end of each group */
    std::vector<int> group_index;
public:
    /*! \brief default constructor */
    RMatrix( void ) {}

    /*! \brief get the number of instances */
    inline size_t Size() const {
        return labels.size();
    }

    /*!
    * \brief load from text file
    * \param fname name of text data
    * \param silent whether print information or not
    */
    inline void LoadText( const char* fname, bool silent = false ) {
        data.Clear();
        FILE* file = utils::FopenCheck( fname, "r" );
        float label;
        bool init = true;
        char tmp[ 1024 ];
        int group_size,group_size_acc = 0;
        std::vector<booster::bst_uint> findex;
        std::vector<booster::bst_float> fvalue;
        group_index.push_back(0);
        while(fscanf(file, "%d",group_size) == 1) {
            group_size_acc += group_size;
            group_index.push_back(group_size_acc);
            unsigned index;
            float value;
            if( sscanf( tmp, "%u:%f", &index, &value ) == 2 ) {
                findex.push_back( index );
                fvalue.push_back( value );
            } else {
                if( !init ) {
                    labels.push_back( label );
                    data.AddRow( findex, fvalue );
                }
                findex.clear();
                fvalue.clear();
                utils::Assert( sscanf( tmp, "%f", &label ) == 1, "invalid format" );
                init = false;
            }
        }


        labels.push_back( label );
        data.AddRow( findex, fvalue );
        // initialize column support as well
        data.InitData();

        if( !silent ) {
            printf("%ux%u matrix with %lu entries is loaded from %s\n",
                   (unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname );
        }
        fclose(file);
    }
    /*!
    * \brief load from binary file
    * \param fname name of binary data
    * \param silent whether print information or not
    * \return whether loading is success
    */
    inline bool LoadBinary( const char* fname, bool silent = false ) {
        FILE *fp = fopen64( fname, "rb" );
        int group_index_size = 0;
        if( fp == NULL ) return false;
        utils::FileStream fs( fp );
        data.LoadBinary( fs );
        labels.resize( data.NumRow() );
        utils::Assert( fs.Read( &labels[0], sizeof(float) * data.NumRow() ) != 0, "DMatrix LoadBinary" );

        utils::Assert( fs.Read( &group_index_size, sizeof(int) ) != 0, "Load group indice size" );
        group_index.resize(group_index_size);
        utils::Assert( fs.Read( &group_index, sizeof(int) * group_index_size) != 0, "Load group indice" d);

        fs.Close();
        // initialize column support as well
        data.InitData();

        if( !silent ) {
            printf("%ux%u matrix with %lu entries is loaded from %s\n",
                   (unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname );
        }
        return true;
    }
    /*!
    * \brief save to binary file
    * \param fname name of binary data
    * \param silent whether print information or not
    */
    inline void SaveBinary( const char* fname, bool silent = false ) {
        // initialize column support as well
        data.InitData();

        utils::FileStream fs( utils::FopenCheck( fname, "wb" ) );
        data.SaveBinary( fs );
        fs.Write( &labels[0], sizeof(float) * data.NumRow() );

        fs.Write( &(group_index.size()), sizeof(int));
        fs.Write( &group_index[0], sizeof(int) * group_index.size() );

        fs.Close();
        if( !silent ) {
            printf("%ux%u matrix with %lu entries is saved to %s\n",
                   (unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname );
        }
    }
    /*!
    * \brief cache load data given a file name, if filename ends with .buffer, direct load binary
    *        otherwise the function will first check if fname + '.buffer' exists,
    *        if binary buffer exists, it will reads from binary buffer, otherwise, it will load from text file,
    *        and try to create a buffer file
    * \param fname name of binary data
    * \param silent whether print information or not
    * \param savebuffer whether do save binary buffer if it is text
    */
    inline void CacheLoad( const char *fname, bool silent = false, bool savebuffer = true ) {
        int len = strlen( fname );
        if( len > 8 && !strcmp( fname + len - 7,  ".buffer") ) {
            this->LoadBinary( fname, silent );
            return;
        }
        char bname[ 1024 ];
        sprintf( bname, "%s.buffer", fname );
        if( !this->LoadBinary( bname, silent ) ) {
            this->LoadText( fname, silent );
            if( savebuffer ) this->SaveBinary( bname, silent );
        }
    }
private:
    /*! \brief update num_feature info */
    inline void UpdateInfo( void ) {
        this->num_feature = 0;
        for( size_t i = 0; i < data.NumRow(); i ++ ) {
            booster::FMatrixS::Line sp = data[i];
            for( unsigned j = 0; j < sp.len; j ++ ) {
                if( num_feature <= sp[j].findex ) {
                    num_feature = sp[j].findex + 1;
                }
            }
        }
    }
};
};
};
#endif
