#ifndef XGBOOST_REGRANK_DATA_H
#define XGBOOST_REGRANK_DATA_H

/*!
 * \file xgboost_regrank_data.h
 * \brief input data structure for regression, binary classification, and rankning.
 *     Format:
 *        The data should contain each data instance in each line.
 *		  The format of line data is as below:
 *        label <nonzero feature dimension> [feature index:feature value]+
 *     When using rank, an addtional group file with suffix group must be provided, giving the number of instances in each group
 *     When using weighted aware classification(regression), an addtional weight file must be provided, giving the weight of each instance
 * 
 * \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include "../booster/xgboost_data.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost{
    /*! \brief namespace to handle regression and rank */
    namespace regrank{
        /*! \brief data matrix for regression content */
        struct DMatrix{
        public:
            /*! \brief data information besides the features */
            struct Info{
                /*! \brief label of each instance */
                std::vector<float> labels;
                /*! \brief the index of begin and end of a groupneeded when the learning task is ranking */
                std::vector<unsigned> group_ptr;
                /*! \brief weights of each instance, optional */            
                std::vector<float> weights;
                /*! \brief specified root index of each instance, can be used for multi task setting*/
                std::vector<unsigned> root_index;
                /*! \brief get weight of each instances */
                inline float GetWeight( size_t i ) const{
                    if( weights.size() != 0 ) return weights[i];
                    else return 1.0f;
                }
                inline float GetRoot( size_t i ) const{
                    if( root_index.size() != 0 ) return static_cast<float>(root_index[i]);
                    else return 0;
                }
            };
        public:
            /*! \brief feature data content */
            booster::FMatrixS data;
            /*! \brief information fields */
            Info info;
        public:
            /*! \brief default constructor */
            DMatrix(void){}
            /*! \brief get the number of instances */
            inline size_t Size() const{
                return data.NumRow();
            }           
            /*!
             * \brief load from text file
             * \param fname name of text data
             * \param silent whether print information or not
             */
            inline void LoadText(const char* fname, bool silent = false){
                data.Clear();
                FILE* file = utils::FopenCheck(fname, "r");
                float label; bool init = true;
                char tmp[1024];
                std::vector<booster::bst_uint> findex;
                std::vector<booster::bst_float> fvalue;
                
                while (fscanf(file, "%s", tmp) == 1){
                    unsigned index; float value;
                    if (sscanf(tmp, "%u:%f", &index, &value) == 2){
                        findex.push_back(index); fvalue.push_back(value);
                    }
                    else{
                        if (!init){
                            info.labels.push_back(label);
                            data.AddRow(findex, fvalue);
                        }
                        findex.clear(); fvalue.clear();
                        utils::Assert(sscanf(tmp, "%f", &label) == 1, "invalid format");
                        init = false;
                    }
                }
            
                info.labels.push_back(label);
                data.AddRow(findex, fvalue);
                // initialize column support as well
                data.InitData();
                
                if (!silent){
                    printf("%ux%u matrix with %lu entries is loaded from %s\n",
                           (unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname);
                }
                fclose(file);
                this->TryLoadGroup(fname, silent);
                this->TryLoadWeight(fname, silent);
            }
            /*!
             * \brief load from binary file
             * \param fname name of binary data
             * \param silent whether print information or not
             * \return whether loading is success
             */
            inline bool LoadBinary(const char* fname, bool silent = false){
                FILE *fp = fopen64(fname, "rb");
                if (fp == NULL) return false;
                utils::FileStream fs(fp);
                data.LoadBinary(fs);
                info.labels.resize(data.NumRow());
                utils::Assert(fs.Read(&info.labels[0], sizeof(float)* data.NumRow()) != 0, "DMatrix LoadBinary");
                {// load in group ptr
                    unsigned ngptr;
                    if( fs.Read(&ngptr, sizeof(unsigned) ) != 0 ){
                        info.group_ptr.resize( ngptr );
                        if( ngptr != 0 ){
                            utils::Assert( fs.Read(&info.group_ptr[0], sizeof(unsigned) * ngptr) != 0, "Load group file");
                            utils::Assert( info.group_ptr.back() == data.NumRow(), "number of group must match number of record" );
                        }
                    }
                }
                {// load in weight
                    unsigned nwt;
                    if( fs.Read(&nwt, sizeof(unsigned) ) != 0 ){
                        utils::Assert( nwt == 0 || nwt == data.NumRow(), "invalid weight" );
                        info.weights.resize( nwt );
                        if( nwt != 0 ){
                            utils::Assert( fs.Read(&info.weights[0], sizeof(unsigned) * nwt) != 0, "Load weight file");
                        }
                    }
                }
                fs.Close();
                
                if (!silent){
                    printf("%ux%u matrix with %lu entries is loaded from %s\n",
                           (unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname);
                    if( info.group_ptr.size() != 0 ){
                        printf("data contains %u groups\n", (unsigned)info.group_ptr.size()-1 );
                    }
                }
                return true;
            }
            /*!
             * \brief save to binary file
             * \param fname name of binary data
             * \param silent whether print information or not
             */
            inline void SaveBinary(const char* fname, bool silent = false){
                // initialize column support as well
                data.InitData();
                
                utils::FileStream fs(utils::FopenCheck(fname, "wb"));
                data.SaveBinary(fs);
                utils::Assert( info.labels.size() == data.NumRow(), "label size is not consistent with feature matrix size" );
                fs.Write(&info.labels[0], sizeof(float) * data.NumRow());
                {// write out group ptr
                    unsigned ngptr = static_cast<unsigned>( info.group_ptr.size() );
                    fs.Write(&ngptr, sizeof(unsigned) );
                    if( ngptr != 0 ){
                        fs.Write(&info.group_ptr[0], sizeof(unsigned) * ngptr);
                    }
                }                
                {// write out weight
                    unsigned nwt = static_cast<unsigned>( info.weights.size() );
                    fs.Write( &nwt, sizeof(unsigned) );
                    if( nwt != 0 ){
                        fs.Write(&info.weights[0], sizeof(float) * nwt);
                    }
                }
                fs.Close();
                if (!silent){
                    printf("%ux%u matrix with %lu entries is saved to %s\n",
                       (unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname);
                    if( info.group_ptr.size() != 0 ){
                        printf("data contains %u groups\n", (unsigned)info.group_ptr.size()-1 );
                    }
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
            inline void CacheLoad(const char *fname, bool silent = false, bool savebuffer = true){
                int len = strlen(fname);
                if (len > 8 && !strcmp(fname + len - 7, ".buffer")){
                    if( !this->LoadBinary(fname, silent) ){
                        fprintf(stderr,"can not open file \"%s\"", fname);
                        utils::Error("DMatrix::CacheLoad failed");
                    }
                    return;
                }
                char bname[1024];
                sprintf(bname, "%s.buffer", fname);
                if (!this->LoadBinary(bname, silent)){
                    this->LoadText(fname, silent);
                    if (savebuffer) this->SaveBinary(bname, silent);
                }
            }
        private:
            inline bool TryLoadGroup(const char* fname, bool silent = false){
                std::string name = fname;
                if (name.length() > 8 && !strcmp(fname + name.length() - 7, ".buffer")){
                    name.resize( name.length() - 7 );
                }
                name += ".group";
                //if exists group data load it in
                FILE *fi = fopen64(name.c_str(), "r");
                if (fi == NULL) return false;                
                info.group_ptr.push_back(0);
                unsigned nline;
                while (fscanf(fi, "%u", &nline) == 1){
                    info.group_ptr.push_back(info.group_ptr.back()+nline);
                }
                if(!silent){
                    printf("%lu groups are loaded from %s\n", info.group_ptr.size()-1, name.c_str());
                }
                fclose(fi);
                utils::Assert( info.group_ptr.back() == data.NumRow(), "DMatrix: group data does not match the number of rows in feature matrix" );
                return true;
            }
            inline bool TryLoadWeight(const char* fname, bool silent = false){
                std::string name = fname;
                if (name.length() > 8 && !strcmp(fname + name.length() - 7, ".buffer")){
                    name.resize( name.length() - 7 );
                }
                name += ".weight";
                //if exists group data load it in
                FILE *fi = fopen64(name.c_str(), "r");
                if (fi == NULL) return false;                
                float wt;
                while (fscanf(fi, "%f", &wt) == 1){
                    info.weights.push_back( wt );
                }
                if(!silent){
                    printf("loading weight from %s\n", name.c_str());
                }
                fclose(fi);
                utils::Assert( info.weights.size() == data.NumRow(), "DMatrix: weight data does not match the number of rows in feature matrix" );
                return true;
            }
        };
    };
};
#endif
