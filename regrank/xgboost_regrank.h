#ifndef XGBOOST_REGRANK_H
#define XGBOOST_REGRANK_H
/*!
* \file xgboost_regrank.h
* \brief class for gradient boosted regression and ranking
* \author Kailong Chen: chenkl198812@gmail.com, Tianqi Chen: tianqi.tchen@gmail.com
*/
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "xgboost_regrank_data.h"
#include "xgboost_regrank_eval.h"
#include "xgboost_regrank_obj.h"
#include "../utils/xgboost_omp.h"
#include "../booster/xgboost_gbmbase.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"

namespace xgboost{
    namespace regrank{
        /*! \brief class for gradient boosted regression and ranking */
        class RegRankBoostLearner{
        public:
            /*! \brief constructor */
            RegRankBoostLearner(void){
                silent = 0;
                obj_ = NULL;
                name_obj_ = "reg:linear";
            }
            /*!
             * \brief a regression booter associated with training and evaluating data
             * \param mats  array of pointers to matrix whose prediction result need to be cached
             */
            RegRankBoostLearner(const std::vector<const DMatrix *>& mats){
                silent = 0;
                obj_ = NULL;
                name_obj_ = "reg:linear";
                this->SetCacheData(mats);
            }            
            /*!
             * \brief add internal cache space for mat, this can speedup prediction for matrix,
             *        please cache prediction for training and eval data
             *    warning: if the model is loaded from file from some previous training history
             *             set cache data must be called with exactly SAME 
             *             data matrices to continue training otherwise it will cause error
             * \param mats  array of pointers to matrix whose prediction result need to be cached
             */          
            inline void SetCacheData(const std::vector<const DMatrix *>& mats){
                // estimate feature bound
                int num_feature = 0;
                // assign buffer index
                unsigned buffer_size = 0;
                
                utils::Assert( cache_.size() == 0, "can only call cache data once" );
                for( size_t i = 0; i < mats.size(); ++i ){
                    bool dupilicate = false;
                    for( size_t j = 0; j < i; ++ j ){
                        if( mats[i] == mats[j] ) dupilicate = true;
                    }
                    if( dupilicate ) continue;
                    cache_.push_back( CacheEntry( mats[i], buffer_size ) );
                    buffer_size += static_cast<unsigned>(mats[i]->Size());
                    num_feature = std::max(num_feature, (int)(mats[i]->data.NumCol()));
                }
                
                char str_temp[25];
                if (num_feature > mparam.num_feature){
                    mparam.num_feature = num_feature;
                    sprintf(str_temp, "%d", num_feature);
                    base_gbm.SetParam("bst:num_feature", str_temp);
                }

                sprintf(str_temp, "%u", buffer_size);
                base_gbm.SetParam("num_pbuffer", str_temp);
                if (!silent){
                    printf("buffer_size=%u\n", buffer_size);
                }
            }

            /*!
             * \brief set parameters from outside
             * \param name name of the parameter
             * \param val  value of the parameter
             */
            inline void SetParam(const char *name, const char *val){
                if (!strcmp(name, "silent"))  silent = atoi(val);
                if (!strcmp(name, "eval_metric"))  evaluator_.AddEval(val);
                if (!strcmp(name, "objective") )   name_obj_ = val;
                if (!strcmp(name, "num_class") )   base_gbm.SetParam("num_booster_group", val );
                mparam.SetParam(name, val);
                base_gbm.SetParam(name, val);
                cfg_.push_back( std::make_pair( std::string(name), std::string(val) ) );
            }
            /*!
            * \brief initialize solver before training, called before training
            * this function is reserved for solver to allocate necessary space and do other preparation
            */
            inline void InitTrainer(void){
                if( mparam.num_class != 0 ){
                    if( name_obj_ != "multi:softmax" ){
                        name_obj_ = "multi:softmax";
                        printf("auto select objective=softmax to support multi-class classification\n" );
                    }
                }
                base_gbm.InitTrainer();                
                obj_ = CreateObjFunction( name_obj_.c_str() );
                for( size_t i = 0; i < cfg_.size(); ++ i ){
                    obj_->SetParam( cfg_[i].first.c_str(), cfg_[i].second.c_str() );
                }
                evaluator_.AddEval( obj_->DefaultEvalMetric() );
            }
            /*!
             * \brief initialize the current data storage for model, if the model is used first time, call this function
             */
            inline void InitModel(void){
                base_gbm.InitModel();
                mparam.AdjustBase(name_obj_.c_str());
            }
            /*!
             * \brief load model from file 
             * \param fname file name
             */
            inline void LoadModel(const char *fname){
                utils::FileStream fi(utils::FopenCheck(fname, "rb"));
                this->LoadModel(fi);
                fi.Close();                
            }
            /*!
             * \brief load model from stream
             * \param fi input stream
             */
            inline void LoadModel(utils::IStream &fi){
                base_gbm.LoadModel(fi);
                utils::Assert(fi.Read(&mparam, sizeof(ModelParam)) != 0);
            }
            /*!
             * \brief DumpModel
             * \param fo text file
             * \param fmap feature map that may help give interpretations of feature
             * \param with_stats whether print statistics as well
             */
            inline void DumpModel(FILE *fo, const utils::FeatMap& fmap, bool with_stats){
                base_gbm.DumpModel(fo, fmap, with_stats);
            }
            /*!
             * \brief Dump path of all trees
             * \param fo text file
             * \param data input data
             */
            inline void DumpPath(FILE *fo, const DMatrix &data){
                base_gbm.DumpPath(fo, data.data);
            }
            /*!
            * \brief save model to stream
            * \param fo output stream
            */
            inline void SaveModel(utils::IStream &fo) const{
                base_gbm.SaveModel(fo);
                fo.Write(&mparam, sizeof(ModelParam));
            }
            /*!
             * \brief save model into file
             * \param fname file name
             */
            inline void SaveModel(const char *fname) const{
                utils::FileStream fo(utils::FopenCheck(fname, "wb"));
                this->SaveModel(fo);
                fo.Close();                
            }
            /*!
             * \brief update the model for one iteration
             */
            inline void UpdateOneIter(const DMatrix &train){
                this->PredictRaw(preds_, train);
                obj_->GetGradient(preds_, train.info, base_gbm.NumBoosters(), grad_, hess_);
                if( grad_.size() == train.Size() ){
                    base_gbm.DoBoost(grad_, hess_, train.data, train.info.root_index);
                }else{
                    int ngroup = base_gbm.NumBoosterGroup();
                    utils::Assert( grad_.size() == train.Size() * (size_t)ngroup, "BUG: UpdateOneIter: mclass" );
                    std::vector<float> tgrad( train.Size() ), thess( train.Size() );
                    for( int g = 0; g < ngroup; ++ g ){
                        memcpy( &tgrad[0], &grad_[g*tgrad.size()], sizeof(float)*tgrad.size() );
                        memcpy( &thess[0], &hess_[g*tgrad.size()], sizeof(float)*tgrad.size() );
                        base_gbm.DoBoost(tgrad, thess, train.data, train.info.root_index, g );
                    }
                }
            }
            /*!
             * \brief evaluate the model for specific iteration
             * \param iter iteration number
             * \param evals datas i want to evaluate
             * \param evname name of each dataset
             * \param fo file to output log
             */
            inline void EvalOneIter(int iter,
                                    const std::vector<const DMatrix*> &evals,
                                    const std::vector<std::string> &evname,
                                    FILE *fo=stderr ){
                fprintf(fo, "[%d]", iter);
                for (size_t i = 0; i < evals.size(); ++i){
                    this->PredictRaw(preds_, *evals[i]);
                    obj_->PredTransform(preds_);
                    evaluator_.Eval(fo, evname[i].c_str(), preds_, evals[i]->info);
                }
                fprintf(fo, "\n");
                fflush(fo);
            }
            /*! 
             * \brief get prediction
             * \param storage to store prediction
             * \param data input data
             * \param bst_group booster group we are in
             */
            inline void Predict(std::vector<float> &preds, const DMatrix &data, int bst_group = -1){
                this->PredictRaw( preds, data, bst_group );
                obj_->PredTransform( preds );
            }            
        public:
            /*!
             * \brief interactive update 
             * \param action action type 
             * \parma train training data
             */
            inline void UpdateInteract(std::string action, const DMatrix& train){
                for(size_t i = 0; i < cache_.size(); ++i){
                    this->InteractPredict(preds_, *cache_[i].mat_);
                }

                if (action == "remove"){
                    base_gbm.DelteBooster(); return;
                }

                obj_->GetGradient(preds_, train.info, base_gbm.NumBoosters(), grad_, hess_);
                std::vector<unsigned> root_index;
                base_gbm.DoBoost(grad_, hess_, train.data, root_index);

                for(size_t i = 0; i < cache_.size(); ++i){
                    this->InteractRePredict(*cache_[i].mat_);
                }
            }
        private:
            /*! \brief get the transformed predictions, given data */
            inline void InteractPredict(std::vector<float> &preds, const DMatrix &data){
                int buffer_offset = this->FindBufferOffset(data);
                utils::Assert( buffer_offset >=0, "interact mode must cache training data" );
                preds.resize(data.Size());
                const unsigned ndata = static_cast<unsigned>(data.Size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    preds[j] = mparam.base_score + base_gbm.InteractPredict(data.data, j, buffer_offset + j);                    
                }
                obj_->PredTransform( preds );
            }
            /*! \brief repredict trial */
            inline void InteractRePredict(const DMatrix &data){
                int buffer_offset = this->FindBufferOffset(data);
                utils::Assert( buffer_offset >=0, "interact mode must cache training data" );
                const unsigned ndata = static_cast<unsigned>(data.Size());
                #pragma omp parallel for schedule( static )
                for (unsigned j = 0; j < ndata; ++j){
                    base_gbm.InteractRePredict(data.data, j, buffer_offset + j);
                }
            }
            /*! \brief get un-transformed prediction*/
            inline void PredictRaw(std::vector<float> &preds, const DMatrix &data, int bst_group = -1 ){
                int buffer_offset =  this->FindBufferOffset(data);
                if( bst_group < 0 ){
                    int ngroup = base_gbm.NumBoosterGroup();
                    preds.resize( data.Size() * ngroup );
                    for( int g = 0; g < ngroup; ++ g ){ 
                        this->PredictBuffer(&preds[ data.Size() * g ], data, buffer_offset, g );
                    }
                }else{
                    preds.resize( data.Size() );
                    this->PredictBuffer(&preds[0], data, buffer_offset, bst_group );
                }
            }
            /*! \brief get the un-transformed predictions, given data */
            inline void PredictBuffer(float *preds, const DMatrix &data, int buffer_offset, int bst_group ){
                const unsigned ndata = static_cast<unsigned>(data.Size());
                if( buffer_offset >= 0 ){  
                    #pragma omp parallel for schedule( static )
                    for (unsigned j = 0; j < ndata; ++j){
                        preds[j] = mparam.base_score + base_gbm.Predict(data.data, j, buffer_offset + j, data.info.GetRoot(j), bst_group );

                    }
                }else
                    #pragma omp parallel for schedule( static )
                    for (unsigned j = 0; j < ndata; ++j){
                        preds[j] = mparam.base_score + base_gbm.Predict(data.data, j, -1, data.info.GetRoot(j), bst_group );
                    }{
                }
            }
        private:
            /*! \brief training parameter for regression */
            struct ModelParam{
                /* \brief global bias */
                float base_score;
                /* \brief type of loss function */
                int loss_type;
                /* \brief number of features  */
                int num_feature;  
                /* \brief number of class, if it is multi-class classification  */
                int num_class; 
                /*! \brief reserved field */
                int reserved[15];
                /*! \brief constructor */
                ModelParam(void){
                    base_score = 0.5f;
                    loss_type = -1;
                    num_feature = 0;
                    num_class = 0;
                    memset(reserved, 0, sizeof(reserved));
                }
                /*!
                 * \brief set parameters from outside
                 * \param name name of the parameter
                 * \param val  value of the parameter
                 */
                inline void SetParam(const char *name, const char *val){
                    if (!strcmp("base_score", name))  base_score = (float)atof(val);
                    if (!strcmp("num_class", name))   num_class = atoi(val);
                    if (!strcmp("loss_type", name))   loss_type = atoi(val);
                    if (!strcmp("bst:num_feature", name)) num_feature = atoi(val);
                }
                /*!
                * \brief adjust base_score based on loss type and objective function
                */
                inline void AdjustBase(const char *obj){
                    // some tweaks for loss type
                    if( loss_type == -1 ){
                        loss_type = 1;
                        if( !strcmp("reg:linear", obj ) ) loss_type = 0;
                    }
                    if (loss_type == 1 || loss_type == 2|| loss_type == 3){
                        utils::Assert(base_score > 0.0f && base_score < 1.0f, "sigmoid range constrain");
                        base_score = -logf(1.0f / base_score - 1.0f);
                    }
                }
            };
        private:
            struct CacheEntry{
                const DMatrix *mat_;
                int buffer_offset_;
                CacheEntry(const DMatrix *mat, int buffer_offset)
                    :mat_(mat), buffer_offset_(buffer_offset){}
            };           
            /*! \brief the entries indicates that we have internal prediction cache */
            std::vector<CacheEntry> cache_;
        private:
            // find internal bufer offset for certain matrix, if not exist, return -1
            inline int FindBufferOffset(const DMatrix &mat){
                for(size_t i = 0; i < cache_.size(); ++i){
                    if( cache_[i].mat_ == &mat ) return cache_[i].buffer_offset_; 
                }
                return -1;
            } 
        protected:
            int silent;
            EvalSet evaluator_;
            booster::GBMBase base_gbm;
            ModelParam   mparam;           
            // objective fnction
            IObjFunction *obj_;
            // name of objective function
            std::string name_obj_;
            std::vector< std::pair<std::string, std::string> > cfg_;
        protected:
            std::vector<float> grad_, hess_, preds_;
        };
    }
};

#endif
