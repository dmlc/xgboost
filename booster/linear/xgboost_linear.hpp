#ifndef XGBOOST_LINEAR_HPP
#define XGBOOST_LINEAR_HPP
/*!
 * \file xgboost_linear.h
 * \brief Implementation of Linear booster, with L1/L2 regularization: Elastic Net
 *        the update rule is coordinate descent, require column major format
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <vector>
#include <algorithm>

#include "../xgboost.h"
#include "../../utils/xgboost_utils.h"

namespace xgboost{
    namespace booster{
        /*! \brief linear model, with L1/L2 regularization */
        template<typename FMatrix>
        class LinearBooster : public InterfaceBooster<FMatrix>{
        public:
            LinearBooster( void ){ silent = 0;}
            virtual ~LinearBooster( void ){}
        public:
            virtual void SetParam( const char *name, const char *val ){
                if( !strcmp( name, "silent") )  silent = atoi( val );
                if( model.weight.size() == 0 )  model.param.SetParam( name, val );
                param.SetParam( name, val );
            }
            virtual void LoadModel( utils::IStream &fi ){
                model.LoadModel( fi );
            }
            virtual void SaveModel( utils::IStream &fo ) const{
                model.SaveModel( fo );
            }
            virtual void InitModel( void ){
                model.InitModel();
            }
        public:
            virtual void DoBoost( std::vector<float> &grad, 
                                  std::vector<float> &hess,
                                  const FMatrix &fmat,
                                  const std::vector<unsigned> &root_index ){
                utils::Assert( grad.size() < UINT_MAX, "number of instance exceed what we can handle" );
                this->UpdateWeights( grad, hess, fmat );
            }
            inline float Predict( const FMatrix &fmat, bst_uint ridx, unsigned root_index ){
                float sum = model.bias();
                for( typename FMatrix::RowIter it = fmat.GetRow(ridx); it.Next(); ){ 
                    sum += model.weight[ it.findex() ] * it.fvalue();
                }
                return sum;
            }
            virtual float Predict( const std::vector<float> &feat, 
                                   const std::vector<bool>  &funknown,
                                   unsigned rid = 0 ){
                float sum = model.bias();
                for( size_t i = 0; i < feat.size(); i ++ ){
                    if( funknown[i] ) continue;
                    sum += model.weight[ i ] * feat[ i ];
                }
                return sum;
            }
            
        protected:
            // training parameter
            struct ParamTrain{
                /*! \brief learning_rate */
                float learning_rate;
                /*! \brief regularization weight for L2 norm */
                float reg_lambda;
                /*! \brief regularization weight for L1 norm */
                float reg_alpha;
                 /*! \brief regularization weight for L2 norm  in bias */               
                float reg_lambda_bias;
                
                ParamTrain( void ){
                    reg_alpha = 0.0f; reg_lambda = 0.0f; reg_lambda_bias = 0.0f;
                    learning_rate = 1.0f;
                }            
                inline void SetParam( const char *name, const char *val ){
                    // sync-names
                    if( !strcmp( "eta", name ) )    learning_rate = (float)atof( val );
                    if( !strcmp( "lambda", name ) ) reg_lambda = (float)atof( val );
                    if( !strcmp( "alpha", name ) )  reg_alpha  = (float)atof( val );
                    if( !strcmp( "lambda_bias", name ) ) reg_lambda_bias = (float)atof( val );
                    // real names
                    if( !strcmp( "learning_rate", name ) ) learning_rate = (float)atof( val );     
                    if( !strcmp( "reg_lambda", name ) )    reg_lambda = (float)atof( val );
                    if( !strcmp( "reg_alpha", name ) )     reg_alpha = (float)atof( val );
                    if( !strcmp( "reg_lambda_bias", name ) )    reg_lambda_bias = (float)atof( val );
                }
                // given original weight calculate delta 
                inline double CalcDelta( double sum_grad, double sum_hess, double w ){
                    if( sum_hess < 1e-5f ) return 0.0f;
                    double tmp = w - ( sum_grad + reg_lambda*w )/( sum_hess + reg_lambda );
                    if ( tmp >=0 ){
                        return std::max(-( sum_grad + reg_lambda*w + reg_alpha)/(sum_hess+reg_lambda),-w);
                    }else{
                        return std::min(-( sum_grad + reg_lambda*w - reg_alpha)/(sum_hess+reg_lambda),-w);
                    }
                }
                // given original weight calculate delta bias
                inline double CalcDeltaBias( double sum_grad, double sum_hess, double w ){
                    return - (sum_grad + reg_lambda_bias*w) / (sum_hess + reg_lambda_bias );
                }
            };
            
            // model for linear booster
            class Model{
            public:
                // model parameter
                struct Param{
                    // number of feature dimension
                    int num_feature;
                    // reserved field
                    int reserved[ 32 ];
                    // constructor
                    Param( void ){
                        num_feature = 0;
                        memset( reserved, 0, sizeof(reserved) );
                    }
                    inline void SetParam( const char *name, const char *val ){
                        if( !strcmp( name, "num_feature" ) )  num_feature = atoi( val );
                    }
                };
            public:
                Param param;
                // weight for each of feature, bias is the last one
                std::vector<float> weight;
            public:
                // initialize the model parameter
                inline void InitModel( void ){
                    // bias is the last weight
                    weight.resize( param.num_feature + 1 );
                    std::fill( weight.begin(), weight.end(), 0.0f );
                }
                // save the model to file 
                inline void SaveModel( utils::IStream &fo ) const{
                    fo.Write( &param, sizeof(Param) );
                    fo.Write( &weight[0], sizeof(float) * weight.size() );
                }
                // load model from file
                inline void LoadModel( utils::IStream &fi ){
                    utils::Assert( fi.Read( &param, sizeof(Param) ) != 0, "Load LinearBooster" );
                    weight.resize( param.num_feature + 1 );
                    utils::Assert( fi.Read( &weight[0], sizeof(float) * weight.size() ) != 0, "Load LinearBooster" );
                }
                // model bias
                inline float &bias( void ){
                    return weight.back();
                }
            };
        private:
            int silent;
        protected:
            Model model;
            ParamTrain param;
        protected:
            // update weights, should work for any FMatrix
            inline void UpdateWeights( std::vector<float> &grad,                       
                                       const std::vector<float> &hess,
                                       const FMatrix &smat ){
                {// optimize bias
                    double sum_grad = 0.0, sum_hess = 0.0;
                    for( size_t i = 0; i < grad.size(); i ++ ){
                        sum_grad += grad[ i ]; sum_hess += hess[ i ];
                    }
                    // remove bias effect
                    double dw = param.learning_rate * param.CalcDeltaBias( sum_grad, sum_hess, model.bias() );
                    model.bias() += dw;
                    // update grad value 
                    for( size_t i = 0; i < grad.size(); i ++ ){
                        grad[ i ] += dw * hess[ i ];
                    }
                }

                // optimize weight
                const unsigned nfeat= (unsigned)smat.NumCol();                           
                for( unsigned i = 0; i < nfeat; i ++ ){
                    if( !smat.GetSortedCol( i ).Next() ) continue;
                    double sum_grad = 0.0, sum_hess = 0.0;
                    for( typename FMatrix::ColIter it = smat.GetSortedCol(i); it.Next(); ){
                        const float v = it.fvalue();
                        sum_grad += grad[ it.rindex() ] * v;
                        sum_hess += hess[ it.rindex() ] * v * v;
                    }
                    float w = model.weight[ i ];
                    double dw = param.learning_rate * param.CalcDelta( sum_grad, sum_hess, w );
                    model.weight[ i ] += dw;
                    // update grad value 
                    for( typename FMatrix::ColIter it = smat.GetSortedCol(i); it.Next(); ){
                        const float v = it.fvalue();
                        grad[ it.rindex() ] += hess[ it.rindex() ] * v * dw;
                    }
                }
            }
        };
    };
};
#endif
