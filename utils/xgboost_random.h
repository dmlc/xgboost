#ifndef XGBOOST_RANDOM_H
#define XGBOOST_RANDOM_H
/*!
 * \file xgboost_random.h
 * \brief PRNG to support random number generation
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 *
 * Use standard PRNG from stdlib
 */
#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef _MSC_VER
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int  uint32_t;
#else
#include <inttypes.h>
#endif

/*! namespace of PRNG */
namespace xgboost{
    namespace random{
        /*! \brief seed the PRNG */
        inline void Seed(uint32_t seed){
            srand(seed);
        }

        /*! \brief return a real number uniform in [0,1) */
        inline double NextDouble(){
            return static_cast<double>(rand()) / (static_cast<double>(RAND_MAX)+1.0);
        }
        /*! \brief return a real numer uniform in (0,1) */
        inline double NextDouble2(){
            return (static_cast<double>(rand()) + 1.0) / (static_cast<double>(RAND_MAX)+2.0);
        }
    };

    namespace random{
        /*! \brief return a random number */
        inline uint32_t NextUInt32(void){
            return (uint32_t)rand();
        }
        /*! \brief return a random number in n */
        inline uint32_t NextUInt32(uint32_t n){
            return (uint32_t)floor(NextDouble() * n);
        }
        /*! \brief return  x~N(0,1) */
        inline double SampleNormal(){
            double x, y, s;
            do{
                x = 2 * NextDouble2() - 1.0;
                y = 2 * NextDouble2() - 1.0;
                s = x*x + y*y;
            } while (s >= 1.0 || s == 0.0);

            return x * sqrt(-2.0 * log(s) / s);
        }

        /*! \brief return iid x,y ~N(0,1) */
        inline void SampleNormal2D(double &xx, double &yy){
            double x, y, s;
            do{
                x = 2 * NextDouble2() - 1.0;
                y = 2 * NextDouble2() - 1.0;
                s = x*x + y*y;
            } while (s >= 1.0 || s == 0.0);
            double t = sqrt(-2.0 * log(s) / s);
            xx = x * t;
            yy = y * t;
        }
        /*! \brief return  x~N(mu,sigma^2) */
        inline double SampleNormal(double mu, double sigma){
            return SampleNormal() * sigma + mu;
        }

        /*! \brief  return 1 with probability p, coin flip */
        inline int SampleBinary(double p){
            return NextDouble() < p;
        }

        /*! \brief  return distribution from Gamma( alpha, beta ) */
        inline double SampleGamma(double alpha, double beta) {
            if (alpha < 1.0) {
                double u;
                do {
                    u = NextDouble();
                } while (u == 0.0);
                return SampleGamma(alpha + 1.0, beta) * pow(u, 1.0 / alpha);
            }
            else {
                double d, c, x, v, u;
                d = alpha - 1.0 / 3.0;
                c = 1.0 / sqrt(9.0 * d);
                do {
                    do {
                        x = SampleNormal();
                        v = 1.0 + c*x;
                    } while (v <= 0.0);
                    v = v * v * v;
                    u = NextDouble();
                } while ((u >= (1.0 - 0.0331 * (x*x) * (x*x)))
                    && (log(u) >= (0.5 * x * x + d * (1.0 - v + log(v)))));
                return d * v / beta;
            }
        }

        template<typename T>
        inline void Exchange(T &a, T &b){
            T c;
            c = a;
            a = b;
            b = c;
        }

        template<typename T>
        inline void Shuffle(T *data, size_t sz){
            if (sz == 0) return;
            for (uint32_t i = (uint32_t)sz - 1; i > 0; i--){
                Exchange(data[i], data[NextUInt32(i + 1)]);
            }
        }
        // random shuffle the data inside, require PRNG 
        template<typename T>
        inline void Shuffle(std::vector<T> &data){
            Shuffle(&data[0], data.size());
        }
    };
    
    namespace random{
        /*! \brief random number generator with independent random number seed*/
        struct Random{
            /*! \brief set random number seed */
            inline void Seed( unsigned sd ){
                this->rseed = sd;
            }
            /*! \brief return a real number uniform in [0,1) */
            inline double RandDouble( void ){               
                return static_cast<double>( rand_r( &rseed ) ) / (static_cast<double>( RAND_MAX )+1.0);
            }
            // random number seed
            unsigned rseed;
        };
    };
};

#endif
