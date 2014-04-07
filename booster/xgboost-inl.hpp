#ifndef XGBOOST_INL_HPP
#define XGBOOST_INL_HPP
/*!
 * \file xgboost-inl.hpp
 * \brief bootser implementations
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
// implementation of boosters go to here 

// A good design should have minimum functions defined interface, user should only operate on interface
// I break it a bit, by using template and let user 'see' the implementation
// The user should pretend that they only can use the interface, and we are all cool
// I find this is the only way so far I can think of to make boosters invariant of data structure, 
// while keep everything fast
#include "xgboost.h"
#include "../utils/xgboost_utils.h"
#include "tree/xgboost_tree.hpp"
#include "linear/xgboost_linear.hpp"

namespace xgboost{
    namespace booster{
        /*!
         * \brief create a gradient booster, given type of booster
         * \param booster_type type of gradient booster, can be used to specify implements
         * \tparam FMatrix input data type for booster
         * \return the pointer to the gradient booster created
         */
        template<typename FMatrix>
        inline InterfaceBooster<FMatrix> *CreateBooster(int booster_type){
            switch (booster_type){
            case 0: return new RegTreeTrainer<FMatrix>();
            case 1: return new LinearBooster<FMatrix>();
            default: utils::Error("unknown booster_type"); return NULL;
            }
        }
    }; // namespace booster
}; // namespace xgboost

#endif // XGBOOST_INL_HPP
