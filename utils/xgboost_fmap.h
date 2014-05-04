#ifndef XGBOOST_FMAP_H
#define XGBOOST_FMAP_H
/*!
 * \file xgboost_fmap.h
 * \brief helper class that holds the feature names and interpretations
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#include <vector>
#include <string>
#include <cstring>
#include "xgboost_utils.h"

namespace xgboost{
    namespace utils{
        /*! \brief helper class that holds the feature names and interpretations */
        class FeatMap{
        public:
            enum Type{
                kIndicator = 0,
                kQuantitive = 1,
                kInteger = 2,
                kFloat = 3
            };
        public:
            /*! \brief load feature map from text format */
            inline void LoadText(const char *fname){
                FILE *fi = utils::FopenCheck(fname, "r");
                this->LoadText(fi);
                fclose(fi);
            }
            /*! \brief load feature map from text format */
            inline void LoadText(FILE *fi){
                int fid;
                char fname[1256], ftype[1256];
                while (fscanf(fi, "%d\t%[^\t]\t%s\n", &fid, fname, ftype) == 3){
                    utils::Assert(fid == (int)names_.size(), "invalid fmap format");
                    names_.push_back(std::string(fname));
                    types_.push_back(GetType(ftype));
                }
            }
            /*! \brief number of known features */
            size_t size(void) const{
                return names_.size();
            }
            /*! \brief return name of specific feature */
            const char* name(size_t idx) const{
                utils::Assert(idx < names_.size(), "utils::FMap::name feature index exceed bound");
                return names_[idx].c_str();
            }
            /*! \brief return type of specific feature */
            const Type& type(size_t idx) const{
                utils::Assert(idx < names_.size(), "utils::FMap::name feature index exceed bound");
                return types_[idx];
            }
        private:
            inline static Type GetType(const char *tname){
                if (!strcmp("i", tname)) return kIndicator;
                if (!strcmp("q", tname)) return kQuantitive;
                if (!strcmp("int", tname)) return kInteger;
                if (!strcmp("float", tname)) return kFloat;
                utils::Error("unknown feature type, use i for indicator and q for quantity");
                return kIndicator;
            }
        private:
            /*! \brief name of the feature */
            std::vector<std::string> names_;
            /*! \brief type of the feature */
            std::vector<Type>        types_;
        };
    }; // namespace utils

    namespace utils{
        /*! \brief feature constraint, allow or disallow some feature during training */
        class FeatConstrain{
        public:
            FeatConstrain(void){
                default_state_ = +1;
            }
            /*!\brief set parameters */
            inline void SetParam(const char *name, const char *val){
                int a, b;
                if (!strcmp(name, "fban")){
                    this->ParseRange(val, a, b);
                    this->SetRange(a, b, -1);
                }
                if (!strcmp(name, "fpass")){
                    this->ParseRange(val, a, b);
                    this->SetRange(a, b, +1);
                }
                if (!strcmp(name, "fdefault")){
                    default_state_ = atoi(val);
                }
            }
            /*! \brief whether constrain is specified */
            inline bool HasConstrain(void) const {
                return state_.size() != 0 && default_state_ == 1;
            }
            /*! \brief whether a feature index is banned or not */
            inline bool NotBanned(unsigned index) const{
                int rt = index < state_.size() ? state_[index] : default_state_;
                if (rt == 0) rt = default_state_;
                return rt == 1;
            }
        private:
            inline void SetRange(int a, int b, int st){
                if (b >(int)state_.size()) state_.resize(b, 0);
                for (int i = a; i < b; ++i){
                    state_[i] = st;
                }
            }
            inline void ParseRange(const char *val, int &a, int &b){
                if (sscanf(val, "%d-%d", &a, &b) == 2) return;
                utils::Assert(sscanf(val, "%d", &a) == 1);
                b = a + 1;
            }
            /*! \brief default state */
            int default_state_;
            /*! \brief whether the state here is, +1:pass, -1: ban, 0:default */
            std::vector<int> state_;
        };
    }; // namespace utils
}; // namespace xgboost
#endif // XGBOOST_FMAP_H
