#ifndef XGBOOST_CONFIG_H
#define XGBOOST_CONFIG_H
/*!
 * \file xgboost_config.h
 * \brief helper class to load in configures from file
 * \author Tianqi Chen: tianqi.tchen@gmail.com
 */
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstring>
#include <string>
#include "xgboost_utils.h"
#include <vector>

namespace xgboost{
    namespace utils{
        /*!
         * \brief an iterator that iterates over a configure file and gets the configures
         */
        class ConfigIterator{
        public:
            /*!
             * \brief constructor
             * \param fname name of configure file
             */
            ConfigIterator(const char *fname){
                fi = FopenCheck(fname, "r");
                ch_buf = fgetc(fi);
            }
            /*! \brief destructor */
            ~ConfigIterator(){
                fclose(fi);
            }
            /*!
             * \brief get current name, called after Next returns true
             * \return current parameter name
             */
            inline const char *name(void)const{
                return s_name;
            }
            /*!
             * \brief get current value, called after Next returns true
             * \return current parameter value
             */
            inline const char *val(void) const{
                return s_val;
            }
            /*!
             * \brief move iterator to next position
             * \return true if there is value in next position
             */
            inline bool Next(void){
                while (!feof(fi)){
                    GetNextToken(s_name);
                    if (s_name[0] == '=')  return false;
                    if (GetNextToken(s_buf) || s_buf[0] != '=') return false;
                    if (GetNextToken(s_val) || s_val[0] == '=') return false;
                    return true;
                }
                return false;
            }
        private:
            FILE *fi;
            char ch_buf;
            char s_name[256], s_val[256], s_buf[246];

            inline void SkipLine(){
                do{
                    ch_buf = fgetc(fi);
                } while (ch_buf != EOF && ch_buf != '\n' && ch_buf != '\r');
            }

            inline void ParseStr(char tok[]){
                int i = 0;
                while ((ch_buf = fgetc(fi)) != EOF){
                    switch (ch_buf){
                    case '\\': tok[i++] = fgetc(fi); break;
                    case '\"': tok[i++] = '\0';
                        return;
                    case '\r':
                    case '\n': Error("unterminated string"); break;
                    default: tok[i++] = ch_buf;
                    }
                }
                Error("unterminated string");
            }
            // return newline 
            inline bool GetNextToken(char tok[]){
                int i = 0;
                bool new_line = false;
                while (ch_buf != EOF){
                    switch (ch_buf){
                    case '#': SkipLine(); new_line = true; break;
                    case '\"':
                        if (i == 0){
                            ParseStr(tok); ch_buf = fgetc(fi); return new_line;
                        }
                        else{
                            Error("token followed directly by string");
                        }
                    case '=':
                        if (i == 0) {
                            ch_buf = fgetc(fi);
                            tok[0] = '=';
                            tok[1] = '\0';
                        }
                        else{
                            tok[i] = '\0';
                        }
                        return new_line;
                    case '\r':
                    case '\n':
                        if (i == 0) new_line = true;
                    case '\t':
                    case ' ':
                        ch_buf = fgetc(fi);
                        if (i > 0){
                            tok[i] = '\0';
                            return new_line;
                        }
                        break;
                    default:
                        tok[i++] = ch_buf;
                        ch_buf = fgetc(fi);
                        break;
                    }
                }
                return true;
            }
        };
    };

    namespace utils{
        /*!
         * \brief a class that save parameter configurations
         *        temporally and allows to get them out later
         *        there are two kinds of priority in ConfigSaver
         */
        class ConfigSaver{
        public:
            /*! \brief constructor */
            ConfigSaver(void){ idx = 0; }
            /*! \brief clear all saves */
            inline void Clear(void){
                idx = 0;
                names.clear(); values.clear();
                names_high.clear(); values_high.clear();
            }
            /*!
             * \brief push back a parameter setting
             * \param name name of parameter
             * \param val  value of parameter
             * \param priority whether the setting has higher priority: high priority occurs
             *        latter when read from ConfigSaver, and can overwrite existing settings
             */
            inline void PushBack(const char *name, const char *val, int priority = 0){
                if (priority == 0){
                    names.push_back(std::string(name));
                    values.push_back(std::string(val));
                }
                else{
                    names_high.push_back(std::string(name));
                    values_high.push_back(std::string(val));
                }
            }
            /*! \brief set pointer to beginning of the ConfigSaver */
            inline void BeforeFirst(void){
                idx = 0;
            }
            /*!
             * \brief move iterator to next position
             * \return true if there is value in next position
             */
            inline bool Next(void){
                if (idx >= names.size() + names_high.size()){
                    return false;
                }
                idx++;
                return true;
            }
            /*!
             * \brief get current name, called after Next returns true
             * \return current parameter name
             */
            inline const char *name(void) const{
                Assert(idx > 0, "can't call name before first");
                size_t i = idx - 1;
                if (i >= names.size()){
                    return names_high[i - names.size()].c_str();
                }
                else{
                    return names[i].c_str();
                }
            }
            /*!
             * \brief get current value, called after Next returns true
             * \return current parameter value
             */
            inline const char *val(void) const{
                Assert(idx > 0, "can't call name before first");
                size_t i = idx - 1;
                if (i >= values.size()){
                    return values_high[i - values.size()].c_str();
                }
                else{
                    return values[i].c_str();
                }
            }
        private:
            std::vector<std::string> names;
            std::vector<std::string> values;
            std::vector<std::string> names_high;
            std::vector<std::string> values_high;
            size_t idx;
        };
    };
};
#endif

