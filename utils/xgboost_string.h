#ifndef _XGBOOST_STRING_H_
#define _XGBOOST_STRING_H_
#include<vector>
#include<sstream>

namespace xgboost{
	namespace utils{
		class StringProcessing{

		public:
			static std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
				std::stringstream ss(s);
				std::string item;
				while (std::getline(ss, item, delim)) {
					elems.push_back(item);
				}
				return elems;
			}


			static std::vector<std::string> split(const std::string &s, char delim) {
				std::vector<std::string> elems;
				split(s, delim, elems);
				return elems;
			}

		};
	}
}

#endif