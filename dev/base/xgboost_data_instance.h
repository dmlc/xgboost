#ifndef XGBOOST_DATA_INSTANCE_H
#define XGBOOST_DATA_INSTANCE_H

#include <cstdio>
#include <vector>
#include "../booster/xgboost_data.h"
#include "../utils/xgboost_utils.h"
#include "../utils/xgboost_stream.h"


namespace xgboost{
	namespace base{
		/*! \brief data matrix for regression,classification,rank content */
		struct  DMatrix{
		public:
			/*! \brief maximum feature dimension */
			unsigned num_feature;
			/*! \brief feature data content */
			booster::FMatrixS data;
			/*! \brief label of each instance */
			std::vector<float> labels;
			/*! \brief the index of begin and end of a group,
			 * needed when the learning task is ranking*/
			std::vector<int> group_index;
		public:
			/*! \brief default constructor */
			DMatrix(void){}

			/*! \brief get the number of instances */
			inline size_t Size() const{
				return labels.size();
			}
			/*!
			* \brief load from text file
			* \param fname file of instances data
			* \param fgroup file of the group data
			* \param silent whether print information or not
			*/
			inline void LoadText(const char* fname, const char* fgroup, bool silent = false){
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
							labels.push_back(label);
							data.AddRow(findex, fvalue);
						}
						findex.clear(); fvalue.clear();
						utils::Assert(sscanf(tmp, "%f", &label) == 1, "invalid format");
						init = false;
					}
				}

				labels.push_back(label);
				data.AddRow(findex, fvalue);
				// initialize column support as well
				data.InitData();

				if (!silent){
					printf("%ux%u matrix with %lu entries is loaded from %s\n",
						(unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname);
				}
				fclose(file);

				//if exists group data load it in
				FILE *file_group = fopen64(fgroup, "r");
				if (file_group != NULL){
					group_index.push_back(0);
					int tmp = 0, acc = 0;
					while (fscanf(file_group, "%d", tmp) == 1){
						acc += tmp;
						group_index.push_back(acc);
					}
				}
			}
			/*!
			* \brief load from binary file
			* \param fname name of binary data
			* \param silent whether print information or not
			* \return whether loading is success
			*/
			inline bool LoadBinary(const char* fname, const char* fgroup, bool silent = false){
				FILE *fp = fopen64(fname, "rb");
				if (fp == NULL) return false;
				utils::FileStream fs(fp);
				data.LoadBinary(fs);
				labels.resize(data.NumRow());
				utils::Assert(fs.Read(&labels[0], sizeof(float)* data.NumRow()) != 0, "DMatrix LoadBinary");
				fs.Close();
				// initialize column support as well
				data.InitData();

				if (!silent){
					printf("%ux%u matrix with %lu entries is loaded from %s\n",
						(unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname);
				}

				//if group data exists load it in
				FILE *file_group = fopen64(fgroup, "r");
				if (file_group != NULL){
					int group_index_size = 0;
					utils::FileStream group_stream(file_group);
					utils::Assert(group_stream.Read(&group_index_size, sizeof(int)) != 0, "Load group indice size");
					group_index.resize(group_index_size);
					utils::Assert(group_stream.Read(&group_index, sizeof(int)* group_index_size) != 0, "Load group indice");

					if (!silent){
						printf("the group index of %d groups is loaded from %s\n",
							group_index_size - 1, fgroup);
					}
				}
				return true;
			}
			/*!
			* \brief save to binary file
			* \param fname name of binary data
			* \param silent whether print information or not
			*/
			inline void SaveBinary(const char* fname, const char* fgroup, bool silent = false){
				// initialize column support as well
				data.InitData();

				utils::FileStream fs(utils::FopenCheck(fname, "wb"));
				data.SaveBinary(fs);
				fs.Write(&labels[0], sizeof(float)* data.NumRow());
				fs.Close();
				if (!silent){
					printf("%ux%u matrix with %lu entries is saved to %s\n",
						(unsigned)data.NumRow(), (unsigned)data.NumCol(), (unsigned long)data.NumEntry(), fname);
				}

				//save group data
				if (group_index.size() > 0){
					utils::FileStream file_group(utils::FopenCheck(fgroup, "wb"));
					int group_index_size = group_index.size();
					file_group.Write(&(group_index_size), sizeof(int));
					file_group.Write(&group_index[0], sizeof(int) * group_index_size);
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
			inline void CacheLoad(const char *fname, const char *fgroup, bool silent = false, bool savebuffer = true){
				int len = strlen(fname);
				if (len > 8 && !strcmp(fname + len - 7, ".buffer")){
					this->LoadBinary(fname, fgroup, silent); return;
				}
				char bname[1024];
				sprintf(bname, "%s.buffer", fname);
				if (!this->LoadBinary(bname, fgroup, silent)){
					this->LoadText(fname, fgroup, silent);
					if (savebuffer) this->SaveBinary(bname, fgroup, silent);
				}
			}
		private:
			/*! \brief update num_feature info */
			inline void UpdateInfo(void){
				this->num_feature = 0;
				for (size_t i = 0; i < data.NumRow(); i++){
					booster::FMatrixS::Line sp = data[i];
					for (unsigned j = 0; j < sp.len; j++){
						if (num_feature <= sp[j].findex){
							num_feature = sp[j].findex + 1;
						}
					}
				}
			}
		};



	}
};

#endif