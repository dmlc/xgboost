#include"xgboost_data.h"
#include<stdio.h>
#include<vector>

using namespace xgboost::booster;
/*!
 * \file xgboost_gbmbase.h
 * \brief A reader to read the data for regression task from a specified file
 *        The data should contain each data instance in each line.
 *		  The format of line data is as below:
 *        label nonzero feature dimension[ feature index:feature value]+
 * \author Kailong Chen: chenkl198812@gmail.com
 */

class xgboost_regression_data_reader{

public:
	xgboost_regression_data_reader(const char* file_path){
		Load(file_path);
	}

	void Load(const char* file_path){
		data_matrix.Clear();
		FILE* file = fopen(file_path,"r");
		if(file == NULL){
			printf("The file is missing at %s",file_path);
			return;
		}
		float label;
		int nonzero_dimension,index,value,num_row = 0;
		std::vector<bst_uint> findex;
		std::vector<bst_float> fvalue;
	
		while(fscanf(file,"%f %i",label,nonzero_dimension)){
			findex.clear();
			fvalue.clear();
			findex.resize(nonzero_dimension);
			fvalue.resize(nonzero_dimension);
			for(int i = 0; i < nonzero_dimension; i++){
				if(!fscanf(file," %i:%f",index,value)){
					printf("The feature dimension is not coincident \
						with the indicated one");
					return;
				}
				findex.push_back(index);
				fvalue.push_back(value);
			}
			data_matrix.AddRow(findex, fvalue);
			labels.push_back(label);
			num_row++;
		}
		printf("%i rows of data is loaded from %s",num_row,file_path);
		fclose(file);
	}

	
	float GetLabel(int index){
		return labels[index];
	}

	FMatrixS::Line GetLine(int index){
		return data_matrix[index];
	}

	int InsNum(){
		return labels.size();
	}

	FMatrixS::Image GetImage(){
		return FMatrixS::Image(data_matrix);
	}

private:
	FMatrixS data_matrix;
	std::vector<float> labels;
};