#pragma once
#include <xgboost/tree_updater.h>
#include "../../src/tree/param.h"

namespace xgboost {

	namespace tree{

		struct gpu_gpair;
		struct GPUData;

		class GPUBuilder
		{
		public:
			GPUBuilder();
			void Init(const TrainParam& param);
			~GPUBuilder();

			void Update(const std::vector<bst_gpair>& gpair,
				DMatrix* p_fmat,
				RegTree* p_tree);

		private:
			void InitData(const std::vector<bst_gpair>& gpair,
				DMatrix& fmat,
				const RegTree& tree);

			void UpdateNodeId();
			void InitFirstNode();
			void CopyTree(RegTree &tree);

			TrainParam param;
			GPUData *gpu_data;
		};

	}
}