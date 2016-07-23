#pragma once
#include <cub/cub.cuh>
#include <xgboost/base.h>
#include "gpu_gpair.cuh"
#include "cuda_helpers.cuh"

namespace xgboost{
	namespace tree{
		
		__device__ DeviceTimerGlobal timer_global;

		struct Node;

		typedef uint32_t BitFlagSet;

		__device__ __inline__ void set_bit(BitFlagSet &bf, int index)
		{
			bf |= 1 << index;
		}

		__device__ __inline__ bool check_bit(BitFlagSet bf, int index)
		{
			return (bf >> index) & 1;
		}

        // Carryover prefix for scanning multiple tiles of bit flags
		struct FlagPrefixCallbackOp
		{
			BitFlagSet tile_carry;

			__device__ FlagPrefixCallbackOp() : tile_carry(0) {}

			__device__ BitFlagSet operator()(BitFlagSet block_aggregate){
				BitFlagSet old_prefix = tile_carry;
				tile_carry |= block_aggregate;
				return old_prefix;
			}
		};

        //Scan op for bit flags that resets if the final bit is set
		struct FlagScanOp
		{
			__device__ __forceinline__ BitFlagSet operator()(const BitFlagSet &a, const BitFlagSet &b){
				if (check_bit(b, 31))
				{
					return b;
				}
				else
				{
					return a | b;
				}
			}
		};

		struct GPUTrainingParam
		{
			
		  // minimum amount of hessian(weight) allowed in a child
		  float min_child_weight;
		  // L2 regularization factor
		  float reg_lambda;
		  // L1 regularization factor
		  float reg_alpha;
		  // maximum delta update we can add in weight estimation
		  // this parameter can be used to stabilize update
		  // default=0 means no constraint on weight delta
		  float max_delta_step;
		  
		  __host__ __device__ GPUTrainingParam() {}
		 
		  __host__ __device__ GPUTrainingParam(float min_child_weight_in, float reg_lambda_in, float reg_alpha_in, float max_delta_step_in) : 
			  min_child_weight(min_child_weight_in),
			  reg_lambda(reg_lambda_in),
			  reg_alpha(reg_alpha_in),
			  max_delta_step(max_delta_step_in)
		  {}
		};


		// functions for L1 cost
		__host__ __device__ float device_threshold_L1(float w, float lambda)
		{
			if (w > +lambda) return w - lambda;
			if (w < -lambda) return w + lambda;
			return 0.0f;
		}

		__host__ __device__ float device_sqr(float a)
		{
			return a * a;
		}

		// calculate weight given the statistics
		template <typename GpairT>
		__host__ __device__  float device_calc_weight(const GPUTrainingParam& p, const GpairT& sum)
		{
			if (sum.hess() < p.min_child_weight) return 0.0;
			float dw;
			if (p.reg_alpha == 0.0f)
			{
				dw = -sum.grad() / (sum.hess() + p.reg_lambda);
			}
			else
			{
				dw = -device_threshold_L1(sum.grad(), p.reg_alpha) / (sum.hess() + p.reg_lambda);
			}
			if (p.max_delta_step != 0.0f)
			{
				if (dw > p.max_delta_step) dw = p.max_delta_step;
				if (dw < -p.max_delta_step) dw = -p.max_delta_step;
			}
			return dw;
		}

		template <typename GpairT>
		__host__ __device__ float device_calc_gain(const GPUTrainingParam& p, const GpairT& sum)
		{
			if (sum.hess() < p.min_child_weight) return 0.0f;
			if (p.max_delta_step == 0.0f)
			{
				if (p.reg_alpha == 0.0f)
				{
					return device_sqr(sum.grad()) / (sum.hess() + p.reg_lambda);
				}
				else
				{
					return device_sqr(device_threshold_L1(sum.grad(), p.reg_alpha)) / (sum.hess() + p.reg_lambda);
				}
			}
			else
			{
				float w = device_calc_weight(p, sum);
				float ret = sum.grad() * w + 0.5f * (sum.hess() + p.reg_lambda) * device_sqr(w);
				if (p.reg_alpha == 0.0f)
				{
					return - 2.0f * ret;
				}
				else
				{
					return - 2.0f * (ret + p.reg_alpha * fabsf(w));
				}
			}
		}

        // Holds information about node split
		struct Split
		{
			float loss_chg;
			bool missing_left;
			float fvalue;
			int findex;
			gpu_gpair left_sum;
			gpu_gpair right_sum;

			__host__ __device__ Split() : loss_chg(-FLT_MAX), missing_left(true), fvalue(0){}

			__device__ void Update(float loss_chg_in, bool missing_left_in, float fvalue_in, int findex_in, gpu_gpair left_sum_in, gpu_gpair right_sum_in, const GPUTrainingParam &param) 			
            {
				if (loss_chg_in > loss_chg && left_sum_in.hess() > param.min_child_weight && right_sum_in.hess() > param.min_child_weight)
				{
					loss_chg = loss_chg_in;
					missing_left = missing_left_in;
					fvalue = fvalue_in;
					left_sum = left_sum_in;
					right_sum = right_sum_in;
					findex = findex_in;
				}
			}

			//Does not check minimum weight
			__device__ void Update(Split &s) 
			{
				if (s.loss_chg > loss_chg)
				{
					loss_chg = s.loss_chg;
					missing_left = s.missing_left;
					fvalue = s.fvalue;
					findex = s.findex;
					left_sum = s.left_sum;
					right_sum = s.right_sum;
				}
			}

			__device__ void Print()
			{
					printf("Loss: %1.4f\n", loss_chg);
					printf("Missing left: %d\n", missing_left);
					printf("fvalue: %1.4f\n", fvalue);
					printf("Left sum: ");
					left_sum.print();

					printf("Right sum: ");
					right_sum.print();
			}
		};

		struct split_reduce_op
		{
			template <typename T>
			__device__ __forceinline__ T operator()(T &a, T b){
				b.Update(a);
				return b;
			}
		};

		struct Node
		{

			gpu_gpair sum_gradients;
			float root_gain;
			float weight;

			Split split;

			__host__ __device__ Node(): weight(0), root_gain(0) {}

			__host__ __device__ Node(gpu_gpair sum_gradients_in, const GPUTrainingParam& param)
			{
				sum_gradients = sum_gradients_in;
				CalcGain(param);
				CalcWeight(param);
			}

			__host__ __device__ void CalcGain(const  GPUTrainingParam& param)
			{
				root_gain = device_calc_gain(param, sum_gradients);
			}

			__host__ __device__ void CalcWeight(const GPUTrainingParam& param)
			{
				weight = device_calc_weight(param, sum_gradients);
			}

			__host__ __device__ bool IsLeaf()
			{
				return split.loss_chg == -FLT_MAX;
			}
		};

		__device__ float device_calc_loss_chg(const  GPUTrainingParam& param, const gpu_gpair& scan, const gpu_gpair& missing, const gpu_gpair& parent_sum, const float& parent_gain, bool missing_left)
		{
			gpu_gpair left = scan;

			if (missing_left)
			{
				left += missing;
			}

			gpu_gpair right = parent_sum - left;

			float left_gain = device_calc_gain(param, left);
			float right_gain = device_calc_gain(param, right);
			return left_gain + right_gain - parent_gain;
		}

		__device__ float loss_chg_missing(const gpu_gpair &scan, const gpu_gpair &missing, const gpu_gpair& parent_sum, const float& parent_gain, const GPUTrainingParam&param, bool &missing_left_out)
		{
			float missing_left_loss = device_calc_loss_chg(param, scan, missing, parent_sum, parent_gain, true);
			float missing_right_loss = device_calc_loss_chg(param, scan, missing, parent_sum, parent_gain, false);
			if (missing_left_loss >= missing_right_loss)
			{
				missing_left_out = true;
				return missing_left_loss;
			}
			else
			{
				missing_left_out = false;
				return missing_right_loss;
			}
		}

		template<
			int _BLOCK_THREADS,
			int _N_NODES,
			bool _DEBUG_VALIDATE
		>
		struct FindSplitParams
		{
			enum
			{
				BLOCK_THREADS = _BLOCK_THREADS,
				TILE_ITEMS = BLOCK_THREADS,
				N_NODES = _N_NODES,
				N_WARPS = _BLOCK_THREADS / 32,
				DEBUG_VALIDATE = _DEBUG_VALIDATE,
				ITEMS_PER_THREAD = 1
			};
		};

		template<
			int _BLOCK_THREADS,
			int _ITEMS_PER_THREAD,
			int _N_NODES,
			bool _DEBUG_VALIDATE
		>
		struct ReduceParams
		{
			enum
			{
				BLOCK_THREADS = _BLOCK_THREADS,
				ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
				TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
				N_NODES = _N_NODES,
				N_WARPS = _BLOCK_THREADS / 32,
				DEBUG_VALIDATE = _DEBUG_VALIDATE
			};
		};

		template<
			typename ParamsT,
			typename GpairT,
			typename NodeIdIterT,
			typename OffsetT
		>
		struct ReduceEnactor
		{
			//typedef cub::BlockReduce<GpairT, ParamsT::BLOCK_THREADS> BlockReduceT;
			typedef cub::WarpReduce<GpairT> WarpReduceT;

			struct  _TempStorage
			{
				typename WarpReduceT::TempStorage warp_reduce[ParamsT::N_WARPS];
				GpairT partial_sums[ParamsT::N_NODES][ParamsT::N_WARPS];
			};

			struct TempStorage : cub::Uninitialized<_TempStorage> {};

			struct _Reduction
			{
				GpairT node_sums[ParamsT::N_NODES];
			};

			struct Reduction : cub::Uninitialized <_Reduction> {};

			//Thread local member variables
			const GpairT*d_gpair_in;
			const NodeIdIterT node_id;
			_TempStorage &temp_storage;
			_Reduction &reduction;
			GpairT gpair_thread_data[ParamsT::ITEMS_PER_THREAD];
			int8_t nodeid_thread_data[ParamsT::ITEMS_PER_THREAD];
			const int node_begin;

			__device__ __forceinline__ ReduceEnactor(
				TempStorage& temp_storage, 
				Reduction& reduction,
				const GpairT*d_gpair_in,
				const NodeIdIterT node_id,
				const int node_begin
				) 
			: 
				temp_storage(temp_storage.Alias()), 
				reduction(reduction.Alias()),
				d_gpair_in(d_gpair_in),
				node_id(node_id),
				node_begin(node_begin)
			{}

			__device__ __forceinline__ void ResetPartials()
			{
				if (threadIdx.x < ParamsT::N_WARPS)
				{
					for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){
						temp_storage.partial_sums[NODE][threadIdx.x] = GpairT();
					}
				}
			}

			__device__ __forceinline__ void ResetReduction()
			{
				if (threadIdx.x < ParamsT::N_NODES)
				{
					reduction.node_sums[threadIdx.x] = GpairT();
				}
			}

			template<bool IS_FULL_TILE>
			__device__ __forceinline__ void LoadTile(const OffsetT &offset, const OffsetT &num_remaining)
			{
				if (IS_FULL_TILE){
					cub::LoadDirectStriped<ParamsT::BLOCK_THREADS>(threadIdx.x, d_gpair_in + offset, gpair_thread_data);
					cub::LoadDirectStriped<ParamsT::BLOCK_THREADS>(threadIdx.x, node_id + offset, nodeid_thread_data);
				}
				else{
					cub::LoadDirectStriped<ParamsT::BLOCK_THREADS>(threadIdx.x, d_gpair_in + offset, gpair_thread_data, num_remaining, GpairT());
					cub::LoadDirectStriped<ParamsT::BLOCK_THREADS>(threadIdx.x, node_id + offset, nodeid_thread_data, num_remaining, (int8_t)0);
				}
				__syncthreads();
			}

			template<bool IS_FULL_TILE>
			__device__ __forceinline__ void ProcessTile(const OffsetT &offset, const OffsetT &num_remaining)
			{
				LoadTile<IS_FULL_TILE>(offset, num_remaining);

				//Warp synchronous reduction
				for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){

					for (int ITEM = 0; ITEM < ParamsT::ITEMS_PER_THREAD; ITEM++)
					{
						int8_t nodeid_adjusted = nodeid_thread_data[ITEM] - node_begin;

						bool active = nodeid_adjusted == NODE;

						unsigned int ballot = __ballot(active);

						int warp_id = threadIdx.x / 32;
						int lane_id = threadIdx.x % 32;

						if (ballot == 0)
						{
							continue;
						}
						else if(__popc(ballot) == 1)
						{
							if (active)
							{
								temp_storage.partial_sums[NODE][warp_id] += gpair_thread_data[ITEM];
							}
						}
						else
						{
							GpairT sum = WarpReduceT(temp_storage.warp_reduce[warp_id]).Sum(active ? gpair_thread_data[ITEM] : GpairT());
							if (lane_id == 0)
							{
								temp_storage.partial_sums[NODE][warp_id] += sum;
							}
						}
							
					}

				}
			}

			__device__ __forceinline__ void ReducePartials()
			{
				//Use single warp to reduce partials
				if (threadIdx.x < 32){
					for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){
						GpairT sum = WarpReduceT(temp_storage.warp_reduce[0]).Sum(threadIdx.x < ParamsT::N_WARPS ? temp_storage.partial_sums[NODE][threadIdx.x] : GpairT());

						if (threadIdx.x == 0)
						{
							reduction.node_sums[NODE] = sum;
						}
					}
				}
			}

			//Compare result against single threaded result
			__device__ void DebugValidate(OffsetT begin, OffsetT end)
			{
				if (threadIdx.x > 0)
				{
					return;
				}

				GpairT single_thread_reduction[ParamsT::N_NODES];

				for (OffsetT i = begin; i < end; i++)
				{
					GpairT gpair = d_gpair_in[i];
					int8_t nodeid_adjusted = node_id[i] - node_begin;

					if (nodeid_adjusted < ParamsT::N_NODES)
					{
						single_thread_reduction[nodeid_adjusted] += gpair;
					}

				}

				for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++)
				{
					if (!single_thread_reduction[NODE].approximate_compare(reduction.node_sums[NODE], 1, 1))
					{
						printf("Block %d single thread reduction:\n", blockIdx.x);
						single_thread_reduction[NODE].print();
						printf("Block %d multi-thread reduction:\n", blockIdx.x);
						reduction.node_sums[NODE].print();
					}
				}

			}
			__device__ __forceinline__ void ProcessRegion(const OffsetT &segment_begin, const OffsetT &segment_end)
			{
				//Current position
				OffsetT offset = segment_begin;

				ResetReduction();
				ResetPartials();

				__syncthreads();

				//Process full tiles
				while (offset + ParamsT::TILE_ITEMS <= segment_end)
				{
					ProcessTile<true>(offset, segment_end - offset);
					offset += ParamsT::TILE_ITEMS;
				}

				//Process partial tile if exists
				ProcessTile<false>(offset, segment_end - offset);

				__syncthreads();

				ReducePartials();

				__syncthreads();

			}
		};

		template<
			typename ParamsT,
			typename GpairT,
			typename NodeIdIterT,
			typename OffsetT,
			typename ReductionT
		>
		struct FindSplitEnactor
		{
			typedef cub::BlockScan<BitFlagSet, ParamsT::BLOCK_THREADS> FlagsBlockScanT;

			typedef cub::WarpReduce<Split> WarpSplitReduceT;

			typedef cub::WarpReduce<float> WarpReduceT;

			typedef cub::WarpScan<GpairT> WarpScanT;

			struct  _TempStorage
			{
				union{
					typename WarpSplitReduceT::TempStorage warp_split_reduce;
					typename FlagsBlockScanT::TempStorage flags_scan;
					typename WarpScanT::TempStorage warp_gpair_scan[ParamsT::N_WARPS];
					typename WarpReduceT::TempStorage warp_reduce[ParamsT::N_WARPS]; 
				};

				Split warp_best_splits[ParamsT::N_NODES][ParamsT::N_WARPS];
				GpairT partial_sums[ParamsT::N_NODES][ParamsT::N_WARPS];
				GpairT top_level_sum[ParamsT::N_NODES]; //Sum of current partial sums
				GpairT tile_carry[ParamsT::N_NODES]; //Contains top-level sums from previous tiles
				Split best_splits[ParamsT::N_NODES];
				//Cache current level nodes into shared memory
				float node_root_gain[ParamsT::N_NODES];
				GpairT node_parent_sum[ParamsT::N_NODES];
			};

			struct TempStorage : cub::Uninitialized<_TempStorage> {};

			//Thread local member variables
			const GpairT*d_gpair_in;
			Split* d_split_candidates_out;
			const float *d_fvalues;
			const NodeIdIterT node_id_iter;
			const Node*d_nodes;
			_TempStorage &temp_storage;
			GpairT gpair_thread;
			int8_t node_id_thread;
			float fvalue_thread;
			const int8_t node_begin;
			const GPUTrainingParam& param;
			const ReductionT &reduction;
			const int level;
			const int iteration;
			FlagPrefixCallbackOp flag_prefix_op;

			__device__ __forceinline__ FindSplitEnactor(
				TempStorage& temp_storage, 
				const GpairT*d_gpair_in,
				Split *d_split_candidates_out,
				const float *d_fvalues,
				const NodeIdIterT node_id_iter,
				const Node *d_nodes,
				const int8_t node_begin,
				const GPUTrainingParam&param,
				const ReductionT reduction,
				const int level,
				const int iteration
				) 
			: 
				temp_storage(temp_storage.Alias()), 
				d_gpair_in(d_gpair_in),
				d_split_candidates_out(d_split_candidates_out),
				d_fvalues(d_fvalues),
				node_id_iter(node_id_iter),
				d_nodes(d_nodes),
				node_begin(node_begin),
				param(param),
				reduction(reduction),
				level(level),
				iteration(iteration),
				flag_prefix_op()
			{}
			
			__device__ __forceinline__ void UpdateTileCarry()
			{
				if (threadIdx.x < ParamsT::N_NODES)
				{
					temp_storage.tile_carry[threadIdx.x] += temp_storage.top_level_sum[threadIdx.x];
				}
			}

			__device__ __forceinline__ void ResetTileCarry()
			{
				if (threadIdx.x < ParamsT::N_NODES)
				{
					temp_storage.tile_carry[threadIdx.x] = GpairT();
				}
			}

			__device__ __forceinline__ void ResetPartials()
			{
				if (threadIdx.x < ParamsT::N_WARPS){
					for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++)
					{
						temp_storage.partial_sums[NODE][threadIdx.x] = GpairT();
					}
				}

				if (threadIdx.x < ParamsT::N_NODES)
				{
					temp_storage.top_level_sum[threadIdx.x] = GpairT();
				}
			}

			__device__ __forceinline__ void ResetSplits()
			{
				if (threadIdx.x < ParamsT::N_WARPS){
					for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++)
					{
						temp_storage.warp_best_splits[NODE][threadIdx.x] = Split();
					}
				}

				if (threadIdx.x < ParamsT::N_NODES)
				{
					temp_storage.best_splits[threadIdx.x] = Split();
				}
			}

			//Cache d_nodes array for this level into shared memory
			__device__ __forceinline__ void CacheNodes()
			{
				//Get pointer to nodes on the current level
				const Node* d_nodes_level = d_nodes + node_begin;

				if (threadIdx.x < ParamsT::N_NODES)
				{
					temp_storage.node_root_gain[threadIdx.x] = d_nodes_level[threadIdx.x].root_gain;
					temp_storage.node_parent_sum[threadIdx.x] = d_nodes_level[threadIdx.x].sum_gradients;
				}

			}

			template<bool IS_FULL_TILE>
			__device__ __forceinline__ void LoadTile(OffsetT offset, OffsetT num_remaining)
			{
				OffsetT index = offset + threadIdx.x;
				if (IS_FULL_TILE || threadIdx.x < num_remaining){
					gpair_thread = d_gpair_in[index];
					node_id_thread = node_id_iter[index];
					fvalue_thread = d_fvalues[index];
				}
				else{

					gpair_thread = GpairT();
					node_id_thread = -1;
					fvalue_thread = -FLT_MAX;
				}
			}

			//Is this node being processed by current kernel iteration?
			__device__ __forceinline__ bool NodeActive(const int8_t nodeid_adjusted)
			{
				return nodeid_adjusted < ParamsT::N_NODES && nodeid_adjusted >= 0;
			}

			//Is current fvalue different from left fvalue
			__device__ __forceinline__ bool LeftMostFvalue(const OffsetT &segment_begin, const OffsetT &offset, const OffsetT &num_remaining)
			{

				int left_index = offset + threadIdx.x - 1;
				float left_fvalue = left_index >= segment_begin && threadIdx.x < num_remaining ? d_fvalues[left_index] : -FLT_MAX;

				return left_fvalue != fvalue_thread;

			}

			//Prevent splitting in the middle of same valued instances
			__device__ __forceinline__ bool CheckSplitValid(const OffsetT &segment_begin, const OffsetT &offset, const OffsetT &num_remaining)
			{
				BitFlagSet bit_flag = 0;

				bool valid_split;

				if (LeftMostFvalue(segment_begin, offset, num_remaining))
				{
					valid_split = true;
					//Use MSB bit to flag if fvalue is leftmost
					set_bit(bit_flag, 31);
				}
				else
				{
					valid_split = false;
				}

				//Flag nodeid
				int8_t nodeid_adjusted = node_id_thread - node_begin;
				if (NodeActive(nodeid_adjusted)){
					set_bit(bit_flag, nodeid_adjusted);
				}

				BitFlagSet block_aggregate;
				FlagsBlockScanT(temp_storage.flags_scan).ExclusiveScan(bit_flag, bit_flag, BitFlagSet(), FlagScanOp(), block_aggregate, flag_prefix_op);
				__syncthreads();

				if (!valid_split && NodeActive(nodeid_adjusted))
				{
					if (!check_bit(bit_flag, nodeid_adjusted))
					{
						valid_split = true;
					}
				}

				return valid_split;
			}

			//Perform warp reduction to find if this lane contains best loss_chg in warp
			__device__ __forceinline__ bool QueryLaneBestLoss(const float &loss_chg){
				int lane_id = threadIdx.x % 32;
				int warp_id = threadIdx.x / 32;
				
				//Possible source of bugs. Not all threads in warp are active here. Not sure if reduce function will behave correctly
				float best = WarpReduceT(temp_storage.warp_reduce[warp_id]).Reduce(loss_chg, cub::Max());

				//Its possible for more than one lane to contain the best value, so make sure only one lane returns true
				unsigned int ballot = __ballot(loss_chg == best);

				if (lane_id == (__ffs(ballot) - 1))
				{
					return true;
				}
				else
				{
					return false;
				}
			}

			//Which thread in this warp should update the current best split, if any
			//Returns true for one thread or none 
			__device__ __forceinline__ bool QueryUpdateWarpSplit(const float &loss_chg, const volatile float &warp_best_loss, const int8_t &nodeid_adjusted)
			{
				bool update = false;

				for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){

					bool active = nodeid_adjusted == NODE;

					unsigned int ballot = __ballot(loss_chg > warp_best_loss && active);

					//No lane has improved loss_chg
					if (__popc(ballot) == 0)
					{
						 continue;
					}
					//A single lane has improved loss_chg, set true for this lane
					else if (__popc(ballot) == 1)
					{
						int lane_id = threadIdx.x % 32;

						if (lane_id == __ffs(ballot) - 1)
						{
							update =  true;
						}
					}
					//More than one lane has improved loss_chg, perform a reduction. 
					else
					{
						if (QueryLaneBestLoss(active ? loss_chg : -FLT_MAX))
						{
							update = true;
						}
					}

				}

				return update;
			}

            /*
            __device__  void PrintTile(int block) {

                if (blockIdx.x != block) {
                    return;
                }

                for (int WARP = 0; WARP < ParamsT::N_WARPS; WARP++) {
                    int warp_id = threadIdx.x / 32;

                    if (warp_id == WARP) {
                        for (int i = 0; i < 32; i++) {
                            GpairT g = cub::ShuffleIdx(gpair_thread, i);

                            if (cub::LaneId() == 0) {
                                g.print();
                            }
                        }
                    }

                    __syncthreads();
                }
			}
            */

			__device__ __forceinline__ void EvaluateSplits(const OffsetT &segment_begin, const OffsetT &offset, const OffsetT &num_remaining)
			{
				bool valid_split = CheckSplitValid(segment_begin, offset, num_remaining);

				const int8_t nodeid_adjusted = node_id_thread - node_begin;

				const int warp_id = threadIdx.x / 32;


				if (NodeActive(nodeid_adjusted) && valid_split && threadIdx.x < num_remaining){

					GpairT parent_sum = temp_storage.node_parent_sum[nodeid_adjusted];
					float parent_gain = temp_storage.node_root_gain[nodeid_adjusted];
					GpairT missing = parent_sum - reduction.node_sums[nodeid_adjusted];

					bool missing_left;

					float loss_chg = loss_chg_missing(gpair_thread, missing, parent_sum, parent_gain, param, missing_left);

					if(QueryUpdateWarpSplit(loss_chg, temp_storage.warp_best_splits[nodeid_adjusted][warp_id].loss_chg, nodeid_adjusted))
					{
						if (missing_left){
							GpairT left_sum = missing + gpair_thread;
							GpairT right_sum = parent_sum - left_sum;
							temp_storage.warp_best_splits[nodeid_adjusted][warp_id].Update(loss_chg, missing_left, fvalue_thread, blockIdx.x, left_sum, right_sum, param);
						}
						else
						{
							GpairT left_sum = gpair_thread;
							GpairT right_sum = parent_sum - left_sum;
							temp_storage.warp_best_splits[nodeid_adjusted][warp_id].Update(loss_chg, missing_left, fvalue_thread, blockIdx.x, left_sum, right_sum, param);
						}
					}
				}
				
			}

            /*
			__device__ __forceinline__ void WarpExclusiveScan(bool active, GpairT input, GpairT &output, GpairT &sum)
			{

				output = input;

				for (int offset = 1; offset < 32; offset <<= 1){
					float tmp1 = __shfl_up(output.grad(), offset);

					float tmp2 = __shfl_up(output.hess(), offset);
					if (cub::LaneId() >= offset)
					{
						output.grad += tmp1;
						output.hess += tmp2;
					}
				}

				sum.grad = __shfl(output.grad, 31);
				sum.hess = __shfl(output.hess, 31);

				output -= input;
			}
            */

			__device__ __forceinline__ void BlockExclusiveScan()
			{
				ResetPartials();

				__syncthreads();
				int warp_id = threadIdx.x / 32;
				int lane_id = threadIdx.x % 32;

				int8_t nodeid_adjusted = node_id_thread - node_begin;

				for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){

					bool node_active = nodeid_adjusted == NODE;

					unsigned int ballot = __ballot(node_active);

					GpairT warp_sum = GpairT();
					GpairT scan_result = GpairT();

					if (ballot > 0){
						WarpScanT(temp_storage.warp_gpair_scan[warp_id]).ExclusiveSum(node_active ? gpair_thread : GpairT(), scan_result, warp_sum);
						//WarpExclusiveScan( node_active, node_active ? gpair_thread : GpairT(), scan_result, warp_sum);
					}

					if (node_active)
					{
						gpair_thread = scan_result;
					}

					if (lane_id == 0){
						temp_storage.partial_sums[NODE][warp_id] = warp_sum;
					}
				}

				__syncthreads();

				if (threadIdx.x < 32)
				{
					for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){
						GpairT top_level_sum;
						bool warp_active = threadIdx.x < ParamsT::N_WARPS;
						GpairT scan_result;
						WarpScanT(temp_storage.warp_gpair_scan[warp_id]).ExclusiveSum(warp_active ? temp_storage.partial_sums[NODE][threadIdx.x] : GpairT(), 
							scan_result, 
							top_level_sum);

						if (warp_active)
						{
							temp_storage.partial_sums[NODE][threadIdx.x] = scan_result;
						}

						if (threadIdx.x == 0){
							temp_storage.top_level_sum[NODE] = top_level_sum;
						}
					}
				}

				__syncthreads();

				if (NodeActive(nodeid_adjusted)){
					gpair_thread += temp_storage.partial_sums[nodeid_adjusted][warp_id] + temp_storage.tile_carry[nodeid_adjusted];
				}

				__syncthreads();

				UpdateTileCarry();

				__syncthreads();


			}

			//Perform a full scan for this tile
			template<bool IS_FULL_TILE>
			__device__ __forceinline__ void ProcessTile(const OffsetT &segment_begin, const OffsetT &offset, const OffsetT &num_remaining)
			{
				LoadTile<IS_FULL_TILE>(offset, num_remaining);
				DeviceTimer t0(timer_global, 0);
				BlockExclusiveScan();
				t0.End();
				DeviceTimer t1(timer_global, 1);

				EvaluateSplits(segment_begin, offset, num_remaining);

				t1.End();
			}

			__device__ __forceinline__ void ReduceSplits()
			{

				for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){
					if (threadIdx.x < 32){
						Split s = Split();
						if (threadIdx.x < ParamsT::N_WARPS){
							s = temp_storage.warp_best_splits[NODE][threadIdx.x];
						}
						Split best = WarpSplitReduceT(temp_storage.warp_split_reduce).Reduce(s, split_reduce_op());
						if (threadIdx.x == 0)
						{
							temp_storage.best_splits[NODE] = best;
						}
					}
				}
			}

			__device__ __forceinline__ void WriteBestSplits()
			{
				const int nodes_level = 1 << level;

				if (threadIdx.x < ParamsT::N_NODES)
				{
					d_split_candidates_out[blockIdx.x * nodes_level + (iteration * ParamsT::N_NODES) + threadIdx.x] = temp_storage.best_splits[threadIdx.x];
				}
			}

			__device__ void SequentialAlgorithm(OffsetT segment_begin, OffsetT segment_end)
			{
				if (threadIdx.x != 0)
				{
					return;
				}

				__shared__ Split best_split[ParamsT::N_NODES];

				__shared__ GpairT scan[ParamsT::N_NODES];

				__shared__ Node nodes[ParamsT::N_NODES];

				__shared__ GpairT missing[ParamsT::N_NODES];

				float previous_fvalue[ParamsT::N_NODES];

				//Initialise counts
				for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){
					best_split[NODE] = Split();
					scan[NODE] = GpairT();
					nodes[NODE] = d_nodes[node_begin + NODE];
					missing[NODE] = nodes[NODE].sum_gradients - reduction.node_sums[NODE];
					previous_fvalue[NODE] = FLT_MAX;
				}

				for (OffsetT i = segment_begin; i < segment_end; i++)
				{
					int8_t nodeid_adjusted = node_id_iter[i] - node_begin;
					float fvalue = d_fvalues[i];

					if (NodeActive(nodeid_adjusted)){

						if (fvalue != previous_fvalue[nodeid_adjusted]){
							float f_split;
							if (previous_fvalue[nodeid_adjusted] != FLT_MAX){
								f_split = (previous_fvalue[nodeid_adjusted] + fvalue)  * 0.5;
							}
							else
							{
								f_split = fvalue;
							}

							best_split[nodeid_adjusted].UpdateCalcLoss(f_split, scan[nodeid_adjusted], missing[nodeid_adjusted], nodes[nodeid_adjusted], param);
						}

						scan[nodeid_adjusted] += d_gpair_in[i];
						previous_fvalue[nodeid_adjusted] = fvalue;

					}

				}


				for (int NODE = 0; NODE < ParamsT::N_NODES; NODE++){
					temp_storage.best_splits[NODE] = best_split[NODE];
				}

			}

			__device__ __forceinline__ void ProcessRegion(const OffsetT &segment_begin, const OffsetT &segment_end)
			{
				//Current position
				OffsetT offset = segment_begin;

				ResetTileCarry();
				ResetSplits();
				CacheNodes();
				__syncthreads();

				//Process full tiles
				while (offset + ParamsT::TILE_ITEMS <= segment_end)
				{
					ProcessTile<true>(segment_begin, offset, segment_end - offset);
					__syncthreads();
					offset += ParamsT::TILE_ITEMS;
				}

				//Process partial tile if exists
				ProcessTile<false>(segment_begin, offset, segment_end - offset);

				__syncthreads();
				ReduceSplits();

				__syncthreads();
				WriteBestSplits();
			}
		};

		template <
			typename FindSplitParamsT,
			typename ReduceParamsT,
			typename GpairT, 
			typename NodeIdIterT, 
			typename OffsetT
		>
		__global__ void  
#if __CUDA_ARCH__ <= 530
		__launch_bounds__(1024, 2)
#endif
		find_split_candidates_kernel(const GpairT* d_gpair_in,  Split* d_split_candidates_out, const float* d_fvalues, const NodeIdIterT node_id, const Node *d_nodes, const int node_begin, xgboost::bst_uint num_items, int num_features, const OffsetT* d_feature_offsets, const GPUTrainingParam param, const int level, const int iteration)
		{

			timer_global.Init();

			if (num_items <= 0)
			{
				return;
			}


			OffsetT segment_begin = d_feature_offsets[blockIdx.x];
			OffsetT segment_end = d_feature_offsets[blockIdx.x + 1];

			typedef ReduceEnactor<ReduceParamsT, GpairT, NodeIdIterT, OffsetT> ReduceT;
			typedef FindSplitEnactor<FindSplitParamsT, GpairT, NodeIdIterT, OffsetT, typename ReduceT::_Reduction> FindSplitT;

			__shared__ union
			{
				typename ReduceT::TempStorage reduce;
				typename FindSplitT::TempStorage find_split;
			} temp_storage;

			__shared__  typename ReduceT::Reduction reduction;

			ReduceT(temp_storage.reduce, reduction, d_gpair_in, node_id, node_begin).ProcessRegion(segment_begin, segment_end);
			__syncthreads();

			FindSplitT find_split(temp_storage.find_split, d_gpair_in, d_split_candidates_out, d_fvalues, node_id, d_nodes, node_begin, param, reduction.Alias(), level, iteration);
			find_split.ProcessRegion(segment_begin, segment_end);

		}


		__global__ void reduce_split_candidates_kernel(Split* d_split_candidates, Node *d_current_nodes, Node *d_new_nodes, int n_current_nodes, int n_features, const GPUTrainingParam param)
		{
			int nid = threadIdx.x;

			if (nid >= n_current_nodes)
			{
				return;
			}

			//Find best split for each node
			Split best;

			for (int i = 0; i < n_features; i++)
			{
				best.Update(d_split_candidates[n_current_nodes * i + nid]);
			}

			//Update current node
			d_current_nodes[nid].split = best;

			//Generate new nodes
			d_new_nodes[nid * 2] = Node(best.left_sum, param);
			d_new_nodes[nid * 2 + 1] = Node(best.right_sum, param);

		}

		void reduce_split_candidates(Split* d_split_candidates, Node *d_nodes, int level , int n_features, const GPUTrainingParam param)
		{
			Node *d_current_nodes = d_nodes + (1 << (level)) - 1; //Current level nodes (before split)
			Node *d_new_nodes = d_nodes + (1 << (level + 1)) - 1; //Next level nodes (after split)
			int n_current_nodes = 1 << level; //Number of existing nodes on this level

			int block_threads = n_current_nodes;

			reduce_split_candidates_kernel << <1, block_threads >> >(d_split_candidates, d_current_nodes, d_new_nodes, n_current_nodes, n_features, param);
			safe_cuda(cudaDeviceSynchronize());
		}

		template <int N_NODES, typename GpairT, typename NodeIdIterT, typename OffsetT>
		void find_split_candidates(const GpairT* d_gpair_in,  Split* d_split_candidates_out, const float* d_fvalues, const NodeIdIterT node_id, const Node *d_nodes, int node_begin, int node_end, OffsetT num_items, int num_features, const OffsetT* d_feature_offsets, const GPUTrainingParam param, const int level, const int iteration){

			const int BLOCK_THREADS = 512;

			CHECK((node_end - node_begin) <= N_NODES) << "Multiscan: N_NODES template parameter is too small for given node range.";
			CHECK(BLOCK_THREADS/32 < 32) << "Too many active warps. See FindSplitEnactor - ReduceSplits.";

			typedef FindSplitParams<BLOCK_THREADS, N_NODES, false> find_split_params;
			typedef ReduceParams<BLOCK_THREADS, 1, N_NODES, false> reduce_params;
			int grid_size = num_features;

			find_split_candidates_kernel<find_split_params, reduce_params> << <grid_size, find_split_params::BLOCK_THREADS >> >(d_gpair_in, d_split_candidates_out, d_fvalues, node_id, d_nodes, node_begin,num_items, num_features, d_feature_offsets, param, level, iteration);

			safe_cuda(cudaDeviceSynchronize());

			timer_global.HostPrint();
		}

		template <typename GpairT, typename NodeIdIterT, typename OffsetT>
		void find_split(const GpairT* d_gpair_in,  Split *d_split_candidates, const float* d_fvalues, const NodeIdIterT node_id, Node *d_nodes, OffsetT num_items, int num_features, const OffsetT* d_feature_offsets, const GPUTrainingParam param, const int level)
		{
			//Select templated variation of split finding algorithm
			if (level == 0){
				find_split_candidates<1>(d_gpair_in, d_split_candidates, d_fvalues, node_id, d_nodes, 0, 1, num_items, num_features, d_feature_offsets, param, level, 0);
			}
			else if (level == 1){
				find_split_candidates<2>(d_gpair_in, d_split_candidates, d_fvalues, node_id, d_nodes, 1, 3, num_items, num_features, d_feature_offsets, param, level, 0);
			}
			else if (level == 2){
				find_split_candidates<4>(d_gpair_in, d_split_candidates, d_fvalues, node_id, d_nodes, 3, 7, num_items, num_features, d_feature_offsets, param, level, 0);
			}
			else if (level == 3){
				find_split_candidates<8>(d_gpair_in, d_split_candidates, d_fvalues, node_id, d_nodes, 7, 15, num_items, num_features, d_feature_offsets, param, level, 0);
			}
			else if (level == 4){
				find_split_candidates<16>(d_gpair_in, d_split_candidates, d_fvalues, node_id, d_nodes, 15, 31, num_items, num_features, d_feature_offsets, param, level, 0);
			}
			else if (level == 5){
				find_split_candidates<32>(d_gpair_in, d_split_candidates, d_fvalues, node_id, d_nodes, 31, 63, num_items, num_features, d_feature_offsets, param, level, 0);
			}

			//Find the best split for each node
			reduce_split_candidates(d_split_candidates, d_nodes, level, num_features, param);
		}
	}
}